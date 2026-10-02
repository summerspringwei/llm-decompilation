"""Resumable binary -> contracts -> LLVM IR -> C project campaign.

Coreutils is a validation adapter: source bodies are never model inputs.
Generated functions are linked with a source scaffold for isolation tests.
Coverage is reported explicitly; a passing hybrid is not a fully recovered app.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import networkx as nx
from openai import OpenAI
import requests

from models.binary_decompilation.analyze_binary import (
    DEFAULT_TRAIN_ROOT, objdump_function_assembly, run_ghidra,
)
from models.binary_decompilation.export_source_references import find_definition_span
from models.binary_decompilation.prompt_budget import fit_prompt


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def run(command: list[str], cwd: Path | None = None, timeout=90, env=None) -> dict:
    env = dict(os.environ if env is None else env)
    env["TMPDIR"] = "/tmp"
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                                text=True, timeout=timeout)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "returncode": 124,
                "stdout": str(error.stdout or ""), "stderr": "Timed out"}


def compiler_feedback(text: str) -> str:
    """Keep diagnostics but do not send scaffold source excerpts to the model."""
    return "\n".join(line for line in text.splitlines()
                     if not re.match(r"^\s*(?:\d+\s*\||\|)", line))[-12000:]


def model_client(config: dict):
    key_env = config.get("api_key_env")
    key = os.environ.get(key_env, "") if key_env else "local"
    if key_env and not key:
        raise ValueError("Missing model API key environment variable: " + key_env)
    return OpenAI(base_url=config["base_url"], api_key=key, timeout=600, max_retries=0)


def dependency_components(functions: list[dict], selected: set[str]) -> list[list[str]]:
    """True SCC condensation, so unrelated callers do not form a fake cycle."""
    graph = nx.DiGraph()
    graph.add_nodes_from(f["name"] for f in functions if f["name"] in selected)
    for f in functions:
        if f["name"] in selected:
            graph.add_edges_from((f["name"], c["name"]) for c in f["calls"]
                                 if c["name"] in selected)
    condensed = nx.condensation(graph)
    return [sorted(condensed.nodes[n]["members"])
            for n in reversed(list(nx.topological_sort(condensed)))]


def code_block(text: str, languages: tuple[str, ...]) -> str:
    for lang in languages:
        match = re.search(r"```" + re.escape(lang) + r"\s*\n(.*?)```", text, re.S)
        if match:
            return match.group(1).strip() + "\n"
    if "```" not in text:
        return text.strip() + "\n"
    raise ValueError("Response did not contain the requested code block")


class Retriever:
    def __init__(self, config: dict):
        self.config = config
        self.shards = None

    def retrieve(self, binary: Path, name: str) -> list[dict]:
        if not self.config.get("enabled", True):
            return []
        from datasets import load_from_disk
        from qdrant_client import QdrantClient

        assembly = objdump_function_assembly(binary, name)
        response = requests.post(self.config["embedding_url"], json=[assembly], timeout=300)
        response.raise_for_status()
        data = response.json()
        if data.get("success_indices") != [0]:
            raise ValueError("HermesSim could not embed the function")
        if self.shards is None:
            root = Path(self.config.get("train_root", DEFAULT_TRAIN_ROOT))
            self.shards = [load_from_disk(str(root / (
                f"train_synth_rich_io_filtered_{i}_llvm_extract_func_ir_assembly_O2_llvm_diff"
            ))) for i in range(8)]
        client = QdrantClient(url=self.config["qdrant_url"], timeout=30)
        examples = []
        for i, shard in enumerate(self.shards):
            collection = self.config["collection_template"].format(idx=i)
            if hasattr(client, "query_points"):
                hits = client.query_points(collection, query=data["embeddings"][0], limit=1).points
            else:
                hits = client.search(collection, query_vector=data["embeddings"][0], limit=1)
            for hit in hits:
                item = shard[int(hit.payload["id"])]
                examples.append({"score": hit.score, "collection": collection,
                                 "index": int(hit.payload["id"]), "fname": item["fname"],
                                 "c": item["func_def"], "llvm_ir": item["llvm_ir"]["code"][-1]})
        return sorted(examples, key=lambda e: e["score"], reverse=True)[:2]


class CoreutilsAdapter:
    def __init__(self, config: dict, program: str, output: Path):
        self.config, self.program, self.output = config, program, output
        self.source = Path(config["source_dir"]).resolve()
        self.build = Path(config["build_dir"]).resolve()
        self.clang = config["clang"]
        self.binary = self.build / "src" / program
        self.extra_objects = ["src/find-mount-point.o"] if program == "df" else []

    def compile_source(self, source: Path, output: Path, optimization="-O3") -> dict:
        return run([self.clang, "-I.", "-I..", "-I./lib", "-Ilib", "-I../lib",
                    "-Isrc", "-I../src", "-Wno-format-extra-args", optimization,
                    "-fno-omit-frame-pointer", "-femit-all-decls", "-c", str(source), "-o", str(output)], self.build)

    def link(self, objects: list[Path], executable: Path, source_file: str) -> dict:
        extras = self.extra_objects[:]
        if source_file != f"src/{self.program}.c":
            extras.insert(0, f"src/{self.program}.o")
        return run([self.clang, "-o", str(executable), *map(str, objects), *extras,
                    "src/libver.a", "lib/libcoreutils.a", "lib/libcoreutils.a"], self.build)

    def test(self, executable: Path, directory: Path) -> dict:
        root = directory / "testroot"
        root.mkdir(parents=True, exist_ok=True)
        (root / "src").mkdir(exist_ok=True)
        link = root / "src" / self.program
        if link.is_symlink():
            link.unlink()
        link.symlink_to(executable)
        env = dict(os.environ, LC_ALL="C", srcdir=str(self.source),
                   built_programs=self.program, CC=self.clang,
                   PATH=str(root / "src") + os.pathsep + str(self.build / "src") +
                   os.pathsep + os.environ.get("PATH", ""))
        tests = []
        for test in self.config["programs"][self.program]["tests"]:
            if test.endswith(".pl"):
                perl_env = dict(env, VERBOSE="1", LOCALE_FR_UTF8=env.get("LOCALE_FR_UTF8", "none"))
                result = run(["perl", "-w", "-I" + str(self.source / "tests"), "-MCuSkip", "-MCoreutils",
                              str(self.source / test)], root, timeout=180, env=perl_env)
                result["unit_vector_cases"] = len(re.findall(r"^[A-Za-z0-9_.-]+\.\.\.$", result["stderr"], re.M))
                result["locale_fr_utf8"] = perl_env["LOCALE_FR_UTF8"]
            else:
                result = run(["bash", str(self.source / test)], root, timeout=180, env=env)
            result["test"] = test
            tests.append(result)
        # Differential CLI probes supplement upstream scripts and work for basename.
        for case in self.config["programs"][self.program]["cases"]:
            stdin = case.get("stdin", "")
            expected = subprocess.run([self.program, *case["args"]], executable=str(self.binary), input=stdin,
                                      capture_output=True, text=True, env=env, timeout=30)
            actual = subprocess.run([self.program, *case["args"]], executable=str(link), input=stdin,
                                    capture_output=True, text=True, env=env, timeout=30)
            equal = (actual.returncode, actual.stdout, actual.stderr) == (
                expected.returncode, expected.stdout, expected.stderr)
            tests.append({"test": "differential:" + json.dumps(case), "returncode": 0 if equal else 1,
                          "stdout": actual.stdout, "stderr": actual.stderr,
                          "expected_stdout": expected.stdout, "expected_stderr": expected.stderr})
        return {"passed": sum(t["returncode"] == 0 for t in tests),
                "skipped": sum(t["returncode"] == 77 for t in tests),
                "failed": sum(t["returncode"] not in (0, 77) for t in tests), "tests": tests,
                "unit_vector_cases": sum(t.get("unit_vector_cases", 0) for t in tests)}

    def integrate(self, functions: list[dict], directory: Path) -> dict:
        """Validate accepted replacements together, retaining explicit hybrid scope."""
        directory.mkdir(parents=True, exist_ok=True)
        grouped = {}
        for function in functions:
            grouped.setdefault(function["source_file"], []).append(function)
        if not grouped:
            return {"accepted": False, "replacements": 0, "feedback": "No accepted C functions"}
        objects = []
        for index, (source_file, replacements) in enumerate(grouped.items()):
            source = (self.source / source_file).read_text()
            edits = []
            for function in replacements:
                span = find_definition_span(source, function["name"])
                if span is None:
                    raise ValueError("Missing definition: " + function["name"])
                edits.append((*span, Path(function["candidate"]).read_text()))
            for start, end, candidate in sorted(edits, reverse=True):
                source = source[:start] + candidate + source[end:]
            staged = directory / f"combined_{index}.c"
            staged.write_text(source)
            obj = directory / f"combined_{index}.o"
            result = self.compile_source(staged, obj)
            write_json(directory / f"compile_{index}.json", result)
            if result["returncode"]:
                return {"accepted": False, "feedback": result["stderr"][-12000:]}
            objects.append(obj)
        executable = directory / "recovered"
        primary = f"src/{self.program}.c"
        result = self.link(objects, executable, primary if primary in grouped else next(iter(grouped)))
        write_json(directory / "link.json", result)
        if result["returncode"]:
            return {"accepted": False, "feedback": result["stderr"][-12000:]}
        tests = self.test(executable, directory)
        write_json(directory / "tests.json", tests)
        return {"accepted": tests["failed"] == 0 and tests["passed"] > 0,
                "replacements": sum(map(len, grouped.values())), "tests": tests,
                "validation_mode": "combined generated C with original support scaffold"}

    def validate(self, name: str, source_file: str, signature: str, candidate: Path,
                 directory: Path, stage: str) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        source = (self.source / source_file).read_text()
        span = find_definition_span(source, name)
        if span is None:
            return {"accepted": False, "feedback": "No source-backed definition for target"}
        object_path = directory / "scaffold.o"
        staged = directory / "scaffold.c"
        if stage == "c":
            replacement = candidate.read_text()
            if find_definition_span(replacement, name) is None:
                return {"accepted": False, "feedback": "Candidate does not define target " + name}
        else:
            replacement = re.sub(r"\bstatic\s+", "", signature.strip()).rstrip(";") + ";\n"
        staged.write_text(source[:span[0]] + replacement + source[span[1]:])
        compile_result = self.compile_source(staged, object_path, "-O0" if stage == "ir" else "-O3")
        write_json(directory / "compile.json", compile_result)
        if compile_result["returncode"]:
            return {"accepted": False, "feedback": compiler_feedback(compile_result["stderr"])}
        if stage == "signature":
            return {"accepted": True, "feedback": "Declaration accepted by harness"}
        objects = [object_path]
        if stage == "ir":
            candidate_object = directory / "candidate.o"
            result = run([self.clang, "-Wno-override-module", "-c", str(candidate),
                          "-o", str(candidate_object)])
            write_json(directory / "ir_compile.json", result)
            if result["returncode"]:
                return {"accepted": False, "feedback": result["stderr"][-12000:]}
            defined = run(["nm", "--defined-only", str(candidate_object)])["stdout"]
            if not any(len(parts := line.split()) == 3 and parts[1] in "tT" and parts[2] == name
                       for line in defined.splitlines()):
                return {"accepted": False, "feedback": "LLVM module does not define target " + name}
            # Make retained static functions/data visible to the generated IR object.
            symbols = run(["nm", "--defined-only", str(object_path)])["stdout"]
            local = [line.split()[-1] for line in symbols.splitlines()
                     if len(line.split()) == 3 and line.split()[1] in "tdbr"
                     and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", line.split()[-1])]
            globalized = directory / "scaffold.global.o"
            result = run(["objcopy", *["--globalize-symbol=" + symbol for symbol in local],
                          str(object_path), str(globalized)])
            if result["returncode"]:
                return {"accepted": False, "feedback": result["stderr"]}
            objects = [candidate_object, globalized]
        executable = directory / "recovered"
        linked = self.link(objects, executable, source_file)
        write_json(directory / "link.json", linked)
        if linked["returncode"]:
            return {"accepted": False, "feedback": linked["stderr"][-12000:]}
        try:
            tests = self.test(executable, directory)
        except subprocess.TimeoutExpired:
            return {"accepted": False, "feedback": "Differential test timed out"}
        write_json(directory / "tests.json", tests)
        return {"accepted": tests["failed"] == 0 and tests["passed"] > 0,
                "feedback": json.dumps([t for t in tests["tests"] if t["returncode"] not in (0, 77)])[-12000:],
                "tests": {key: tests[key] for key in ("passed", "failed", "skipped")}}


def trace_cases(adapter: CoreutilsAdapter, names: list[str], output: Path) -> dict:
    """Capture named frames at each target entry and observed caller edges."""
    script = output / "trace.py"
    event_file = output / "trace-events.jsonl"
    script.write_text(
        "import gdb,json\n"
        f"destination={str(event_file)!r}\n"
        "class Trace(gdb.Breakpoint):\n"
        " def stop(self):\n"
        "  frame=gdb.newest_frame(); stack=[]\n"
        "  registers={}\n"
        "  for name in ('rdi','rsi','rdx','rcx','r8','r9','rsp'):\n"
        "   try: registers[name]=int(frame.read_register(name))\n"
        "   except gdb.error: pass\n"
        "  while frame is not None and len(stack)<32:\n"
        "   stack.append(frame.name()); frame=frame.older()\n"
        "  with open(destination,'a') as f: f.write(json.dumps({'stack':stack,'registers':registers})+'\\n')\n"
        "  return False\n"
        f"for name in {names!r}: Trace(name,internal=True)\n"
    )
    event_file.write_text("")
    invocations = []
    for i, case in enumerate(adapter.config["programs"][adapter.program]["cases"]):
        stdin = output / f"trace-input-{i}.txt"
        stdin.write_text(case.get("stdin", ""))
        result = run(["gdb", "--batch", "-ex", "set pagination off",
                      "-ex", "set breakpoint pending on", "-ex", "source " + str(script),
                      "-ex", "run " + shlex.join(case["args"]) + " < " + shlex.quote(str(stdin)),
                      "--args", str(adapter.binary), *case["args"]],
                     timeout=120)
        invocations.append(result)
    events = [json.loads(line) for line in event_file.read_text().splitlines()]
    stacks = [event["stack"] for event in events]
    hits = Counter(stack[0] for stack in stacks if stack)
    edges = sorted({(stack[1], stack[0]) for stack in stacks if len(stack) > 1
                    and stack[0] and stack[1]})
    result = {"hits": dict(hits), "observed_edges": edges, "invocations": invocations,
              "trace_valid": bool(events), "schema_version": 3,
              "entry_register_samples": {name: [event["registers"] for event in events
                                                 if event["stack"] and event["stack"][0] == name][:8]
                                         for name in names}}
    write_json(output / "runtime.json", result)
    return result


def prepare(config: dict, program: str, output: Path) -> dict:
    adapter = CoreutilsAdapter(config, program, output)
    profile_path = output / "ghidra_export.json"
    if not profile_path.exists() or json.loads(profile_path.read_text()).get("schema_version", 0) < 2:
        jdk = config.get("java_home", "/data1/xiachunwei/Software/jdk-21.0.9")
        env = dict(os.environ, JAVA_HOME=jdk, PATH=jdk + "/bin:" + os.environ.get("PATH", ""),
                   XDG_CONFIG_HOME=str(output / "ghidra-config"),
                   XDG_CACHE_HOME=str(output / "ghidra-cache"))
        run_ghidra(adapter.binary, profile_path, config["ghidra_home"], [], env=env)
    profile = json.loads(profile_path.read_text())
    nm = run(["nm", "--defined-only", str(adapter.build / "src" / (program + ".o"))])
    primary = {line.split()[-1] for line in nm["stdout"].splitlines()
               if len(line.split()) == 3 and line.split()[1] in "tT"}
    sources = {name: f"src/{program}.c" for name in primary}
    sources.update(config["programs"][program].get("extra_functions", {}))
    targets = []
    for f in profile["functions"]:
        name = f["name"]
        if name in sources and find_definition_span((adapter.source / sources[name]).read_text(), name):
            targets.append({**f, "source_file": sources[name]})
            disassembly = run(["objdump", "-d", "-w", "--no-show-raw-insn", "--disassemble=" + name,
                               str(adapter.binary)])["stdout"]
            targets[-1]["symbolized_disassembly"] = disassembly
            targets[-1]["ghidra_body_may_be_incomplete"] = sum(
                bool(re.match(r"^\s*[0-9a-f]+:", line)) for line in disassembly.splitlines()
            ) > len(f["assembly"].splitlines()) + 2
            known = {call["name"] for call in f["calls"]}
            for address, called in re.findall(r"\bcall\s+([0-9a-f]+)\s+<([^>]+)>", disassembly):
                external = called.endswith("@plt")
                called = called.removesuffix("@plt")
                if "+" not in called and called not in known:
                    f["calls"].append({"name": called, "entry": address, "external": external})
                    known.add(called)
    names = [f["name"] for f in targets]
    runtime_path = output / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    if not runtime.get("trace_valid") or runtime.get("schema_version") != 3:
        runtime = trace_cases(adapter, names, output)
    # Add observed edges without deleting statically reachable but unobserved code.
    by_name = {f["name"]: f for f in profile["functions"]}
    for caller, callee in runtime["observed_edges"]:
        if caller in by_name and callee in by_name:
            by_name[caller]["calls"].append({"name": callee, "entry": by_name[callee]["entry"], "external": False})
    components = dependency_components(profile["functions"], set(names))
    for component in components:
        component.sort(key=lambda name: (-runtime["hits"].get(name, 0), name))
    plan = {"program": program, "binary": str(adapter.binary),
            "binary_sha256": hashlib.sha256(adapter.binary.read_bytes()).hexdigest(),
            "targets": targets, "components": components, "runtime": runtime,
            "scope": "source-defined out-of-line application functions and configured support functions"}
    write_json(output / "plan.json", plan)
    return plan


def infer_stage(client, config: dict, adapter: CoreutilsAdapter, f: dict, context: dict,
                stage: str, directory: Path, signature: str = "", ir: str = "") -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / "result.json"
    identity = {"model": config["model"], "base_url": config.get("base_url"),
                "stage": stage, "target": f["name"],
                "assembly_sha256": hashlib.sha256((f.get("symbolized_disassembly") or f["assembly"]).encode()).hexdigest(),
                "signature": signature, "ir_sha256": hashlib.sha256(ir.encode()).hexdigest()}
    identity_path = directory / "input_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Stage inputs changed; use a new output directory rather than reusing stale responses")
    write_json(identity_path, identity)
    if final_path.exists():
        cached = json.loads(final_path.read_text())
        if stage == "signature" and cached.get("accepted") and "contract" not in cached:
            contract_path = directory / f"attempt_{cached['attempts'] - 1}" / "contract.json"
            if contract_path.exists():
                cached["contract"] = json.loads(contract_path.read_text())
                write_json(final_path, cached)
        if not cached.get("accepted") and cached.get("attempts", 0) >= config["max_attempts"]:
            return cached
        if cached.get("accepted"):
            refreshed = adapter.validate(f["name"], f["source_file"], cached["signature"],
                                         Path(cached["candidate"]), directory / "revalidation", stage)
            if refreshed["accepted"]:
                cached["validation"] = refreshed
                write_json(final_path, cached)
                return cached
    if stage == "signature":
        instruction = ('Infer the SysV x86-64 ABI and types. Return ONLY JSON with keys '
                       'c_declaration (a complete C function declaration ending in ;), '
                       'argument_evidence, return_evidence, memory_layouts, uncertainties. '
                       'Use the exact target function name. Distinguish evidence from guesses.')
    elif stage == "ir":
        instruction = ("Translate the target assembly to a complete standalone LLVM 18 IR module. "
                       "Use opaque pointers, the target symbol and inferred ABI; declare dependencies. "
                       "Preserve global symbol names, widths, signedness, memory side effects and control flow. "
                       "Return only a ```llvm code block. Do not substitute the retrieved function body.")
    else:
        instruction = ("Translate the validated LLVM IR into one C function definition. Preserve its ABI "
                       "and exact target name. Use the inferred contract. The harness provides Coreutils "
                       "headers, structures, globals and helper declarations. Return a ```c code block "
                       "containing only the replacement function, no main wrapper or include directives.")
    evidence = {"target": f["name"], "assembly": f.get("symbolized_disassembly") or f["assembly"],
                "ghidra_cfg": f.get("cfg", []),
                "ghidra_body_may_be_incomplete": f.get("ghidra_body_may_be_incomplete", False),
                "binary_references": [{"address": r["address"], "references": r["references"]}
                                      for r in f.get("instruction_records", []) if r["references"]],
                "ghidra_signature_hint": f["signature"], "calls": f["calls"], **context}
    if signature:
        evidence["inferred_declaration"] = signature
    if ir:
        evidence["validated_llvm_ir"] = ir
    feedback = ""
    for attempt in range(config["max_attempts"]):
        attempt_dir = directory / f"attempt_{attempt}"
        attempt_dir.mkdir(exist_ok=True)
        prompt, budget = fit_prompt(config, instruction, evidence, feedback)
        if not (attempt_dir / "context_budget.json").exists():
            write_json(attempt_dir / "context_budget.json", budget)
        response_path = attempt_dir / "response.json"
        if response_path.exists():
            response = json.loads(response_path.read_text())
        else:
            (attempt_dir / "prompt.txt").write_text(prompt)
            write_json(attempt_dir / "generation_config.json", {"model": config["model"], "n": 1,
                       "temperature": 0, "max_tokens": config.get("max_tokens", 8192), "enable_thinking": False})
            response = client.chat.completions.create(
                model=config["model"], messages=[{"role": "user", "content": prompt}],
                n=1, temperature=0, max_tokens=config.get("max_tokens", 8192),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ).model_dump()
            write_json(response_path, response)
        text = response["choices"][0]["message"]["content"] or ""
        try:
            if stage == "signature":
                contract = json.loads(code_block(text, ("json",)))
                signature = contract["c_declaration"]
                write_json(attempt_dir / "contract.json", contract)
                candidate = attempt_dir / "unused.c"
            else:
                candidate = attempt_dir / ("candidate.ll" if stage == "ir" else "candidate.c")
                candidate.write_text(code_block(text, ("llvm", "llvm-ir", "ll") if stage == "ir" else ("c",)))
            validation = adapter.validate(f["name"], f["source_file"], signature, candidate,
                                          attempt_dir / "validation", stage)
        except (ValueError, KeyError) as error:
            validation = {"accepted": False, "feedback": str(error)}
        write_json(attempt_dir / "validation.json", validation)
        if validation["accepted"]:
            result = {"accepted": True, "attempts": attempt + 1,
                      "candidate": str(candidate), "signature": signature,
                      "validation": validation}
            if stage == "signature":
                result["contract"] = contract
            write_json(final_path, result)
            return result
        feedback = validation["feedback"] + "\nPrevious candidate:\n" + text[-16000:]
    result = {"accepted": False, "attempts": config["max_attempts"], "feedback": feedback}
    write_json(final_path, result)
    return result


def campaign(config: dict, prepare_only=False, baseline_only=False) -> None:
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "toolchain.json", {"compiler": run([config["clang"], "--version"]),
                                          "gdb": run(["gdb", "--version"]),
                                          "git": run(["git", "rev-parse", "HEAD"]),
                                          "python": sys.version})
    retriever = Retriever(config["retrieval"])
    client = None if prepare_only or baseline_only else model_client(config)
    if client is not None:
        from models.binary_decompilation.service_preflight import check_services
        preflight = check_services(client, config)
        write_json(output / "service_preflight.json", preflight)
        if not preflight["ready"]:
            raise RuntimeError("Service/tracing preflight failed; see service_preflight.json. No generation was requested.")
    for program in config["programs"]:
        program_dir = output / program
        program_dir.mkdir(exist_ok=True)
        if baseline_only:
            adapter = CoreutilsAdapter(config, program, program_dir)
            baseline = adapter.test(adapter.binary, program_dir / "baseline")
            write_json(program_dir / "baseline.json", baseline)
            print(program, "baseline", {key: baseline[key] for key in ("passed", "failed", "skipped")}, flush=True)
            continue
        plan = prepare(config, program, program_dir)
        if prepare_only:
            continue
        adapter = CoreutilsAdapter(config, program, program_dir)
        baseline = adapter.test(adapter.binary, program_dir / "baseline")
        write_json(program_dir / "baseline.json", baseline)
        if baseline["failed"] or baseline["passed"] == 0:
            raise RuntimeError(program + ": original-binary baseline failed; repair harness before inference")
        if not plan["runtime"]["trace_valid"]:
            raise RuntimeError(program + ": runtime tracing unavailable; inspect runtime.json before inference")
        functions = {f["name"]: f for f in plan["targets"]}
        contracts = {}
        summary = []
        for component in plan["components"]:
            # Infer all interfaces in a recursive component before generating bodies.
            for name in component:
                f = functions[name]
                function_dir = program_dir / "functions" / name
                function_dir.mkdir(parents=True, exist_ok=True)
                retrieval_path = function_dir / "retrieval.json"
                try:
                    if not retrieval_path.exists():
                        write_json(retrieval_path, retriever.retrieve(adapter.binary, name))
                    context = {"retrieved_examples": json.loads(retrieval_path.read_text()),
                               "callee_contracts": contracts,
                               "runtime_hits": plan["runtime"]["hits"].get(name, 0),
                               "entry_register_samples": plan["runtime"].get("entry_register_samples", {}).get(name, [])}
                    contracts[name] = infer_stage(client, config, adapter, f, context,
                                                  "signature", function_dir / "signature")
                except Exception as error:
                    contracts[name] = {"accepted": False, "error": str(error)}
                    write_json(function_dir / "failure.json", contracts[name])
            for name in component:
                f = functions[name]
                function_dir = program_dir / "functions" / name
                result = {"name": name, "signature": contracts[name],
                          "runtime_observed": bool(plan["runtime"]["hits"].get(name, 0))}
                try:
                    if contracts[name]["accepted"]:
                        context = {"retrieved_examples": json.loads((function_dir / "retrieval.json").read_text()),
                                   "callee_contracts": contracts}
                        result["ir"] = infer_stage(client, config, adapter, f, context, "ir",
                                                   function_dir / "ir", contracts[name]["signature"])
                        if result["ir"]["accepted"]:
                            result["c"] = infer_stage(client, config, adapter, f, context, "c",
                                                      function_dir / "c", contracts[name]["signature"],
                                                      Path(result["ir"]["candidate"]).read_text())
                except Exception as error:
                    result["error"] = str(error)
                write_json(function_dir / "result.json", result)
                reference = (adapter.source / f["source_file"]).read_text()
                span = find_definition_span(reference, name)
                (function_dir / "source_reference.c").write_text(reference[span[0]:span[1]] + "\n")
                if result.get("c", {}).get("accepted", False):
                    compared = run(["diff", "-u", str(function_dir / "source_reference.c"),
                                    result["c"]["candidate"]])
                    (function_dir / "source_comparison.diff").write_text(compared["stdout"])
                summary.append(result)
                write_json(program_dir / "results.json", summary)
                print(program, name, "IR", result.get("ir", {}).get("accepted", False),
                      "C", result.get("c", {}).get("accepted", False), flush=True)
        generated = [{"name": x["name"], "source_file": functions[x["name"]]["source_file"],
                      "candidate": x["c"]["candidate"]} for x in summary
                     if x.get("c", {}).get("accepted", False)]
        integration = adapter.integrate(generated, program_dir / "integration")
        write_json(program_dir / "integration.json", integration)
        standalone = {"accepted": False, "reason": "standalone recovery disabled"}
        if config.get("recover_standalone", True):
            from models.binary_decompilation.standalone_recovery import recover_standalone
            standalone = recover_standalone(client, config, adapter, plan, summary, program_dir / "standalone")
        write_json(program_dir / "summary.json", {
            "targets": len(functions), "runtime_observed": sum(x["runtime_observed"] for x in summary),
            "signature_harness_accepted": sum(x["signature"].get("accepted", False) for x in summary),
            "ir_regression_passed": sum(x.get("ir", {}).get("accepted", False) for x in summary),
            "c_regression_passed": sum(x.get("c", {}).get("accepted", False) for x in summary),
            "ir_runtime_observed_and_regression_passed": sum(x["runtime_observed"] and x.get("ir", {}).get("accepted", False) for x in summary),
            "c_runtime_observed_and_regression_passed": sum(x["runtime_observed"] and x.get("c", {}).get("accepted", False) for x in summary),
            "validation_mode": "isolated generated function with original support scaffold",
            "complete_project_recovery": False,
            "combined_c_accepted": integration["accepted"],
            "standalone_application_accepted": standalone["accepted"],
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if not 1 <= config["max_attempts"] <= 10:
        raise ValueError("max_attempts must be between 1 and 10")
    if config.get("num_generate", 1) != 1 or config.get("temperature", 0) != 0:
        raise ValueError("This campaign requires one deterministic generation per request")
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.no_retrieval:
        config["retrieval"]["enabled"] = False
        if not args.output_dir:
            config["output_dir"] += "_no_retrieval"
    campaign(config, args.prepare_only, args.baseline_only)
    if config.get("run_retrieval_ablation") and not (args.prepare_only or args.baseline_only or args.no_retrieval):
        ablation = copy.deepcopy(config)
        ablation["output_dir"] += "_no_retrieval"
        ablation["retrieval"]["enabled"] = False
        campaign(ablation)
        comparison = {}
        for program in config["programs"]:
            comparison[program] = {
                variant: json.loads((Path(settings["output_dir"]) / program / "summary.json").read_text())
                for variant, settings in (("retrieval", config), ("no_retrieval", ablation))}
        write_json(Path(config["output_dir"]) / "retrieval_ablation.json", comparison)


if __name__ == "__main__":
    main()
