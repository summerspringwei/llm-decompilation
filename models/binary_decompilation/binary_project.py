"""Reusable binary-first recovery campaign; source is evaluation-only metadata."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from models.binary_decompilation.analyze_binary import run_ghidra
from models.binary_decompilation.native_validation import build_candidate, elf_symbols, make_wrapper, validate_probe
from models.binary_decompilation.project_pipeline import (
    CoreutilsAdapter, Retriever, dependency_components, generation_config, generation_metadata,
    infer_stage, model_client, run, write_json,
)
from models.binary_decompilation.source_inventory import clang_inventory
from models.binary_decompilation.standalone_recovery import (
    binary_data_evidence, generated_includes_only, recover_standalone,
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path: Path) -> dict:
    raw = json.loads(path.read_text())
    base = load_config(path.parent / raw.pop("extends")) if "extends" in raw else {}
    overrides = raw.pop("program_overrides", {})
    base.update(raw)
    for name, values in overrides.items():
        additional_cases = values.pop("additional_cases", [])
        base["programs"][name].update(values)
        base["programs"][name]["cases"].extend(additional_cases)
    return base


def native_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize_trace(events: list[dict], functions: list[dict]) -> dict:
    names = {f["name"] for f in functions}
    hits = {name: 0 for name in names}
    registers, edges, samples = {}, set(), {}
    for event in events:
        if event.get("event") != "entry":
            continue
        name = event["name"]
        hits[name] += 1
        # Upstream traces select coverage/dependencies, but their inputs do not
        # enter inference prompts. Only configured feedback probes supply values.
        if str(event.get("test_id", "")).startswith("differential:"):
            registers.setdefault(name, []).append(event["registers"])
            samples.setdefault(name, [])
            if len(samples[name]) < 4:
                samples[name].append(event)
        stack = event.get("stack", [])
        for caller in stack[1:]:
            if caller in names and caller != name:
                edges.add((caller, name))
                break
    return {"schema_version": 4, "trace_valid": any(hits.values()), "hits": hits,
            "hits_are_bounded_samples": True, "observed_edges": sorted(edges),
            "entry_register_samples": {n: r[:8] for n, r in registers.items()},
            "samples": samples,
            "test_ids": sorted({e["test_id"] for e in events if e.get("test_id")}),
            "events": len(events)}


class BinaryCoreutilsAdapter(CoreutilsAdapter):
    """Coreutils tests are an adapter; the generated-code validator reads no source."""

    binary_only = True

    def __init__(self, config, program, output):
        super().__init__(config, program, output)
        self.extra_objects = []
        self.header = ""
        self.direct_probes = {}
        self.probe_errors = {}
        self.translation_contracts = {}
        self.semantic_facts = {}
        self.feedback_config = copy.deepcopy(config)
        self.feedback_config["programs"][program]["tests"] = []

    def feedback_test(self, executable: Path, directory: Path) -> dict:
        adapter = CoreutilsAdapter(self.feedback_config, self.program, directory)
        return adapter.test(executable, directory)

    def validate(self, name, source_file, signature, candidate, directory, stage):
        directory.mkdir(parents=True, exist_ok=True)
        if stage == "probes":
            contract = json.loads((candidate.parent / "contract.json").read_text())
            probes, errors = self.parse_probes(contract.get("direct_probes", []))
            if errors or not probes:
                return {"accepted": False, "failure_kind": "probe_schema",
                        "feedback": json.dumps(errors or ["At least one valid direct probe is required"])}
            self.direct_probes[name], self.probe_errors[name] = probes, []
            return {"accepted": True, "feedback": "Probe schema accepted; execution remains unverified"}
        if stage == "signature":
            contract = json.loads((candidate.parent / "contract.json").read_text())
            header = contract.get("project_header", "")
            if not header:
                return {"accepted": False, "feedback": "project_header is required"}
            generated_includes_only(header)
            unit = directory / "declarations.c"
            unit.write_text(header + "\n" + signature + "\n")
            result = run([self.clang, "-std=gnu17", "-D_GNU_SOURCE", "-Xclang", "-ast-dump=json",
                          "-fsyntax-only", str(unit)])
            write_json(directory / "contract_compile.json", result)
            if result["returncode"]:
                return {"accepted": False, "feedback": result["stderr"][-12000:]}
            # System inline functions are permitted; generated function bodies are not.
            ast = json.loads(result["stdout"])
            last_file = str(unit)
            for node in ast.get("inner", []):
                location = node.get("loc", {})
                last_file = location.get("file", last_file)
                if node.get("kind") == "FunctionDecl" and last_file == str(unit) and any(
                        n["kind"] == "CompoundStmt" for n in node.get("inner", [])):
                    return {"accepted": False, "feedback": "project_header must contain declarations, not function bodies"}
            if not any(node.get("kind") == "FunctionDecl" and node.get("name") == name
                       for node in ast.get("inner", [])):
                return {"accepted": False, "feedback": "Contract must declare the exact target: " + name}
            self.header = header
            probes, errors = self.parse_probes(contract.get("direct_probes", []))
            self.direct_probes[name] = probes
            self.probe_errors[name] = errors
            return {"accepted": True, "feedback": "Contract is syntax-valid; ABI remains behaviorally unverified",
                    "probe_errors": errors, "valid_probes": len(probes),
                    "abi_verified": False, "validation_mode": "binary-only declarations"}
        if stage == "ir" and self.config.get("translation_scaffold"):
            from models.binary_decompilation.translation_contract import inspect_ir
            structural = inspect_ir(candidate.read_text(), self.translation_contracts.get(name))
            write_json(directory / "structural_validation.json", structural)
            if not structural["accepted"]:
                return {"accepted": False, "failure_kind": "ir_contract",
                        "feedback": "\n".join(structural["errors"])}
        built = build_candidate(self.clang, self.binary, name, candidate, self.header, directory, stage)
        if not built["accepted"]:
            if stage == "c" and any(marker in built.get("feedback", "") for marker in
                                    ("project.h", "no member named", "incomplete definition")):
                built["repair_contract"] = True
            return built
        semantic = None
        if stage == "ir" and self.config.get("semantic_checks"):
            from models.binary_decompilation.semantic_validation import check_candidate
            # Candidate ABI width bounds the check; it is not independent ABI proof.
            import re
            match = re.search(r'\bdefine\s+(?:\w+\s+)*?i(8|16|32|64)\s+@(?:"' + re.escape(name) +
                              r'"|' + re.escape(name) + r')\s*\(', candidate.read_text())
            if match:
                semantic = check_candidate(self.binary, built["manifest"]["library"], name,
                                           int(match.group(1)), directory,
                                           self.config.get("semantic_timeout", 30))
            else:
                semantic = {"status": "unknown", "reason": "No supported scalar return ABI"}
                write_json(directory / "semantic_validation.json", semantic)
        wrapper = make_wrapper(self.clang, self.binary, [], [built["manifest"]], directory / "runtime")
        tests = self.feedback_test(wrapper, directory / "feedback")
        write_json(directory / "tests.json", tests)
        events = native_events(directory / "runtime/events.jsonl")
        entered = sum(e.get("event") == "replacement" and e.get("name") == name for e in events)
        if not entered:
            for index, probe in enumerate(self.direct_probes.get(name, [])):
                probe = {**probe, "name": name}
                original_wrapper = make_wrapper(self.clang, self.binary, [], [], directory / f"direct_{index}/original", direct_probe=probe)
                candidate_wrapper = make_wrapper(self.clang, self.binary, [], [built["manifest"]], directory / f"direct_{index}/candidate", direct_probe=probe)
                expected = run([str(original_wrapper)])
                actual = run([str(candidate_wrapper)])
                direct_events = native_events(directory / f"direct_{index}/candidate/events.jsonl")
                original_events = native_events(directory / f"direct_{index}/original/events.jsonl")
                executed = any(e.get("event") == "direct_probe" for e in original_events) and any(
                    e.get("event") == "direct_probe" for e in direct_events)
                equal = executed and expected["returncode"] == actual["returncode"] == 0 and (
                    expected["stdout"], expected["stderr"]) == (actual["stdout"], actual["stderr"])
                tests["tests"].append({"test": "direct_probe:" + str(index), "probe": probe,
                                       "target_executed": executed,
                                       "returncode": 0 if equal else 1, "expected": expected, "actual": actual})
                tests["passed" if equal else "failed"] += 1
                entered += sum(e.get("event") == "replacement" for e in direct_events)
                events.extend(direct_events)
            write_json(directory / "tests.json", tests)
        from models.binary_decompilation.validation_feedback import runtime_feedback
        diagnostic, feedback = runtime_feedback(tests, events, name)
        if semantic and semantic["status"] == "counterexample":
            feedback += "\nAdvisory scalar counterexample (caller reachability unproved): " + json.dumps(semantic)
        write_json(directory / "coverage_and_feedback.json", diagnostic)
        loader_failure = diagnostic["failure_kind"] == "loader_error"
        return {"accepted": tests["failed"] == 0 and entered > 0,
                "needs_probe": entered == 0 and tests["failed"] == 0,
                "failure_kind": diagnostic["failure_kind"],
                "repair_contract": bool(stage == "c" and loader_failure and any(
                    "undefined symbol" in e for e in diagnostic["loader_errors"])),
                "coverage": diagnostic["coverage"],
                "semantic_validation": semantic,
                "feedback": feedback if tests["failed"] or loader_failure else
                            ("Candidate was executed" if entered else "Target not executed by feedback probes"),
                "execution_observed": entered > 0, "replacement_calls": entered,
                "manifest": built["manifest"], "validation_mode": "binary-only native replacement",
                "tests": {key: tests[key] for key in ("passed", "failed", "skipped")}}

    @staticmethod
    def parse_probes(probes):
        if not isinstance(probes, list) or len(probes) > 8:
            return [], ["direct_probes must be an array of at most eight bounded probes"]
        valid, errors = [], []
        for index, probe in enumerate(probes):
            try:
                validate_probe(probe)
                valid.append(probe)
            except (ValueError, TypeError, KeyError) as error:
                errors.append({"probe": index, "error": str(error)})
        return valid, errors


def prepare_binary(config: dict, program: str, output: Path) -> dict:
    adapter = BinaryCoreutilsAdapter(config, program, output)
    binary_hash = hashlib.sha256(adapter.binary.read_bytes()).hexdigest()
    profile_path = output / "ghidra_export.json"
    inventory = clang_inventory(config, program, output / "ownership")
    if not profile_path.exists():
        old = Path(config.get("analysis_cache", "")) / program
        if (old / "plan.json").exists() and (old / "ghidra_export.json").exists() and json.loads(
                (old / "plan.json").read_text()).get("binary_sha256") == binary_hash:
            write_json(profile_path, json.loads((old / "ghidra_export.json").read_text()))
        else:
            jdk = config.get("java_home", "/data1/xiachunwei/Software/jdk-21.0.9")
            env = dict(os.environ, JAVA_HOME=jdk, PATH=jdk + "/bin:" + os.environ.get("PATH", ""),
                       XDG_CONFIG_HOME=str(output / "ghidra-config"), XDG_CACHE_HOME=str(output / "ghidra-cache"))
            run_ghidra(adapter.binary, profile_path, config["ghidra_home"], [], env=env)
    profile = json.loads(profile_path.read_text())
    symbols = elf_symbols(adapter.binary)
    sample = next(f for f in profile["functions"] if f["name"] in symbols and symbols[f["name"]]["kind"] == "STT_FUNC")
    delta = int(sample["entry"], 16) - symbols[sample["name"]]["address"]
    profile = normalize_ghidra_addresses(profile, delta)
    profile_functions = {f["name"]: f for f in profile["functions"]}
    targets, missing = [], []
    for metadata in inventory["functions"]:
        name = metadata["name"]
        if name not in symbols:
            missing.append(metadata)
            continue
        function = copy.deepcopy(profile_functions.get(name, {"name": name, "assembly": "", "signature": "",
                                                               "calls": [], "cfg": []}))
        function.update(metadata)
        function["entry"] = hex(symbols[name]["address"])
        function["function_id"] = binary_hash + ":" + function["entry"]
        assembly = run(["objdump", "-d", "-w", "--no-show-raw-insn", "--disassemble=" + name,
                        str(adapter.binary)])["stdout"]
        function["symbolized_disassembly"] = assembly
        function["ghidra_body_may_be_incomplete"] = sum(":" in line for line in assembly.splitlines()) > len(
            function["assembly"].splitlines()) + 2
        import re
        known = {c["name"] for c in function["calls"]}
        for address, called in re.findall(r"\bcall\s+([0-9a-f]+)\s+<([^>]+)>", assembly):
            external = called.endswith("@plt")
            called = called.removesuffix("@plt")
            if "+" not in called and called not in known:
                function["calls"].append({"name": called, "entry": address, "external": external})
                known.add(called)
        # Address-taken source functions include callbacks invoked by libraries.
        address_taken = {called for called in re.findall(r"#\s*[0-9a-f]+\s+<([^>]+)>", assembly)}
        for called in sorted(address_taken - known):
            if called in symbols and symbols[called]["kind"] == "STT_FUNC":
                function["calls"].append({"name": called, "entry": hex(symbols[called]["address"]),
                                          "external": False, "evidence": "address-taken conservative dependency"})
        targets.append(function)
    data = binary_data_evidence(adapter.binary, adapter.build / "src" / (program + ".o"), targets)
    wrapper = make_wrapper(adapter.clang, adapter.binary, [f["name"] for f in targets], [], output / "trace",
                           [g["name"] for g in data["objects"]])
    trace_config = copy.deepcopy(config)
    excluded = config["programs"][program].get("instrumentation_excluded_tests", {})
    trace_config["programs"][program]["tests"] = [t for t in config["programs"][program]["tests"] if t not in excluded]
    traced_tests = CoreutilsAdapter(trace_config, program, output).test(wrapper, output / "trace/tests")
    traced_tests["instrumentation_exclusions"] = excluded
    write_json(output / "trace/tests.json", traced_tests)
    runtime = summarize_trace(native_events(output / "trace/events.jsonl"), targets)
    by_name = {f["name"]: f for f in targets}
    for caller, callee in runtime["observed_edges"]:
        if callee not in {c["name"] for c in by_name[caller]["calls"]}:
            by_name[caller]["calls"].append({"name": callee, "entry": by_name[callee]["entry"], "external": False})
    plan = {"schema_version": 4, "program": program, "binary": str(adapter.binary),
            "binary_sha256": binary_hash, "targets": targets, "not_retained": missing,
            "inlined_or_not_emitted": inventory["inlined_or_not_emitted"],
            "components": dependency_components(targets, set(by_name)), "runtime": runtime,
            "binary_data": data, "instrumented_baseline": traced_tests,
            "source_scope": "application objects and private header helpers; approved libraries excluded"}
    write_json(output / "plan.json", plan)
    return plan


def run_stage(client, config, adapter, function, context, stage, directory, signature="", ir=""):
    """Input-addressed revisions share one generation budget per target/stage."""
    directory.mkdir(parents=True, exist_ok=True)
    config = generation_config(config, stage)
    if stage == "probes":
        config["max_attempts"] = min(config["max_attempts"], config.get("max_probe_attempts", 3))
    identity = {"schema": 3, "function_id": function["function_id"], "context": context,
                "function_evidence": {key: function.get(key) for key in ("symbolized_disassembly", "assembly", "cfg", "calls", "instruction_records")},
                "prompt_implementation": hashlib.sha256(Path(__file__).with_name("project_pipeline.py").read_bytes()).hexdigest(),
                "stage": stage, "signature": signature, "ir": digest(ir),
                "model": config["model"], "base_url": config["base_url"],
                "max_tokens": config["max_tokens"], "context_window": config["context_window"],
                "repetition_penalty": config.get("repetition_penalty", 1.0),
                "generation": generation_metadata(config),
                "tests": adapter.feedback_config["programs"][adapter.program],
                "header": adapter.header if stage != "signature" else ""}
    identity["guidance"] = {key: config.get(key, False) for key in (
        "guided_decompilation", "compiler_analysis", "semantic_analysis", "translation_scaffold", "semantic_checks")}
    identity["analysis_implementation"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("guided_prompts.py", "compiler_evidence.py", "candidate_search.py", "native_validation.py",
                     "semantic_analysis.py", "translation_contract.py", "validation_feedback.py", "semantic_validation.py",
                     "repair_evidence.py")}
    revision = directory / digest(identity)
    used = len(list(directory.glob("*/attempt_*/request.json")))
    current_used = len(list(revision.glob("attempt_*/request.json")))
    remaining = config["max_attempts"] - used + current_used
    if remaining <= 0:
        result = {"accepted": False, "attempts": used, "total_generation_requests": used,
                  "input_identity": digest(identity), "reason": "stage generation budget exhausted"}
        write_json(directory / "latest.json", result)
        return result
    result = infer_stage(client, {**config, "max_attempts": remaining}, adapter, function, context,
                         stage, revision, signature, ir)
    result["total_generation_requests"] = len(list(directory.glob("*/attempt_*/request.json")))
    result["total_requested_candidates"] = sum(json.loads(p.read_text()).get("n", 1)
                                               for p in directory.glob("*/attempt_*/request.json"))
    result["input_identity"] = digest(identity)
    write_json(directory / "latest.json", result)
    return result


def compact_contracts(contracts: dict) -> dict:
    return {name: {"signature": value.get("signature"), "contract": value.get("contract"),
                   "accepted": value.get("accepted", False)} for name, value in contracts.items()}


def normalize_ghidra_addresses(value, delta, key=""):
    import re
    if isinstance(value, dict):
        return {k: normalize_ghidra_addresses(v, delta, k) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_ghidra_addresses(v, delta, key) for v in value]
    if key in ("address", "entry", "start", "end") and isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]+", value):
        address = int(value, 16)
        if address >= delta:
            return hex(address - delta)
    return value


def run_program(client, config, program, directory, track):
    directory.mkdir(parents=True, exist_ok=True)
    from models.binary_decompilation.project_interfaces import resolve_factory
    factory = resolve_factory(config["adapter_factory"]) if config.get("adapter_factory") else BinaryCoreutilsAdapter
    analyzer = resolve_factory(config["analysis_factory"]) if config.get("analysis_factory") else prepare_binary
    adapter = factory(config, program, directory)
    baseline = adapter.test(adapter.binary, directory / "baseline")
    write_json(directory / "baseline.json", baseline)
    if baseline["failed"] or not baseline["passed"]:
        raise RuntimeError(program + ": original baseline failed")
    plan_path = directory / "plan.json"
    identity = {"binary": hashlib.sha256(adapter.binary.read_bytes()).hexdigest(),
                "program_config": config["programs"][program], "schema": 6}
    if plan_path.exists() and (directory / "analysis_identity.json").exists() and json.loads(
            (directory / "analysis_identity.json").read_text()) == identity:
        plan = json.loads(plan_path.read_text())
    else:
        primary = Path(config.get("evidence_cache", config["output_dir"])) / "retrieval" / program
        primary_identity = primary / "analysis_identity.json"
        if primary != directory and primary_identity.exists() and json.loads(primary_identity.read_text()) == identity:
            plan = json.loads((primary / "plan.json").read_text())
            write_json(plan_path, plan)
            write_json(directory / "paired_evidence.json", {"source": str(primary), "identity": identity,
                                                           "plan_sha256": digest(plan)})
        else:
            plan = analyzer(config, program, directory)
        write_json(directory / "analysis_identity.json", identity)
    if plan["instrumented_baseline"]["failed"] or not plan["runtime"]["trace_valid"]:
        raise RuntimeError(program + ": binary instrumentation baseline failed; no inference allowed")
    if track == "assisted":
        from models.binary_decompilation.assisted_validation import SourceScaffoldAdapter
        adapter = SourceScaffoldAdapter(config, program, directory)
    retriever = Retriever({**config["retrieval"], "enabled": track != "no_retrieval"})
    functions = {f["name"]: f for f in plan["targets"]}
    selected = set(config.get("function_filters", {}).get(program) or config.get("function_filter") or functions)
    if selected - set(functions):
        raise ValueError("Unknown target functions: " + ", ".join(sorted(selected - set(functions))))
    from models.binary_decompilation.public_abi import public_abi
    system_abi = public_abi(config["clang"], directory / "public_abi")
    contracts, results = {}, []
    for component in plan["components"]:
        component = [name for name in component if name in selected]
        component_contexts = {}
        for name in component:
            function = functions[name]
            root = directory / "functions" / name
            root.mkdir(parents=True, exist_ok=True)
            retrieval_path = root / "retrieval.json"
            retrieval_identity = {"query_sha256": digest(function["symbolized_disassembly"]),
                                  "config": config["retrieval"], "enabled": track != "no_retrieval"}
            cached_retrieval = json.loads(retrieval_path.read_text()) if retrieval_path.exists() else {}
            shared_path = None
            if config.get("retrieval_cache"):
                shared_path = Path(config["retrieval_cache"]) / digest(retrieval_identity) / "retrieval.json"
                if shared_path.exists():
                    shared = json.loads(shared_path.read_text())
                    if all(shared.get(k) == v for k, v in retrieval_identity.items()):
                        cached_retrieval = shared
                        write_json(retrieval_path, shared)
            if any(cached_retrieval.get(k) != v for k, v in retrieval_identity.items()):
                started = time.monotonic()
                examples = retriever.retrieve(adapter.binary, name)
                write_json(retrieval_path, {**retrieval_identity, "examples": examples,
                           "elapsed_seconds": time.monotonic() - started})
                if shared_path:
                    write_json(shared_path, json.loads(retrieval_path.read_text()))
            retrieval = json.loads(retrieval_path.read_text())
            context = {"retrieved_examples": retrieval["examples"], "binary_data": plan["binary_data"],
                       "system_library_abi": {call["name"]: system_abi["functions"][call["name"]]
                                              for call in function["calls"] if call["name"] in system_abi["functions"]},
                       "callee_contracts": compact_contracts(contracts),
                       "entry_register_samples": plan["runtime"]["entry_register_samples"].get(name, []),
                       "runtime_samples": plan["runtime"].get("samples", {}).get(name, []),
                       "feedback_runtime_hits": len(plan["runtime"]["entry_register_samples"].get(name, [])),
                       "runtime_hits": plan["runtime"]["hits"].get(name, 0)}
            if config.get("compiler_analysis"):
                from models.binary_decompilation.compiler_evidence import binary_facts
                context["compiler_cfg"] = binary_facts(adapter.binary, function, root / "compiler_analysis")
                context["public_type_layouts"] = system_abi["layouts"]
            if config.get("semantic_analysis"):
                from models.binary_decompilation.semantic_analysis import analyze_function, prompt_packet
                facts = analyze_function(adapter.binary, function, root / "semantic_analysis")
                adapter.semantic_facts[name] = facts
                context["semantic_evidence"] = prompt_packet(facts)
                if config.get("translation_scaffold"):
                    from models.binary_decompilation.translation_contract import create_contract
                    contract = create_contract(adapter.binary, facts, root / "semantic_analysis", context['system_library_abi'])
                    context["translation_contract"] = contract
                    adapter.translation_contracts[name] = contract
            component_contexts[name] = context
            adapter.header = ""
            try:
                contracts[name] = run_stage(client, config, adapter, function, context, "signature", root / "signature")
            except Exception as error:
                contracts[name] = {"accepted": False, "error": str(error)}
                write_json(root / "signature/failure.json", contracts[name])
            print(program, track, name, "signature", contracts[name]["accepted"], flush=True)
        for name in component:
            root = directory / "functions" / name
            result = {"name": name, "function_id": functions[name]["function_id"], "signature": contracts[name],
                      "runtime_observed": bool(plan["runtime"]["hits"].get(name))}
            try:
                if contracts[name]["accepted"]:
                    adapter.header = contracts[name]["contract"]["project_header"]
                    context = {**component_contexts[name], "callee_contracts": compact_contracts(contracts),
                               "project_header": adapter.header}
                    result["ir"] = run_stage(client, config, adapter, functions[name], context, "ir", root / "ir",
                                             contracts[name]["signature"])
                    if result["ir"].get("needs_probe"):
                        probe_context = {**context, "probe_repair_feedback": result["ir"]["feedback"],
                                         "probe_errors": adapter.probe_errors.get(name, []),
                                         "compiled_llvm_ir": Path(result["ir"]["candidate"]).read_text()}
                        result["probes"] = run_stage(client, config, adapter, functions[name], probe_context,
                                                     "probes", root / "probes", contracts[name]["signature"])
                        if result["probes"].get("accepted"):
                            checked = adapter.validate(name, functions[name]["source_file"], contracts[name]["signature"],
                                                       Path(result["ir"]["candidate"]), root / "probe_revalidation", "ir")
                            result["ir"].update(accepted=checked["accepted"], validation=checked,
                                                 feedback=checked["feedback"], needs_probe=checked.get("needs_probe", False))
                            write_json(root / "ir/latest.json", result["ir"])
                            write_json(root / "ir" / result["ir"]["input_identity"] / "result.json", result["ir"])
                    if result["ir"]["accepted"] and not config.get("ir_only"):
                        result["c"] = run_stage(client, config, adapter, functions[name], context, "c", root / "c",
                                                contracts[name]["signature"], Path(result["ir"]["candidate"]).read_text())
                        repairs = 0
                        while result["c"].get("repair_contract") and repairs < config["max_attempts"]:
                            repairs += 1
                            previous_signature = contracts[name]["signature"]
                            repair_context = {**context, "contract_repair_feedback": result["c"]["feedback"],
                                              "validated_llvm_ir": Path(result["ir"]["candidate"]).read_text()}
                            repaired = run_stage(client, config, adapter, functions[name], repair_context,
                                                 "signature", root / "signature")
                            if not repaired.get("accepted"):
                                result["contract_repair"] = repaired
                                break
                            contracts[name] = result["signature"] = repaired
                            adapter.header = repaired["contract"]["project_header"]
                            context = {**context, "project_header": adapter.header, "callee_contracts": compact_contracts(contracts)}
                            if previous_signature != repaired["signature"]:
                                result["ir"] = run_stage(client, config, adapter, functions[name], context, "ir", root / "ir",
                                                         repaired["signature"])
                            else:
                                refreshed = adapter.validate(name, functions[name]["source_file"], repaired["signature"],
                                    Path(result["ir"]["candidate"]), root / f"contract_revalidation_{repairs}", "ir")
                                result["ir"]["accepted"] = refreshed["accepted"]
                                result["ir"]["validation"] = refreshed
                                write_json(root / "ir/latest.json", result["ir"])
                            if not result["ir"]["accepted"]:
                                break
                            result["c"] = run_stage(client, config, adapter, functions[name], context, "c", root / "c",
                                                    repaired["signature"], Path(result["ir"]["candidate"]).read_text())
            except Exception as error:
                result["error"] = str(error)
            results.append(result)
            write_json(root / "result.json", result)
            write_json(directory / "results.json", results)
            # Evaluation-only access occurs after recovery of this function.
            from models.binary_decompilation.export_source_references import find_definition_span
            source_root = getattr(adapter, "source", None)
            original = (source_root / functions[name]["source_file"]).read_text() if source_root else ""
            span = find_definition_span(original, name) if original else None
            if span:
                reference = root / "source_reference.c"
                reference.write_text(original[span[0]:span[1]] + "\n")
                if result.get("c", {}).get("accepted"):
                    difference = run(["diff", "-u", str(reference), result["c"]["candidate"]])
                    (root / "source_comparison.diff").write_text(difference["stdout"])
            print(program, track, name, "IR", result.get("ir", {}).get("accepted", False),
                  "C", result.get("c", {}).get("accepted", False), flush=True)
    standalone_builder = resolve_factory(config["standalone_factory"]) if config.get("standalone_factory") else recover_standalone
    standalone = ({"accepted": False, "skipped": True, "reason": "IR-only function evaluation"}
                  if config.get("ir_only") else
                  standalone_builder(client, config, adapter, plan, results, directory / "standalone"))
    heldout_passed = False
    if standalone.get("accepted"):
        heldout = {}
        for representation in ("ir", "c"):
            executable = Path(standalone["validation"][representation]["executable"])
            heldout[representation] = adapter.test(executable, directory / "heldout" / representation)
        write_json(directory / "heldout.json", heldout)
        heldout_passed = all(t["failed"] == 0 and t["passed"] > 0 for t in heldout.values())
        standalone["heldout_passed"] = heldout_passed
    completed = {r["name"] for r in results if r.get("c", {}).get("accepted") and r.get("ir", {}).get("accepted")}
    summary = {"track": track, "targets": len(functions), "evaluated_targets": len(results),
               "ir_only": bool(config.get("ir_only")),
               "accepted_ir": sum(r.get("ir", {}).get("accepted", False) for r in results),
               "selected_functions": sorted(selected), "partial_run": selected != set(functions),
               "accepted_c": len(completed), "signature_syntax_valid": sum(r["signature"]["accepted"] for r in results),
               "runtime_observed": sum(r["runtime_observed"] for r in results),
               "standalone": standalone, "complete_project_recovery": heldout_passed and completed == set(functions),
               "limitations": ["Passing tests is behavioral evidence, not proof of equivalence",
                               "Inlined functions have no independent recovered body",
                               "Approved support libraries are reused; entire ELF is not regenerated"]}
    write_json(directory / "summary.json", summary)
    return summary


def parallel_programs(args, config, programs, tracks, output):
    """Keep function checkpoints in place and isolate worker bookkeeping."""
    workers = {}
    logs = []
    status = {}
    try:
        for program in programs:
            metadata = output / "workers" / program
            metadata.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-m", "models.binary_decompilation.binary_project",
                       "--config", str(Path(args.config).resolve()), "--program", program,
                       "--metadata-dir", str(metadata)]
            for track in tracks:
                command.extend(["--track", track])
            for function in args.function or []:
                command.extend(["--function", function])
            if args.prepare_only:
                command.append("--prepare-only")
            log = (metadata / "campaign.log").open("a")
            logs.append(log)
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            workers[program] = worker
            status[program] = {"pid": worker.pid, "phase": "running", "command": command}
        write_json(output / "parallel_status.json", status)
        pending = set(workers)
        while pending:
            for program in list(pending):
                code = workers[program].poll()
                if code is None:
                    continue
                pending.remove(program)
                status[program].update(phase="finished" if code == 0 else "process_failed", exit_code=code)
                path = output / "campaign.json"
                campaign = json.loads(path.read_text()) if path.exists() else {}
                worker_path = output / "workers" / program / "campaign.json"
                if code == 0 and worker_path.exists():
                    campaign.update(json.loads(worker_path.read_text()))
                elif code != 0:
                    for track in tracks:
                        campaign[track + "/" + program] = {
                            "complete_project_recovery": False, "error": f"Worker exited with {code}"}
                write_json(path, campaign)
                write_json(output / "parallel_status.json", status)
            if pending:
                time.sleep(2)
        campaign = json.loads((output / "campaign.json").read_text())
        write_json(output / "acceptance.json", {
            "goal_complete": all(campaign.get("retrieval/" + p, {}).get("complete_project_recovery", False)
                                 for p in config["programs"]),
            "programs": list(config["programs"]), "generation_limit": config["max_attempts"],
            "n": config.get("num_generate", 1)})
        if any(value["exit_code"] for value in status.values()):
            raise RuntimeError("One or more benchmark workers failed; see workers/*/campaign.log")
    finally:
        for worker in workers.values():
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--program", action="append")
    parser.add_argument("--track", choices=["retrieval", "no_retrieval", "assisted"], action="append")
    parser.add_argument("--function", action="append", help="Recover selected retained functions; full application inventory remains required")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--parallel-programs", action="store_true", help="Run one isolated worker per benchmark")
    parser.add_argument("--metadata-dir", type=Path, help="Isolate worker metadata without moving function checkpoints")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    generation_config(config, "ir")
    if not 1 <= config["max_attempts"] <= 10:
        raise ValueError("At most ten generation rounds per function/stage are required")
    if args.function:
        config["function_filter"] = args.function
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    metadata = args.metadata_dir or output
    metadata.mkdir(parents=True, exist_ok=True)
    write_json(metadata / "config.json", config)
    programs = args.program or list(config["programs"])
    tracks = args.track or ["retrieval", "no_retrieval", "assisted"]
    if len(set(programs)) != len(programs) or any(p not in config["programs"] for p in programs):
        raise ValueError("Programs must be unique configured benchmark names")
    if args.parallel_programs:
        if args.metadata_dir:
            raise ValueError("A parallel coordinator cannot use worker metadata mode")
        parallel_programs(args, config, programs, tracks, output)
        return
    if tracks == ["no_retrieval"]:
        config["retrieval"]["enabled"] = False
    client = None if args.prepare_only else model_client(config)
    if client:
        from models.binary_decompilation.service_preflight import check_services
        # HermesSim can serialize embeddings and block its health handler.
        with (output / "service_preflight.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            preflight = check_services(client, config)
        write_json(metadata / "service_preflight.json", preflight)
        if not preflight["ready"]:
            raise RuntimeError("Service preflight failed")
        fixture = run([sys.executable, "-m", "unittest", "models.binary_decompilation.test_native_validation", "-v"])
        write_json(metadata / "native_backend_fixtures.json", fixture)
        if fixture["returncode"]:
            raise RuntimeError("Native backend fixtures failed; no inference allowed")
        write_json(metadata / "toolchain.json", {"compiler": run([config["clang"], "--version"]),
                   "git": run(["git", "rev-parse", "HEAD"]), "python": sys.version,
                   "dependencies": run([sys.executable, "-m", "pip", "show", "frida", "pyelftools", "capstone"]),
                   "framework_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                   "external_library_boundary": "libc/CRT, src/libver.a, lib/libcoreutils.a"})
    campaign_path = metadata / "campaign.json"
    campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
    for track in tracks:
        for program in programs:
            directory = output / track / program
            if args.prepare_only:
                directory.mkdir(parents=True, exist_ok=True)
                from models.binary_decompilation.project_interfaces import resolve_factory
                analyzer = resolve_factory(config["analysis_factory"]) if config.get("analysis_factory") else prepare_binary
                analyzer(config, program, directory)
                continue
            try:
                campaign[track + "/" + program] = run_program(client, config, program, directory, track)
            except Exception as error:
                campaign[track + "/" + program] = {"complete_project_recovery": False, "error": str(error)}
                write_json(directory / "failure.json", campaign[track + "/" + program])
                print(program, track, "FAILED", repr(error), flush=True)
            write_json(metadata / "campaign.json", campaign)
    primary = [campaign.get("retrieval/" + p, {}) for p in config["programs"]]
    write_json(metadata / "acceptance.json", {"goal_complete": bool(primary) and all(
        s.get("complete_project_recovery", False) for s in primary), "programs": list(config["programs"]),
        "generation_limit": config["max_attempts"], "n": config.get("num_generate", 1)})


if __name__ == "__main__":
    main()
