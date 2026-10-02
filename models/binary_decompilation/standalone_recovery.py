"""Generated-only application builds, with explicit external support dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection

from models.binary_decompilation.export_source_references import find_definition_span
from models.binary_decompilation.project_pipeline import code_block, compiler_feedback, run, write_json
from models.binary_decompilation.prompt_budget import fit_prompt


def binary_data_evidence(binary: Path, ownership_object: Path, targets: list[dict]) -> dict:
    """Use build-object names for ownership only; obtain values from the binary."""
    with ownership_object.open("rb") as stream:
        elf = ELFFile(stream)
        table = elf.get_section_by_name(".symtab")
        owned = {symbol.name for symbol in table.iter_symbols()
                 if symbol["st_info"]["type"] == "STT_OBJECT"} if table else set()
    references = {int(address, 16) for f in targets for address in re.findall(
        r"#\s*([0-9a-f]+)\s+<", f.get("symbolized_disassembly", ""))}
    with binary.open("rb") as stream:
        elf = ELFFile(stream)
        sections = list(elf.iter_sections())
        symbols = elf.get_section_by_name(".symtab")
        relocations = []
        for section in sections:
            if not isinstance(section, RelocationSection):
                continue
            symtab = elf.get_section(section["sh_link"])
            for relocation in section.iter_relocations():
                index = relocation["r_info_sym"]
                relocations.append({"address": relocation["r_offset"],
                                    "type": relocation["r_info_type"],
                                    "addend": relocation.entry.get("r_addend"),
                                    "symbol": symtab.get_symbol(index).name if index else None})
        objects = []
        for symbol in symbols.iter_symbols() if symbols else []:
            index = symbol["st_shndx"]
            if symbol.name not in owned or not isinstance(index, int):
                continue
            section = sections[index]
            address, size = symbol["st_value"], symbol["st_size"]
            offset = address - section["sh_addr"]
            associated = [r for r in relocations if address <= r["address"] < address + size]
            references.update(r["addend"] for r in associated if r["addend"] is not None)
            objects.append({"name": symbol.name, "address": address, "size": size,
                            "section": section.name, "zero_initialized": section["sh_type"] == "SHT_NOBITS",
                            "initial_bytes_hex": "" if section["sh_type"] == "SHT_NOBITS" else
                            section.data()[offset:offset + size].hex(), "relocations": associated})
        strings = []
        for address in sorted(references):
            for section in sections:
                if section.name != ".rodata" or not section["sh_addr"] <= address < section["sh_addr"] + section["sh_size"]:
                    continue
                data = section.data()[address - section["sh_addr"]:]
                terminator = data.find(b"\0")
                if terminator < 0:
                    continue
                value = data[:terminator]
                if value and all(byte in (9, 10, 13, 27) or 32 <= byte <= 126 for byte in value):
                    strings.append({"address": address, "text": value.decode("ascii")})
        return {"elf_class": elf.elfclass, "little_endian": elf.little_endian,
                "machine": elf["e_machine"], "objects": objects, "referenced_strings": strings,
                "ownership_oracle": "build object symbol names; no source bodies or declared types"}


def generated_includes_only(text: str) -> None:
    for delimiter, name in re.findall(r'^\s*#\s*include\s*([<"])([^>"\n]+)[>"]', text, re.M):
        if ".." in name.split("/") or name.startswith("/") or not name.endswith(".h"):
            raise ValueError("Disallowed generated include: " + name)
        if delimiter == '"' and name != "project.h":
            raise ValueError("Only project.h may be included as a local header")
    for line in text.splitlines():
        if re.match(r"^\s*#\s*include\b", line) and not re.match(r'^\s*#\s*include\s*[<"]', line):
            raise ValueError("Macro-expanded includes are not allowed in generated application code")


def support_dependencies(adapter) -> list[Path]:
    options = adapter.config["programs"][adapter.program].get("standalone", {})
    defaults = [*adapter.extra_objects, "src/libver.a", "lib/libcoreutils.a"]
    dependencies = [((adapter.build / p) if not Path(p).is_absolute() else Path(p)).resolve()
                    for p in options.get("support_objects", defaults)]
    original = (adapter.build / "src" / (adapter.program + ".o")).resolve()
    if original in dependencies:
        raise ValueError("Original application object is not an external support dependency")
    return dependencies


def build_generated_project(adapter, functions: list[dict], header: str, globals_c: str,
                            directory: Path, kind: str) -> dict:
    """No source scaffold or original application object enters this build."""
    if kind not in ("c", "ir"):
        raise ValueError("Unknown project representation: " + kind)
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    generated_includes_only(header)
    generated_includes_only(globals_c)
    dependencies = support_dependencies(adapter)
    required = {f["name"] for f in functions}
    if "main" not in required or len(required) != len(functions):
        return {"accepted": False, "feedback": "Unique application targets including main are required"}
    for dependency in dependencies:
        symbols = run(["nm", "--defined-only", str(dependency)])
        if symbols["returncode"]:
            return {"accepted": False, "feedback": symbols["stderr"]}
        for line in symbols["stdout"].splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[1] in "tT" and parts[2] in required:
                return {"accepted": False, "feedback": "Support dependency defines application target " + parts[2]}
    (directory / "project.h").write_text(header)
    globals_file = directory / "globals.c"
    globals_file.write_text('#include "project.h"\n' + globals_c)
    glue_ast = run([adapter.clang, "-std=gnu17", "-D_GNU_SOURCE", "-Xclang", "-ast-dump=json",
                    "-fsyntax-only", str(globals_file)])
    if glue_ast["returncode"]:
        return {"accepted": False, "feedback": compiler_feedback(glue_ast["stderr"])}
    glue_definitions = {node.get("name") for node in json.loads(glue_ast["stdout"])["inner"]
                        if node["kind"] == "FunctionDecl" and any(
                            child["kind"] == "CompoundStmt" for child in node.get("inner", []))}
    if required & glue_definitions:
        return {"accepted": False, "feedback": "Glue must not substitute application functions: " +
                str(sorted(required & glue_definitions))}
    provenance = {"representation": kind, "application_source_scaffold": False,
                  "targets": sorted(required), "generated_inputs": [],
                  "support_dependencies": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                           for p in dependencies]}
    objects = []
    if kind == "c":
        pieces = ['#include "project.h"\n', globals_c]
        for function in functions:
            path = Path(function["c_candidate"])
            body = path.read_text()
            generated_includes_only(body)
            if find_definition_span(body, function["name"]) is None:
                return {"accepted": False, "feedback": "Missing generated C definition " + function["name"]}
            pieces.append(body)
            provenance["generated_inputs"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        source = directory / "recovered.c"
        source.write_text("\n".join(pieces))
        obj = directory / "application.o"
        command = [adapter.clang, "-std=gnu17", "-D_GNU_SOURCE", "-O3", "-Werror=implicit-function-declaration",
                   "-c", str(source), "-o", str(obj)]
        compiled = run(command)
        write_json(directory / "compile.json", compiled)
        if compiled["returncode"]:
            return {"accepted": False, "feedback": compiler_feedback(compiled["stderr"])}
        ast = run([adapter.clang, "-std=gnu17", "-D_GNU_SOURCE", "-Xclang", "-ast-dump=json",
                   "-fsyntax-only", str(source)])
        if ast["returncode"]:
            return {"accepted": False, "feedback": compiler_feedback(ast["stderr"])}
        definitions = {node.get("name") for node in json.loads(ast["stdout"])["inner"]
                       if node["kind"] == "FunctionDecl" and any(
                           child["kind"] == "CompoundStmt" for child in node.get("inner", []))}
        if required - definitions:
            return {"accepted": False, "feedback": "Missing C AST definitions: " + str(sorted(required - definitions))}
        objects.append(obj)
    else:
        globals_obj = directory / "globals.o"
        compiled = run([adapter.clang, "-std=gnu17", "-D_GNU_SOURCE", "-O0", "-femit-all-decls",
                        "-c", str(globals_file), "-o", str(globals_obj)])
        write_json(directory / "globals_compile.json", compiled)
        if compiled["returncode"]:
            return {"accepted": False, "feedback": compiler_feedback(compiled["stderr"])}
        symbols = run(["nm", "--defined-only", str(globals_obj)])["stdout"]
        local = []
        for line in symbols.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[1] in "tT" and parts[2] in required:
                return {"accepted": False, "feedback": "Global-data unit must not substitute application functions"}
            if len(parts) == 3 and parts[1] in "bdr" and re.fullmatch(r"[A-Za-z_]\w*", parts[2]):
                local.append(parts[2])
        public_globals = directory / "globals.public.o"
        published = run(["objcopy", *["--globalize-symbol=" + n for n in local], str(globals_obj), str(public_globals)])
        if published["returncode"]:
            return {"accepted": False, "feedback": published["stderr"]}
        objects.append(public_globals)
        for index, function in enumerate(functions):
            path = Path(function["ir_candidate"])
            obj = directory / f"function_{index}.o"
            compiled = run([adapter.clang, "-Wno-override-module", "-c", str(path), "-o", str(obj)])
            write_json(directory / f"ir_compile_{index}.json", compiled)
            if compiled["returncode"]:
                return {"accepted": False, "feedback": compiled["stderr"][-12000:]}
            defined = run(["nm", "--defined-only", str(obj)])["stdout"]
            if not any(len(p := line.split()) == 3 and p[1] in "tT" and p[2] == function["name"]
                       for line in defined.splitlines()):
                return {"accepted": False, "feedback": "IR module does not define " + function["name"]}
            public = directory / f"function_{index}.public.o"
            published = run(["objcopy", "--globalize-symbol=" + function["name"], str(obj), str(public)])
            if published["returncode"]:
                return {"accepted": False, "feedback": published["stderr"]}
            objects.append(public)
            provenance["generated_inputs"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    provenance["generated_inputs"].extend({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                         for p in (directory / "project.h", globals_file))
    write_json(directory / "build_provenance.json", provenance)
    executable = directory / "recovered"
    options = adapter.config["programs"][adapter.program].get("standalone", {})
    linked = run([adapter.clang, *map(str, objects), "-Wl,--start-group", *map(str, dependencies),
                  "-Wl,--end-group", *options.get("link_flags", []), "-o", str(executable)])
    write_json(directory / "link.json", linked)
    if linked["returncode"]:
        return {"accepted": False, "feedback": linked["stderr"][-12000:], "provenance": provenance}
    tests = adapter.test(executable, directory)
    write_json(directory / "tests.json", tests)
    return {"accepted": tests["failed"] == 0 and tests["passed"] > 0,
            "feedback": json.dumps([t for t in tests["tests"] if t["returncode"] not in (0, 77)])[-12000:],
            "tests": {key: tests[key] for key in ("passed", "failed", "skipped")},
            "executable": str(executable), "provenance": provenance}


def recover_standalone(client, config: dict, adapter, plan: dict, results: list[dict], directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    required = {f["name"] for f in plan["targets"]}
    complete = {r["name"] for r in results if r.get("ir", {}).get("accepted") and r.get("c", {}).get("accepted")}
    if required - complete:
        result = {"accepted": False, "reason": "missing validated functions",
                  "missing": sorted(required - complete), "application_source_scaffold": False}
        write_json(directory / "result.json", result)
        return result
    final_path = directory / "result.json"
    if final_path.exists():
        cached = json.loads(final_path.read_text())
        if cached.get("accepted") or cached.get("attempts") == config["max_attempts"]:
            return cached
    functions = [{"name": r["name"], "c_candidate": r["c"]["candidate"], "ir_candidate": r["ir"]["candidate"]}
                 for r in results]
    data = binary_data_evidence(adapter.binary, adapter.build / "src" / (adapter.program + ".o"), plan["targets"])
    write_json(directory / "binary_data.json", data)
    evidence = {"binary_data": data, "functions": [{"name": r["name"], "signature": r["signature"]["signature"],
                "validated_llvm_ir": Path(r["ir"]["candidate"]).read_text(),
                "generated_c": Path(r["c"]["candidate"]).read_text()} for r in results],
                "support_dependencies": [str(p) for p in support_dependencies(adapter)]}
    instruction = (
        "Recover application types, declarations and global initializers from binary evidence. "
        "Build an application without any original application C or object files. External support libraries "
        "listed in the evidence remain available. Return ONLY JSON with project_header, globals_c, and "
        "function_replacements (a map of target name to revised C definition; use only when necessary). "
        "The header may include standard system headers, but not original Coreutils headers or C files. "
        "Preserve linkage and layouts in the validated LLVM IR. Declare all target functions consistently. "
        "Do not define application functions in globals_c or the header. Any C revision must implement "
        "the provided validated LLVM IR, not delegate to original functions. Do not invent missing target names."
    )
    feedback = ""
    for attempt in range(config["max_attempts"]):
        root = directory / f"attempt_{attempt}"
        root.mkdir(exist_ok=True)
        try:
            prompt, budget = fit_prompt(config, instruction, evidence, feedback)
        except ValueError as error:
            result = {"accepted": False, "reason": "context_window_exceeded", "feedback": str(error),
                      "attempts": attempt, "application_source_scaffold": False}
            write_json(final_path, result)
            return result
        if not (root / "context_budget.json").exists():
            write_json(root / "context_budget.json", budget)
        response_path = root / "response.json"
        if response_path.exists():
            response = json.loads(response_path.read_text())
        else:
            (root / "prompt.txt").write_text(prompt)
            write_json(root / "generation_config.json", {"model": config["model"], "n": 1,
                       "temperature": 0, "max_tokens": config.get("max_tokens", 8192), "enable_thinking": False})
            response = client.chat.completions.create(model=config["model"], n=1, temperature=0,
                max_tokens=config.get("max_tokens", 8192), messages=[{"role": "user", "content": prompt}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}).model_dump()
            write_json(response_path, response)
        text = response["choices"][0]["message"]["content"] or ""
        try:
            candidate = json.loads(code_block(text, ("json",)))
            revised = [dict(f) for f in functions]
            for name, body in candidate.get("function_replacements", {}).items():
                if name not in required:
                    raise ValueError("Unknown application target: " + name)
                replacement = root / (name + ".c")
                replacement.write_text(body)
                next(f for f in revised if f["name"] == name)["c_candidate"] = str(replacement)
            validations = {kind: build_generated_project(adapter, revised, candidate["project_header"],
                          candidate["globals_c"], root / kind, kind) for kind in ("ir", "c")}
        except (ValueError, KeyError) as error:
            validations = {"error": str(error)}
        write_json(root / "validation.json", validations)
        if validations.get("ir", {}).get("accepted") and validations.get("c", {}).get("accepted"):
            result = {"accepted": True, "attempts": attempt + 1, "validation": validations,
                      "application_source_scaffold": False, "standalone_application_build": True,
                      "complete_project_recovery": False,
                      "remaining_gate": "whole-suite coverage and recovery-scope audit"}
            write_json(final_path, result)
            return result
        feedback = json.dumps(validations)[-16000:] + "\nPrevious glue:\n" + text[-16000:]
    result = {"accepted": False, "attempts": config["max_attempts"], "feedback": feedback,
              "application_source_scaffold": False, "complete_project_recovery": False}
    write_json(final_path, result)
    return result
