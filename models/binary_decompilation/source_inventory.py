"""Evaluation-only ownership metadata, extracted using Clang rather than regex."""

from pathlib import Path
import json

from models.binary_decompilation.project_pipeline import run, write_json


def clang_inventory(config: dict, program: str, directory: Path) -> dict:
    source, build = Path(config["source_dir"]), Path(config["build_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    units = [f"src/{program}.c", *config["programs"][program].get("ownership_units", [])]
    definitions = {}
    commands = []
    for unit in units:
        result = run([config["clang"], "-I.", "-I..", "-I./lib", "-Ilib", "-I../lib",
                      "-Isrc", "-I../src", "-Wno-format-extra-args", "-Xclang", "-ast-dump=json",
                      "-fsyntax-only", str(source / unit)], build)
        commands.append({"command": result["command"], "returncode": result["returncode"],
                         "stderr": result["stderr"]})
        if result["returncode"]:
            raise ValueError("Source inventory failed: " + result["stderr"][-4000:])
        last_file = str(source / unit)
        for node in json.loads(result["stdout"]).get("inner", []):
            loc = node.get("loc", {})
            loc = loc.get("spellingLoc", loc)
            if loc.get("file"):
                last_file = loc["file"]
            if node.get("kind") != "FunctionDecl" or not any(
                    c["kind"] == "CompoundStmt" for c in node.get("inner", [])):
                continue
            path = Path(last_file)
            if not path.is_absolute():
                path = build / path
            try:
                relative = path.resolve().relative_to(source.resolve())
            except ValueError:
                continue
            definitions[node["name"]] = {"source_file": str(relative),
                                          "inline": bool(node.get("inline", False))}
    inventory = []
    for unit in units:
        obj = build / Path(unit).with_suffix(".o")
        nm = run(["nm", "--defined-only", str(obj)])
        if nm["returncode"]:
            raise ValueError(nm["stderr"])
        for line in nm["stdout"].splitlines():
            parts = line.split()
            if len(parts) != 3 or parts[1] not in "tT":
                continue
            name = parts[2]
            if name in definitions:
                inventory.append({"name": name, **definitions[name], "object": str(obj),
                                  "application_helper": definitions[name]["source_file"] != units[0]})
    retained = {item["name"] for item in inventory}
    result = {"functions": inventory, "source_definitions": sorted(definitions),
              "inlined_or_not_emitted": [{"name": name, **metadata} for name, metadata in definitions.items()
                                         if metadata["source_file"] in units and name not in retained],
              "commands": commands, "inference_visibility": "names and ownership only; no types or bodies"}
    write_json(directory / "source_inventory.json", result)
    return result
