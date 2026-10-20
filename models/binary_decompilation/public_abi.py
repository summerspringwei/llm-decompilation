"""Compiler-derived system-library ABI hints, never application source types."""

from pathlib import Path
import json
import re

from models.binary_decompilation.project_pipeline import run, write_json


HEADERS = ("stdio.h", "stdlib.h", "string.h", "unistd.h", "ctype.h", "errno.h", "getopt.h", "locale.h", "sys/stat.h")


def public_layouts(clang: str, directory: Path) -> dict:
    """Ask the configured native compiler for layouts of public system records."""
    records = {"struct stat": ["st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid", "st_size"],
               "struct option": ["name", "has_arg", "flag", "val"]}
    source, binary = directory / "public_layouts.c", directory / "public_layouts"
    lines = ['#define _GNU_SOURCE', '#include <stdio.h>', '#include <stddef.h>',
             '#include <sys/stat.h>', '#include <getopt.h>', 'int main(void) {', 'puts("{");']
    for index, (record, fields) in enumerate(records.items()):
        prefix = ("," if index else "") + json.dumps(record) + ':{"size":%zu,"alignment":%zu,"offsets":{'
        lines.append('printf(' + json.dumps(prefix) + ', sizeof(' + record + '), _Alignof(' + record + '));')
        for field_index, field in enumerate(fields):
            text = ("," if field_index else "") + json.dumps(field) + ':%zu'
            lines.append('printf(' + json.dumps(text) + ', offsetof(' + record + ', ' + field + '));')
        lines.append('puts("}}");')
    lines.extend(['puts("}");', 'return 0;', '}'])
    source.write_text('\n'.join(lines) + '\n')
    compiled = run([clang, str(source), '-o', str(binary)])
    write_json(directory / "layouts_compile.json", compiled)
    if compiled["returncode"]:
        raise ValueError(compiled["stderr"])
    measured = run([str(binary.resolve())])
    if measured["returncode"]:
        raise ValueError(measured["stderr"])
    result = {"records": json.loads(measured["stdout"]),
              "provenance": "sizeof/offsetof from system headers and the native configured compiler; no application source"}
    write_json(directory / "public_layouts.json", result)
    return result


def public_abi(clang: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "#define _GNU_SOURCE\n" + "".join(f"#include <{h}>\n" for h in HEADERS)
    source = directory / "public_headers.c"
    source.write_text(prefix)
    ast = run([clang, "-Xclang", "-ast-dump=json", "-fsyntax-only", str(source)])
    write_json(directory / "public_headers_compile.json", {k: v for k, v in ast.items() if k != "stdout"})
    if ast["returncode"]:
        raise ValueError(ast["stderr"])
    functions = {}
    for node in json.loads(ast["stdout"]).get("inner", []):
        if node.get("kind") == "FunctionDecl" and re.fullmatch(r"[A-Za-z_]\w*", node.get("name", "")):
            functions[node["name"]] = node["type"]["qualType"]
    source.write_text(prefix + "void *public_abi_anchors[] = {\n" +
                      ",\n".join("(void *)&" + name for name in functions) + "\n};\n")
    ir = directory / "public_abi.ll"
    compiled = run([clang, "-S", "-emit-llvm", "-O0", str(source), "-o", str(ir)])
    write_json(directory / "public_abi_compile.json", compiled)
    if compiled["returncode"]:
        raise ValueError(compiled["stderr"])
    declarations = {}
    for line in ir.read_text().splitlines():
        if line.startswith("declare "):
            name = line.split("@", 1)[1].split("(", 1)[0]
            declarations[name] = re.sub(r" #[0-9]+$", "", line)
    result = {"headers": list(HEADERS), "functions": {name: {"c_type": functions[name], "llvm_declaration": declaration}
              for name, declaration in declarations.items() if name in functions},
              "provenance": "system headers compiled by the configured toolchain; no application source"}
    result["layouts"] = public_layouts(clang, directory)
    write_json(directory / "public_abi.json", result)
    return result
