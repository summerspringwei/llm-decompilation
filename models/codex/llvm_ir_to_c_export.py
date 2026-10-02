"""Direct LLVM IR to low-level C exporter for ExeBench samples.

This is intentionally not a Ghidra/object-code decompiler.  It translates the
dataset LLVM IR text into C statements that preserve the SSA/control-flow shape:
basic-block labels, gotos, explicit PHI handoff variables, byte-offset GEPs,
and typed load/store operations.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from config import HOME_DIR


DATASET_PATHS = {
    "sampled_dataset_with_loops_164": os.path.join(
        HOME_DIR, "Datasets/filtered_exebench/sampled_dataset_with_loops_164"
    ),
    "sampled_dataset_without_loops_164": os.path.join(
        HOME_DIR, "Datasets/filtered_exebench/sampled_dataset_without_loops_164"
    ),
    "sampled_dataset_with_loops_and_only_one_bb_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
    ),
}

PRELUDE = """#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
typedef float llvmtoc_v2f __attribute__((vector_size(8)));
typedef float llvmtoc_v4f __attribute__((vector_size(16)));
typedef double llvmtoc_v2d __attribute__((vector_size(16)));
typedef int32_t llvmtoc_v2i __attribute__((vector_size(8)));
typedef int32_t llvmtoc_v4i __attribute__((vector_size(16)));
typedef int64_t llvmtoc_v2l __attribute__((vector_size(16)));
typedef uint8_t llvmtoc_v16u8 __attribute__((vector_size(16)));
"""


@dataclass
class Instr:
    text: str
    result: str = ""
    op: str = ""


@dataclass
class Block:
    name: str
    instrs: list[Instr] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export direct LLVM IR-to-C predictions.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def default_output_dir(dataset_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        HOME_DIR, "Projects", "validation", "codex", f"{timestamp}_{dataset_name}_llvm-ir-to-c"
    )


def selected_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.sample_indices.strip():
        return [int(item.strip()) for item in args.sample_indices.split(",") if item.strip()]
    if args.max_samples > 0:
        return list(range(min(args.max_samples, dataset_len)))
    return list(range(dataset_len))


def split_top(s: str, sep: str = ",") -> list[str]:
    out: list[str] = []
    start = depth = angle = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle -= 1
        elif ch == sep and depth == 0 and angle == 0:
            out.append(s[start:i].strip())
            start = i + 1
    tail = s[start:].strip()
    if tail:
        out.append(tail)
    return out


def strip_meta(line: str) -> str:
    parts = split_top(line, ",")
    keep = []
    for part in parts:
        if part.startswith("!") or re.match(r"^#[0-9]+$", part) or part.startswith("align "):
            continue
        keep.append(part)
    return ", ".join(keep)


def clean_name(name: str) -> str:
    name = name.strip()
    if name.startswith("%") or name.startswith("@"):
        name = name[1:]
    name = name.replace(".", "_").replace("-", "_")
    if name and name[0].isdigit():
        name = "v" + name
    return re.sub(r"\W", "_", name)


def label_name(name: str) -> str:
    return "bb_" + clean_name(name)


def cty(ty: str) -> str:
    ty = ty.strip()
    ty = re.sub(r"\s+(noundef|nonnull|nocapture|readonly|writeonly|byval\([^)]*\))\b", "", ty)
    if ty == "void":
        return "void"
    if ty == "i1":
        return "bool"
    if ty == "i8":
        return "uint8_t"
    if ty == "i16":
        return "uint16_t"
    if ty == "i32":
        return "uint32_t"
    if ty == "i64":
        return "uint64_t"
    if ty == "float":
        return "float"
    if ty == "double":
        return "double"
    if ty == "ptr" or ty.startswith("%struct.") or ty.startswith("["):
        return "void *"
    if ty == "<2 x float>":
        return "llvmtoc_v2f"
    if ty == "<4 x float>":
        return "llvmtoc_v4f"
    if ty == "<2 x double>":
        return "llvmtoc_v2d"
    if ty == "<2 x i32>":
        return "llvmtoc_v2i"
    if ty == "<4 x i32>":
        return "llvmtoc_v4i"
    if ty == "<2 x i64>":
        return "llvmtoc_v2l"
    if ty == "<16 x i8>":
        return "llvmtoc_v16u8"
    return "uint64_t"


def sizeof_ty(ty: str, structs: dict[str, list[str]]) -> int:
    ty = ty.strip()
    if ty in ("i1", "i8"):
        return 1
    if ty == "i16":
        return 2
    if ty in ("i32", "float"):
        return 4
    if ty in ("i64", "double", "ptr"):
        return 8
    m = re.match(r"<(\d+) x (.+)>", ty)
    if m:
        return int(m.group(1)) * sizeof_ty(m.group(2), structs)
    m = re.match(r"\[(\d+) x (.+)\]", ty)
    if m:
        return int(m.group(1)) * sizeof_ty(m.group(2), structs)
    if ty in structs:
        off = 0
        max_align = 1
        for field_ty in structs[ty]:
            align = min(sizeof_ty(field_ty, structs), 8)
            max_align = max(max_align, align)
            off = (off + align - 1) // align * align
            off += sizeof_ty(field_ty, structs)
        return (off + max_align - 1) // max_align * max_align
    return 8


def field_offset(struct_ty: str, field_idx: int, structs: dict[str, list[str]]) -> int:
    off = 0
    for i, field_ty in enumerate(structs.get(struct_ty, [])):
        align = min(sizeof_ty(field_ty, structs), 8)
        off = (off + align - 1) // align * align
        if i == field_idx:
            return off
        off += sizeof_ty(field_ty, structs)
    return 0


def parse_module(ir: str) -> tuple[dict[str, list[str]], str, str, list[str], list[Block], list[str]]:
    structs: dict[str, list[str]] = {}
    declares: list[str] = []
    func_header = ""
    body: list[str] = []
    in_func = False
    for raw in ir.splitlines():
        line = raw.strip()
        m = re.match(r"(%struct\.[\w.]+) = type \{(.+)\}", line)
        if m:
            structs[m.group(1)] = split_top(m.group(2))
        if line.startswith("declare "):
            declares.append(line)
        if line.startswith("define "):
            func_header = line
            in_func = True
            continue
        if in_func:
            if line == "}":
                in_func = False
            elif line and not line.startswith(";"):
                body.append(strip_meta(line))
    hm = re.search(r"define .*? ([%@\w<>\[\].*]+) @([\w.$]+)\((.*?)\)", func_header)
    if not hm:
        raise ValueError("could not parse define header")
    ret_ty, fname, arg_s = hm.group(1), hm.group(2), hm.group(3)
    blocks: list[Block] = []
    current = Block("entry")
    for line in body:
        lm = re.match(r"([-\w.$]+):", line)
        if lm:
            if current.instrs or current.name != "entry":
                blocks.append(current)
            current = Block(lm.group(1))
            continue
        if not line:
            continue
        result = ""
        rhs = line
        am = re.match(r"(%[-\w.$]+)\s*=\s*(.+)", line)
        if am:
            result, rhs = am.group(1), am.group(2)
        om = re.match(r"(\w+)", rhs)
        op = om.group(1) if om else ""
        if op == "tail":
            op = "call"
        current.instrs.append(Instr(rhs, result, op))
    blocks.append(current)
    args = split_top(arg_s) if arg_s.strip() else []
    return structs, ret_ty, fname, args, blocks, declares


def val(x: str) -> str:
    x = x.strip()
    x = re.sub(r"^(noundef|nonnull|nocapture|readonly|writeonly)\s+", "", x)
    tm = re.match(r"^(i\d+|float|double|ptr|<[^>]+>)\s+(.+)$", x)
    if tm:
        return val(tm.group(2))
    if x.startswith("%"):
        return "v_" + clean_name(x)
    if x.startswith("@"):
        return "&" + clean_name(x)
    if x == "true":
        return "1"
    if x == "false":
        return "0"
    if x == "null":
        return "0"
    if x == "poison" or x == "undef" or x == "zeroinitializer":
        return "0"
    return x


def typed_value(part: str) -> tuple[str, str]:
    toks = part.strip().split(None, 1)
    if len(toks) == 1:
        return "void", toks[0]
    ty = toks[0]
    rest = toks[1]
    if ty.startswith("<"):
        pieces = part.split(">", 1)
        return pieces[0] + ">", pieces[1].strip()
    return ty, rest


def split_type_value(part: str) -> tuple[str, str]:
    part = part.strip()
    if part.startswith("<"):
        ty, rest = part.split(">", 1)
        return ty + ">", rest.strip()
    return typed_value(part)


def call_expr(text: str) -> str:
    text = re.sub(r"^(tail|musttail|notail)\s+", "", text)
    m = re.match(r"call\s+(.+?)\s+@([\w.$]+)\((.*)\)", text)
    if not m:
        return "0"
    ret_ty, name, args_s = m.group(1), m.group(2), m.group(3)
    args = []
    for arg in split_top(args_s):
        if not arg or arg.startswith("..."):
            continue
        clean = re.sub(r"\b(noundef|nonnull|nocapture|readonly|writeonly)\b", "", arg)
        clean = re.sub(r"\balign\s+\d+\b", "", clean)
        clean = re.sub(r"\bdereferenceable\([^)]*\)", "", clean).strip()
        _, argv = typed_value(clean)
        args.append(val(argv))
    if name.startswith("llvm.lifetime."):
        return ""
    if name.startswith("llvm.memset"):
        return f"memset({args[0]}, {args[1]}, {args[2]})"
    if name.startswith("llvm.memcpy"):
        return f"memcpy({args[0]}, {args[1]}, {args[2]})"
    if name == "llvm.fmuladd.f32" or name == "llvm.fmuladd.f64":
        return f"(({args[0]}) * ({args[1]}) + ({args[2]}))"
    if name.startswith("llvm.abs."):
        return f"(({args[0]}) < 0 ? -({args[0]}) : ({args[0]}))"
    if name.startswith("llvm.smin."):
        return f"(({args[0]}) < ({args[1]}) ? ({args[0]}) : ({args[1]}))"
    if name.startswith("llvm.smax."):
        return f"(({args[0]}) > ({args[1]}) ? ({args[0]}) : ({args[1]}))"
    if name.startswith("llvm."):
        return "0"
    return f"{clean_name(name)}({', '.join(args)})"


def operands_after_type(text: str) -> str:
    rest = text.split(None, 1)[1]
    while True:
        m = re.match(r"(nuw|nsw|exact|fast|nnan|ninf|reassoc|arcp|contract|afn)\s+(.+)", rest)
        if not m:
            break
        rest = m.group(2)
    if rest.startswith("<"):
        return rest.split(">", 1)[1].strip()
    return rest.split(None, 1)[1]


def expr_for(instr: Instr, structs: dict[str, list[str]]) -> str:
    t = instr.text
    op = instr.op
    if op in {"add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr", "sdiv", "udiv", "srem", "urem", "fadd", "fsub", "fmul", "fdiv"}:
        a, b = split_top(operands_after_type(t))[:2]
        sym = {
            "add": "+", "sub": "-", "mul": "*", "and": "&", "or": "|", "xor": "^",
            "shl": "<<", "lshr": ">>", "ashr": ">>", "sdiv": "/", "udiv": "/",
            "srem": "%", "urem": "%", "fadd": "+", "fsub": "-", "fmul": "*", "fdiv": "/",
        }[op]
        return f"({val(a)} {sym} {val(b)})"
    if op == "fneg":
        _, rest = typed_value(t.split(None, 1)[1])
        return f"(-({val(rest)}))"
    if op in {"zext", "sext", "trunc", "fptrunc", "fpext", "sitofp", "uitofp", "fptosi", "ptrtoint", "inttoptr", "bitcast"}:
        m = re.match(r"\w+\s+(.+?)\s+(.+?)\s+to\s+(.+)", t)
        if not m:
            return "0"
        return f"(({cty(m.group(3))})({val(m.group(2))}))"
    if op == "icmp" or op == "fcmp":
        m = re.match(r"\w+\s+(\w+)\s+(.+?)\s+(.+)", t)
        if not m:
            return "0"
        pred, ty, rest = m.group(1), m.group(2), m.group(3)
        a, b = split_top(rest)[:2]
        rel = {
            "eq": "==", "ne": "!=", "ugt": ">", "uge": ">=", "ult": "<", "ule": "<=",
            "sgt": ">", "sge": ">=", "slt": "<", "sle": "<=", "ogt": ">", "oge": ">=",
            "olt": "<", "ole": "<=", "oeq": "==", "one": "!=", "ord": "==", "uno": "!=",
        }.get(pred, "!=")
        return f"({val(a)} {rel} {val(b)})"
    if op == "select":
        m = re.match(r"select\s+i1\s+(.+?),\s+(.+?)\s+(.+?),\s+(.+?)\s+(.+)", t)
        if not m:
            return "0"
        return f"({val(m.group(1))} ? {val(m.group(3))} : {val(m.group(5))})"
    if op == "load":
        m = re.match(r"load\s+(.+?),\s+ptr\s+(.+)", t)
        if not m:
            return "0"
        ty, ptr = m.group(1), m.group(2)
        return f"(*(({cty(ty)}*)({val(ptr)})))"
    if op == "getelementptr":
        m = re.match(r"getelementptr(?: inbounds)?\s+(.+?),\s+ptr\s+(.+)", t)
        if not m:
            return "0"
        base_ty, rest = m.group(1), m.group(2)
        parts = split_top(rest)
        base = parts[0]
        off = "0"
        cur_ty = base_ty
        for idx_part in parts[1:]:
            idx_ty, idx_val = typed_value(idx_part)
            idx = val(idx_val)
            if cur_ty in structs and idx_ty.startswith("i32") and re.match(r"^-?\d+$", idx_val):
                off = f"({off} + {field_offset(cur_ty, int(idx_val), structs)})"
                cur_ty = structs[cur_ty][int(idx_val)] if int(idx_val) < len(structs[cur_ty]) else "i8"
            else:
                elem_size = sizeof_ty(cur_ty, structs)
                off = f"({off} + ({idx}) * {elem_size})"
                am = re.match(r"\[(\d+) x (.+)\]", cur_ty)
                if am:
                    cur_ty = am.group(2)
        return f"((void*)((char*)({val(base)}) + ({off})))"
    if op == "call":
        return call_expr(t)
    if op == "freeze":
        return val(t.split()[-1])
    if op == "insertelement":
        return "0"
    if op == "shufflevector":
        parts = split_top(t.split(None, 1)[1])
        _, vec_val = split_type_value(parts[0])
        return val(vec_val)
    if op == "extractelement":
        parts = split_top(t.split(None, 1)[1])
        vec_ty, vec_val = typed_value(parts[0])
        _, idx = typed_value(parts[1])
        return f"({val(vec_val)}[{val(idx)}])"
    return "0"


def result_type(instr: Instr) -> str:
    t = instr.text
    op = instr.op
    if op in {"icmp", "fcmp"}:
        return "bool"
    if op == "getelementptr" or op == "alloca" or op == "inttoptr":
        return "void *"
    if op == "call":
        m = re.match(r"(?:tail\s+)?call\s+(.+?)\s+@", t)
        return cty(m.group(1)) if m else "uint64_t"
    if op in {"zext", "sext", "trunc", "fptrunc", "fpext", "sitofp", "uitofp", "fptosi", "ptrtoint", "bitcast"}:
        m = re.search(r"\s+to\s+(.+)$", t)
        return cty(m.group(1)) if m else "uint64_t"
    if op == "load":
        m = re.match(r"load\s+(.+?),", t)
        return cty(m.group(1)) if m else "uint64_t"
    if op == "select":
        m = re.match(r"select\s+i1\s+.+?,\s+(.+?)\s+", t)
        return cty(m.group(1)) if m else "uint64_t"
    if op == "phi":
        m = re.match(r"phi\s+(.+?)\s+", t)
        return cty(m.group(1)) if m else "uint64_t"
    if op in {"fadd", "fsub", "fmul", "fdiv", "fneg"}:
        m = re.search(r"\w+\s+(.+?)\s+", t)
        return cty(m.group(1)) if m else "double"
    if op in {"insertelement", "shufflevector"}:
        rest = t.split(None, 1)[1]
        if rest.startswith("<"):
            ty = rest.split(">", 1)[0] + ">"
            return cty(ty)
        m = re.match(r"\w+\s+(.+?)\s+", t)
        return cty(m.group(1)) if m else "uint64_t"
    if op == "extractelement":
        m = re.match(r"extractelement\s+<\d+ x (.+?)>", t)
        return cty(m.group(1)) if m else "uint64_t"
    m = re.search(r"\w+\s+(<[^>]+>|i\d+|float|double|ptr)\s+", t)
    return cty(m.group(1)) if m else "uint64_t"


def parse_phi(instr: Instr) -> list[tuple[str, str]]:
    return [(val(v.strip()), b.strip().lstrip("%")) for v, b in re.findall(r"\[\s*(.+?),\s*%([-\w.$]+)\s*\]", instr.text)]


def emit_branch(instr: Instr, block: Block, phi_inputs: dict[str, list[tuple[str, str, str]]]) -> list[str]:
    t = instr.text
    out: list[str] = []

    def assign_phis(dst: str) -> None:
        for phi_var, pred, value in phi_inputs.get(dst, []):
            if pred == block.name:
                out.append(f"  phi_{phi_var} = {value};")

    if t.startswith("br i1"):
        m = re.match(r"br i1 (.+?), label %([-\w.$]+), label %([-\w.$]+)", t)
        if not m:
            return ["  return;"]
        cond, tlabel, flabel = m.group(1), m.group(2), m.group(3)
        out.append(f"  if ({val(cond)}) {{")
        assign_phis(tlabel)
        out.append(f"    goto {label_name(tlabel)};")
        out.append("  } else {")
        assign_phis(flabel)
        out.append(f"    goto {label_name(flabel)};")
        out.append("  }")
        return out
    if t.startswith("br label"):
        dst = re.search(r"%([-\w.$]+)", t).group(1)
        assign_phis(dst)
        out.append(f"  goto {label_name(dst)};")
        return out
    if t.startswith("switch"):
        m = re.match(r"switch\s+.+?\s+(.+?),\s+label %([-\w.$]+)\s+\[(.*)\]", t)
        if not m:
            return ["  return;"]
        expr, default = m.group(1), m.group(2)
        cases = re.findall(r"(?:i\d+)\s+(-?\d+),\s+label %([-\w.$]+)", m.group(3))
        out.append(f"  switch ({val(expr)}) {{")
        for cv, dst in cases:
            out.append(f"    case {cv}:")
            assign_phis(dst)
            out.append(f"      goto {label_name(dst)};")
        out.append("    default:")
        assign_phis(default)
        out.append(f"      goto {label_name(default)};")
        out.append("  }")
        return out
    if t.startswith("ret void"):
        return ["  return;"]
    if t.startswith("ret "):
        _, rv = typed_value(t[4:])
        return [f"  return {val(rv)};"]
    return []


def prototype_from_declare(line: str) -> str:
    m = re.match(r"declare\s+(.+?)\s+@([\w.$]+)\((.*?)\)", line)
    if not m:
        return ""
    ret, name, args_s = m.group(1), m.group(2), m.group(3)
    if name.startswith("llvm."):
        return ""
    args = []
    for i, arg in enumerate(split_top(args_s)):
        if not arg or arg == "...":
            continue
        ty = arg.split()[0]
        if ty.startswith("<"):
            ty = arg.split(">", 1)[0] + ">"
        args.append(f"{cty(ty)} a{i}")
    return f"extern {cty(ret)} {clean_name(name)}({', '.join(args)});"


def translate(ir: str) -> str:
    structs, ret_ty, fname, args, blocks, declares = parse_module(ir)
    arg_decls = []
    for i, arg in enumerate(args):
        arg = re.sub(r"\b(noundef|nonnull|nocapture|readonly|writeonly|local_unnamed_addr)\b", "", arg).strip()
        m = re.match(r"(.+?)\s+%([-\w.$]+)$", arg)
        if m:
            arg_decls.append(f"{cty(m.group(1))} v_{clean_name(m.group(2))}")
        else:
            arg_decls.append(f"uint64_t arg{i}")

    decls: dict[str, str] = {}
    phi_inputs: dict[str, list[tuple[str, str, str]]] = {}
    for block in blocks:
        for instr in block.instrs:
            if instr.result:
                name = clean_name(instr.result)
                decls[name] = result_type(instr)
                if instr.op == "phi":
                    for value, pred in parse_phi(instr):
                        phi_inputs.setdefault(block.name, []).append((name, pred, value))

    out = [PRELUDE]
    for proto in filter(None, (prototype_from_declare(d) for d in declares)):
        pass
    out.append(f"{cty(ret_ty)} {clean_name(fname)}({', '.join(arg_decls)}) {{")
    for name, ty in sorted(decls.items()):
        out.append(f"  {ty} v_{name} = 0;")
        if any(name == item[0] for items in phi_inputs.values() for item in items):
            out.append(f"  {ty} phi_{name} = 0;")

    for block in blocks:
        out.append(f"{label_name(block.name)}:")
        for instr in block.instrs:
            if instr.op == "phi" and instr.result:
                out.append(f"  v_{clean_name(instr.result)} = phi_{clean_name(instr.result)};")
                continue
            if instr.op == "alloca" and instr.result:
                out.append(f"  v_{clean_name(instr.result)} = __builtin_alloca(1024);")
                continue
            if instr.op == "store":
                m = re.match(r"store\s+(.+?),\s+ptr\s+(.+)", instr.text)
                if m:
                    ty, value = split_type_value(m.group(1))
                    ptr = m.group(2)
                    out.append(f"  *(({cty(ty)}*)({val(ptr)})) = {val(value)};")
                continue
            if instr.op in {"br", "ret", "switch"}:
                out.extend(emit_branch(instr, block, phi_inputs))
                continue
            if instr.op == "call" and not instr.result:
                ce = call_expr(instr.text)
                if ce:
                    out.append(f"  {ce};")
                continue
            if instr.result:
                out.append(f"  v_{clean_name(instr.result)} = {expr_for(instr, structs)};")
        if not block.instrs or block.instrs[-1].op not in {"br", "ret", "switch"}:
            out.append("  return;")
    out.append("}")
    return "\n".join(out) + "\n"


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(DATASET_PATHS[args.dataset_name])
    output_dir = Path(args.output_dir or default_output_dir(args.dataset_name))
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    errors = []
    for idx in tqdm(selected_indices(len(dataset), args), desc="Translating LLVM IR", unit="sample"):
        sample_dir = output_dir / f"sample_{idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            c_code = translate(dataset[idx]["llvm_ir"]["code"][-1])
            (predictions_dir / f"sample_{idx}.c").write_text(c_code)
            (sample_dir / "response.txt").write_text(c_code)
            ok += 1
        except Exception as exc:
            errors.append(idx)
            (sample_dir / "translate.error").write_text(str(exc))
    summary = f"decompile_success: {ok}\ntotal: {len(selected_indices(len(dataset), args))}\nfailed_indices: {errors}\n"
    (output_dir / "export_summary.txt").write_text(summary)
    print(summary)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
