"""Apply conservative syntax normalization to Ghidra C predictions.

The normalizer keeps the decompiled C as the prediction source, but adds common
Ghidra scalar typedefs/macros and rewrites leading-underscore global references
when the dataset dependencies define the non-underscored symbol.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from datasets import load_from_disk

from config import HOME_DIR


DATASET_PATHS = {
    "sampled_dataset_with_loops_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_164",
    ),
    "sampled_dataset_without_loops_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_without_loops_164",
    ),
    "sampled_dataset_with_loops_and_only_one_bb_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
    ),
}

PRELUDE = """#include <stdbool.h>
typedef unsigned char undefined1;
typedef unsigned short undefined2;
typedef unsigned int undefined4;
typedef unsigned long undefined8;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned char byte;
#define SUB84(x,n) (x)
#define SUB82(x,n) (x)
#define CONCAT11(a,b) ((((unsigned int)(a)) << 8) | (unsigned char)(b))
#define CONCAT44(a,b) ((((unsigned long)(a)) << 32) | (unsigned int)(b))
#define SCARRY8(a,b) (((long)(a) > 0 && (long)(b) > 0 && (long)((a) + (b)) < 0) || ((long)(a) < 0 && (long)(b) < 0 && (long)((a) + (b)) >= 0))
#define SBORROW4(a,b) (((int)(a) < 0 && (int)(b) > 0 && (int)((a) - (b)) >= 0) || ((int)(a) >= 0 && (int)(b) < 0 && (int)((a) - (b)) < 0))
"""

IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Ghidra C predictions.")
    parser.add_argument("--dataset_name", default="sampled_dataset_with_loops_164")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def normalize_one(code: str, record: dict) -> str:
    names = set(IDENT_RE.findall(record["synth_deps"]))
    names.update(fn.get("name", "") for fn in record.get("func_info", {}).get("functions", []))
    for name in sorted(names, key=len, reverse=True):
        if not name or name.startswith("_"):
            continue
        code = re.sub(rf"\b_{re.escape(name)}\b", name, code)
    code = code.replace("undefined ", "undefined1 ")
    return PRELUDE + "\n" + code


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(DATASET_PATHS[args.dataset_name])
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, record in enumerate(dataset):
        src = input_dir / f"sample_{idx}.c"
        if not src.exists():
            continue
        (output_dir / f"sample_{idx}.c").write_text(normalize_one(src.read_text(), record))
        count += 1
    print(f"normalized: {count}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
