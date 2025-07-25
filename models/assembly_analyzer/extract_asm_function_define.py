import re
from pathlib import Path
from typing import Dict, List

FUNC_START  = re.compile(r"""^
    \s*          # optional leading whitespace
    \.globl\s+   # '.globl ' directive
    (?P<name>\S+)  # the symbol that follows
    .*?          # anything up to end‑of‑line
    $""", re.X)

LABEL        = re.compile(r"^\s*(?P<name>\S+):")       # e.g. 'timer_setup:'
FUNC_END     = re.compile(r"^\s*\.cfi_endproc")        # reliable end‑marker

def split_elf_functions(text: str) -> Dict[str, List[str]]:
    """
    Parse the textual assembly listing in *text* and
    return { function_name : [ lines … ] }.
    """
    functions: Dict[str, List[str]] = {}
    cur_name: str | None = None

    for line in text.splitlines():
        # 1) See a '.globl <sym>'  → remember that the next label is <sym>.
        m_globl = FUNC_START.match(line)
        if m_globl:
            pending_name = m_globl.group("name")
            continue

        # 2) As soon as we meet a label, that starts the body.
        m_label = LABEL.match(line)
        if m_label:
            label = m_label.group("name")
            # If this is the label that follows a '.globl', open a new bucket.
            if "pending_name" in locals() and label == pending_name:
                cur_name = label
                functions[cur_name] = [line]
                del pending_name
                continue

        # 3) If we are inside a function, keep collecting lines.
        if cur_name is not None:
            functions[cur_name].append(line)
            if FUNC_END.match(line):
                # Reached the canonical end of the function.
                cur_name = None

    return functions


# ---------------- demo ----------------
if __name__ == "__main__":
    asm_text = Path("/tmp/exebench_dss3090_1000482mo8b34e6.s").read_text(encoding="utf-8")
    funcs = split_elf_functions(asm_text)

    # Pretty‑print the result
    for name, body in funcs.items():
        # print(f"=== {name} ({len(body)} lines) ===")
        # for l in body:
        #     print(l)
        # print()
        print(name)
