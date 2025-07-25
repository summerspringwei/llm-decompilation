import re
from typing import List
from models.assembly_analyzer.parse_assembly import extract_called_functions
import argparse

def find_call_related(lines: List[str],
                      func_name: str,
                      arch: str = "amd64_sysv") -> List[str]:
    """
    Return all lines related to calling `func_name` in an assembly listing.

    Args:
      lines:      List of assembly lines (strings).
      func_name:  The target function name to look for.
      arch:       Calling convention; currently supports:
                    - "amd64_sysv" (Linux/macOS, args in RDI, RSI, RDX, RCX, R8, R9)
                    - "amd64_win"  (Windows: args in RCX, RDX, R8, R9)
                    - "x86"        (32-bit: push args on stack)
    """
    # define per-ABI argument patterns
    abi_arg_regs = {
        "amd64_sysv": [
            # 64-bit registers
            "rdi", "rsi", "rdx", "rcx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
            # 32-bit registers
            "edi", "esi", "edx", "ecx", "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
            # 16-bit registers
            "di", "si", "dx", "cx", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w",
            # 8-bit registers
            "dil", "sil", "dl", "cl", "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
            # Vector registers used for argument passing
            "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
            # Variable registers
            "al"
        ],
        
        "amd64_win":  ["rcx","rdx","r8","r9"],
    }
    call_re = re.compile(rf"\b(?:callq?|jmp)\s+(?:\w+:)?\b{re.escape(func_name)}(?:@PLT)?\b", re.IGNORECASE)
    related_idxs = set()

    for idx, line in enumerate(lines):
        if call_re.search(line):
            # mark the call itself
            related_idxs.add(idx)

            # scan backward for argument setup / stack adjust
            j = idx - 1
            while j >= 0:
                L = lines[j].strip().lower()
                matched = False

                # stack-based args for 32-bit
                if arch == "x86" and L.startswith("push "):
                    matched = True

                # register args for amd64
                regs = abi_arg_regs.get(arch, [])
                
                # pattern = rf"^\s*((?:v?mov(?:b|l|q|sd|vps))|leaq|pushq|orq?)\s+[^,]+\s*,\s*%(?:{'|'.join(regs)})\b"
                # Match instructions that end with any of the registers
                pattern = rf"^\s*(?:v?mov(?:b|l|q|sd|ss|ups|aps|shdup|pd|ps)|lea|push|or|shl|vshuf|vextract|vcvt|vzeroupper)[a-z]*\s+.*,\s*%(?:{'|'.join(regs)})\b"
                # Also match ymm registers for vector operations
                pattern = rf"^\s*(?:v?mov(?:b|l|q|sd|ss|ups|aps|shdup|pd|ps|aps)|lea|push|or|shl|vshuf|vextract|vcvt|vzeroupper)[a-z]*\s+.*,\s*%(?:{'|'.join(regs)}|ymm[0-9]+)\b"
                if regs and re.match(pattern, L):
                    matched = True

                if matched:
                    related_idxs.add(j)
                    j -= 1
                else:
                    break

            # scan forward for stack cleanup
            # j = idx + 1
            # while j < len(lines):
            #     L = lines[j].strip().lower()
            #     # cdecl cleanup
            #     if (arch == "x86" and L.startswith("add esp")) or (arch != "x86" and L.startswith("add rsp")):
            #         related_idxs.add(j)
            #         j += 1
            #     else:
            #         break

    # return the related lines in original order
    return [lines[i] for i in sorted(related_idxs)]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Find function calls in x86 assembly file')
    # parser.add_argument('asm_file', help='Path to the assembly file to analyze')
    # args = parser.parse_args()

    # First get all the called functions
    asm = "/home/xiachunwei/Projects/alpaca-lora-decompilation/models/gemini/example.s"
    with open(asm, "r") as f:
        asm_str = f.read()
    function_list = extract_called_functions(asm_str)
    for func_name in function_list:
        if func_name.find("@PLT") != -1:
            func_name = func_name.split("@PLT")[0]
        if func_name == "main":
            continue
        rel = find_call_related(asm_str.splitlines(), func_name, arch="amd64_sysv")
        print("\n".join(rel))
        print("==" * 20)