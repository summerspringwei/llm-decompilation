"""Artifact-oriented decompilation procedures; no application source solutions."""

COMMON = """
Work as a binary compatibility engineer. The objective is observational equivalence
to this optimized ELF function, not reconstruction of familiar source code.
Follow the procedure below, but output only the requested artifact, not a reasoning
transcript. Keep the result finite and compact. Do not spend the output budget on
repeated analysis, alternate solutions, giant register snapshots, or commentary.

Evidence priority: target instructions and relocations; compiler-derived public
ABI/layout facts; observed execution; inferred contracts; retrieved examples.
Retrieval is an analogy, never permission to substitute another function's logic.
An unexecuted branch is unknown, not dead. A syntax-valid contract is a hypothesis.
Do not infer arguments from arbitrary register values or types from function names.
Compiler analysis marked incomplete must not be treated as a complete CFG proof.
Never hardcode runtime addresses, expected test output, or delegate to the original
target. Calls to its existing dependencies are permitted at function-validation
time. Do not introduce an additional original function body.
"""

STAGES = {
    "signature": """
1. Inventory input registers that are read before definition, caller argument setup,
   stack arguments, return-register uses, and tail calls. Distinguish live-in
   callee-saved registers and stack pointers from actual arguments. Mark uncertainty.
2. Record memory accesses as base + byte offset, width, and read/write direction.
   Compare against supplied sizeof/offsetof facts before inventing a record layout.
   In particular, a 144-byte record with mode at offset 24 can be struct stat on
   this platform ONLY when the supplied compiler layout confirms it. Do not invent
   a struct mntent containing stat. Use public headers for public types.
3. Preserve every referenced ELF symbol, including dots. For example, a 4-byte
   signed flag may be declared as:
     extern int help_no_sgr __asm__("oputs_.help_no_sgr");
   Choose its type from observed operations. Never simply rename the ELF symbol,
   omit it because its name is not a C identifier, or give it a local initializer.
4. Declare the exact target and required dependencies once. Match compiler-provided
   libc declarations (including variadic arguments); do not guess libc internals.
   Unknown private records may stay opaque if accesses can use explicit offsets.
5. Check that the header is self-contained with standard includes, forward
   declarations, and extern declarations only. No function bodies or data definitions.
6. Supply bounded direct probes when feedback coverage is absent. Follow the probe
   schema exactly; do not invent setup helpers or native type names. Empty probes
   are better than a knowingly unsafe invocation, but do not establish correctness.
Return the required JSON contract, with brief evidence summaries and uncertainties.
""",
    "ir": r"""
1. Build a block/edge map from the complete symbol's assembly and compiler_cfg.
   Separate calls from control-flow successors. Mark entry, exits, tail calls,
   loop headers/latches, and unresolved indirect jumps. Do not omit option-dispatch
   cases, error exits, or loop updates because they look similar.
2. Establish the function ABI and external declarations. Preserve all shared
   state: @"oputs_.help_no_sgr" = external global i32 has NO initializer.
   ELF GOT slots resolve addresses; they are not additional application dereferences.
3. Translate one basic block at a time using uniquely named SSA values. If joins
   are difficult, use a finite set of entry-block allocas, stores, and loads; the
   compiler can promote them later. Never allocate a new slot per loop iteration.
4. Preserve operation width and signedness. Zero-extending a 32-bit register write
   differs from sign extension. Do not add nsw/nuw/inbounds promises without proof.
   LLVM branches have the complete form: br i1 %cond, label %yes, label %no.
   A GEP using a runtime value is an instruction, not a constant expression:
     %next = getelementptr i8, ptr %p, i64 -1
     %byte = load i8, ptr %next, align 1
   Never write: load i8, ptr getelementptr i8, ptr %p, i64 -1.
5. Each block ends in exactly one terminator. Each phi has one incoming value per
   predecessor; all definitions dominate uses. Values and labels share a local
   namespace: use descriptive names with unique suffixes throughout the function.
6. Preserve every load/store to option flags and persistent globals before calls
   to original helpers. For a dispatcher, trace each case's state updates to the
   consuming call. Correct default output does not validate non-default options.
7. Audit literal bytes and lengths, pointer-vs-integer operations, call signatures,
   and return paths. Return ONE complete LLVM 18 module in a llvm fence.
When repairing, fix the earliest compiler diagnostic first; for runtime failures,
use the first differing byte and failing input to locate a specific path. Preserve
already passing paths. Do not copy expected output into the implementation.
""",
    "c": r"""
1. Treat the validated LLVM IR as the semantic specification. Compiler-normalized
   IR and loop/dominator reports are optional structural aids, not new behavior.
2. Map integer widths explicitly with stdint types, signed comparisons, extension
   operations, truncation, and unsigned wraparound. Preserve the inferred entry ABI.
3. Initially lower basic blocks to C labels/gotos when structure is uncertain.
   Lower phis as edge assignments; cyclic copies require temporaries. Only then
   simplify well-supported loops and conditionals. Structured elegance is secondary.
4. Preserve byte offsets, alignment, and aliasing. Use memcpy for potentially
   unaligned typed accesses; do not dereference invented or incomplete structs.
   Pointer GEP offsets are byte offsets unless the IR explicitly scales them.
5. Preserve external-global identity. Use existing header aliases or a function-
   scope extern with __asm__("exact.ELF.symbol"). Do not substitute a static local
   or an initialized global for live application storage. Standard FILE objects
   are not arbitrary byte arrays; use supplied system declarations where possible.
6. Keep call order, state updates, failure exits, and string bytes identical to IR.
   Do not replace a long function with a plausible standard-library shortcut.
7. Output only the target C definition. Local typedefs and extern declarations are
   allowed if necessary; no other function definitions or source includes. Check
   that every identifier resolves through the project header or a correct extern.
""",
    "probes": """
Use native ABI types, not source spellings: pointer, int, uint64, void (not char*,
struct stat*, string, or integer). Use return_kind cstring ONLY with pointer return.
For a function proved to take (path, struct stat*), this is a schema example, not
a universal probe; use the supplied compiler sizeof(struct stat) for the buffer:
{"return_type":"pointer","argument_types":["pointer","pointer"],
 "arguments":[{"string":"/"},{"ref":"st"}],"return_kind":"cstring",
 "setup":[{"symbol":"stat","return_type":"int",
 "argument_types":["pointer","pointer"],
 "arguments":[{"string":"/"},{"buffer":144,"id":"st"}]}],
 "memory_outputs":[]}
Only stat/lstat/memcpy/memset are allowed setup symbols. Buffers must be allocated
before references; argument counts must match. Never invent malloc/chdir setup,
raw pointer constants, or target-specific helper calls. Choose existing paths and
bounded valid inputs, not guesses at application-global initialization.
""",
}


def guidance(stage):
    return COMMON + STAGES.get(stage, "")


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        (args.output_dir / (stage + '.txt')).write_text(guidance(stage))
