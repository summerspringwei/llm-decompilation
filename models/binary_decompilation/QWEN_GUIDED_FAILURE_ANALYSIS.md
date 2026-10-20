# Qwen Coreutils Failure Analysis and Guided Recovery

## Conclusion

The previous experiment does not isolate Qwen model capability. Its final current
pass accepted 14/20 LLVM IR functions and 12/20 C functions, but reconstructed
0/3 standalone applications. Of eight unsuccessful functions, six ended with a
recorded timeout exception, one lacked execution coverage and valid probes, and
one exhausted its IR budget with compiler/runtime failures. Earlier candidates
inside the timeout cases also contain genuine model mistakes.

The appropriate intervention combines clearer artifact-oriented instructions,
compiler-derived evidence, and correct failure attribution. Increasing the output
ceiling alone did not solve the problem. No conclusion about a different model's
relative ability follows from these results.

Baseline: `validation/project_level_binary_v3_candidates8`.
New experiment: `validation/project_level_binary_v4_guided8`.
The baseline artifacts are preserved. The new experiment uses fresh generations,
not copied accepted bodies or manually repaired candidates.

## Final Baseline Outcomes

| Program | IR Accepted | C Accepted | Standalone |
| --- | ---: | ---: | ---: |
| basename | 3/4 | 2/4 | 0/1 |
| uniq | 4/5 | 3/5 | 0/1 |
| df | 7/11 | 7/11 | 0/1 |
| Total | 14/20 | 12/20 | 0/3 |

Authoritative counts come from `retrieval/<program>/results.json` and
`summary.json`. A historical `latest.json` success may survive a newer failed
request; the report now prefers current per-function results over such artifacts.
In particular, the earlier `perform_basename` pilot is not an extra success in
the final current pass.

| Unsuccessful Function | Terminal Outcome | Interpretation |
| --- | --- | --- |
| basename::oputs_ | Timeout during C stage | Earlier C candidates also used the wrong global symbol |
| basename::perform_basename | Signature timeout | No semantic verdict on a new body |
| uniq::usage | C-stage timeout | IR accepted; C stage interrupted |
| uniq::main | Ten IR rounds exhausted | Genuine syntax and behavior errors remain |
| df::oputs_ | IR-stage timeout | Earlier compiler errors; no final accepted IR |
| df::get_dev | IR-stage timeout | No final accepted IR |
| df::main | IR-stage timeout | No final accepted IR |
| df::find_mount_point | No target execution; probe repair exhausted | Not evidence that the compiled body is wrong |

Regenerate this inventory without inference:

```bash
python -m models.binary_decompilation.failure_audit \
  validation/project_level_binary_v3_candidates8 \
  --output validation/project_level_binary_v4_guided8/diagnostics/baseline_failure_analysis.json
```

Across all saved baseline revisions, including the earlier pilot and restarts,
the IR selections contain 225 compiler/contract failures, 82 runtime failures,
73 response-format failures, 51 accepted candidates, and one missing-execution
candidate. These are candidate counts, not independent functions or final success
rates. Timeout requests without a completed selection are outside these counts.

## Concrete Examples

### Incorrect Global Identity and Misleading Feedback

The binary's persistent flag is `oputs_.help_no_sgr`. Multiple C candidates used:

```c
extern int oputs_help_no_sgr;
```

That declares a different linker symbol. The shared library linked but failed to
load. The saved runtime event contains `undefined symbol: oputs_help_no_sgr`, while
the previous wrapper surfaced `unable to find method 'ping'`. This obscured the
actionable failure and was labeled missing execution instead of a loader error.

The correct binding, when the observed operations support `int`, is:

```c
extern int oputs_help_no_sgr __asm__("oputs_.help_no_sgr");
```

A diagnostic revalidation used the saved baseline candidate
`basename/functions/oputs_/c/b3ddadb4f5c8713376506a5c95e523dc7a2ce10220f7b71fab901f9b028c0fe2/attempt_8/choice_7/candidate.c`.
Changing only that declaration in the supplied header, with the model body
unchanged, changed the result from 0/11 to 11/11 matching CLI cases. The replacement
was entered five times, all during `--help`. This is limited execution evidence,
not exhaustive equivalence. The diagnostic is saved under
`diagnostics/alias_revalidation/comparison.json` in the new output directory and
is explicitly excluded from new model success rates.

Changes: the prompt teaches assembler aliases, binary analysis lists referenced
ELF global names, the wrapper preserves original loader errors, and undefined
symbols in C loader failures can trigger contract repair.

### LLVM Syntax Is Not Assembly Syntax

Saved errors include missing false destinations in `br`, repeated SSA names,
initialized `external global` declarations, and runtime GEP operands written as
constant expressions. For example:

```llvm
; Invalid: runtime indexing written as a constant expression.
; %v = load i8, ptr getelementptr i8, ptr %p, i64 -1

; Valid instruction sequence:
%previous = getelementptr i8, ptr %p, i64 -1
%v = load i8, ptr %previous, align 1
```

The new instructions explicitly require named SSA values, one terminator per
block, correct predecessor lists for phi nodes, typed memory operations, and a
finite set of stack slots when manual SSA is difficult. LLVM verification remains
mandatory; no invalid candidate is silently repaired and counted as model output.

### Option Dispatch Was Not Preserved

`uniq::main` candidate 7 in its last baseline IR round produced `a\nb\n` for
`uniq -c` with input `a\na\nb\n`; the original produced occurrence counts.
Other options also produced incorrect output. This is a real semantic failure.

The optimized function uses a relative jump table at `0x8010`, dispatched at
`0x2d3e`. New binary analysis recognizes the bounds check and signed 32-bit
relative entries and recovers all 130 index-to-target mappings. Known-edge loop
analysis changes from four to nine natural back-edge loops after those edges are
included. The prompt receives explicit case destinations and is instructed to
preserve the stores consumed by downstream functions, not just default output.

The recognizer is deliberately narrow: a local RIP-relative base, scale four,
signed i32 offsets, a matching unsigned upper-bound check dominating the dispatch
under known edges, and every target inside the symbol on an instruction boundary.
Other indirect jumps remain explicitly unresolved. No claim of general jump-table
or indirect-call recovery is made.

### Incorrect Public Record Hypotheses and Probe Schemas

The baseline `find_mount_point` signature described a 144-byte input with a field
at offset 24, but invented a `struct mntent` containing a `struct stat`. Probe
repair then proposed disallowed setup functions and unsupported native types.

The framework now compiles a tiny program using only system headers to measure
`sizeof`, alignment, and field offsets of `struct stat` and `struct option`.
This platform reports `sizeof(struct stat) == 144` and `st_mode` at offset 24.
These are type constraints, not proof that every matching pointer is a stat.
Private records must still be inferred from binary evidence.

The probe prompt supplies an exact schema example that allocates a bounded stat
buffer, initializes it through permitted `stat` setup, and passes it by reference.
Unsupported helper calls and arbitrary raw pointers remain rejected. No original
function is counted correct just because unrelated application cases pass.

### Timeouts and Excessive Output

The old batch loop could lose already completed sibling candidates when a later
stream failed, then abandon the function. The new stream handler saves completed
choices, marks incomplete ones interrupted, records `request_error.json`, and
continues within the same ten-round limit. A timeout still consumes a round;
there is no unlimited hidden retry or reset of a paid generation budget.

The new preset disables extended thinking at all stages and requests a compact
final artifact. It retains the 260000-token output ceiling, bounded by remaining
context, but does not instruct the model to fill it. Read inactivity timeout is
600 seconds rather than 3600; continuously streaming requests can run longer.
This is an operational choice, not a claim that thinking is generally harmful.

Repair feedback now includes extracted code rather than an entire previous
reasoning transcript. Runtime counterexamples preserve exit differences and the
first differing raw-output byte with bounded surrounding excerpts, rather than
truncating away important failures behind a long help message.

## Detailed Instruction Prompts

The executable prompt procedure is [guided_prompts.py](guided_prompts.py).
Export the exact reusable stage instructions:

```bash
python -m models.binary_decompilation.guided_prompts \
  --output-dir validation/project_level_binary_v4_guided8/protocol
```

The exported `signature.txt`, `ir.txt`, `c.txt`, and `probes.txt` are appended to
the corresponding existing output-schema instructions. Each real request saves
the complete instruction, target evidence, and feedback in `attempt_N/prompt.txt`.
The procedure asks for checkable artifacts and concise evidence summaries, not
a transcript of private reasoning.

The common sequence is:

1. Identify entry ABI evidence; separate observations from hypotheses.
2. Reconcile public layouts and private byte accesses; preserve symbol identity.
3. Map blocks, branches, calls, loops, and unresolved indirect transfers.
4. Translate each block with precise widths, side effects, and join values.
5. Verify and compile the complete LLVM module before any execution claim.
6. Execute the replacement and repair from specific compiler or runtime evidence.
7. Translate accepted IR to C, using labels and edge assignments when needed.
8. Validate C independently; never treat IR acceptance as C acceptance.

## Compiler Analysis Implementation

| Component | Implementation | Evidence or Behavior |
| --- | --- | --- |
| Native CFG | `compiler_evidence.py` | ELF symbol bytes decoded with Capstone; basic blocks and direct edges |
| Graph analysis | `compiler_evidence.py` | NetworkX immediate dominators, natural back-edge loops, cyclic components |
| Dispatch tables | `compiler_evidence.py` | Bounded relative-i32 table recognition with explicit residual uncertainty |
| Memory accesses | `compiler_evidence.py` | Base/index/scale/displacement, access width, read/write flags, RIP targets |
| Public ABI | `public_abi.py` | Existing Clang AST/LLVM declarations plus compiler-measured public layouts |
| IR structure | `compiler_evidence.py` | LLVM verify, mem2reg, simplifycfg, instnamer, dominator/loop reports |
| Diagnostic feedback | `validation_feedback.py` | Bounded raw-byte differences, loader errors, per-test replacement coverage |
| Candidate loop | `candidate_search.py` | Finished sibling retention, counted transport failures, artifact-only repairs |
| Reporting | `project_report.py` | Current outcomes override historical successes |

Normalized IR is an aid for the C stage, not a replacement for the accepted IR
artifact. It is produced from model-generated accepted IR, not source-compiled
reference IR. Failed normalization falls back to the original IR with diagnostics.
The framework does not use original application bodies as inference evidence.

Limits: instruction-local registers are not full alias/dataflow recovery; unknown
indirect edges can invalidate graph conclusions; matching offsets do not prove a
type; bounded runtime checks do not prove equivalence; compiler normalization
does not make semantically wrong code correct. Full SSA lifting, symbolic
equivalence checking, and general indirect-call resolution remain future work.

## Reproducible Rerun

Configuration: `configs/coreutils_binary_guided8.json`, extending the previous
eight-candidate preset. Model, binaries, 260K context/output ceilings, retrieval
threshold/collections, tests, eight candidates, and ten-round limits are retained.
All three programs run in parallel; service startup probes remain serialized.
The guided run gives service health checks 60 seconds: a ten-second check timed
out during initial startup despite a subsequent successful embedding probe.
Inference credentials stay in `QWEN_PROJECT_API_KEY`, not saved files.

```bash
python -m models.binary_decompilation.run_experiment \
  --config models/binary_decompilation/configs/coreutils_binary_guided8.json \
  --baseline validation/project_level_binary_v3_candidates8
```

The supervisor saves `runner_status.json`, `parallel_status.json`, worker logs,
full prompts/configuration/compiler evidence, and all candidates/validation
results. On exit it generates `report.json`, `REPORT.md`, and `comparison.json`.
Reusing cached binary analysis is allowed; model responses and acceptance results
are not imported from the baseline. The manual alias diagnostic is not counted.

This rerun is a bundled engineering intervention: prompting, compiler evidence,
diagnostics, transport recovery, and thinking settings change together. It cannot
attribute an improvement to RAG, prompting, compiler analysis, or model capability
individually. A subsequent controlled ablation would hold the repaired harness
and decoding settings fixed and toggle guidance and compiler evidence separately.
Do not report an improved final success rate until the new run finishes.

## Verification

The implementation passed 87 tests, including native PIE replacement tests:

```bash
python -m unittest \
  models.binary_decompilation.test_compiler_evidence \
  models.binary_decompilation.test_candidate_search \
  models.binary_decompilation.test_project_pipeline \
  models.binary_decompilation.test_standalone_recovery \
  models.binary_decompilation.test_native_validation \
  models.binary_decompilation.test_binary_project -q
```

New regressions cover dominators and loop membership, explicit unresolved jumps,
bounded relative table decoding and rejection of invalid targets, memory widths,
compiler-measured public layouts, verified IR normalization and failure fallback,
raw-output counterexamples and coverage, preserved sibling completions after a
stream timeout, bounded transport retry, exact global aliases, loader diagnostics,
and current-run reporting precedence. These are framework tests, not 87 recovered
benchmark functions or evidence of a higher model success rate.
