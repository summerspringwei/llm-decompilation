# Program-Analysis-Assisted Assembly to LLVM IR

## Purpose and Current Scope

This experimental path retains Qwen as the author of recovered function bodies.
It uses binary analysis to constrain generation, exposes uncertainty, and compares
matched inference settings instead of attributing a bundled change to one cause.
The previous 14/20 and 6/20 accepted-IR outcomes are historical context, not the
control and treatment measurements for this experiment.

The first implemented version provides raw P-code dataflow, argument/global/field
evidence, literal/declaration scaffolding, bounded scalar semantic checks,
compiler-location repair slices, and a resumable controlled campaign. It does not
implement a general binary-to-LLVM equivalence prover. The limitations below are
part of the experiment, not hidden successful checks.

## Analysis Implementation

`semantic_analysis.py` reads ELF function bytes and lifts individual x86-64
instructions with pypcode/SLEIGH. It never reads application source bodies or
Ghidra decompiled C. Capstone CFG analysis remains a separate input.

Each P-code operation records its source instruction, input/output address spaces,
byte offsets, widths, and register names. LOAD/STORE space handles are removed
because their opaque process-local pointer values are not reproducible evidence.
Unique temporaries are instruction-scoped.

The forward analysis joins reaching definitions per register byte. This handles
overlapping AL/AX/EAX/RAX storage and explicit zero extension in raw P-code.
Constants are must-facts: disagreement at a join drops the fact. Known bytes yield
known-bit masks and conservative unsigned intervals. Exact constant operations
use modular widths; unsupported operations produce unknown output values.
SysV caller-clobbered registers are killed at calls. Calls and returns contribute
implicit argument/return-register uses to slices and block interfaces.

Backward liveness and dependency slices expose the definitions feeding branches,
stores, calls, and returns. A separate affine analysis tracks entry-register plus
constant byte offsets. Differing expressions widen to unknown; stack offsets and
argument-relative access widths form layout constraints, not named C structures.
The call model restores RSP after an ordinary returning call's pushed return PC.

Internal P-code branches invalidate block abstract facts. Unresolved indirect
control flow prevents completeness claims. The relative jump-table recognizer
also requires a dominating upper-half-zero definition when a 32-bit comparison
guards a 64-bit index; `cmp ecx, bound` alone is not enough.

Full facts are saved as `semantic_analysis/analysis_facts.json`. Cache identity
includes binary hash, pypcode version, and analysis implementation hashes. Prompt
packets compact byte sets into ranges and select at most 600 relevant operations;
included and omitted operation counts are recorded. The original assembly is not
silently truncated. A conditional fact always carries its assumptions.

## Generation and Repair

`translation_contract.py` supplies exact public LLVM declarations and optional
byte-escaped string literal definitions from referenced read-only ELF bytes.
NUL-terminated string extent is a hypothesis; copied literals must not substitute
for live-global storage or observable pointer identity.

The model still emits a complete LLVM function/module. No lifted executable body
is inserted. Contracts advise named SSA values or finite stack slots and accept
`asm-block` comments as an unverified correspondence map. Literal declarations
are checked byte-for-byte when their supplied symbols are used. Unproved overflow,
alias, pointer, assumption, and explicit strong memory-alignment annotations are
rejected conservatively; exact verified public declarations are exempt.

`repair_evidence.py` uses a compiler's LLVM line number and a model-provided block
comment to retrieve a bounded binary dependency slice. This is advisory: the map
is not a proof. A stdout mismatch alone is explicitly reported as unlocalized.

RAG retrieves a global pool of up to 20 neighbors from the existing eight Qdrant
collections and retains at most two. Available numeric structural payload fields
provide a small reranking penalty; absent metadata falls back to similarity.
Retrieved examples are frozen in a shared content-addressed cache across arms.
No claim of structural reranking is made when payload metadata is absent.

## Validation and Acceptance

Existing LLVM verification, native compilation, live dependency binding, and
Frida replacement tests remain mandatory. CLI output is compared byte-for-byte.
Candidate ranking uses target-exercising successes/failures rather than unrelated
passing tests. Direct-probe feedback now includes nested return/output mismatches.
Infrastructure failures are distinct from semantic errors and stop further rounds
for that stage rather than prompting the model to repair a ptrace failure.

`semantic_validation.py` uses angr/VEX and Claripy for a deliberately small domain:
integer, call-free functions without explicit memory accesses or stack frames.
It compares candidate and original machine-code return bits under shared symbolic
entry registers, checks path coverage, and concretely replays solver mismatches.
Limits are 30 seconds, 64 paths, and 256 execution steps, with a subprocess wall
timeout. Unsupported instructions, unmodeled initial flags, nontermination within
bounds, memory effects, and calls produce `unknown`.

Results are `bounded_equivalent`, `counterexample`, or `unknown`. Return width
comes from the candidate and is not independent ABI proof. Concrete replay uses
VEX, not a second native invocation. Caller reachability is not established, so
counterexamples are advisory feedback and do not independently override observed
native test acceptance. An unknown result never upgrades acceptance.

## Controlled Experiment

The supervisor is `controlled_ir_experiment.py`. It runs three benchmark workers
in parallel within each campaign, but serializes arms to avoid inference load
confounding. Each candidate stage has eight choices and at most ten rounds.
IR thinking stays enabled in every arm; signature/probe thinking remains disabled.
The context is 260000 tokens and the output cap is bounded by remaining context.
Generation seeds reach the API and are saved with request metadata.

| Arm | Intervention |
| --- | --- |
| A | Repaired harness and existing compiler CFG evidence |
| B | A plus P-code, dataflow, memory, and call-ABI evidence |
| C | B plus translation contracts and conservative IR attribute checks |
| D | C plus bounded scalar checks and compiler repair slices |

Screening uses eight functions: basename oputs_/perform_basename/main; uniq
find_field/writeline/main; df find_mount_point/get_dev. Select B/C/D by development
acceptance, then measured tokens, then simpler intervention. Missing token usage
cannot win a cost tie. Selection never reads held-out results.

Then run A and the selected arm on all 20 targets with seeds 1729 and 1730.
Screening responses are not copied into full runs: changed callee contexts would
invalidate generation identity. At most 112 function evaluations are scheduled
across screening and full runs, excluding signature/probe sub-stages. Execution
stops early for accepted functions within their generation budget.

Held-out cases are frozen before inference and kept outside generation configs.
After full generation, evaluate selected IR candidates without feeding failures
back. Missing held-out target coverage is reported as unvalidated, not accepted.
DF cases avoid volatile used-space/inode counts. These tests are not the complete
upstream Coreutils test suite. Successful replacement is not standalone recovery.

Configurations and implementation hashes are immutable once frozen. A restart
with changed code/configuration requires a new output root. Existing request
markers prevent resending interrupted paid batches.

```bash
# Freeze the protocol without inference.
python -m models.binary_decompilation.controlled_ir_experiment \
  --output validation/program_analysis_ir_v2

# Credentials must already be in QWEN_PROJECT_API_KEY; they are never saved.
python -m models.binary_decompilation.controlled_ir_experiment \
  --output validation/program_analysis_ir_v2 --run
```

Artifacts include protocol/config/implementation snapshots, shared RAG results,
every request and candidate, per-function facts and validation, screening selection,
held-out results, and final paired comparison. `runner_status.json` reports the
active arm; process completion is not a success claim.

## Tests and Remaining Work

The regression suite currently covers 107 tests, including real native replacement,
modular constants, partial registers, flags, loops, call clobbers, affine stack and
field accesses, uncertain jumps, literal/attribute checks, counterexample replay,
repair-map uncertainty, frozen protocols, disjoint held-out cases, and seed passing.

```bash
python -m unittest \
  models.binary_decompilation.test_semantic_analysis \
  models.binary_decompilation.test_controlled_ir_experiment \
  models.binary_decompilation.test_compiler_evidence \
  models.binary_decompilation.test_candidate_search \
  models.binary_decompilation.test_project_pipeline \
  models.binary_decompilation.test_standalone_recovery \
  models.binary_decompilation.test_native_validation \
  models.binary_decompilation.test_binary_project -q
```

Not yet implemented from the broader design: arbitrary region-to-region semantic
correspondence, symbolic memory/call-effect equivalence, automatic branch-directed
function-input generation, and a joint interprocedural ABI/type constraint solver.
Current ABI/layout evidence informs Qwen but does not prove inferred signatures.
Constant/known-byte intervals are not a relational range analysis. Runtime indirect
targets are not yet merged into the P-code CFG. These capabilities must not be
claimed from the current artifacts or from passing differential tests.

The initial analysis-only inventory lifted all 20 targets without decoding errors
and converged; df get_dev/main retain explicit uncertainty. The v1 directory is an
analysis/protocol draft with no inference; the corrected controlled protocol is v2.
No improved decompilation rate is claimed before the paired runs finish.
