# Project-Level Binary Recovery Workflow

The revised eight-candidate preset is `coreutils_binary_candidates8.json`.
The completed v2 experiment below remains a historical single-candidate result.

## Scope and Evidence Boundary

The executable is the primary semantic input. Coreutils source is used to
identify ownership, configure tests, and export references after each recovery
attempt. Original application declarations, bodies, and Ghidra-generated C are
not primary model inputs. Public system headers supply external-library ABI
declarations. Source-scaffold-assisted recovery is a separate comparison track.

This implementation targets the existing Coreutils 9.10 Clang 18 build with
`-O3 -fno-omit-frame-pointer`. It does not promise recovery of the complete ELF:
libc, startup code, `src/libver.a`, and `lib/libcoreutils.a` are approved external
dependencies. The `df` application helper `find_mount_point` is a recovery
target, not an approved original object to include in standalone builds.

The inventories contain four `basename`, five `uniq`, and eleven `df` targets.
Retained private header helpers are included. Functions completely removed or
inlined by optimization are recorded separately, not assigned invented bodies.

## Run Commands

Use the repository's Python environment with Frida 17.4.0, pyelftools, Capstone,
and the configured Clang/LLVM toolchain installed. Set the remote model key in
the environment; configuration and artifacts must not contain credentials.

```bash
export QWEN_PROJECT_API_KEY='<remote service key>'
python -m models.binary_decompilation.binary_project \
  --config models/binary_decompilation/configs/coreutils_binary_candidates8.json \
  --prepare-only
python -m models.binary_decompilation.binary_project \
  --config models/binary_decompilation/configs/coreutils_binary_candidates8.json
python -m models.binary_decompilation.project_report \
  validation/project_level_binary_v3_candidates8
```

The configured remote endpoint is `http://10.208.130.107:9001/v1`; the model is
`Qwen3.8-27B-FP8`. Qdrant and HermesSim remain local. The HermesSim collections
have 384-dimensional vectors; older 4096-dimensional collections are not
interchangeable. Preflight checks the model, embedding service, collections,
and dimensions before generating candidates.

The configured context window is 260,000 tokens per sequence. It includes
prompt and completion tokens, including reasoning. Every stage now has a
260,000-token output ceiling. The `remaining_context` policy sends the smaller
of that ceiling and `260000 - prompt_tokens - 64` as the request's `max_tokens`.
Requested and effective output limits are saved separately; the historical
pilot used smaller stage limits. Eight candidates do not multiply the
per-sequence context limit.
Add `--program basename --function perform_basename --track retrieval` for a
focused run; the full target inventory still governs application acceptance.

## 1. Inventory and Analyze

The ownership inventory uses Clang AST metadata and emitted object symbols.
Ghidra exports provide CFG evidence and static calls; exports are reused only
for a matching executable hash. Ghidra addresses are normalized to ELF-relative
addresses. `objdump` supplies the complete symbolized disassembly. Conservative
address-taken dependencies augment direct-call edges.

Each function receives a stable executable-hash/address ID. Strongly connected
components are scheduled callee-first. Contracts for all members of a component
are attempted before its bodies, supporting mutually dependent declarations.

## 2. Establish the Runtime Baseline

Run upstream tests and differential CLI cases against the original executable.
Then run instrumented baselines before any model request. A preload constructor
holds the process before `main`, allowing Frida to install entry/return hooks.
Events record bounded argument registers, readable memory, global snapshots,
and fuzzy call stacks, tagged by test ID.

Upstream traces contribute coverage and call edges. Only differential feedback
cases supply value snapshots to prompts. Entry sampling stops after eight
observations per function, so counts are not execution frequencies. The `df`
test that masks `/proc` is excluded only from instrumentation: it remains in
original and final standalone suites. Locale-dependent skips remain skips.

## 3. Infer Contracts and Recover Bodies

For each target, retrieve assembly neighbors through HermesSim/Qdrant. Save the
query identity, collection configuration, examples, and latency. The no-retrieval
track omits this step while retaining the same binary analysis evidence.
The revised preset keeps at most two examples with scores of at least 0.7.
This is a relevance heuristic, not a semantic-equivalence threshold. Cache
reuse requires a matching query and retrieval configuration.

Generate a JSON contract containing the signature, an inferred project header,
and optional bounded direct probes. Probe type names follow the
[Frida NativeFunction API](https://frida.re/docs/javascript-api/#nativefunction).
Validate types, allocated-buffer references, and output bounds before execution.
Compile declarations and reject function
bodies masquerading as header declarations. Syntax acceptance is not ABI proof.
Malformed probes are reported separately and do not invalidate an otherwise
syntax-valid signature. If a compiled body lacks execution evidence, a separate
probe-repair stage generates bounded probes and revalidates that same body.
Probe schema acceptance alone never accepts a function.

Generate LLVM IR from complete assembly, CFG/call information, inferred callee
contracts, binary data, public ABI declarations, and runtime/retrieval evidence.
Execute `opt -passes=verify -disable-output`, compile PIC code, and test the
candidate in the original process. Only accepted IR may advance to IR-to-C.
Generate C from that IR and inferred contracts, then compile and test it using
the same native isolation backend. Header/layout compilation failures may
trigger contract repair and IR revalidation under the existing budgets.

Signature, IR, and C stages use `n=8`, temperature 0.6, top-p 0.95,
and at most ten request rounds each. Thinking is enabled for IR and C but
disabled for signature/probe JSON: in the live pilot all eight thinking-enabled
signature candidates exhausted 8,192 tokens without a final answer.
Probe repair has at most three
rounds of eight candidates. Integration uses one glue candidate per round,
with at most ten rounds. Each streamed choice is isolated by index; a repetitive
choice does not cancel other candidates. All returned candidates are retained,
and duplicate responses reuse validation evidence with explicit provenance.
Passing candidates take precedence. Failed rounds select the candidate with
greatest runtime progress for compiler/runtime feedback. Named SSA values,
typed pointer offsets, and correct GOT/global indirection are explicit IR rules.

The parser removes inline reasoning, including responses whose opening
`<think>` tag was prefilled by the server. With `ir_only_c_evidence` enabled,
the C stage receives accepted IR and interface declarations, without redundant
assembly, retrieval examples, or runtime traces.

Interrupted requests count. Repeated identical failed candidate sets can stop early. Changed
inputs get separate revisions but share a stage's total request budget. Saved
streams, prompts, responses, compiler diagnostics, and runtime crashes make
failures inspectable; restarting does not manufacture additional attempts.

## 4. Validate Isolated Functions

Native function bridges preserve argument registers, including variadic ABI
state. Candidate globals are rebound to the original executable's live data.
Redirect the target entry to the recovered code and compare stdout, stderr,
and exit status with the original. Require an observed replacement entry.
For otherwise unentered functions, inferred direct probes compare the original
and recovered function, including requested bounded output buffers.

Compilation alone is insufficient. Crashes, output differences, missing entry
observations, and unsuccessful probes reject candidates. The assisted track
instead compiles replacements within original source scaffolding and reports
regression-test acceptance separately; it is not binary-first recovery proof.
Its IR scaffold is compiled at `-O1` with retained declarations: `-O0` can leave
gnulib header-only inline helpers such as `mbszero` unresolved. Initial affected
`df` diagnostics are preserved in `assisted_df_before_inline_fix`; resumption
revalidates saved responses without resetting request budgets.

## 5. Reconstruct and Test Applications

Only when every required function has accepted IR and C does integration infer
consistent declarations and global initializers. Primary integration cannot
rewrite accepted function bodies. Build separate IR and C applications using
generated application code/data and audited support archives. Audit generated
definitions and reject archives that supply required application functions.

Integration uses differential feedback cases. Both independently built final
executables must then pass the configured upstream suites. Per-function passes
inside the original executable do not imply standalone application success.
The campaign is complete only when all three primary applications pass this
gate; failure within the generation budgets leaves the goal unmet.

## Artifacts and Interpretation

`validation/project_level_binary_v3_candidates8` contains resolved configuration, service
and toolchain checks, per-track plans and baselines, per-function revision and
attempt directories, source references, standalone build provenance, and
campaign/acceptance summaries. `REPORT.md` and `report.json` are generated from
saved outcomes, not estimated from compiler success or original test passes.
Candidate artifacts live under `attempt_N/choice_K`; `selection.json` records
outcomes and selection. Reports distinguish request rounds, requested
candidates (including interrupted batches), and returned candidates. Focused
runs are labeled partial. The v2 output directory retains the original experiment.

Initial retrieval pilots exposed invalid LLVM pointer arithmetic, incorrect
string constant lengths, excessive repetitive generation, and candidates that
compiled but crashed. Those attempts remain charged to their budgets. Because
validation and prompt safeguards evolved during these pilots, this campaign is
an exploratory comparison, not a controlled causal measurement of RAG benefit.
A future controlled rerun should freeze code/configuration and use a new,
explicit experiment directory rather than erase or reset these attempts.

## Completed Exploratory Campaign

All nine v2 tool/track runs finished on October 3, 2026, using `n=1` and
temperature zero. These are not results of the revised eight-candidate preset.
Counts below require
acceptance of both recovered IR and recovered C, not compilation alone.

| Track | basename | uniq | df | Overall | Standalone Applications |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retrieval, native replacement | 0/4 | 0/5 | 1/11 | 1/20 (5%) | 0/3 |
| No retrieval, native replacement | 0/4 | 0/5 | 1/11 | 1/20 (5%) | 0/3 |
| Source-scaffold-assisted regression | 0/4 | 0/5 | 2/11 | 2/20 (10%) | 0/3 |

The retrieval success is `df:devlist_hash`; the no-retrieval success is
`df:replace_invalid_chars`. Each IR/C version passes all 12 differential cases
and has observed replacement entries (eight and four bounded entries,
respectively). These cases do not prove equivalence for all inputs.
Assisted successes are `df:me_for_dev` and `df:devlist_hash`; they must not be
added to the binary-first count. Missing validated functions prevent standalone
integration, so the application suites have not been passed by recovered apps.

There are 513 saved model-request markers across the three tracks, including
interrupted requests and initial pilots. The maximum for any function/stage is
ten, with no budget violations. No-retrieval and assisted runs use fewer
requests than the retrieval pilots; equal final counts therefore do not establish
equal model capability or a causal retrieval effect. Main failure classes are
invalid or incomplete LLVM IR, invalid inferred contracts, and runtime failures
or absent execution evidence. The strict three-application goal remains unmet.

Inspect `validation/project_level_binary_v2/REPORT.md`, `report.json`,
`campaign.json`, and per-function `source_comparison.diff` files for details.

## Eight-Candidate Pilot

The focused `basename:perform_basename` run completed on October 4, 2026
(Asia/Shanghai). Both recovered versions passed all 11 configured differential
cases with 10 observed replacement calls each. IR candidate eight and C
candidate seven were selected (zero-based positions 7 and 6). In each body
stage, the other seven candidates exhausted their output allowance. Selecting
only candidate zero would have missed both passing implementations.

The signature stage used four request rounds, including two interrupted
batches and an all-truncated thinking-enabled batch. Switching signature JSON
to non-thinking generation produced five syntax-valid declarations out of
eight. IR and C each used one eight-candidate round. In total, 48 candidate
slots were requested and 32 complete response records returned; interrupted
requests remain charged. Reasoning-related truncation remains a substantial
cost even with a 260,000-token context window.

Retrieval ran, but no neighbor met the 0.7 threshold, so no retrieved example
entered this pilot's prompt. This is not evidence of a RAG benefit or a
controlled comparison with v2. Only one function was selected: coverage is
1/1 selected functions, or 1/4 of the full retained `basename` inventory.
`main`, `oputs_`, and `usage` remain unattempted here; no standalone application
has been recovered. See the [saved report](../../validation/project_level_binary_v3_candidates8/REPORT.md)
and per-function `signature/latest.json`, `ir/latest.json`, and `c/latest.json`
under `retrieval/basename/functions/perform_basename` for artifact paths and
validation evidence.

## Known Limits

- The default analyzer requires symbolized Linux x86-64 ELF and source/build
  metadata for ownership; stripped-binary discovery is not implemented.
- Fuzzy stacks and conservative address-taken edges do not resolve every
  indirect call, callback, or optimized tail call.
- Large-function CFG partitioning is not implemented; excessive context is an
  explicit failure, not evidence that a function was recovered.
- Output limits and heuristic repetition/stagnation guards can stop generation
  before a complete candidate; these stops are failures, not semantic verdicts.
- Inferred layouts and signatures are hypotheses until behavior validates them.
- Tests cover observed behaviors, not all inputs or undefined-behavior cases.
- Reused support libraries mean application reconstruction, not whole-ELF
  regeneration or recovery of library implementations.
- Assisted acceptance may cover an unentered replacement; it must not be
  presented as an independently observed function-equivalence result.
- Native validation executes generated machine code and is not a security
  sandbox. Use an isolated worker for untrusted inputs and candidates.

## Infrastructure Verification

```bash
python -m unittest \
  models.binary_decompilation.test_project_pipeline \
  models.binary_decompilation.test_standalone_recovery \
  models.binary_decompilation.test_native_validation \
  models.binary_decompilation.test_binary_project \
  models.binary_decompilation.test_candidate_search -v
```

The current suite has 73 passing checks, including live-global binding,
recursion, tiny entries, variadic bridges, direct probes, crash evidence,
generation budget persistence, streaming/repetition guards, held-out-value
isolation, cached-contract repair, and timeout cleanup for looping code and
hanging constructors. Generated entry shims also cover very short LLVM IR bodies.
Assisted inline-helper compilation policy and partial-campaign resumption are covered.
Additional checks cover interleaved candidate streams, per-choice repetition
guards, best-candidate feedback, eight-candidate budgets, thinking-aware token
accounting, independent probe validation, and native acceptance of candidate
eight followed by cache revalidation without another generation request.
These infrastructure passes are not counted as model recovery successes.
