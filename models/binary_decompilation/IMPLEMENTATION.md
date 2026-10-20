# Project Level Binary Decompilation Implementation

Implementation snapshot: October 4, 2026.

This guide explains the code behind the Coreutils binary-first decompilation
experiments. It is intended for reviewing the implementation, tracing a saved
result back to its producer, and adapting the framework to another project.
It describes the current code, not an idealized architecture or a claim that
the full applications have been recovered.

The active experiment uses `basename`, `uniq`, and `df` from the existing
Clang `-O3` Coreutils build. Its entry point is `binary_project.py`, not the
older campaign entry point in `project_pipeline.py`. The latter still supplies
shared generation, retrieval, testing, and prompt helpers.

For operating instructions, see [WORKFLOW.md](WORKFLOW.md). For the historical
failure analysis, see [COREUTILS_FAILURE_ANALYSIS.md](COREUTILS_FAILURE_ANALYSIS.md).
Changing inference code while a campaign is running can mix implementations
and checkpoint identities; review or edit it after the run, or use an explicit
new experiment directory for a changed protocol.

## Code Reading Order

| File | Main symbols | Responsibility |
| --- | --- | --- |
| [binary_project.py](binary_project.py) | `main`, `run_program`, `run_stage` | Campaign orchestration, stages, checkpoints, acceptance |
| [source_inventory.py](source_inventory.py) | `clang_inventory` | Source-defined ownership and retained function inventory |
| [analyze_binary.py](analyze_binary.py) | `run_ghidra`, `objdump_function_assembly` | Headless Ghidra invocation and retrieval assembly |
| [ghidra_export_program.py](ghidra_export_program.py) | Ghidra export script | Functions, instructions, references, and CFG evidence |
| [project_pipeline.py](project_pipeline.py) | `infer_stage`, `generate`, `Retriever`, `CoreutilsAdapter` | Prompts, model transport, retrieval, Coreutils test harness |
| [candidate_search.py](candidate_search.py) | `search_candidates`, `validation_rank` | Independent candidate evaluation and iterative feedback |
| [prompt_budget.py](prompt_budget.py) | `fit_prompt` | Token accounting and evidence retention |
| [native_validation.py](native_validation.py) | `build_candidate`, `make_wrapper`, `launch`, `AGENT` | Compilation, process instrumentation, and native replacement |
| [public_abi.py](public_abi.py) | `public_abi` | Compiler-derived public library declarations |
| [standalone_recovery.py](standalone_recovery.py) | `recover_standalone`, `build_generated_project` | Generated application builds and support-library audit |
| [assisted_validation.py](assisted_validation.py) | `SourceScaffoldAdapter` | Separately labeled source-scaffold comparison |
| [project_interfaces.py](project_interfaces.py) | `ProjectAdapter`, `resolve_factory` | Extension contracts and factory loading |
| [project_report.py](project_report.py) | `report` | Results aggregated from saved artifacts |

Start with `run_program`, then follow its calls into `run_stage`, `infer_stage`,
`search_candidates`, and `BinaryCoreutilsAdapter.validate`. Read the native
validator before interpreting an accepted function as an executed replacement.

## Campaign Entry Point

`binary_project.main` loads the selected JSON configuration, validates the
generation limits, resolves programs and tracks, and writes the resolved
configuration. A normal inference run then checks services and executes the
native-validator fixture suite before making decompilation requests.

The service checks in `service_preflight.check_services` verify the advertised
model, available context length, Qdrant readiness, HermesSim model readiness,
384-dimensional embedding compatibility, and basic debugger execution. The
dimension probe calls HermesSim, not the language model. Preflight does not
generate a decompiled function. Native fixtures separately exercise the Frida
replacement backend.

The main loop runs programs sequentially within each selected track. Without
`--track`, the CLI runs `retrieval`, `no_retrieval`, and `assisted`. The currently
requested benchmark run selects only `retrieval`:

```bash
python -m models.binary_decompilation.binary_project \
  --config models/binary_decompilation/configs/coreutils_binary_candidates8.json \
  --track retrieval
```

The API key is read from `QWEN_PROJECT_API_KEY`; do not put its value in the
configuration or this document. Run the command from the repository root with
the project Python environment. `--program` restricts tools and `--function`
restricts retained functions. `--prepare-only` performs analysis without LLM
generation. Do not launch concurrent writers into the same output directory.

## Configuration Resolution

`load_config` recursively follows `extends` relative to the containing JSON
file. It applies a shallow `dict.update`, not a recursive merge of every nested
object. A replacement `retrieval` object must therefore contain its required
settings. `program_overrides` updates individual program options and appends
`additional_cases` to existing differential cases.

The [eight-candidate preset](configs/coreutils_binary_candidates8.json) extends
`coreutils_binary_first.json`, which extends `coreutils_qwen38_remote.json`.
`generation_config` applies stage overrides to the resolved configuration:

| Stage | Candidates per round | Maximum rounds | Output tokens per candidate | Thinking |
| --- | ---: | ---: | ---: | --- |
| Signature | 8 | 10 | Up to 260000 | Disabled |
| Probe repair | 8 | 3 | Up to 260000 | Disabled |
| Assembly to IR | 8 | 10 | Up to 260000 | Enabled |
| IR to C | 8 | 10 | Up to 260000 | Enabled |
| Application integration | 1 | 10 | Up to 260000 | Enabled |

Temperature is 0.6, top-p is 0.95, and the configured context window is
260,000 tokens. Eight candidates multiply generation work, not the allowed
context length of a single candidate. The ten-round bound is per function and
stage, shared across its input revisions, not ten total requests per project.
Integration deliberately accepts only one glue candidate per round.

## Program Control Flow

The following is a simplified map of `run_program`, not executable code:

```text
original executable baseline
  -> load or create analysis plan
  -> require valid instrumented baseline
  -> build public system ABI evidence
  -> visit dependency components in callee-first order
       -> retrieve examples and infer signatures for component members
       -> for each accepted signature
            -> generate and validate LLVM IR
            -> repair probes if compiled IR lacks execution evidence
            -> generate and validate C only after IR passes
            -> repair declarations if C exposes a contract problem
       -> save function results and evaluation-only source comparisons
  -> require every retained target to have accepted IR and C
  -> generate integration declarations and initializers
  -> build independent IR-based and C-based applications
  -> run configured upstream and differential tests on both applications
  -> save program summary and campaign acceptance
```

Within a recursive strongly connected component, signatures are inferred
before its bodies. This ordering helps expose interfaces to body generation,
but it is not a joint ABI constraint solver or a fixed-point proof of mutually
recursive signatures.

## Function Inventory and Static Analysis

`clang_inventory` parses application translation units with Clang's JSON AST
and intersects source definitions with `nm --defined-only` results from their
build objects. The inventory records names, source ownership, object paths,
and inline metadata. It does not provide original types or bodies to inference.
Private header helpers retained in application objects can be included.

`prepare_binary` intersects this inventory with executable ELF symbols. A
source function optimized away or fully inlined is recorded separately rather
than assigned an invented independent machine-code body. The default backend
therefore needs the source/build tree for ownership and retained symbols in
the binary; it is not a stripped-binary discovery implementation.

Ghidra supplies static functions, CFGs, calls, and instruction references.
The code normalizes Ghidra addresses against ELF addresses and supplements
the disassembly with `objdump`. It adds direct calls missed in the export and
conservative dependencies for address-taken functions. It does not request
Ghidra C for the binary-first recovery prompt.

`dependency_components` constructs a NetworkX directed graph, condenses its
strongly connected components, and reverses topological order so callees are
processed before callers. Runtime edges supplement this graph; they do not
resolve every indirect call or prove that every static edge is feasible.

`binary_data_evidence` uses application-object symbol names to identify owned
data, then reads values from the executable: global sizes, initial bytes,
zero-initialized sections, relocations, and referenced printable strings.
`public_abi` separately compiles public system headers to recover C types and
LLVM declarations for known library calls. Neither mechanism recovers all
application structure layouts automatically.

## Runtime Evidence Collection

`make_wrapper` generates a launcher and runtime specification. `build_gate`
compiles a preload library whose constructor waits on a pipe. `launch` starts
the original executable with that library, attaches Frida, installs its agent,
and releases the gate only after instrumentation reports readiness.

The embedded JavaScript `AGENT` records bounded function-entry events,
argument registers, readable memory snapshots, global snapshots, return values,
and fuzzy backtraces. It records at most eight sampled entries per function
per process. These counts describe sampled coverage, not call frequency.
Snapshots are bounded; they are not complete process-state captures.

`summarize_trace` uses upstream tests and differential cases for coverage and
observed dependencies, but includes value snapshots in prompts only for
`differential:` test IDs. Thus upstream inputs are excluded from iterative
value feedback, although their coverage information is used. The final suite
is not wholly unseen in the strict statistical sense.

The `df` test that masks `/proc` is excluded from instrumentation because it
prevents attachment. It remains in the original and standalone application
test lists. The plan records this exclusion explicitly.

## Retrieval and Prompt Construction

`Retriever.retrieve` normalizes one function's disassembly into an assembly
fragment and submits it to HermesSim's `/embed/batch`. It queries the best hit
from each of eight Qdrant collections. Payload IDs index matching local
ExeBench dataset shards, from which the code loads the function name, C body,
and LLVM IR. The database and local dataset indexing must agree.

The results are sorted by score; the active preset keeps at most two hits with
scores of at least 0.7. This is a score filter, not ABI-aware reranking or an
equivalence probability. An enabled retrieval run may legitimately include no
example. `retrieval.json` caches the query identity, retrieval configuration,
selected examples, and elapsed time. Rejected neighbors are not all retained
in that artifact.

`infer_stage` builds stage-specific instructions and evidence. Signature and
IR recovery can receive assembly, CFGs, binary references, public ABI hints,
runtime samples, retrieval examples, and inferred contracts. For the binary
track it removes Ghidra's signature hint. With `ir_only_c_evidence`, the C
stage retains accepted IR and interface evidence rather than the assembly,
retrieval examples, or runtime traces.

`fit_prompt` filters contract evidence to the target and listed callees, then
counts tokens using the local tokenizer and the selected thinking template:

```text
effective output allowance = min(configured max_tokens, context window - prompt tokens - 64)
```

This is the active `remaining_context` policy. All stage ceilings are 260,000;
the old 8,192/32,768/16,384 limits remain only in historical artifacts.
`with_output_budget` passes the effective allowance to generation in the
single-candidate, multi-candidate, and integration paths. The request metadata
records `requested_max_tokens` separately from the actual `max_tokens`.

If necessary it drops retrieved examples, then shortens repair feedback to
4,000 characters. It does not truncate the target binary or accepted IR.
If that still does not fit, the stage fails explicitly. CFG partitioning is
not implemented. This policy requires a local tokenizer and at least one
output token after the safety margin. Legacy configurations default to `fixed`,
which reserves the full stage output allowance before fitting the prompt;
that legacy policy does not measure tokens without a configured tokenizer.

## Signature and Data Structure Recovery

The model returns a JSON contract with `c_declaration`, argument and return
evidence, memory-layout observations, uncertainties, `project_header`, and
optional `direct_probes`. The generated header contains inferred declarations,
structures, and extern globals; it is not copied from the application source.

`BinaryCoreutilsAdapter.validate` compiles the header and target declaration
using Clang's AST output. It checks that the target is declared and rejects
generated function bodies masquerading as header declarations. Standard
system inline definitions are allowed. This establishes syntax validity only;
the result explicitly records `abi_verified: false`.

Probe validation is independent. Valid probes are retained and malformed
ones get specific errors without rejecting an otherwise valid declaration.
A compiled body with no observed execution may enter the separate probe
repair stage. Accepted probe syntax triggers execution of the same IR again;
it does not itself mark the function as recovered.

For example, the supported bounded-buffer shape is:

```json
{
  "return_type": "void",
  "argument_types": ["pointer"],
  "arguments": [{"buffer": 16, "id": "buf", "bytes": [65, 0]}],
  "return_kind": "integer",
  "memory_outputs": [{"id": "buf", "size": 16}]
}
```

This illustrates the schema, not a valid test for every target. Probe ABI
types must match the inferred function. Buffers are limited to 4,096 bytes;
setup calls are restricted to `stat`, `lstat`, `memcpy`, and `memset`. Bounds
checks cannot establish that a guessed layout, pointer, or setup is meaningful.

## Candidate Generation and Selection

`generate` sends an OpenAI-compatible chat request to the configured vLLM
endpoint. In streaming mode it maintains separate content, reasoning, finish
status, and repetition detection for every choice index. Raw chunks are saved
to `stream.jsonl`. A repetitive candidate is marked independently rather than
canceling valid siblings. The batch normally completes before validation;
this is not an asynchronous early-acceptance scheduler.

`code_block` removes inline reasoning before extracting JSON, LLVM IR, or C.
It handles a closing `</think>` tag when the server has prefilled the opening
tag. `search_candidates` rejects truncated, interrupted, filtered, and guarded
responses rather than treating draft code inside reasoning as final code.

Each completion gets its own `choice_K` directory and validation result.
Identical full response texts within a batch can reuse validation with a
`duplicate_of` reference. This is textual deduplication, not semantic hashing.

`validation_rank` selects candidates in this order:

1. Accepted candidates, preferring fewer probe errors and then more passes.
2. Candidates with observed execution, ranked by test pass ratio and count.
3. Compiled candidates with a manifest, or those needing probes.
4. Other compiler or contract failures.
5. Response-format failures.

All returned choices are evaluated before selecting a contract-repair or
probe-repair outcome. Signature and probe validation mutate adapter state, so
the selected choice is validated again to restore its header/probes. A failed
round supplies the strongest candidate's diagnostics and bounded summaries of
the others to the next round. Repeated identical failed response sets can
trigger stagnation termination. Exhausted stages preserve their best failed
candidate; they do not become successes merely because the budget is spent.

## Native Function Validation

`build_candidate` performs the following checks and transformations:

1. Require an x86-64 ELF target symbol and the exact generated target definition.
2. For IR, run `opt -passes=verify -disable-output` before compilation.
3. Compile PIC code at `-O1`; C receives the inferred `project.h` and errors on
   implicit function declarations. This validation optimization differs from
   the original binary's `-O3` build.
4. Reject definitions of other original functions or shadow copies of original
   globals. Rename the recovered target and add a patchable entry shim.
5. Generate assembly tail-jump bridges for unresolved internal functions and
   identify GOT relocations that need binding to live original globals.
6. Link `candidate.so` and save a manifest of entries, bridges, and bindings.

At runtime the Frida agent loads the shared object, binds addresses with PIE
adjustment, and redirects the original target entry. The jump bridges preserve
native argument registers, including variadic-call state, rather than invoking
every function through a guessed Python or JavaScript signature. Supported
short entries use a checked jump into nearby executable memory. Calls to other
application functions still reach the original implementation during isolated
validation; this is why isolated success is not whole-application recovery.

`feedback_test` runs only configured differential cases. `CoreutilsAdapter.test`
compares raw stdout, stderr, and exit status against the original binary.
An accepted native candidate requires zero failed cases and observed entry into
the replacement. If normal cases never enter it, direct probes invoke original
and recovered versions in separate processes and compare serialized returns,
requested memory outputs, stdout, stderr, and successful exit.

Probe dispatch replaces `main`, initializes the program name when available,
and performs allowed setup calls. It does not replay the application's full
normal initialization sequence. This limits tests of stateful helpers.

The agent records crash addresses, registers, memory-access information, and
fuzzy stacks. A watchdog kills hung processes, including constructor-time
hangs; timeout and instrumentation failures have distinct exit statuses.
These diagnostics feed later generations. Generated native code still runs
on the host: the validation backend is not a security sandbox.

## Repair and Resume Behavior

`run_stage` hashes an input identity containing function evidence, context,
model settings, tests, declarations, and the prompt implementation hash.
Revisions get separate directories, but `request.json` markers across all
revisions consume the same stage budget. A marker is written before invoking
the model, so a disconnected or interrupted request is charged even when no
`response.json` was saved.

`infer_stage` revalidates accepted cached candidates. Saved responses can be
replayed through validation without a new inference request; interrupted
markers are not silently resent. If context or declarations change, a new
revision can require new generation even for a previously successful target.
In particular, the identity includes broader context than the final pruned
prompt, so unrelated contract changes can invalidate reuse. Do not assume
every partial-run success is reused unchanged in a later full run.

C errors involving headers or incomplete layouts can request contract repair.
The orchestrator infers a revised contract within the remaining signature
budget, then regenerates IR if the signature changed or revalidates existing
IR if only the header changed. C generation resumes only after IR passes.
There is no general solver that propagates layout constraints to every caller.

The saved identities are not a complete content-addressed build system:
`run_stage` hashes `project_pipeline.py`, but not every validator/search module
in its revision key. Toolchain metadata records framework hashes separately.
Some exhausted failed caches return without replay. Freeze code and settings
for controlled experiments and preserve the old directory when starting a new
protocol; deleting markers would change the experimental budget.

## Standalone Application Reconstruction

`recover_standalone` refuses to start integration until every target in the
full plan has accepted IR and C. A `--function` run does not shrink that
required inventory. Missing functions are reported explicitly.

Integration generates consistent declarations and global initializers from
binary data and the recovered functions. In the binary-first track,
`function_replacements` must be empty: integration cannot quietly replace
tested bodies with new implementations. Two applications are built:

- The C build combines accepted C bodies, generated declarations, and globals.
- The IR build compiles accepted LLVM modules and a generated global-data unit.

`support_dependencies` excludes the original application object. The builder
audits approved support objects/archives and rejects any that define a required
application target. Glue is also checked so it cannot substitute target bodies.
Support libraries such as `libcoreutils.a`, libc, and the CRT remain external
dependencies; the experiment does not reconstruct the entire ELF ecosystem.

After both generated applications pass differential feedback cases,
`run_program` runs the configured upstream and differential suite on each.
Only then, with every target covered, is `complete_project_recovery` true.
Upstream exit status 77 means skipped, not passed. Baseline successes and
source-scaffold-assisted successes cannot satisfy binary-first completion.

## Artifact Layout and Result Inspection

The active output root is `validation/project_level_binary_v3_candidates8`.
Important artifacts are organized as follows; not every stage reaches every
file, and an interrupted request may contain only its marker and stream.

```text
output/
  config.json, service_preflight.json, toolchain.json
  native_backend_fixtures.json, campaign.json, acceptance.json
  retrieval/<program>/
    baseline.json, plan.json, analysis_identity.json
    paired_evidence.json                 # when prior analysis was reused
    public_abi/
    functions/<function>/
      retrieval.json, result.json
      source_reference.c, source_comparison.diff
      <signature|probes|ir|c>/
        latest.json
        <input-hash>/
          input_identity.json, result.json
          attempt_N/
            request.json, generation_config.json, context_budget.json
            prompt.txt, stream.jsonl, response.json, selection.json
            choice_K/
              contract.json or candidate.ll or candidate.c
              validation.json
              validation/
                compiler and linker diagnostics
                candidate_manifest.json, tests.json, runtime/
    results.json, summary.json
    standalone/, heldout.json
```

`source_reference.c` and its diff are exported after the function recovery
attempt for comparison, not supplied as ground truth to the primary prompt.
Source text similarity is not the acceptance criterion.

To inspect a failure, read the stage's `latest.json`, then its revision's
`selection.json`, the selected `choice_K/validation.json`, and detailed compile
or runtime artifacts. Compare `generation_config.json` with `finish_reason`
before calling a truncated response a semantic error. `context_budget.json`
shows whether the target fit and whether retrieval examples were removed.

Generate the aggregate report explicitly:

```bash
python -m models.binary_decompilation.project_report \
  validation/project_level_binary_v3_candidates8
```

The report counts request rounds, requested candidate slots, returned choices,
accepted IR, paired accepted IR/C, and standalone completion separately.
Its denominator is the full retained inventory, including targets not attempted
in a partial run. Reports and program summaries are saved snapshots, not live
job monitors; during an active program they can lag per-stage artifacts.

For this long-running campaign, an operational
[runner](../../validation/project_level_binary_v3_candidates8/run_campaign.py)
is saved with the experiment artifacts. It holds an exclusive runner lock,
starts the same retrieval CLI with inherited credentials, and generates the
aggregate report when the campaign process exits. `campaign.log` contains its
output and `runner_status.json` records its PID, child PID, phase, timestamps,
and exit codes. A runner phase of `finished` describes process completion, not
successful application recovery. This wrapper does not alter generation
settings, candidate selection, or checkpoint budgets.

## Extending to Another Project

The factory mechanism accepts `module:callable` values for `adapter_factory`,
`analysis_factory`, and `standalone_factory`. `ProjectAdapter` documents the
core test and validation interface, but the implementation still contains
Coreutils-specific assumptions beyond that minimal protocol.

For another project, supply an adapter constructed as `(config, program,
output)` that locates its binary, manages declarations/probes, implements
reference and feedback tests, and returns the validation fields used by
candidate selection. Supply an analyzer returning the expected plan keys:
`targets`, `components`, `runtime`, `binary_data`, and
`instrumented_baseline`. Targets need stable function IDs, source ownership,
addresses, disassembly, call information, and CFG evidence. Replace the
standalone factory if the project uses different build/support-library rules.

The defaults assume Coreutils build paths, source inventory, tests, and Linux
x86-64 replacement. Loading a custom factory alone does not make those defaults
portable. The orchestration also expects optional probe-repair state such as
`probe_errors`, which is not fully captured by the minimal protocol today.
Add adapter-specific fixtures before enabling inference.

## Tests and Review Checklist

The implementation test command is:

```bash
python -m unittest \
  models.binary_decompilation.test_project_pipeline \
  models.binary_decompilation.test_standalone_recovery \
  models.binary_decompilation.test_native_validation \
  models.binary_decompilation.test_binary_project \
  models.binary_decompilation.test_candidate_search -v
```

At the 260,000-token output-ceiling revision, 73 tests passed. They cover
candidate stream isolation, budget persistence, token accounting, probe schema
handling, native bridges and live globals, recursion, short entries, timeout
cleanup, integration boundaries, and cache revalidation. One native fixture
makes only the eighth candidate correct. Infrastructure tests are not model
decompilation successes; rerun them after executable-code changes.

For a code review, check the following boundaries first:

- Only source-defined retained targets enter the inventory; inlined functions
  and support libraries are not silently counted as recovered bodies.
- Original application declarations/bodies do not enter binary-first prompts
  or generated-only builds; the ownership oracle is still disclosed.
- Every candidate has its own state and diagnostics, and budget accounting
  survives interruptions, repairs, and input revisions.
- Compilation, observed execution, matching behavior, and standalone success
  remain separate acceptance levels.
- Pointer widths, calling conventions, layouts, GOT bindings, and global
  initialization are tested rather than assumed from a plausible declaration.
- Fuzzy stacks, bounded probes, conservative dependencies, and available test
  cases are not described as complete semantic coverage.
- Large-function partitioning, stripped-binary discovery, general ABI solving,
  semantic retrieval reranking, and a native-code security sandbox remain
  unimplemented capabilities, not promises of the current framework.

The full benchmark campaign was still running when this document was written.
Use its saved results for final success counts; this implementation guide does
not substitute the earlier successful single-function pilot for those results.
