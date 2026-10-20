# Binary Project Pipeline

For the code architecture, control flow, native validation internals, and
extension interfaces, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Binary-First Project Recovery

`binary_project.py` runs source-isolated recovery through inferred contracts,
LLVM IR, and C. The initial benchmark is `basename`, `uniq`, and `df` from
the existing Clang `-O3` Coreutils 9.10 build. Retained private header helpers
are included; compiler scaffolding and approved library implementations are not.

```bash
python -m pip install frida==17.4.0
export QWEN_PROJECT_API_KEY='<your remote service key>'
python -m models.binary_decompilation.binary_project \
  --config models/binary_decompilation/configs/coreutils_binary_candidates8.json
python -m models.binary_decompilation.project_report \
  validation/project_level_binary_v3_candidates8
```

Use `--prepare-only` for inventory and instrumented baseline checks without model
generation. `--program basename` selects one tool. `--track retrieval`,
`--track no_retrieval`, and `--track assisted` select independent comparisons;
the default runs all three. Credentials are not written into configuration files.
Use `--program basename --function perform_basename --track retrieval` for a
focused run. It retains the full application inventory and cannot claim full
project recovery. The original `coreutils_binary_first.json` preset and v2
artifacts describe the earlier single-candidate experiment.

The native validator loads PIC candidates into the original binary and redirects
internal entries by ELF address. Function bridges preserve native argument
registers, and GOT bindings share live globals rather than copied data. A preload
gate preserves process I/O and installs instrumentation before `main`. Fixtures
must pass before inference. This backend currently requires Linux x86-64 ELF.
The default analyzer requires retained function symbols and a source/build tree
for ownership filtering; stripped-binary function discovery is not implemented.
Large-function CFG partitioning and general indirect-call resolution also remain
limitations rather than completed recovery features.

Source-derived declarations and bodies are unavailable to primary prompts.
Clang AST ownership metadata is evaluation-only. System-library ABI declarations
come from public system headers. The assisted track deliberately uses original
source scaffolding and is labeled separately; it is not evidence of binary-only
recovery. Ghidra C is not a primary input.

The revised preset generates eight candidates per function/stage round with
thinking enabled for IR/C (disabled for signature/probe JSON), temperature 0.6
and top-p 0.95. There are at most ten rounds
per signature/IR/C stage (80 requested candidates); probe repair has three
rounds. Application glue uses one candidate per integration round, up to ten.
Interrupted requests count and are not resent automatically. Input-addressed
revisions share the same stage budget. Each choice has separate code and
validation artifacts; passing choices take priority, and failed rounds feed
back the candidate with greatest runtime progress. Probe schema errors are
recorded separately from signature syntax validity. Missing execution evidence
triggers probe repair and revalidation of the compiled IR.

The context limit is 260,000 tokens per sequence, including completion tokens.
The preset sets a 260,000-token output ceiling for every stage, including
reasoning. With `max_tokens_policy: remaining_context`, each request uses
`min(260000, 260000 - prompt_tokens - 64)` output tokens. This avoids requesting
an output allowance that leaves no room for the prompt. Requested and effective
limits are saved separately. Token counting uses the same
thinking template as generation. Retrieval drops neighbors below score 0.7;
this is a configured heuristic, not proof of semantic similarity. Retrieval
caches check query/config identity before reuse.
Streaming responses, compiler diagnostics, runtime crashes, retrieval records,
and test commands remain in the output directory. A repetition penalty of 1.05
is identical across tracks. Complete target evidence is not truncated; oversized
functions fail explicitly rather than being claimed recovered.

Feedback uses differential CLI probes, including terminal probes for `df`.
Upstream suites are run on final independent IR/C executables. The `/proc`-masking
test cannot be externally instrumented, but remains in standalone evaluation.
Runtime stacks use fuzzy unwinding and hit counts are bounded samples, not exact
profiles. Repeated held-out runs after repairs are not independent evaluations.

Standalone builds contain generated application code and data, not the original
application objects. Reused support archives and their hashes appear in build
provenance. Completion requires all three primary standalone applications to
pass applicable configured tests. Missing execution coverage, budget exhaustion,
skipped tests, and incomplete runs never become successes.

### Other Projects

The `ProjectAdapter` protocol in `project_interfaces.py` separates tests and
validation from recovery. Configuration can supply `adapter_factory`,
`analysis_factory`, and `standalone_factory` as `module:callable` entrypoints.
The analyzer returns the same normalized plan: binary hash, stable function IDs,
source-defined target inventory, assembly/CFG/call evidence, dependency components,
runtime samples, and instrumented-baseline results. The adapter separates feedback
probes from full tests. A custom standalone builder returns independently built
IR/C executable paths and audited provenance. No Coreutils source scaffold is
required by the native code-loading API.

### Infrastructure Checks

```bash
python -m unittest \
  models.binary_decompilation.test_project_pipeline \
  models.binary_decompilation.test_standalone_recovery \
  models.binary_decompilation.test_native_validation \
  models.binary_decompilation.test_binary_project \
  models.binary_decompilation.test_candidate_search -v
```

Infrastructure and original-binary baseline results are not model decompilation
results. See the generated `REPORT.md` for the current campaign's actual counts.
See [WORKFLOW.md](WORKFLOW.md) for the detailed procedure, commands, validation
gates, evidence boundaries, and experimental limitations.

## Earlier Function-Level Utilities

`analyze_binary.py` imports an ELF into Ghidra and writes a JSON program profile.
It includes per-function assembly, Ghidra signatures, direct call edges, a
callee-before-caller order, and C output for requested functions.

Use `--retrieve` to query the existing Qdrant ExeBench collections for the
requested functions. The returned source signatures are exemplars only; they
are not treated as ground truth for a different binary.

For HermesSim-backed Qdrant collections, select the binary embedding backend.
It reconstructs an assembleable function fragment from `objdump` output before
submitting it to HermesSim:

```bash
python -m models.binary_decompilation.analyze_binary \
  --binary path/to/program \
  --output-dir validation/program \
  --decompile target_function \
  --retrieve \
  --embedding-backend hermessim \
  --embedding-url http://localhost:8125/embed/batch \
  --collection-template 'train_synth_rich_io_filtered_{idx}_preprocessed_hermessim'
```

Example:

```bash
python -m models.binary_decompilation.analyze_binary \
  --binary third_party/coreutils-9.10/build-clang-o3/src/basename \
  --output-dir validation/coreutils/basename \
  --decompile main --retrieve
```

Use `trace_functions.py` to record internal symbols reached by a concrete test
invocation. This creates a JSON trace that can select the subset of recovered
functions that must be linked and tested together.

```bash
python -m models.binary_decompilation.trace_functions \
  --binary path/to/program --output trace.json \
  --function main --function target_function \
  --arg=--option --stdin input.txt
```

Use `build_manifest.py` before a whole-program campaign. It maps each function
from the Ghidra profile to build objects and candidate source files, and marks
startup/runtime functions that cannot be source-compared as such.

### Parallel Benchmarks

The compiler-guided follow-up is documented in
[Qwen Failure Analysis and Guided Recovery](QWEN_GUIDED_FAILURE_ANALYSIS.md),
[Program-Analysis-Assisted IR Experiments](PROGRAM_ANALYSIS_IR.md),
including exact prompt exports, baseline failure examples, and rerun commands.

Add `--parallel-programs` to `binary_project` to launch one process per selected
benchmark (repeat `--program` to select a subset). For the current campaign:

```bash
python -m models.binary_decompilation.binary_project \
  --config models/binary_decompilation/configs/coreutils_binary_candidates8.json \
  --track retrieval --parallel-programs
```

Workers preserve the existing `retrieval/<program>` checkpoints and attempt
counts. Logs and bookkeeping are isolated under `workers/<program>`; the
coordinator merges campaign results and writes `parallel_status.json` with
worker PIDs and exit codes. Stop any previous sequential campaign before
starting this command: do not run two writers for the same benchmark/output.
Concurrency shares the inference server's capacity and does not guarantee
linear speedup. Process completion is not a decompilation success metric.
