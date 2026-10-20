# Coreutils Decompilation Failure Analysis

Historical scope: the results below describe the v2 campaign. The subsequent
eight-candidate implementation and its stage-specific reasoning policy are
documented in [WORKFLOW.md](WORKFLOW.md) and configured separately in
[coreutils_binary_candidates8.json](configs/coreutils_binary_candidates8.json).
Those changes do not revise the v2 success counts or establish a new success rate.

## Scope and Conclusion

This report compares the saved Qwen3-32B ExeBench experiment with the
Qwen3.8-27B-FP8 Coreutils project-level experiment completed on October 3, 2026.
The analysis reads existing responses, evaluation flags, prompts, compiler
diagnostics, and runtime results. It does not rerun ExeBench tests or make new
model requests.

**The Coreutils result does not establish that Qwen3.8 is intrinsically worse
than Qwen3-32B.** The experiments differ in candidate count, reasoning mode,
input size, ABI recovery requirements, feedback, and acceptance criteria.
Several choices in the new framework contributed to failures. LLVM generation
is the main observed bottleneck; C generation is not the main failure stage.

## 1. Where Results Are Stored

The Coreutils experiment root is:

```text
/data1/xiachunwei/Projects/llm-decompilation/validation/project_level_binary_v2/
```

Its layout is:

```text
project_level_binary_v2/
  config.json
  toolchain.json
  service_preflight.json
  campaign.json
  acceptance.json
  report.json
  REPORT.md
  retrieval/
  no_retrieval/
  assisted/
    <basename|uniq|df>/
      plan.json
      baseline.json
      results.json
      summary.json
      functions/<function>/
        retrieval.json
        result.json
        source_reference.c
        source_comparison.diff       # Accepted C only
        <signature|ir|c>/
          latest.json
          <input-revision>/attempt_N/
            prompt.txt
            generation_config.json
            context_budget.json
            request.json
            stream.jsonl
            response.json
            contract.json            # Signature stage, when parsed
            candidate.ll             # IR stage, when extracted
            candidate.c              # C stage, when extracted
            validation.json
            validation/              # Compiler/native/test artifacts
```

Not every file exists for every attempt. An interrupted request may have only
its request marker and partial stream. A malformed response may not yield a
candidate. A failed IR stage does not advance to C, so missing C output does
not mean that an independently generated C function was tested and failed.

Useful entry points:

- [Consolidated Markdown results](../../validation/project_level_binary_v2/REPORT.md)
- [Machine-readable results](../../validation/project_level_binary_v2/report.json)
- [Resolved experiment configuration](../../validation/project_level_binary_v2/config.json)
- [Campaign acceptance](../../validation/project_level_binary_v2/acceptance.json)
- [Framework workflow](WORKFLOW.md)

## 2. What the Saved ExeBench Experiment Shows

The inspected directory is:

```text
/data1/xiachunwei/Projects/validation/Qwen3-32B/
  sample_without_loops_Qwen3-32B-n8-assembly-without-comments/
```

The current batch entry point is
[gemini_decompilation.py](../gemini/gemini_decompilation.py). Generation and
repair are implemented in [llm_decompiler.py](../gemini/llm_decompiler.py).
The current source is not necessarily the exact version used by the historical
experiment, so saved responses take precedence when identifying actual settings.

### Candidate Count and Cached Successes

All 969 inspected `response_*.pkl` files contain **eight choices**, not ten.
There are therefore 7,752 saved candidate completions across initial generation
and corrective rounds. This is a count of completions, not distinct correct
programs or independently validated source functions.

Combining initial flags in `results.pkl` with corrective flags in
`fix_evaluation_list.pkl` gives:

| Included Corrective Rounds | Samples With Any Execution-Success Flag | Rate |
| ---: | ---: | ---: |
| 0: initial generation only | 45/164 | 27.4% |
| 1 | 59/164 | 36.0% |
| 8 | 101/164 | 61.6% |
| 9 | 103/164 | 62.8% |
| 10 | 106/164 | 64.6% |

Thus the reported **101 successes can be reproduced from the saved flags when
including eight corrective rounds**. Counting all ten saved rounds yields 106.
This explains the numerical difference without assuming why an earlier summary
used a particular round limit. These counts have not been independently retested.

For the initial generation, candidate zero alone succeeds on **14/164 (8.5%)**,
whereas accepting any of eight candidates succeeds on **45/164 (27.4%)**.
That is direct evidence that candidate multiplicity matters in this experiment.
It is not a counterfactual measurement of a full single-candidate repair run:
later repair prompts can depend on selection among all eight earlier candidates.

The saved initial prompts have reported sizes of 222 to 2,365 tokens. Responses
contain explicit `<think>` reasoning. The current client invocation does not
explicitly set temperature or a generation token limit, so the historical
effective values cannot be established from that invocation alone.

### What ExeBench Validates

[evaluate_exebench.py](../../utils/evaluate_exebench.py) compiles generated LLVM
IR to assembly and constructs a wrapper with:

```python
Wrapper(
    c_deps=c_deps,
    func_c_signature=row["func_head_types"].replace("extern", ""),
    func_assembly=assembly,
    cpp_wrapper=row["synth_exe_wrapper"],
    assembler_backend=LLVMAssembler(),
)
```

The wrapper supplies known declarations, dependencies, dummy functions, and
synthetic input/output cases. These are evaluation support, not necessarily
inputs shown to the model. The success flags measure assembly-to-IR execution
on these cases, not a completed IR-to-C conversion or reconstructed application.

## 3. Coreutils Outcomes and Settings

Counts below require acceptance of both recovered IR and C:

| Track | basename | uniq | df | Overall | Standalone Applications |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retrieval, native replacement | 0/4 | 0/5 | 1/11 | 1/20 (5%) | 0/3 |
| No retrieval, native replacement | 0/4 | 0/5 | 1/11 | 1/20 (5%) | 0/3 |
| Source-assisted regression | 0/4 | 0/5 | 2/11 | 2/20 (10%) | 0/3 |

The denominator is 20 retained application/private-helper targets, not every
source function or every symbol in the executables. Completely inlined or
removed functions do not have independent recovered bodies. Approved support
libraries remain outside this inventory.

The accepted binary-first functions are:

- Retrieval: `df:devlist_hash`; both representations pass 12 differential cases,
  with eight bounded replacement entries recorded.
- No retrieval: `df:replace_invalid_chars`; both representations pass 12 cases,
  with four replacement entries recorded.
- Assisted: `df:me_for_dev` and `df:devlist_hash`; these are source-scaffold
  regression passes, not additional binary-first equivalence results.

The final retrieval outcomes break down as:

| Final Blocking Stage | Targets |
| --- | ---: |
| Contract/probe validation | 6 |
| IR generation or runtime validation | 13 |
| Accepted IR and C | 1 |

Some contract-blocked targets generated IR during earlier pilot revisions.
The table describes their final status, not a claim that they never reached IR.
Nor does it mean that all 19 failures are semantically incorrect C functions.

The Coreutils client explicitly requests `n=1`, temperature zero,
`enable_thinking=False`, and at most 16,384 output tokens. Each function/stage
has a ten-request budget shared across revisions. Identical failed candidates
can stop after three repetitions; interrupted requests also consume budget.

| Track | Saved Request Markers |
| --- | ---: |
| Retrieval, including initial pilots | 273 |
| No retrieval | 128 |
| Assisted | 112 |
| Total | 513 |

The audit found no function/stage exceeding ten requests. Ten is a maximum,
not a guarantee of ten completed, distinct IR candidates under the final backend.

## 4. Failure Examples

The corrections below illustrate particular LLVM or schema rules. They are
**not newly validated complete replacements** and are not counted as successes.

### Example A: Invalid LLVM Constant Assignment in `get_dev`

The generated IR contains:

```llvm
%col_idx = 0
```

LLVM rejects it:

```text
candidate.ll:391:14: error: expected instruction opcode
  %col_idx = 0
             ^
```

LLVM SSA definitions require instructions; it does not support this C-like
constant alias. A generator should use the typed constant directly at its uses,
or produce an appropriate typed instruction. For example, if an `i32` value is
required, this is syntactically valid:

```llvm
%col_idx = add i32 0, 0
```

The correct width must come from the actual uses, not from this illustration.
This failure occurred before any runtime behavior could be evaluated.

Evidence:
[saved IR](../../validation/project_level_binary_v2/retrieval/df/functions/get_dev/ir/d088f0ff473fd3cea3447272abdb99c32f8c1730f6195ea73c435174db42f2eb/attempt_5/candidate.ll)
and [validation](../../validation/project_level_binary_v2/retrieval/df/functions/get_dev/ir/d088f0ff473fd3cea3447272abdb99c32f8c1730f6195ea73c435174db42f2eb/attempt_5/validation.json).

### Example B: Invalid Numeric SSA Numbering

For `df:main`, the diagnostic is:

```text
candidate.ll:159:3: error: instruction expected to be numbered '%6' or greater
  %2 = load i32, ptr %argc_slot, align 4
  ^
```

For `decode_output_arg`, it is:

```text
candidate.ll:51:3: error: instruction expected to be numbered '%22' or greater
  %10 = phi ptr [ %21, %strchr_found ], [ null, %strchr_not_found ]
  ^
```

Unnamed numeric values and blocks obey LLVM slot-numbering rules. Generating
arbitrary `%2`, `%10`, and `%21` identifiers is not equivalent to using arbitrary
named variables. Named values such as `%argc_value` avoid that numbering issue.
They do not automatically fix invalid PHI predecessors, dominance, or types.

Evidence:
[main validation](../../validation/project_level_binary_v2/retrieval/df/functions/main/ir/fffda74fa1da5ad97c1d503201d252449047382a3365bb7e0e69e26bf78b434a/attempt_5/validation.json)
and [decode_output_arg validation](../../validation/project_level_binary_v2/retrieval/df/functions/decode_output_arg/ir/bfc5e53d83253d0dc9af57b27f93016e91b7c84801c0a08906eb6bf3770df4ec/attempt_3/validation.json).

### Example C: Illegal Pointer Arithmetic in `replace_invalid_chars`

The retrieval-enabled candidate contains:

```llvm
%end_ptr = add ptr %s, %0
```

The compiler reports:

```text
candidate.ll:17:26: error: '%0' defined with type 'i64' but expected 'ptr'
```

LLVM does not express pointer advancement with `add ptr`. If `%offset` is a
byte offset, the appropriate form is:

```llvm
%end_ptr = getelementptr i8, ptr %s, i64 %offset
```

The no-retrieval track later recovered this function successfully. That shows
the function is not inherently unrecoverable in the current infrastructure;
it does not establish that retrieval generally harms recovery.

Evidence:
[failed IR](../../validation/project_level_binary_v2/retrieval/df/functions/replace_invalid_chars/ir/d66cc0ab9d572ec9af90dd8c8fa2e76d43b3e1b0d9727f2da5ab4b183cdb0b03/attempt_0/candidate.ll)
and [successful no-retrieval result](../../validation/project_level_binary_v2/no_retrieval/df/functions/replace_invalid_chars/result.json).

### Example D: Compiled IR Crashes in `perform_basename`

The original source includes:

```c
fputs (name, stdout);
putchar (use_nuls ? '\0' : '\n');
```

The generated IR includes:

```llvm
@stdout = external global ptr, align 8

%stdout_ptr = load ptr, ptr @stdout, align 8
%file_ptr = load ptr, ptr %stdout_ptr, align 8
call i32 @fputs_unlocked(ptr noundef %call, ptr noundef %file_ptr)
```

With that global declaration, the first load obtains the `FILE *`. The second
load interprets the beginning of the `FILE` object as another pointer. That
is a genuine pointer-indirection bug. A call using the first loaded pointer is
the corresponding partial correction:

```llvm
%stream = load ptr, ptr @stdout, align 8
call i32 @fputs_unlocked(ptr %call, ptr %stream)
```

GOT addressing can be misleading here: two machine-level loads involving a
GOT entry do not necessarily require two loads from the logical LLVM global.

The saved attempt passed three cases and failed eight, with exit status 139
on ordinary inputs such as `/usr/bin/sort`, where `sort\n` was expected.
The double load is consistent with those crashes. No newly executed correction
establishes that it was the only bug or that fixing it makes the whole function
pass.

Evidence:
[original source reference](../../validation/project_level_binary_v2/retrieval/basename/functions/perform_basename/source_reference.c),
[generated IR](../../validation/project_level_binary_v2/retrieval/basename/functions/perform_basename/ir/d2717fe00e085bd6087cc2c61fa7b73cb65be1680c1a5ad4df1840cb3a88f8ab/attempt_5/candidate.ll),
and [runtime validation](../../validation/project_level_binary_v2/retrieval/basename/functions/perform_basename/ir/d2717fe00e085bd6087cc2c61fa7b73cb65be1680c1a5ad4df1840cb3a88f8ab/attempt_5/validation.json).

### Example E: A Probe-Schema Failure Blocks Contract Acceptance

`replace_control_chars` generated an argument in this shape:

```json
{
  "buffer": {
    "byte_count": 16,
    "id": "buf1",
    "bytes": [72, 101, 108, 108, 111, 9, 13, 10, 87, 111, 114, 108, 100, 0, 0, 0]
  }
}
```

The required schema uses a numeric `buffer` value and sibling fields:

```json
{
  "buffer": 16,
  "id": "buf1",
  "bytes": [72, 101, 108, 108, 111, 9, 13, 10, 87, 111, 114, 108, 100, 0, 0, 0]
}
```

The validator reported `Probe buffers must contain 1 to 4096 bytes`. In this
example the intended size is 16; the actual problem is the nested JSON shape.
That feedback is insufficiently specific and may make corrective generation
less effective. Other contracts referred to a global symbol where the probe
API required an already allocated buffer.

This is not evidence that the function body is incorrect: it is a failure of
the generated probe contract and the framework's repair workflow. Probe repair
should be separated from otherwise syntax-valid signature acceptance, while
still requiring actual execution before accepting a recovered body.

Evidence:
[generated contract](../../validation/project_level_binary_v2/retrieval/df/functions/replace_control_chars/signature/9eae32c3b1451f5a71d48a504b3523d984cec023e02a4b1c2c0f26e0f662f8ce/attempt_2/contract.json)
and [final contract status](../../validation/project_level_binary_v2/retrieval/df/functions/replace_control_chars/signature/latest.json).

### Example F: `find_mount_point` Is Unvalidated, Not Proven Incorrect

Its final IR rejection is:

```text
Target not executed by feedback probes
```

CLI cases used for corrective feedback did not establish replacement execution.
Upstream coverage is not interchangeable with feedback execution evidence.
Without an executed differential probe, matching application output can be
vacuous: the replacement might never have run.

The rejection is appropriate for the acceptance gate, but the status should
be interpreted as insufficient execution evidence rather than demonstrated
semantic failure. Better input construction and direct probes are needed.

Evidence:
[final IR status](../../validation/project_level_binary_v2/retrieval/df/functions/find_mount_point/ir/latest.json).

### Example G: An Assisted Harness Link Error

An early assisted `devlist_hash` attempt failed with:

```text
scaffold.c:(.text+0x18a8): undefined reference to `mbszero'
scaffold.c:(.text+0x1969): undefined reference to `mbszero'
```

The source scaffold was compiled at `-O0`, which left a gnulib header-only
inline helper unresolved. Revalidating the same saved IR with the scaffold
compiled at `-O1` passed all 12 differential cases. The assisted adapter was
fixed, and its affected prior artifacts were preserved in:

```text
validation/project_level_binary_v2/assisted_df_before_inline_fix/
```

This is an infrastructure failure, not a generated-function error. It affected
the assisted comparison, not the completed binary-first native-replacement
results. The current framework regression suite has 57 passing tests; those
infrastructure passes must not be counted as model recovery successes.

## 5. Why the Tasks Differ Beyond Candidate Count

### ABI, Layouts, and State

ExeBench's evaluator knows the source signature and wrapper dependencies.
The binary-first Coreutils track must infer optimized signatures, private
structures, global objects, callbacks, and external ABI declarations. A
syntax-valid header is not proof of a correct layout or calling convention.

Optimization can also remove unused arguments. An inferred declaration that
matches an optimized binary can conflict with the original source scaffold's
calls. Assisted `oputs_` attempts demonstrated this one-versus-two-argument
compilation problem. Binary ABI and source ABI must be distinguished explicitly.

Call-graph scheduling alone does not solve these issues. The current framework
does not provide a complete interprocedural type/layout constraint solver or
a fixed-point ABI recovery process.

### Function Size and Evidence Volume

Saved Coreutils disassembly line counts include:

| Target | Disassembly Lines |
| --- | ---: |
| basename:perform_basename | 77 |
| uniq:main | 788 |
| df:get_dev | 553 |
| df:main | 1,907 |

These include disassembly formatting and are not exact instruction counts.
For retrieval-track attempts, reported prompt sizes range from 9,683 to
108,308 tokens, with median 28,996. This includes different stages and retries;
it is not a matched comparison with ExeBench's initial prompts.

The Coreutils evidence includes disassembly, CFG/instruction metadata, runtime
snapshots, global data, earlier contracts, and retrieved examples. Some of it
is redundant or weakly relevant. Staying within a context window does not imply
that the evidence is well selected or that 16,384 output tokens suffice.
Large-function CFG partitioning is not implemented. However, failures of short
functions such as `devlist_compare` show that size is not the sole explanation.

### Retrieval Relevance

Saved retrieved examples include:

| Coreutils Target | Example Names | Scores |
| --- | --- | --- |
| df:devlist_hash | `i_nint`, `f_le_rm5` | 0.913, 0.896 |
| df:get_dev | `should_break`, `i8086AndTestOrXorImmed` | 0.386, 0.380 |
| df:main | `putpixel`, `rolop` | 0.661, 0.636 |

Embedding scores are not probabilities of semantic equivalence. The large
functions' neighbors are not convincing application-level guides. The current
retriever selects the best two shard results without an additional ABI/layout
relevance gate. ExeBench-trained function examples do not necessarily supply
Coreutils-specific structures, global invariants, or library behavior.

Evidence:
[get_dev retrieval](../../validation/project_level_binary_v2/retrieval/df/functions/get_dev/retrieval.json)
and [main retrieval](../../validation/project_level_binary_v2/retrieval/df/functions/main/retrieval.json).

### Feedback and Experimental Confounds

The ExeBench repair pipeline can select among compilable candidates and compare
synthetic function outputs. Coreutils uses one candidate, application output,
bounded traces, and limited inferred direct probes. Those observations do not
always localize the first faulty instruction or incorrect data-layout assumption.

Initial retrieval pilots changed prompt safeguards, instrumentation, and
validation behavior while preserving their request charges. The assisted
scaffold required a later repair. Consequently, neither equal retrieval and
no-retrieval success counts nor different request totals establish a causal
effect of RAG. The existing campaign is exploratory, not a controlled ablation.

The checkpoint names and sizes also differ. Any effect of model family, model
size, or FP8 quantization remains unmeasured without matched experiments.

## 6. Recommended Next Experiments and Improvements

1. **Establish a matched model baseline.** Run Qwen3.8 on the same ExeBench
   dataset with the same evaluator, candidate count, reasoning policy, and
   corrective-round budget. Record effective generation settings explicitly.
   Do not attribute the present gap to model quality before this control.
2. **Separate failure categories.** Report malformed contracts, invalid probes,
   incomplete output, LLVM syntax/verification errors, linker/harness errors,
   crashes, output mismatches, and missing execution evidence independently.
   Keep the denominator fixed, but avoid calling every rejection incorrect code.
3. **Improve LLVM emission and focused repair.** Prefer named SSA values,
   typed pointer offsets, correct PHI construction, and exact string lengths.
   Repair the specific failing region instead of repeatedly regenerating the
   entire function. A typed IR construction layer is a possible future design,
   not an implemented feature or validated remedy in this campaign.
4. **Decouple probe generation from signature recovery.** Give precise schema
   feedback, allow bounded probe-only repairs, and create meaningful direct
   tests for unentered helpers. Preserve all existing safety/execution gates.
5. **Strengthen ABI and global-state reasoning.** Track logical globals versus
   GOT indirection; propagate calling conventions and layout constraints across
   callers and callees. Distinguish source declarations from optimized ABIs.
6. **Reduce and partition evidence.** Include target-relevant constants, memory
   accesses, contracts, and traces. Decompose large functions with explicit
   cross-region state and control-flow contracts before recomposition.
7. **Use relevant retrieval and a frozen protocol.** Rerank by ABI, memory-access
   patterns, and semantic evidence; omit weak examples when appropriate. Run
   controlled retrieval/no-retrieval comparisons in new experiment directories
   without resetting or overwriting this campaign's budgets and evidence.

Multi-candidate and reasoning-enabled runs are proposals, not silently changed
settings. Their compute cost and budgets should be explicitly configured for
the next experiment.

## Final Assessment

The saved evidence supports three conclusions:

- Candidate multiplicity and iterative repair substantially improved the
  historical Qwen3-32B ExeBench result.
- The Coreutils framework encounters additional ABI/state/coverage problems,
  as well as avoidable generation and harness problems.
- The current experiment does not isolate Qwen3.8 model quality or prove a
  retrieval benefit. No reconstructed application passed, so the strict
  project-level recovery goal remains unmet.

Repeating the unchanged Coreutils configuration is not the next useful step.
Matched baselines and targeted fixes to the observed failure classes are needed.
