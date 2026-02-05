# Remill Assembly-to-LLVM-IR Lifting

Uses [Remill](https://github.com/lifting-bits/remill) to lift x86 assembly to LLVM IR and evaluate on the exebench dataset, similar to the LLM-based decompilation flow.

## Prerequisites

1. **Remill** must be built and installed. The default path is `~/Software/remill/build`.
   - Ensure `remill-lift-17`, `remill-llvm-link-17`, and `lib/Arch/X86/Runtime/amd64.bc` exist.

2. **Dataset** at `~/Datasets/filtered_exebench/`:
   - `sampled_dataset_with_loops_and_only_one_bb_164`
   - `sampled_dataset_without_loops_164`
   - `sampled_dataset_with_loops_164`

## Usage

```bash
# Run on default dataset (sampled_dataset_with_loops_and_only_one_bb_164)
python models/gemini/remill/remill_decompilation.py

# Specify dataset and Remill path
python models/gemini/remill/remill_decompilation.py \
  --dataset_name sampled_dataset_with_loops_164 \
  --remill_build ~/Software/remill/build

# Limit samples for quick testing
python models/gemini/remill/remill_decompilation.py --limit 10

# Use multiple processes
python models/gemini/remill/remill_decompilation.py --num_processes 4
```

## Pipeline

1. **Load dataset** – Same format as gemini_decompilation
2. **Assembly → bytes** – Assemble with clang, extract .text section with objcopy
3. **Remill lift** – `remill-lift-17 --arch amd64 --bytes <hex>`
4. **Link runtime** – llvm-link with amd64.bc
5. **Compile** – llc to assembly
6. **Verify** – Use exebench `compile_llvm_ir` and `eval_assembly` for target (ground truth)

## Output

Reports the same metrics as gemini_decompilation:

- **predict_compile_success**: Remill lift + link + llc succeeded
- **predict_execution_success**: Typically 0 – Remill produces trace-based IR with `(State*, i64, Memory*)` ABI, which is incompatible with exebench's C-callable harness
- **target_compile_success** / **target_execution_success**: Ground truth baseline

## Note on Execution Success

Remill lifts to a trace-based representation: functions have signature `(State*, i64, Memory*) -> Memory*` and are named `sub_0`, etc. The exebench `Wrapper` expects assembly defining a C-callable function with the original name and signature. Therefore, `predict_execution_success` is expected to be 0 for Remill.
