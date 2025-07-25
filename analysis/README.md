

## Filter Redundant records in the Exebench dataset
We found that after compiling the C to LLVM IR, there are samples the they share the same LLVM IR.
We developed a tool `llvm-mdiff` to check whether two LLVM IR module function are the same.

```bash
python3 analysis/filter_exebench_basedon_bb.py --src_dir=SRC_DIR --dst_dir=DST_DIR --llvm_diff=path/to/llvm-mdiff
```

## Plot the distribution of number of basic blocks and number of instructions of BB=1

```shell
python3 analysis/plot_bb_and_inst_distribution.py --path_to_dataset=DIR --save_path="figures"
    name_hint="train_synth_rich_io_filtered_2_llvm_extract_func_ir_assembly_O2"
```

## Plot the curve on after each fix iteration, how many correct samples are get

```shell
python3 plot_llm_decompile_accuracy.py --output_dir="validation/Qwen3-32B/sample_only_one_bb_Qwen3-32B-n8-assembly-with-comments" --num_retry=10
                                   --save_path="figures/accumulated_success_count.png"
```
Here is one example:
![Accumulated success count for samples without loops](../figures/sample_without_loops_Qwen3-32B-n8-assembly-without-comments-accumulated_success_count.png)


## Plot a pie figure to show the distribution of `llc` compiling errors on the failed samples
```shell
python3 plot_llm_decompile_errors_pie.py 
```
Here is one example:
![Error Type Pie](../figures/sample_only_one_bb_qwen3-32b_gemini_error_type_pie.png)