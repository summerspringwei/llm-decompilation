
## How to run ghidra to decompile

```bash
./support/analyzeHeadless /tmp/myghidra/ qwen -import /home/xiachunwei/Projects/alpaca-lora-decompilation/validation/Qwen3-32B/sample_only_one_bb_Qwen3-32B-n8-assembly-with-comments/sample_92_retry_10/predict.o -postscript /home/xiachunwei/Projects/llm-decompilation/models/ghidra_decompile/ghidra_decompile.py -overwrite
```

```bash
rm -rf /tmp/myghidra/
mkdir /tmp/myghidra/
$GHIDRA_HOME/support/analyzeHeadless /tmp/myghidra/ sample155 -import /data1/xiachunwei/Projects/validation/Qwen3-32B/sample_loops_Qwen3-32B-n8-assembly-without-comments-ghidra-decompile/sample_155_retry_5/target.o -overwrite -postscript /data1/xiachunwei/Projects/llm-decompilation/models/ghidra_decompile/ghidra_decompile_script.py 
```

## How to run ghidra to extract basic blocks of a function and do program analysis

This is the example that can extract the basic block in a function dump the json file in the format required in [binary_function_similarity](https://github.com/Cisco-Talos/binary_function_similarity.git)

Usage:
```bash
$GHIDRA_HOME/support/analyzeHeadless path/to/ghdira_project project_name -import path/to/object_file  -overwrite -postscript llm-decompilation/models/ghidra_decompile/ghidra_extract_bb.py function_name path/to/output_json_file
```


Here is one example:
```bash
$GHIDRA_HOME/support/analyzeHeadless /tmp/myghidra/ sample155 -import /data1/xiachunwei/Projects/validation/Qwen3-32B/sample_loops_Qwen3-32B-n8-assembly-without-comments-ghidra-decompile/sample_155_retry_5/target.o -overwrite -postscript /data1/xiachunwei/Projects/llm-decompilation/models/ghidra_decompile/ghidra_extract_bb.py ointerest 
```