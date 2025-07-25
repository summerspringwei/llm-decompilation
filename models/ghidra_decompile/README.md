
## How to run ghidra to decompile

```bash
./support/analyzeHeadless /tmp/myghidra/ qwen -import /home/xiachunwei/Projects/alpaca-lora-decompilation/validation/Qwen3-32B/sample_only_one_bb_Qwen3-32B-n8-assembly-with-comments/sample_92_retry_10/predict.o -postscript /home/xiachunwei/Projects/llm-decompilation/models/ghidra_decompile/ghidra_decompile.py -overwrite
```