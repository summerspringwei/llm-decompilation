
## Coreutils study
Complie `basename` to llvm IR and assembly:
```bash
FLAGS="-Werror -Wall"
clang  -I. -I./lib  -Ilib -I./lib -Isrc -I./src $FLAGS -O2 -MT src/basename.o -MD -MP -MF $depbase.Tpo -S -fno-asynchronous-unwind-tables -emit-llvm -o src/basename.ll src/basename.c
clang  -I. -I./lib  -Ilib -I./lib -Isrc -I./src $FLAGS -O2 -MT src/basename.o -MD -MP -MF $depbase.Tpo -S -fno-asynchronous-unwind-tables -o src/basename.s src/basename.c
```
Note the option `-fno-asynchronous-unwind-tables` will strip the debug info like ".cfi_xxx"


We add the following lines to coreutils makefile to compile `.c` to `.ll`:
```makefile
%.ll: %.c
	$(AM_V_CC)clang -S -emit-llvm \
	$(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \
	$(lib_libcoreutils_a_CFLAGS) $(CFLAGS) \
	-o $@ $<
```

Extract the cfg
```bash
opt -passes=dot-cfg-only target.ll
```

### The overall workflow:

Compile C file to.ll
```bash
make src/tail.ll
```

1. Extract one function from LLVM IR file:
```bash
llvm-extract --func=main -S src/tail.ll -o src/tail_main.ll
```

2. Delete one function from module:
```bash
llvm-extract -delete -func=main -S src/tail.ll -o src/tail_no_main.ll
```

3. Compile function LLVM IR to assembly:
```bash
llc src/tail_main.ll -o src/tail_main.s
```

4. Use the LLM to decompile assembly to IR


5. link the decompiled `.ll` to one `.ll` file:
```bash
llvm-link src/tail_no_main.ll src/tail_main.ll -o tail.ll
```

6. Compile LLVM IR to object file:
```bash
clang -c src/tail.ll -o tail.o
```

7. link the object file to the executable binary:
```bash
clang  -Wno-format-extra-args -Wno-implicit-const-int-float-conversion -Wno-tautological-constant-out-of-range-compare -g -O2 -Wl,--as-needed  -o src/tail src/tail.o src/iopoll.o src/libver.a lib/libcoreutils.a   lib/libcoreutils.a  -ldl
```

8. Run the test:
```bash
# make TESTS=tests/tail/overlay-headers.sh check VERBOSE=yes
TAILTESTS=`make listtests | tr ' ' '\n' | grep '^tests/tail'`
make check TESTS='${TAILTESTS}'
```
or
```bash
make check TESTS="$(make listtests | tr ' ' '\n' | grep '^tests/tail')"
```

Note we add the following code in the Makefile:
```makefile
listtests:
	$(info VALUE of TESTS = $(TESTS))
```
The reason is that the `TESTS` variable contains all the tests for the `coreutils`.

## How to use ghdrid to analyze assembly:
```bash
./support/analyzeHeadless /tmp/myghidra/ qwen -import /home/xiachunwei/Projects/alpaca-lora-decompilation/validation/Qwen3-32B/sample_only_one_bb_Qwen3-32B-n8-assembly-with-comments/sample_92_retry_10/predict.o -postscript /home/xiachunwei/Projects/alpaca-lora-decompilation/analysis/test_ghidra.py -overwrite
```

## How to use vllm to serve Qwen-32B
```bash
export CUDA_VISIBLE_DEVICES=0,1 && vllm serve /home/xiachunwei/Datasets/Models/Qwen3-32B/ --dtype auto --port 9001 --max-model-len 32000 --api-key token-llm4decompilation-abc123 --gpu-memory-utilization 0.99 --swap-space 8 --tensor-parallel-size 2 --enforce-eager --served-model-name Qwen3-32B
```