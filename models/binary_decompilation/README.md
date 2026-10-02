# Binary Project Pipeline

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
