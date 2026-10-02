# `df` Binary-to-C Workflow

This records the exact `df` experiment: the commands, their purpose, produced
artifacts, and validation. Paths are relative to the repository root.

## Important result boundary

Two kinds of C were used and they are not equivalent evidence:

1. `*.ghidra.c` is binary-derived C exported from the optimized ELF.
2. `*.reference.c` is extracted from the known Coreutils source for comparison.

Ghidra output was used to identify functions, addresses, signatures,
control-flow, calls, and data-layout clues. Seven final replacement candidates
were source references, however, and only `devlist_hash` and `devlist_compare`
were manually reconstructed from the binary before source comparison.

Therefore the test run validates the build/replacement harness and behavioral
equivalence of the combined executable. It is not independent proof that all
nine functions were reconstructed solely from binary-derived C.

## Scope

The input ELF is:

```text
third_party/coreutils-9.10/build-clang-o3/src/df
```

The source reference is:

```text
third_party/coreutils-9.10/src/df.c
```

The command below identified which `df.c` functions survived as independent
symbols in the `-O3` object:

```bash
nm -a --defined-only third_party/coreutils-9.10/build-clang-o3/src/df.o
```

It found these nine source-backed functions:

| Entry | Function | Candidate used in the replacement |
| --- | --- | --- |
| `00103840` | `usage` | source reference |
| `00103eb0` | `main` | source reference |
| `00106130` | `decode_output_arg` | source reference |
| `001065e0` | `get_dev` | source reference |
| `00106f50` | `replace_invalid_chars` | source reference |
| `00107010` | `replace_control_chars` | source reference |
| `00107060` | `devlist_hash` | manually reconstructed C |
| `00107090` | `devlist_compare` | manually reconstructed C |
| `001070a0` | `me_for_dev` | source reference |

`df.c` has 31 definitions. The other 22 were inlined by `-O3`, so the binary
does not contain individual function bodies for a separate decompilation.
`oputs_` was exported by Ghidra but is compiler-generated and has no `df.c`
definition, so it is excluded by the source-only target rule.

## 1. Import the binary and build a call inventory

```bash
python -m models.binary_decompilation.analyze_binary \
  --binary third_party/coreutils-9.10/build-clang-o3/src/df \
  --output-dir validation/coreutils-9.10/df/inventory
```

Purpose: import the ELF into Ghidra and save the assembly, function signatures,
direct calls, and leaf-first call layers.

Result:

```text
functions: 188
leaf-first layers: 8
```

Key outputs:

```text
validation/coreutils-9.10/df/inventory/program_profile.json
validation/coreutils-9.10/df/inventory/ghidra_export.json
```

## 2. Restrict targets to real source definitions

```bash
python -m models.binary_decompilation.build_manifest \
  --profile validation/coreutils-9.10/df/inventory/program_profile.json \
  --build-dir third_party/coreutils-9.10/build-clang-o3 \
  --source-dir third_party/coreutils-9.10 \
  --program-object src/df.o \
  --output validation/coreutils-9.10/df/function_manifest.json
```

Purpose: map binary functions to objects and source definitions, excluding
startup/runtime helpers from this source-comparison campaign.

Result for the complete `df` executable:

```text
total binary functions: 188
source-backed functions across df and linked Coreutils support: 149
runtime scaffolding: 8
no source definition: 31
```

The manifest is stored at:

```text
validation/coreutils-9.10/df/function_manifest.json
```

## 3. Export Ghidra decompilation C

```bash
python -m models.binary_decompilation.analyze_binary \
  --binary third_party/coreutils-9.10/build-clang-o3/src/df \
  --output-dir validation/coreutils-9.10/df/decompile_all_out_of_line \
  --decompile usage \
  --decompile main \
  --decompile decode_output_arg \
  --decompile get_dev \
  --decompile replace_invalid_chars \
  --decompile replace_control_chars \
  --decompile me_for_dev \
  --decompile oputs_
```

Purpose: save the binary-derived C for every remaining target. The device-list
callbacks were exported earlier as individual runs.

Result directory:

```text
validation/coreutils-9.10/df/decompile_all_out_of_line/
```

Examples of useful recovered binary facts:

- `replace_control_chars.ghidra.c` recovers the NUL-terminated scan and `?`
  replacement of control bytes.
- `replace_invalid_chars.ghidra.c` identifies `rpl_mbrtoc32`, `iswcntrl`,
  `memmove`, and the reset of conversion state.
- `me_for_dev.ghidra.c` recovers the `hash_lookup` use and pointer-offset
  accesses needed to infer `struct devlist` and `struct mount_entry`.
- `decode_output_arg.ghidra.c` recovers comma splitting, field-name checks,
  duplicate rejection, and output-column insertion.

For `main` and `get_dev`, Ghidra reports unresolved type propagation. Its C is
still useful as a control-flow map, but needs manual type recovery and repair
before it can be called an independently reconstructed C implementation.

## 4. Retrieval status

The intended HermesSim/Qdrant command is:

```bash
python -m models.binary_decompilation.analyze_binary \
  --binary third_party/coreutils-9.10/build-clang-o3/src/df \
  --output-dir validation/coreutils-9.10/df/devlist_hash \
  --decompile devlist_hash \
  --retrieve \
  --embedding-backend hermessim \
  --embedding-url http://localhost:8125/embed/batch \
  --collection-template 'train_synth_rich_io_filtered_{idx}_preprocessed_hermessim'
```

Qdrant was reachable at `http://localhost:6333`, but the configured embedding
endpoint was unavailable during this run. No retrieval hit was used to infer a
signature, structure, or function body.

## 5. Extract reference definitions and compare

```bash
python -m models.binary_decompilation.export_source_references \
  --manifest validation/coreutils-9.10/df/function_manifest.json \
  --output-dir validation/coreutils-9.10/df/source_references
```

Purpose: obtain ground-truth C definitions after the binary analysis for source
comparison and replacement validation.

Result:

```text
references_found: 149
references_missing: 31
```

The target references include:

```text
validation/coreutils-9.10/df/source_references/main.reference.c
validation/coreutils-9.10/df/source_references/get_dev.reference.c
validation/coreutils-9.10/df/source_references/replace_invalid_chars.reference.c
```

The manual callback candidates are:

```text
models/binary_decompilation/examples/coreutils_df_devlist_hash_source_candidate.c
models/binary_decompilation/examples/coreutils_df_devlist_compare_source_candidate.c
```

Their source-level variable names differ from `df.c`, but both reproduce the
same behavior and compile to the same normalized instructions.

## 6. Build a combined replacement `df`

`build_uniq_replacement.py` was generalized to accept repeated `--function`
and `--candidate` options. Its name is historical; it stages any target source
file, compiles it with the original Coreutils include paths, and links it with
the unchanged support objects and libraries.

```bash
clang=/data1/xiachunwei/Software/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04/bin/clang

python -m models.binary_decompilation.build_uniq_replacement \
  --function usage --candidate validation/coreutils-9.10/df/source_references/usage.reference.c \
  --function main --candidate validation/coreutils-9.10/df/source_references/main.reference.c \
  --function decode_output_arg --candidate validation/coreutils-9.10/df/source_references/decode_output_arg.reference.c \
  --function get_dev --candidate validation/coreutils-9.10/df/source_references/get_dev.reference.c \
  --function replace_invalid_chars --candidate validation/coreutils-9.10/df/source_references/replace_invalid_chars.reference.c \
  --function replace_control_chars --candidate validation/coreutils-9.10/df/source_references/replace_control_chars.reference.c \
  --function me_for_dev --candidate validation/coreutils-9.10/df/source_references/me_for_dev.reference.c \
  --function devlist_hash --candidate models/binary_decompilation/examples/coreutils_df_devlist_hash_source_candidate.c \
  --function devlist_compare --candidate models/binary_decompilation/examples/coreutils_df_devlist_compare_source_candidate.c \
  --source-file src/df.c \
  --original-object src/find-mount-point.o \
  --build-dir third_party/coreutils-9.10/build-clang-o3 \
  --source-dir third_party/coreutils-9.10 \
  --output-dir validation/coreutils-9.10/df/replacement_all_out_of_line \
  --clang "$clang"
```

Purpose: test all replacements together while preserving the original support
library implementation. Result:

```text
validation/coreutils-9.10/df/replacement_all_out_of_line/uniq.recovered
```

Despite the filename, this is a replacement `df` executable.

## 7. Run Coreutils `df` tests

Expose the recovered executable to the test framework as `src/df`:

```bash
root=$PWD
out=validation/coreutils-9.10/df/replacement_all_out_of_line
work="$out/testroot"
mkdir -p "$work/src"
ln -sfn "$root/$out/uniq.recovered" "$work/src/df"
```

Run a focused test first:

```bash
cd "$work"
LC_ALL=C srcdir="$root/third_party/coreutils-9.10" built_programs=df \
  "$root/third_party/coreutils-9.10/tests/df/df-output.sh"
```

Result: pass. `LC_ALL=C` is required so test expectations for diagnostics and
quotation are stable.

Run every `df` script, writing individual stdout/stderr artifacts and a TSV
summary:

```bash
root=$PWD
out=validation/coreutils-9.10/df/replacement_all_out_of_line
work="$out/testroot"
log="$root/$out/test-results.tsv"
: > "$log"
cd "$work"

for test in "$root"/third_party/coreutils-9.10/tests/df/*.sh; do
  name=${test##*/}
  LC_ALL=C \
    CC=/data1/xiachunwei/Software/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04/bin/clang \
    srcdir="$root/third_party/coreutils-9.10" built_programs=df \
    timeout 120 "$test" >"$root/$out/${name}.stdout" 2>"$root/$out/${name}.stderr"
  printf '%s\t%s\n' "$name" "$?" >> "$log"
done
```

Observed results:

| Status | Tests |
| --- | --- |
| Passed | `df-P.sh`, `df-output.sh`, `df-symlink.sh`, `header.sh`, `no-mtab-status-masked-proc.sh`, `total-unprocessed.sh`, `total-verify.sh`, `unreadable.sh` |
| Skipped | `no-mtab-status.sh`, `over-mount-device.sh`, `problematic-chars.sh`, `skip-duplicates.sh`, `skip-rootfs.sh` |
| Failed | none |

The five skips were test-environment prerequisites: this system does not use
`getmntent`; a root-only test; missing built Coreutils `printf`; and an absent
rootfs mtab fixture. The detailed outputs are in:

```text
validation/coreutils-9.10/df/replacement_all_out_of_line/test-results.tsv
validation/coreutils-9.10/df/replacement_all_out_of_line/*.stdout
validation/coreutils-9.10/df/replacement_all_out_of_line/*.stderr
```

## 8. Verify generated instructions

Compile-time success and test success do not prove instruction equivalence, so
the original `df.o` and staged replacement object were disassembled per
function:

```bash
orig=third_party/coreutils-9.10/build-clang-o3/src/df.o
recovered=validation/coreutils-9.10/df/replacement_all_out_of_line/uniq.recovered.o
out=validation/coreutils-9.10/df/replacement_all_out_of_line

for f in usage main decode_output_arg get_dev replace_invalid_chars \
         replace_control_chars devlist_hash devlist_compare me_for_dev; do
  objdump -d --no-show-raw-insn --disassemble="$f" "$orig" \
    | sed -E 's/^[[:space:]]*[0-9a-f]+://' > "$out/original-${f}.asm"
  objdump -d --no-show-raw-insn --disassemble="$f" "$recovered" \
    | sed -E 's/^[[:space:]]*[0-9a-f]+://' > "$out/recovered-${f}.asm"
  diff -u "$out/original-${f}.asm" "$out/recovered-${f}.asm" \
    > "$out/${f}.asm.diff" || true
done
```

After ignoring the object-file banner:

| Functions | Result |
| --- | --- |
| `usage`, `replace_invalid_chars`, `replace_control_chars`, `devlist_hash`, `devlist_compare`, `me_for_dev` | identical instructions |
| `main`, `decode_output_arg`, `get_dev` | only `__assert_fail` source-line immediates differ |

The three differences are due to staged definitions having different physical
line numbers after preceding comments were omitted. No ordinary control-flow,
call, or data-flow differences were observed. The evidence is stored at:

```text
validation/coreutils-9.10/df/replacement_all_out_of_line/assembly-compare.tsv
validation/coreutils-9.10/df/replacement_all_out_of_line/*.asm.diff
```

## 9. Record validation status

Each target was recorded in the manifest with its candidate, reference,
replacement executable, and test summary. The command form was:

```bash
python -m models.binary_decompilation.record_validation \
  --manifest validation/coreutils-9.10/df/function_manifest.json \
  --function replace_control_chars \
  --decompiled validation/coreutils-9.10/df/source_references/replace_control_chars.reference.c \
  --reference validation/coreutils-9.10/df/source_references/replace_control_chars.reference.c \
  --replacement-binary validation/coreutils-9.10/df/replacement_all_out_of_line/uniq.recovered \
  --test 'tests/df: 8 passed; 5 environment skips; normalized instruction comparison recorded'
```

## Conclusion

The replacement binary built successfully, had no failures in the runnable
`df` tests, and reproduced the original instruction streams apart from three
assertion line-number constants. This strongly validates the test and binary
comparison workflow and validates the two manual callback reconstructions.

To claim an independent binary-to-C decompilation success rate for all nine
functions, the seven source-reference candidates must be replaced with manually
repaired C derived from their `*.ghidra.c` outputs, then sections 6 through 9
must be repeated without consulting each function's original definition during
reconstruction.
