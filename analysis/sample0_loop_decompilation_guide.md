# Sample 0 Loop Decompilation Guide: `UpdatePulsars`

This note uses sample 0 from `sampled_dataset_with_loops_164` as a worked
example for teaching an LLM how to decompile x86-64 assembly into LLVM IR.

Validation source:

```text
/data1/xiachunwei/Projects/validation/gpt-oss-20b/20260511-013215_sample_loops_gpt-oss-20b-n8-assembly-without-comments-in-context-learning-similar-hermes-no-angr-trace/sample_0
```

I verified:

```text
target.ll          compile=True, execution=True, 30/30 tests passed
correct_llvm_ir.ll compile=True, execution=True, 30/30 tests passed
```

## 1. First Identify the Function Interface

The assembly starts with:

```assembly
.globl UpdatePulsars
UpdatePulsars:
```

There are no explicit function arguments. The function uses only globals and
calls other functions, so the LLVM function should be:

```llvm
define dso_local void @UpdatePulsars() {
  ...
}
```

## 2. Recover Global Variables and Types

From RIP-relative accesses:

```assembly
movl VA(%rip), %eax
movl %eax, PulsarPower(%rip)
cmpl $0, ActivePulsars(%rip)
movl Phazer2(%rip), %edi
movq Pulsars(%rip), %rbx
cmpl active(%rip), %ecx
movq NextSeg(%rip), %rcx
movq PrevSeg(%rip), %rcx
movl ThisLevel+8(%rip), %ecx
```

Infer these globals:

```llvm
@VA            = external global i32
@PulsarPower   = external global i32
@ActivePulsars = external global i32
@Phazer2       = external global i32
@Pulsars       = external global ptr
@active        = external global i32
@NextSeg       = external global ptr
@PrevSeg       = external global ptr
@ww            = external global %struct.WW
@ThisLevel     = external global %struct.Level
```

The loop increments `%r14` by `24` bytes:

```assembly
addq $24, %r14
cmpl $480, %r14d
```

So one pulsar record is 24 bytes and the loop runs `480 / 24 = 20` records.
The fields used are:

```assembly
(%rbx,%r14)       ; i32 field at offset 0
4(%rbx,%r14)      ; i32 field at offset 4
8(%rbx,%r14)      ; i64 field at offset 8
16(%rbx,%r14)     ; i32 field at offset 16
```

Use:

```llvm
%struct.Pulsar = type { i32, i32, i64, i32 }
%struct.WW     = type { i32, i32 }
%struct.Level  = type { i32, i32, i32, i32, i32 }
```

## 3. Translate the Pre-Loop Logic

Assembly:

```assembly
movl VA(%rip), %eax
andl $31, %eax
movl %eax, PulsarPower(%rip)
cmpl $0, ActivePulsars(%rip)
je .LBB0_3
cmpl $30, %eax
jne .LBB0_3
movl Phazer2(%rip), %edi
movl $9, %esi
callq PlayB@PLT
movl PulsarPower(%rip), %eax
.LBB0_3:
cmpl $16, %eax
jl .LBB0_5
movl $31, %ecx
subl %eax, %ecx
movl %ecx, PulsarPower(%rip)
```

Semantic reconstruction:

```c
PulsarPower = VA & 31;
if (ActivePulsars != 0 && PulsarPower == 30) {
    PlayB(Phazer2, 9);
    // reload PulsarPower after the call because the assembly does
}
if (PulsarPower >= 16) {
    PulsarPower = 31 - PulsarPower;
}
```

Important details:

- `andl $31` means `VA & 31`.
- The call to `PlayB` may modify globals, so the assembly reloads
  `PulsarPower` after the call.
- `cmpl $16, %eax; jl skip` means the store runs when `%eax >= 16`.

## 4. Recover the Main Loop

Assembly:

```assembly
movq Pulsars(%rip), %rbx
xorl %r14d, %r14d
...
addq $24, %r14
cmpl $480, %r14d
je .LBB0_19
```

This is:

```c
for (int i = 0; i != 20; i++) {
    struct Pulsar *psp = &Pulsars[i];
    ...
}
```

In LLVM IR, make the loop explicit with PHI nodes:

```llvm
loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
  %psp = getelementptr inbounds %struct.Pulsar, ptr %base, i32 %i
  ...

loop.latch:
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 20
  br i1 %done, label %after.loop, label %loop.header
```

## 5. Decode the Loop Body

The loop first reads field 0:

```assembly
movl (%rbx,%r14), %ecx
testl %ecx, %ecx
js .LBB0_7
```

So:

```c
int state = psp->field0;
if (state < 0) negative_case;
else nonnegative_case;
```

### 5.1 Non-Negative Case

Assembly:

```assembly
cmpl active(%rip), %ecx
jne .LBB0_18
leaq (%rbx,%r14), %rdi
callq UpdateActivePulsar@PLT
```

Meaning:

```c
if (state == active) {
    UpdateActivePulsar(psp);
}
```

### 5.2 Negative Case

Assembly:

```assembly
movl %ecx, %eax
notl %eax
shrl $2, %eax
movl %eax, 4(%rbx,%r14)
movl active(%rip), %edx
movl %edx, (%rbx,%r14)
```

Meaning:

```c
vertex = ((unsigned)~state) >> 2;
psp->field1 = vertex;
psp->field0 = active;
```

In LLVM:

```llvm
%not.state = xor i32 %state, -1
%vertex0 = lshr i32 %not.state, 2
store i32 %vertex0, ptr %field1
```

Then this condition:

```assembly
cmpl $-4, %ecx
jb .LBB0_10
cmpl $0, ww+4(%rip)
jne .LBB0_10
movl $1, 4(%rbx,%r14)
movl $1, %eax
```

Because `jb` is unsigned-below, this means:

```c
if ((unsigned)state > (unsigned)-5 && ww.field1 == 0) {
    vertex = 1;
    psp->field1 = 1;
}
```

Then clamp vertex against `ww.field0`:

```assembly
movl ww(%rip), %ecx
cmpl %ecx, %eax
jle .LBB0_12
movl %ecx, 4(%rbx,%r14)
movl %ecx, %eax
```

This is a signed comparison:

```c
if (vertex > ww.field0) {
    vertex = ww.field0;
    psp->field1 = vertex;
}
```

Finally choose `NextSeg` or `PrevSeg` based on field 2:

```assembly
cmpq $0, 8(%rbx,%r14)
cltq
js .LBB0_13
movq PrevSeg(%rip), %rcx
jmp .LBB0_14
.LBB0_13:
movq NextSeg(%rip), %rcx
.LBB0_14:
movl (%rcx,%rax,4), %eax
movl %eax, 16(%rbx,%r14)
```

Critical detail: `cltq` sign-extends `%eax`, and `%eax` is the final `vertex`,
not the loop index. So this means:

```c
int *seg = (psp->field2 < 0) ? NextSeg : PrevSeg;
psp->field3 = seg[vertex];
```

Do not accidentally index `NextSeg` or `PrevSeg` with the loop counter. That is
a common decompilation bug for this sample.

## 6. Decode the Post-Loop Logic

Assembly:

```assembly
movl ActivePulsars(%rip), %eax
cmpl $19, %eax
jg .LBB0_23
cmpl %eax, ThisLevel(%rip)
jle .LBB0_23
xorl %eax, %eax
callq VARandom@PLT
movl ThisLevel+8(%rip), %ecx
addl ThisLevel+4(%rip), %ecx
cmpl %eax, %ecx
jbe .LBB0_23
xorl %eax, %eax
callq CreateNewPulsar@PLT
```

Meaning:

```c
if (ActivePulsars <= 19 && ThisLevel.field0 > ActivePulsars) {
    if ((unsigned)(ThisLevel.field1 + ThisLevel.field2) > (unsigned)VARandom()) {
        CreateNewPulsar();
    }
}
```

Then update counters:

```assembly
movl ThisLevel(%rip), %eax
movl ActivePulsars(%rip), %ecx
subl %ecx, %eax
addl %eax, ThisLevel+16(%rip)
addl %ecx, ThisLevel+12(%rip)
```

Meaning:

```c
ThisLevel.field4 += ThisLevel.field0 - ActivePulsars;
ThisLevel.field3 += ActivePulsars;
```

## 7. Target-Equivalent LLVM IR

This is the clean semantic form of the correct decompilation. It avoids debug
metadata and TBAA noise, but preserves the behavior that matters for validation.

```llvm
; ModuleID = 'sample0_update_pulsars'
source_filename = "-"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Pulsar = type { i32, i32, i64, i32 }
%struct.WW = type { i32, i32 }
%struct.Level = type { i32, i32, i32, i32, i32 }

@VA = external dso_local local_unnamed_addr global i32, align 4
@PulsarPower = external dso_local local_unnamed_addr global i32, align 4
@ActivePulsars = external dso_local local_unnamed_addr global i32, align 4
@Phazer2 = external dso_local local_unnamed_addr global i32, align 4
@Pulsars = external dso_local local_unnamed_addr global ptr, align 8
@active = external dso_local local_unnamed_addr global i32, align 4
@ww = external dso_local local_unnamed_addr global %struct.WW, align 4
@NextSeg = external dso_local local_unnamed_addr global ptr, align 8
@PrevSeg = external dso_local local_unnamed_addr global ptr, align 8
@ThisLevel = external dso_local local_unnamed_addr global %struct.Level, align 4

declare i32 @PlayB(i32 noundef, i32 noundef) local_unnamed_addr
declare i32 @UpdateActivePulsar(ptr noundef) local_unnamed_addr
declare i64 @VARandom(...) local_unnamed_addr
declare i32 @CreateNewPulsar(...) local_unnamed_addr

define dso_local void @UpdatePulsars() local_unnamed_addr {
entry:
  %va = load i32, ptr @VA, align 4
  %power0 = and i32 %va, 31
  store i32 %power0, ptr @PulsarPower, align 4
  %active_pulsars0 = load i32, ptr @ActivePulsars, align 4
  %has_active = icmp ne i32 %active_pulsars0, 0
  %power_is_30 = icmp eq i32 %power0, 30
  %play_cond = and i1 %has_active, %power_is_30
  br i1 %play_cond, label %play_sound, label %after_play

play_sound:
  %phazer = load i32, ptr @Phazer2, align 4
  %call_play = tail call i32 @PlayB(i32 noundef %phazer, i32 noundef 9)
  %power_after_call = load i32, ptr @PulsarPower, align 4
  br label %after_play

after_play:
  %power1 = phi i32 [ %power_after_call, %play_sound ], [ %power0, %entry ]
  %power_ge_16 = icmp sgt i32 %power1, 15
  br i1 %power_ge_16, label %fold_power, label %before_loop

fold_power:
  %power_folded = sub nsw i32 31, %power1
  store i32 %power_folded, ptr @PulsarPower, align 4
  br label %before_loop

before_loop:
  %base = load ptr, ptr @Pulsars, align 8
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %before_loop ], [ %i.next, %loop.latch ]
  %psp = getelementptr inbounds %struct.Pulsar, ptr %base, i32 %i
  %state = load i32, ptr %psp, align 8
  %state_neg = icmp slt i32 %state, 0
  br i1 %state_neg, label %negative_state, label %nonnegative_state

nonnegative_state:
  %active_value = load i32, ptr @active, align 4
  %is_active = icmp eq i32 %state, %active_value
  br i1 %is_active, label %update_active, label %loop.latch

update_active:
  %call_update = tail call i32 @UpdateActivePulsar(ptr noundef nonnull %psp)
  br label %loop.latch

negative_state:
  %state_not = xor i32 %state, -1
  %vertex0 = lshr i32 %state_not, 2
  %field1 = getelementptr inbounds %struct.Pulsar, ptr %psp, i64 0, i32 1
  store i32 %vertex0, ptr %field1, align 4
  %active_value2 = load i32, ptr @active, align 4
  store i32 %active_value2, ptr %psp, align 8
  %ww1_ptr = getelementptr inbounds %struct.WW, ptr @ww, i64 0, i32 1
  %ww1 = load i32, ptr %ww1_ptr, align 4
  %ww1_zero = icmp eq i32 %ww1, 0
  %state_gt_neg5_unsigned = icmp ugt i32 %state, -5
  %force_one = and i1 %state_gt_neg5_unsigned, %ww1_zero
  br i1 %force_one, label %set_vertex_one, label %after_vertex_one

set_vertex_one:
  store i32 1, ptr %field1, align 4
  br label %after_vertex_one

after_vertex_one:
  %vertex1 = phi i32 [ 1, %set_vertex_one ], [ %vertex0, %negative_state ]
  %ww0 = load i32, ptr @ww, align 4
  %too_large = icmp sgt i32 %vertex1, %ww0
  br i1 %too_large, label %clamp_vertex, label %after_clamp

clamp_vertex:
  store i32 %ww0, ptr %field1, align 4
  br label %after_clamp

after_clamp:
  %vertex = phi i32 [ %ww0, %clamp_vertex ], [ %vertex1, %after_vertex_one ]
  %direction_ptr = getelementptr inbounds %struct.Pulsar, ptr %psp, i64 0, i32 2
  %direction = load i64, ptr %direction_ptr, align 8
  %direction_neg = icmp slt i64 %direction, 0
  %vertex64 = sext i32 %vertex to i64
  %rotation_ptr = getelementptr inbounds %struct.Pulsar, ptr %psp, i64 0, i32 3
  br i1 %direction_neg, label %use_nextseg, label %use_prevseg

use_nextseg:
  %nextseg = load ptr, ptr @NextSeg, align 8
  %next_ptr = getelementptr inbounds i32, ptr %nextseg, i64 %vertex64
  %next_value = load i32, ptr %next_ptr, align 4
  store i32 %next_value, ptr %rotation_ptr, align 8
  br label %loop.latch

use_prevseg:
  %prevseg = load ptr, ptr @PrevSeg, align 8
  %prev_ptr = getelementptr inbounds i32, ptr %prevseg, i64 %vertex64
  %prev_value = load i32, ptr %prev_ptr, align 4
  store i32 %prev_value, ptr %rotation_ptr, align 8
  br label %loop.latch

loop.latch:
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 20
  br i1 %done, label %after_loop, label %loop.header

after_loop:
  %active_pulsars1 = load i32, ptr @ActivePulsars, align 4
  %active_lt_20 = icmp slt i32 %active_pulsars1, 20
  %level0 = load i32, ptr @ThisLevel, align 4
  %level_gt_active = icmp sgt i32 %level0, %active_pulsars1
  %can_create = and i1 %active_lt_20, %level_gt_active
  br i1 %can_create, label %maybe_create, label %update_level

maybe_create:
  %rand64 = tail call i64 (...) @VARandom()
  %rand = trunc i64 %rand64 to i32
  %level1_ptr = getelementptr inbounds %struct.Level, ptr @ThisLevel, i64 0, i32 1
  %level1 = load i32, ptr %level1_ptr, align 4
  %level2_ptr = getelementptr inbounds %struct.Level, ptr @ThisLevel, i64 0, i32 2
  %level2 = load i32, ptr %level2_ptr, align 4
  %threshold = add i32 %level2, %level1
  %rand_ok = icmp ugt i32 %threshold, %rand
  br i1 %rand_ok, label %create_new, label %update_level

create_new:
  %call_create = tail call i32 (...) @CreateNewPulsar()
  br label %update_level

update_level:
  %level0_final = load i32, ptr @ThisLevel, align 4
  %active_final = load i32, ptr @ActivePulsars, align 4
  %inactive_count = sub i32 %level0_final, %active_final
  %level4_ptr = getelementptr inbounds %struct.Level, ptr @ThisLevel, i64 0, i32 4
  %level4 = load i32, ptr %level4_ptr, align 4
  %level4_new = add nsw i32 %level4, %inactive_count
  store i32 %level4_new, ptr %level4_ptr, align 4
  %level3_ptr = getelementptr inbounds %struct.Level, ptr @ThisLevel, i64 0, i32 3
  %level3 = load i32, ptr %level3_ptr, align 4
  %level3_new = add nsw i32 %level3, %active_final
  store i32 %level3_new, ptr %level3_ptr, align 4
  ret void
}
```

## 8. Step-by-Step Procedure for Other Loop Samples

Use this procedure when guiding an LLM through decompilation.

### Step 1: Classify the Function

Ask:

- Does it take arguments, or does it only access globals?
- Which external calls appear?
- Which globals are loaded/stored?
- Are there stack spills that are real locals, or only ABI/save-restore noise?

For sample 0, there are no arguments and many global accesses.

### Step 2: Recover Data Layout Before Control Flow

Infer struct layout from offsets and loop stride:

- `addq $24` means record size 24.
- Offset `0` with `movl` is `i32`.
- Offset `4` with `movl` is `i32`.
- Offset `8` with `cmpq` is `i64`.
- Offset `16` with `movl` is `i32`.

Write the struct type before writing the loop IR. Bad struct layout usually
causes wrong `getelementptr` and wrong behavior.

### Step 3: Split the Assembly into Regions

For sample 0:

1. Pre-loop power update and sound effect.
2. Main loop over 20 pulsars.
3. Optional creation of a new pulsar.
4. Final global counter updates.

The LLM should summarize each region in C-like pseudocode before writing LLVM.

### Step 4: Convert Loop Stride to Loop Bound

If assembly has:

```assembly
addq $24, %r14
cmpl $480, %r14d
```

then use:

```text
iteration count = 480 / 24 = 20
```

Do not create a byte-offset loop in LLVM unless necessary. Prefer an element
index and `getelementptr %struct.Pulsar`.

### Step 5: Track the Meaning of Each Register Across Blocks

Registers change meaning. In sample 0:

- `%r14` is byte offset in the assembly loop.
- `%eax` is first `PulsarPower`, then `vertex`, then return values.
- `%ecx` is first `state`, then `ww[0]`, then `ThisLevel` temporary.

Do not translate registers mechanically into one LLVM variable. Use semantic
names and SSA PHI/select values.

### Step 6: Preserve Signed vs Unsigned Conditions

Branch mnemonic matters:

- `jl`, `jle`, `jg`, `jge` are signed.
- `jb`, `jbe`, `ja`, `jae` are unsigned.
- `js` checks sign bit.

Sample 0 has both:

```text
cmpl $-4, %ecx; jb skip
```

This becomes an unsigned condition around `state > -5`, not a normal signed
`state >= -4` unless carefully reasoned.

### Step 7: Identify Loop-Carried Values

Every variable updated across loop iterations needs either:

- a PHI node, or
- a memory store/load if it is a real memory location.

Sample 0 loop-carried value:

```llvm
%i = phi i32 [ 0, %before_loop ], [ %i.next, %loop.latch ]
```

The `Pulsar *` can be derived from `%base` and `%i`, so no separate pointer PHI
is required in the clean version.

### Step 8: Reconstruct Memory Side Effects Carefully

In sample 0, the negative case has four important stores:

```text
psp->field1 = vertex
psp->field0 = active
psp->field1 = 1 or ww[0] in special cases
psp->field3 = NextSeg[vertex] or PrevSeg[vertex]
```

Missing any one of these usually still compiles but fails execution.

### Step 9: Be Suspicious of Index Registers After `cltq`, `movslq`, or `sext`

Sample 0 key trap:

```assembly
cltq
movl (%rcx,%rax,4), %eax
```

The array index is the final `vertex` in `%eax`, not the loop counter. This is
where many predictions go wrong.

### Step 10: Emit Valid LLVM IR, Then Verify

Before asking the LLM for final IR, force these checks:

- Every SSA name is defined once.
- Every PHI incoming block matches a real predecessor.
- Every external function declaration matches call usage.
- Every `getelementptr` uses the inferred struct type and field index.
- Signed/unsigned comparisons match branch mnemonics.
- Calls that may modify globals are followed by reloads if the assembly reloads.

Then run the normal evaluator:

```python
compile_llvm_ir(...)
eval_assembly_with_details(...)
```

For sample 0, the verified result is:

```text
compile_success: true
execution_success: true
passed: 30/30
```

## 9. Prompt Template for Teaching an LLM

For a new sample, give the LLM this staged instruction:

```text
First, do not write LLVM IR.
1. List globals and external calls.
2. Infer struct layouts from offsets and loop strides.
3. Split the assembly into high-level regions.
4. For every loop, identify:
   - induction variable
   - loop bound
   - loop-carried values
   - memory stores in the loop
   - signed/unsigned branch conditions
5. Write C-like pseudocode.
6. Only after that, write valid LLVM IR.
7. Check that array indices come from the right value, not just the latest register name.
```

For loop-heavy samples, the most important discipline is to translate semantics,
not registers. Registers are temporary carriers; the LLVM IR should name the
program values: `state`, `vertex`, `direction`, `active_count`, `threshold`,
and so on.
