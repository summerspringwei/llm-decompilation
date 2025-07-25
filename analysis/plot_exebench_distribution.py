"""This module count how many predictions are correct and draw the distribution of the prediction length in the exebench dataset.
The format of the validation results is as follows:


example = {
    "file":"forflo/pocoblaze/pocoblaze/pocoblaze.c",
    "func_head_types":"void store_sX_at_sY()",
    "input":"\t.text\n\t.file\t\"exebench_lscat-ACT41_2020645clp5ih56.c\"\n\t.globl\tstore_sX_at_sY                  # -- Begin function store_sX_at_sY\n\t.p2align\t4, 0x90\n\t.type\tstore_sX_at_sY,@function\nstore_sX_at_sY:                         # @store_sX_at_sY\n\t.cfi_startproc\n# %bb.0:                                # %entry\n\tmovq\tregister_file(%rip), %rax\n\tmovq\tx_register_pointer(%rip), %rcx\n\tmovq\t(%rax,%rcx,8), %rcx\n\tmovq\tram_file(%rip), %rdx\n\tmovq\ty_register_pointer(%rip), %rsi\n\tmovq\t(%rax,%rsi,8), %rax\n\tmovq\t%rcx, (%rdx,%rax,8)\n\tretq\n.Lfunc_end0:\n\t.size\tstore_sX_at_sY, .Lfunc_end0-store_sX_at_sY\n\t.cfi_endproc\n                                        # -- End function\n\t.ident\t\"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 88bf774c565080e30e0a073676c316ab175303af)\"\n\t.section\t\".note.GNU-stack\",\"\",@progbits\n",
    "instruction":"Disassemble this code to LLVM-IR",
    "output":[
        "; ModuleID = '/tmp/exebench_lscat-ACT41_2020645clp5ih56.c.ll'\nsource_filename = \"/tmp/exebench_lscat-ACT41_2020645clp5ih56.c\"\ntarget datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\ntarget triple = \"x86_64-unknown-linux-gnu\"\n\n@register_file = external dso_local local_unnamed_addr global ptr, align 8\n@x_register_pointer = external dso_local local_unnamed_addr global i64, align 8\n@ram_file = external dso_local local_unnamed_addr global ptr, align 8\n@y_register_pointer = external dso_local local_unnamed_addr global i64, align 8\n\n; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable\ndefine dso_local void @store_sX_at_sY() local_unnamed_addr #0 {\nentry:\n  %0 = load ptr, ptr @register_file, align 8, !tbaa !5\n  %1 = load i64, ptr @x_register_pointer, align 8, !tbaa !9\n  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %1\n  %2 = load i64, ptr %arrayidx, align 8, !tbaa !9\n  %3 = load ptr, ptr @ram_file, align 8, !tbaa !5\n  %4 = load i64, ptr @y_register_pointer, align 8, !tbaa !9\n  %arrayidx1 = getelementptr inbounds i64, ptr %0, i64 %4\n  %5 = load i64, ptr %arrayidx1, align 8, !tbaa !9\n  %arrayidx2 = getelementptr inbounds i64, ptr %3, i64 %5\n  store i64 %2, ptr %arrayidx2, align 8, !tbaa !9\n  ret void\n}\n\nattributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable \"min-legal-vector-width\"=\"0\" \"no-trapping-math\"=\"true\" \"stack-protector-buffer-size\"=\"8\" \"target-cpu\"=\"x86-64\" \"target-features\"=\"+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87\" \"tune-cpu\"=\"generic\" }\n\n!llvm.module.flags = !{!0, !1, !2, !3}\n!llvm.ident = !{!4}\n\n!0 = !{i32 1, !\"wchar_size\", i32 4}\n!1 = !{i32 8, !\"PIC Level\", i32 2}\n!2 = !{i32 7, !\"PIE Level\", i32 2}\n!3 = !{i32 7, !\"uwtable\", i32 2}\n!4 = !{!\"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 88bf774c565080e30e0a073676c316ab175303af)\"}\n!5 = !{!6, !6, i64 0}\n!6 = !{!\"any pointer\", !7, i64 0}\n!7 = !{!\"omnipotent char\", !8, i64 0}\n!8 = !{!\"Simple C/C++ TBAA\"}\n!9 = !{!10, !10, i64 0}\n!10 = !{!\"long\", !7, i64 0}\n"
    ],
    "predict":"; ModuleID = '<stdin>'\nsource_filename = \"-\"\ntarget datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\ntarget triple = \"x86_64-unknown-linux-gnu\"\n\n@register_file = external local_unnamed_addr global ptr, align 8\n@x_register_pointer = external local_unnamed_addr global i64, align 8\n@ram_file = external local_unnamed_addr global ptr, align 8\n@y_register_pointer = external local_unnamed_addr global i64, align 8\n\n; Function Attrs: minsize mustprogress nofree norecurse nosync nounwind optsize willreturn memory(readwrite, inaccessiblemem: none) uwtable\ndefine dso_local void @store_sX_at_sY() local_unnamed_addr #0 align 4 {\n  %1 = load ptr, ptr @register_file, align 8, !tbaa !5\n  %2 = load i64, ptr @x_register_pointer, align 8, !tbaa !9\n  %3 = getelementptr inbounds i64, ptr %1, i64 %2\n  %4 = load i64, ptr %3, align 8, !tbaa !9\n  %5 = load ptr, ptr @ram_file, align 8, !tbaa !5\n  %6 = load i64, ptr @y_register_pointer, align 8, !tbaa !9\n  %7 = getelementptr inbounds i64, ptr %1, i64 %6\n  %8 = load i64, ptr %7, align 8, !tbaa !9\n  %9 = getelementptr inbounds i64, ptr %5, i64 %8\n  store i64 %4, ptr %9, align 8, !tbaa !9\n  ret void\n}\n\nattributes #0 = { minsize mustprogress nofree norecurse nosync nounwind optsize willreturn memory(readwrite, inaccessiblemem: none) uwtable \"min-legal-vector-width\"=\"0\" \"no-trapping-math\"=\"true\" \"stack-protector-buffer-size\"=\"8\" \"target-cpu\"=\"x86-64\" \"target-features\"=\"+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87\" \"tune-cpu\"=\"generic\" }\n\n!llvm.module.flags = !{!0, !1, !2, !3}\n!llvm.ident = !{!4}\n\n!0 = !{i32 1, !\"wchar_size\", i32 4}\n!1 = !{i32 8, !\"PIC Level\", i32 2}\n!2 = !{i32 7, !\"PIE Level\", i32 2}\n!3 = !{i32 7, !\"uwtable\", i32 2}\n!4 = !{!\"clang version 17.0.6 (git@github.com:fairinternal/CodeGen.git b05db9bbf7a92019267416c1bb9996fe6134e3f1)\"}\n!5 = !{!6, !6, i64 0}\n!6 = !{!\"any pointer\", !7, i64 0}\n!7 = !{!\"omnipotent char\", !8, i64 0}\n!8 = !{!\"Simple C/C++ TBAA\"}\n!9 = !{!10, !10, i64 0}\n!10 = !{!\"long\", !7, i64 0}\n",
    "predict_compile_success":True,
    "predict_execution_success":True,
    "target_compile_success":True,
    "target_execution_success":True
}

Run the following command:
python3 utils/draw_exebench_distribution.py \
    --data_files exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-beams-8_validate_exebench.json \
    --pretrained_model_path /home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd \
    --fig_file_path exebench_distribution-train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-beams-8.png

    # backup
    data_files="exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd_validate_exebench.json",
    pretrained_model_path = "/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd",
    fig_file_path = "exebench_distribution.png"
python3 utils/draw_exebench_distribution.py \
    --data_files ./validation/exebench_llmcompiler/exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd_validate_exebench.json \
    --pretrained_model_path /home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd \
    --fig_file_path exebench_distribution.png
"""

import fire
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt


def count_accuracy_and_draw(data_files="exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-beams-8_validate_exebench.json",
                            pretrained_model_path = "/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd",
                            fig_file_path = "exebench_distribution.png",
                            title = "LLMCompiler Exebench"
                            ):
    results = load_dataset("json", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    def add_token_length(example):
        example["input_length"] = len(tokenizer(example["input"]).input_ids)
        predict = example["predict"] if isinstance(example["predict"], str) else example["predict"][0]
        example["predict_length"] = len(tokenizer(predict).input_ids)
        return example

    results = results.filter(lambda x: len(x["predict"]) > 0)
    results = results.map(add_token_length, num_proc=1)
    sorted_results = results.sort("predict_length")
    max_seq_length = 4096
    bulk_size = 256
    total_count = [0 for _ in range(max_seq_length // bulk_size + 1)]
    compile_count = [0 for _ in range(max_seq_length // bulk_size + 1)]
    execution_count = [0 for _ in range(max_seq_length // bulk_size + 1)]
    for example in sorted_results["train"]:
        if example["predict_length"] > max_seq_length:
            print(example["predict_length"])
            continue
        total_count[example["predict_length"] // bulk_size] += 1
        compile_count[example["predict_length"] // bulk_size] += any(example["predict_compile_success"]) if isinstance(example["predict_compile_success"], list) else example["predict_compile_success"]
        execution_count[example["predict_length"] // bulk_size] += any(example["predict_execution_success"]) if isinstance(example["predict_execution_success"], list) else example["predict_execution_success"]

    print(np.sum(total_count), np.sum(compile_count), np.sum(execution_count))
    print(np.sum(compile_count)/np.sum(total_count), np.sum(execution_count)/np.sum(total_count))
    print(total_count)
    print(compile_count)
    print(execution_count)

    execution_count = np.array(execution_count)
    compile_count = np.array(compile_count)
    total_count = np.array(total_count)

    compile_wrong_count = total_count - compile_count
    execution_wrong_count = compile_count - execution_count

    # Prepare data for plotting
    categories = list(["Executable", "Compilable", "Error"])

    # Define the positions of the bars on the x-axis
    # x = [(i+1)*bulk_size for i in np.arange(len(execution_count))]
    x =  np.arange(len(execution_count))

    plt.figure(figsize=(10,6))
    # Plot each subcategory as a separate bar
    plt.bar(x, execution_count, label=categories[0])
    plt.bar(x, execution_wrong_count, bottom=execution_count, label=categories[1])
    plt.bar(x, compile_wrong_count, bottom=execution_count + execution_wrong_count, label=categories[2])

    # Adding labels and title
    x_labels = [str(i*bulk_size) for i in range(len(execution_count))]

    plt.xlabel('Seq Length')
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks(x, x_labels)  # Replace x-axis labels with category names
    plt.legend(title="LLMCompiler Exebench")

    # Show the plot
    plt.savefig(fig_file_path)


if __name__ == "__main__":
    fire.Fire(count_accuracy_and_draw)
