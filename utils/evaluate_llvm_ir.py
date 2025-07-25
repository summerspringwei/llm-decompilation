
import os
import json
import subprocess
import re
from pathlib import Path
import fire

def dump_llvm_module_to_cfg(file_path: str):
    try:
        cmd_out = subprocess.run(["opt", "-dot-cfg", "-disable-output", "-enable-new-pm=0", file_path], capture_output=True, text=True)
        if cmd_out.returncode == 0:
            # Get the function name
            match = re.search(r"'(.*?)'", cmd_out.stderr)
            if match:
                function_name = match.group(1)
                graph_file_name = file_path.replace(".ll", ".png")
                Path(os.path.dirname(graph_file_name)).mkdir(parents=True, exist_ok=True)
                cmd_out = subprocess.run(["dot", "-Tpng", f"{function_name}", "-o", graph_file_name])
                if cmd_out.returncode == 0:
                    print(f"sunccesfully generate {graph_file_name}")
                    return 1
        else:
            print("predict failed", file_path, cmd_out.stderr)
    except Exception as e:
            print(e)
    return 0


def verify_perfect_decompilation_text_based(predict_file: str, output_file: str):
    # 1. Compare the content of two llvm ir
    perfect = False
    with open(predict_file, "r") as f:
        predict_content = f.read()
    with open(output_file, "r") as f:
        output_content = f.read()
    if predict_content == output_content:
        perfect = True
    # 2. If not perfect, compare the assembly code
    predict_asm = subprocess.run(["llc", "-march=x86-64", "-filetype=asm", predict_file], capture_output=True, text=True)
    output_asm = subprocess.run(["llc", "-march=x86-64", "-filetype=asm", output_file], capture_output=True, text=True)
    if predict_asm.returncode == 0 and output_asm.returncode == 0:
        with open(predict_file.replace(".ll", ".s"), "r") as f:
            predict_asm_content = f.read()
        with open(output_file.replace(".ll", ".s"), "r") as f:
            output_asm_content = f.read()
        if predict_asm_content == output_asm_content:
            perfect = True
    # TODO(Chunwei): Add more verification methods
    return perfect


def verify_perfect_decompilation_llvm_ir(predict_file: str, output_file: str):
    # 1. Compare the content of two llvm ir
    perfect = False
    cmd_out = subprocess.run(["llvm-diff", predict_file, output_file], capture_output=True, text=True)
    if cmd_out.returncode == 0 and cmd_out.stdout == "":
        perfect = True
    else:
        dir_path = os.path.dirname(predict_file)
        diff_file = os.path.join(dir_path, "ir_diff.txt")
        with open(diff_file, "w") as f:
            f.write(cmd_out.stdout)
            f.write(cmd_out.stderr)
    return perfect


def get_dataset_subdir_path(path: str, dataset_name = "AnghaBench"):
    com = path.split("/")
    start_idx = 0
    for c, idx in zip(com, range(len(com))):
        if c.find(dataset_name) >=0 :
            start_idx = idx
            break
    return "/".join(com[start_idx:])


def pre_process_llvm_ir(content:str, file_path: str):
    with open(file_path, "w") as f:
        # For CodeLlama based tokenizer
        if content.find("</s><s>"):
            content = content.replace("</s><s>", "\n")
        # For DeepSeekCoder based tokenizer
        if content.find("<｜end▁of▁sentence｜><｜begin▁of▁sentence｜>"):
            content = content.replace("<｜end▁of▁sentence｜><｜begin▁of▁sentence｜>", "\n")
        f.write(content)


def main(val_file_path: str = "val.json", out_dir: str = "val_result"):
    programs = json.load(open(val_file_path, 'r'))
    if not os.path.exists(out_dir):
        print(f"Creating directory {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
    success_compile, perfect_compile = 0, 0
    for record in programs:
        dir_path = os.path.join(out_dir, get_dataset_subdir_path(record["file"]).rstrip(".ll"))
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        ret_code_list = []
        for idx, content in zip(range(len(record["predict"])), record["predict"]):
            pre_process_llvm_ir(content, os.path.join(dir_path, f"predict_{idx}.ll"))
            ret_code = dump_llvm_module_to_cfg(os.path.join(dir_path, f"predict_{idx}.ll"))
            ret_code_list.append(ret_code)
        with open(os.path.join(dir_path, "output.ll"), "w") as f:
            f.write(record["output"])
        # opt -dot-cfg -disable-output -enable-new-pm=0 file.ll
        can_compile = False
        for ret_code in ret_code_list:
            if ret_code == 1:
                can_compile = True
                break
        success_compile += 1 if can_compile else 0
        dump_llvm_module_to_cfg(os.path.join(dir_path, "output.ll"))
        if can_compile:
            for idx in range(len(record["predict"])):
                if verify_perfect_decompilation_llvm_ir(os.path.join(dir_path, f"predict_{idx}.ll"), os.path.join(dir_path, "output.ll")):
                    perfect_compile += 1
                    break;
    
    print(f"Total programs: {len(programs)}, of which {perfect_compile} are perfectly decompiled, {success_compile} are successfully compiled.")


def test_dump():
    file_path = "decompilation_val/test.ll"
    dump_llvm_module_to_cfg(file_path)


if __name__ == "__main__":
    fire.Fire(main)
