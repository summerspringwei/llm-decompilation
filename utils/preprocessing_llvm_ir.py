

def preprocessing_llvm_ir(llvm_ir: str) -> str:
    lines = llvm_ir.split("\n")
    new_lines = []
    for line in lines:
        if line.find('; ModuleID') != -1:
            line = "; ModuleID = '<stdin>'"
        elif line.find("source_filename") != -1:
            line = 'source_filename = "-"'
        new_lines.append(line)
    
    return "\n".join(new_lines)