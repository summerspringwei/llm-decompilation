

def preprocessing_assembly(assembly: str):
    lines = assembly.split("\n")
    new_lines = []
    for line in lines:
        if line.find('.ident') != -1 and line.find('clang version') != -1:
            continue
        if line.find('.file') != -1:
            line = '.file   "-"'
        new_lines.append(line)
    
    return "\n".join(new_lines)
