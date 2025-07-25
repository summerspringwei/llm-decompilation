

def preprocessing_assembly(assembly: str, remove_comments: bool = False, comment_character='#') -> str:
    """Preprocessing the assembly code to remove the clang version and file path.
    
    """
    lines = assembly.split("\n")
    new_lines = []
    for line in lines:
        if line.find('.ident') != -1 and line.find('clang version') != -1:
            continue
        if line.find('.file') != -1:
            line = '.file   "-"'
        if remove_comments and comment_character in line:
            line = (line.split('#')[0]).strip()
        new_lines.append(line)
    
    return "\n".join(new_lines)
