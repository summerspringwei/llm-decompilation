
"""
llc: error: llc: /home/xiachunwei/Projects/alpaca-lora-decompilation/tmp_validate_exebench/PRETgroup/goFB/examples/goFB_only/c_tcrest/bottlingplant_mem/c_fbc/CanisterCounter.c/predict.ll:20:56: error: invalid getelementptr indices
  %9 = getelementptr inbounds %struct.CanisterCounter, ptr %0, i32 0, i32 11
                                                       ^
"""

import re

def clean_file_path_position(error_string):
    """
    Remove file paths and optional position information from error messages.
    
    Args:
        error_string (str): The original error message containing file paths and optional positions
        
    Returns:
        str: Cleaned error message with file paths and positions removed
    """
    # Pattern to match file paths with or without positions
    # This matches:
    # 1. Regular paths: /path/to/file
    # 2. Paths with positions: /path/to/file:20:56:
    # The (?::\d+:\d+:)? part makes the position numbers optional
    path_pattern = r'(?:/[^:\s]+)+(?::\d+:\d+:)?'
    
    # Remove the file paths and position information
    cleaned_message = re.sub(path_pattern, '', error_string)
    
    # Remove extra spaces and normalize whitespace
    cleaned_message = ' '.join(cleaned_message.split())
    
    return cleaned_message


def preprocessing_llc_error_msg(llc_error_msg: str) -> str:
    lines = llc_error_msg.split("\n")
    new_lines = []
    for line in lines:
        line = line.strip()
        if line.find("llc: error: ") >= 0:
            line = line.replace("llc: error: ", "")
            line = clean_file_path_position(line)
        elif line.find("^") >= 0:
            continue
        new_lines.append(line)
    
    return "\n".join(new_lines)