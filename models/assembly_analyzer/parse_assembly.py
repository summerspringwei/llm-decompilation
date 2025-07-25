"""
This script extracts all the called function names from an x86 assembly file.
"""

import re
from typing import List
# def extract_called_functions(assembly_file_path):
#     """
#     Reads an x86 assembly file and extracts all the called function names.

#     Args:
#         assembly_file_path (str): Path to the assembly file.

#     Returns:
#         list: A list of unique function names called in the assembly file.
#     """
#     called_functions = set()
#     call_instruction_pattern = re.compile(r'\bcall\s+([a-zA-Z_][a-zA-Z0-9_]*)')

#     with open(assembly_file_path, 'r') as file:
#         for line in file:
#             match = call_instruction_pattern.search(line)
#             if match:
#                 called_functions.add(match.group(1))

#     return sorted(called_functions)

def extract_called_functions(assembly: str)->List[str]:
    """
    Extracts all the called function names from an x86 assembly code.
    Args:
        assembly (str): The assembly code as a string.
    Returns:
        list: A list of unique function names called in the assembly code.
    """
    functions = []
    lines = assembly.splitlines()
    for line in lines:
        line = line.strip()
        if line.find("call") != -1:
            # print(line)
            # Extract the function name from the call instruction
            # The function name is usually the first word after "call"
            parts = line.split()
            if len(parts) > 1:
                function_name = parts[1]
                # Remove any trailing characters (like parentheses)
                function_name = re.sub(r'\(.*\)', '', function_name)
                function_name = re.sub(r',.*', '', function_name)
                functions.append(function_name)
    # Remove duplicates by converting the list to a set and back to a list
    functions = list(set(functions))
    return functions



def get_called_functions_details():
    """
    We may need to find the assembly code of the called functions.
    Then we extract the assembly code that is related with the calling conventions.
    This function is a placeholder for future implementation.
    
    The called functions may be classified into:
    1. External functions: Functions that are defined outside the current file.
    2. Internal functions: Functions that are defined within the current file.
    3. Library functions: Functions that are part of a library (e.g., libc).
    """
    pass  # Placeholder for future implementation

# Example usage:
# functions = extract_called_functions('example.asm')
# print(functions)