# Ghidra Headless Script to Dump Basic Blocks and CFG as JSON
#
# This script is designed to be run from the command line
# using analyzeHeadless. It takes one or two arguments:
# 1. the name or address of the function to analyze
# 2. (optional) output JSON file path

import sys
import json
import base64
import time
import re

from ghidra.program.model.block import BasicBlockModel
from ghidra.util.task import ConsoleTaskMonitor


# --- Helper Functions ---
def normalize_operand(operand_str):
    """Normalize an operand by replacing hex values and specific patterns."""
    # Replace hex immediate values with generic placeholder
    operand_str = re.sub(r'0x[0-9a-fA-F]+', 'himm', operand_str)
    operand_str = re.sub(r'#0x[0-9a-fA-F]+', 'himm', operand_str)
    operand_str = re.sub(r'#-?0x[0-9a-fA-F]+', 'himm', operand_str)
    # Replace decimal immediate values
    operand_str = re.sub(r'#-?\d+', 'himm', operand_str)
    # Normalize spaces
    operand_str = operand_str.replace(' ', '_')
    operand_str = operand_str.replace(',_', ',')
    return operand_str

def normalize_instruction(mnemonic, operands_str):
    """Create normalized instruction representation."""
    norm_operands = normalize_operand(operands_str)
    if norm_operands:
        return "{}_{}".format(mnemonic.lower(), norm_operands)
    else:
        return mnemonic.lower()

# --- 1. Setup ---
# Get the program (this is set by the headless analyzer)
program = currentProgram
if program is None:
    print("Error: 'currentProgram' is not available. Run this via analyzeHeadless.")
    sys.exit(1)

# Get a headless monitor
monitor = ConsoleTaskMonitor()

# --- 2. Get the Function to Analyze from Script Arguments ---
function = None
output_file = None
try:
    # getScriptArgs() retrieves cmd-line args passed *after* the script name
    script_args = getScriptArgs()
    if len(script_args) == 0:
        print("Error: Missing required argument.")
        print("Usage: ... -postScript dump_cfg.py <function_name_or_address> [output_json]")
        sys.exit(1)
        
    target_func_str = script_args[0]
    if len(script_args) > 1:
        output_file = script_args[1]
    print("Script argument received. Target: {}".format(target_func_str))
    if output_file:
        print("Output file: {}".format(output_file))

    # Get the FunctionManager
    func_manager = program.getFunctionManager()

    # First, try to see if the argument is an address
    try:
        function = getGlobalFunctions(target_func_str)[0]
    except Exception as e:
        # Not a valid address, so we'll treat it as a name
        pass

    # If not found by address, search by name
    if function is None:
        print("Searching for function by name: '{}'...".format(target_func_str))
        # getFunctions(True) means iterate forward
        funcs_by_name = func_manager.getFunctions(True)
        for f in funcs_by_name:
            if f.getName() == target_func_str:
                function = f
                break
    
    if function is None:
        print("Error: Could not find function '{}' by name or address.".format(target_func_str))
        sys.exit(1)

except Exception as e:
    print("Error processing arguments: {}".format(e))
    sys.exit(1)

# --- 3. Run the Analysis and Build JSON ---
print("=" * 40)
print("Analyzing Function: {} at {}".format(function.getName(), function.getEntryPoint()))
print("=" * 40)

# Get the Basic Block Model
bb_model = BasicBlockModel(program)
listing = program.getListing()

# Get program file path and architecture
program_path = program.getExecutablePath()
arch_info = program.getLanguage().getLanguageID().toString()

# Determine architecture string (simplified)
if "ARM" in arch_info or "arm" in arch_info:
    if "32" in arch_info or "v7" in arch_info:
        arch = "arm-32"
    else:
        arch = "arm-64"
elif "x86" in arch_info or "X86" in arch_info:
    if "64" in arch_info:
        arch = "x86-64"
    else:
        arch = "x86-32"
else:
    arch = "unknown"

# Start timing
start_time = time.time()

try:
    # Get all CodeBlocks for the function's body
    block_iterator = bb_model.getCodeBlocksContaining(function.getBody(), monitor)

    all_blocks = []
    while block_iterator.hasNext():
        all_blocks.append(block_iterator.next())

    # Initialize data structure
    nodes = []
    edges = []
    basic_blocks = {}

    # 4. --- Process Basic Blocks (Nodes) ---
    print("\n## Processing Basic Blocks ##")
    
    for block in all_blocks:
        start_addr = block.getFirstStartAddress()
        end_addr = block.getMaxAddress()
        block_offset = int(start_addr.getOffset())
        
        print("[+] Block: {} (0x{:x})".format(start_addr, block_offset))
        
        # Add to nodes list
        nodes.append(block_offset)
        
        # Collect instruction data
        bb_mnems = []
        bb_norm = []
        bb_disasm = []
        bb_heads = []
        bb_bytes = bytearray()
        
        # Iterate through all instructions in this basic block
        instruction = listing.getInstructionAt(start_addr)
        current_addr = start_addr
        
        while instruction is not None and current_addr <= end_addr:
            # Get instruction address
            instr_offset = int(instruction.getAddress().getOffset())
            bb_heads.append(instr_offset)
            
            # Get the mnemonic
            mnemonic = instruction.getMnemonicString()
            bb_mnems.append(mnemonic.lower())
            
            # Build full operand string
            full_operands = []
            for i in range(instruction.getNumOperands()):
                full_operands.append(instruction.getDefaultOperandRepresentation(i))
            operands_str = ", ".join(full_operands) if full_operands else ""
            
            # Add to disassembly list
            if operands_str:
                bb_disasm.append("{} {}".format(mnemonic.lower(), operands_str))
            else:
                bb_disasm.append(mnemonic.lower())
            
            # Add normalized form
            bb_norm.append(normalize_instruction(mnemonic, operands_str))
            
            # Get raw bytes
            try:
                instr_bytes = instruction.getBytes()
                for b in instr_bytes:
                    bb_bytes.append(b & 0xFF)
            except Exception as e:
                print("  Warning: Could not get bytes for instruction at {}: {}".format(current_addr, e))
            
            # Move to the next instruction
            instruction = instruction.getNext()
            if instruction is not None:
                current_addr = instruction.getAddress()
            else:
                break
                
            # Stop if we've moved beyond this block
            if current_addr > end_addr:
                break
        
        # Encode bytes to base64
        b64_bytes = base64.b64encode(bytes(bb_bytes)).decode('ascii')
        
        # Calculate block length
        bb_len = len(bb_bytes)
        
        # Store block data
        basic_blocks[str(block_offset)] = {
            "bb_len": bb_len,
            "bb_mnems": bb_mnems,
            "bb_norm": bb_norm,
            "bb_disasm": bb_disasm,
            "b64_bytes": b64_bytes,
            "bb_heads": bb_heads
        }

    # 5. --- Process Jump Relationships (Edges) ---
    print("\n## Processing Edges ##")
    for block in all_blocks:
        block_start = block.getFirstStartAddress()
        block_offset = int(block_start.getOffset())
        
        # Get an iterator for all outgoing edges (destinations)
        dest_iter = block.getDestinations(monitor)
        
        if not dest_iter.hasNext():
            print("{} --> [NONE] (Likely a return block)".format(block_start))
            continue

        while dest_iter.hasNext():
            dest_ref = dest_iter.next()
            dest_addr = dest_ref.getDestinationAddress()
            dest_offset = int(dest_addr.getOffset())
            
            # Add edge
            edges.append([block_offset, dest_offset])
            print("{} --> {}".format(block_start, dest_addr))
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # 6. --- Build final JSON structure ---
    func_addr = "0x{:x}".format(int(function.getEntryPoint().getOffset()))
    
    result = {
        program_path: {
            "arch": arch,
            func_addr: {
                "elapsed_time": elapsed_time,
                "nodes": nodes,
                "edges": edges,
                "basic_blocks": basic_blocks
            }
        }
    }
    
    # 7. --- Output JSON ---
    json_output = json.dumps(result, indent=4)
    
    if output_file:
        print("\n## Writing to file: {} ##".format(output_file))
        with open(output_file, 'w') as f:
            f.write(json_output)
        print("JSON output written successfully!")
    else:
        print("\n## JSON Output ##")
        print(json_output)
            
except Exception as e:
    print("Error during analysis: {}".format(e))
    import traceback
    traceback.print_exc()

print("\nAnalysis complete.")