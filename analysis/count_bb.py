"""
Count the basic block count of the llvm ir code.
This file shall be deprecated, as now we use the a new tool to count the basic block count.
"""
import json
import subprocess
import tempfile

import tqdm
import numpy as np
import matplotlib.pyplot as plt


import logging
logger = logging.getLogger(__name__)

bb_count_binary: str = "/home/xiachunwei/Projects/llm4compiler/src/cpp/build/count_llvm_ir_bb"


def count_llvm_ir_bb(llvm_ir: str)->dict:
    """Get llvm ir basic block count
    Args:
        llvm_ir: llvm ir code
    Returns:
        dict: basic block count
        Example of return: "{"func_name": "BusFault_Handler" ,"bbcount":2,"bb_list_size": [1,1]}"
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:        
        # Get the name of the temporary file
        llvm_ir_file = tmp_file.name
        # Write data to the temporary file
        with open(llvm_ir_file, 'w') as f:
            llvm_ir = f.write(llvm_ir)
            f.flush()
        try:
            cmd_out = subprocess.run([bb_count_binary, llvm_ir_file], stdout=subprocess.PIPE)
            if cmd_out.returncode != 0:
                logger.error(f"Error Counting bb for: {llvm_ir_file} output: {cmd_out.stdout}")
                return {}
            llvm_ir_bb_count = json.loads(cmd_out.stdout.decode("utf-8"))
            return llvm_ir_bb_count
        except :
            logger.info(f"Error Counting bb for: {llvm_ir} output: {cmd_out.stdout}")
    return {}


def set_bb_count(record):
    if "llvm_ir" in record:
        llvm_ir_bb_count = count_llvm_ir_bb(record['llvm_ir']['code'][-1])
        record['llvm_ir']["bb_count"] = llvm_ir_bb_count
    else:
        record['llvm_ir']["bb_count"] = {}
    return record


def get_bulk_list(train_dataset: list)->dict[str, list]:
    """Aggregate records by basic block count
    Args:
        train_dataset: exebench dataset
    Returns:
        dict: key: basic block count, value: list of records
    """
    bulk_len_record = {}
    for record in tqdm.tqdm(train_dataset):
        if 'llvm_ir' not in record:
            continue
        if 'bb_count' not in record['llvm_ir']:
            continue
        if 'bbcount' not in record['llvm_ir']['bb_count']:
            continue
        bb_count = record['llvm_ir']['bb_count']['bbcount']
        if bb_count not in bulk_len_record:
            bulk_len_record[bb_count] = [record, ]
        else:
            bulk_len_record[bb_count].append(record)
    
    return bulk_len_record
