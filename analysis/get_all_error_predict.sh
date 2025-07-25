#!/bin/bash

# Define the directory to search in and the output file
search_dir="/home/xiachunwei/Projects/alpaca-lora-decompilation/tmp_dir/gemini-2.0-split_0-sample100"
output_file="/home/xiachunwei/Projects/alpaca-lora-decompilation/analysis/tmp_all_error_predict_gemini-2.0-split_0-sample100.txt"

# Find all files named "error_predict.error" and save their full paths to the output file
find "$search_dir" -type f -name "error_predict.error" > "$output_file"

# Print a message indicating the task is done
echo "File paths have been saved to $output_file"
