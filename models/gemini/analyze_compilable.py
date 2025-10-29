import pickle
from typing import List

from models.gemini.llm_decompiler import LLMDecompileRecord


def get_pass_list(llm_decompile_record_list: List[LLMDecompileRecord]):
    pass_list = []
    
    for llm_decompile_record in llm_decompile_record_list:
        success = False
        for idx, response_validation in llm_decompile_record.retry_response_validation.items():
            execution_list = [result.execution_success for result in response_validation.predict_evaluation_results_list]
            if any(execution_list):
                pass_list.append((idx, True))
                success = True
                break
        if not success:
            pass_list.append((10, False))

    return pass_list


def get_num_execution_success(llm_decompile_record_list: List[LLMDecompileRecord]) -> int:
    markdown = ""
    markdown += "| record idx | retry -1 | retry 0 | retry 1 | retry 2 | retry 3 | retry 4 | retry 5 | retry 6 | retry 7 | retry 8 | retry 9 | retry 10 | \n"
    markdown += "|-----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------| \n"
    for record_idx, llm_decompile_record in enumerate(llm_decompile_record_list):
        markdown += f"| {record_idx} |" 
        for retry_idx, response_validation in llm_decompile_record.retry_response_validation.items():
            # print(f"record idx: {record_idx}, retry idx: {retry_idx}, num execution success: {response_validation.get_num_compile_success()}")
            markdown += f" {response_validation.get_num_compile_success()} |"
        for _ in range(12 - len(llm_decompile_record.retry_response_validation)):
            markdown += " - |"
        markdown += "\n"
    return markdown



def generate_markdown_table(pass_list1: List[tuple], pass_list2: List[tuple], label1: str = "List 1", label2: str = "List 2"):
    """
    Generate a markdown table showing True/False values for two lists
    """
    # Prepare table data
    indices1 = [x[0] for x in pass_list1]
    bool_values1 = [x[1] for x in pass_list1]
    
    indices2 = [x[0] for x in pass_list2]
    bool_values2 = [x[1] for x in pass_list2]
    
    # Calculate match count
    max_len = max(len(pass_list1), len(pass_list2))
    match_count = 0
    
    for i in range(max_len):
        val1 = bool_values1[i] if i < len(bool_values1) else None
        val2 = bool_values2[i] if i < len(bool_values2) else None
        if val1 == val2 and val1 is not None and val2 is not None:
            match_count += 1
    
    # Generate markdown
    markdown = f"# Compilation Success Comparison\n\n"
    markdown += f"**Match Rate:** {match_count}/{min(len(pass_list1), len(pass_list2))} = {match_count/min(len(pass_list1), len(pass_list2))*100:.1f}%\n\n"
    
    # Header
    markdown += "| Index | Record # | " + label1 + " | " + label2 + " | Match |\n"
    markdown += "|-------|----------|" + "-" * (len(label1) + 2) + "|" + "-" * (len(label2) + 2) + "|-------|\n"
    
    both_true_count = 0
    both_false_count = 0
    ghida_true_in_context_false_count = 0
    ghida_false_in_context_true_count = 0
    # Data rows
    for i in range(max_len):
        # Record number
        row = f"| {i+1} |"
        
        # Index
        idx1 = indices1[i] if i < len(indices1) else '-'
        idx2 = indices2[i] if i < len(indices2) else '-'
        row += f" {idx1}/{idx2} |"
        
        # Bool values
        val1 = bool_values1[i] if i < len(bool_values1) else None
        val2 = bool_values2[i] if i < len(bool_values2) else None
        both_true_count += 1 if val1 and val2 else 0
        both_false_count += 1 if not val1 and not val2 else 0
        ghida_true_in_context_false_count += 1 if val1 and not val2 else 0
        ghida_false_in_context_true_count += 1 if not val1 and val2 else 0
        
        row += " ✓ |" if val1 else " ✗ |"
        row += " ✓ |" if val2 else " ✗ |"
        
        # Match column
        if val1 == val2 and val1 is not None and val2 is not None:
            row += " ✓ |"
        else:
            row += " ✗ |"
        
        markdown += row + "\n"
    
    markdown += f"\n**Both True Count:** {both_true_count}\n"
    markdown += f"**Both False Count:** {both_false_count}\n"
    markdown += f"**Ghidra True In Context False Count:** {ghida_true_in_context_false_count}\n"
    markdown += f"**Ghidra False In Context True Count:** {ghida_false_in_context_true_count}\n"
    return markdown


def load_and_analyze(llm_decompile_record_list1: List[LLMDecompileRecord], llm_decompile_record_list2: List[LLMDecompileRecord]):
    pass


def main(dataset_name: str):
    file_path1 = f"/data1/xiachunwei/Projects/validation/Qwen3-32B/{dataset_name}_Qwen3-32B-n8-assembly-without-comments-ghidra-decompile/llm_decompile_results.pkl"
    file_path2 = f"/data1/xiachunwei/Projects/validation/Qwen3-32B/{dataset_name}_Qwen3-32B-n8-assembly-without-comments-in-context-learning/llm_decompile_results.pkl"
    list1 = pickle.load(open(file_path1, "rb"))
    list2 = pickle.load(open(file_path2, "rb"))
    pass_list1 = get_pass_list(list1)
    pass_list2 = get_pass_list(list2)

    
    # Generate and save markdown table
    markdown = generate_markdown_table(pass_list1, pass_list2, label1="Ghidra Decompile", label2="In-Context Learning")
    with open(f'pass_lists_{dataset_name}.md', 'w') as f:
        f.write(markdown)
    print(f"\nMarkdown table saved to pass_lists_{dataset_name}.md")
    
    markdown = get_num_execution_success(list1)
    with open(f'num_execution_success_{dataset_name}.md', 'w') as f:
        f.write(markdown)
    print(f"\nMarkdown table saved to num_execution_success_{dataset_name}.md")
    
    markdown = get_num_execution_success(list2)
    with open(f'num_execution_success_{dataset_name}.md', 'w') as f:
        f.write(markdown)
    print(f"\nMarkdown table saved to num_execution_success_{dataset_name}.md")

if __name__ == "__main__":
    main(dataset_name="sample_loops")
    main(dataset_name="sample_only_one_bb")
    main(dataset_name="sample_without_loops")