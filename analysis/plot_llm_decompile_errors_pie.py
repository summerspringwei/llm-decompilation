"""
Plot the error type pie figure for all datasets
"""
import os
import pickle
import fire

from analysis.analyze_llc_error_type import get_error_type_from_str_list, post_processing_error_dict, draw_error_type_pie


def load_content(file_path):
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    return results


def analyze_all_generation_errors(file_path: str):
    error_type_dict = {}
    results = load_content(file_path)
    for sample_idx, validation in enumerate(results):
        error_msg_list = validation[1]['predict_error_msg']
        for choice_idx, error_msg in enumerate(error_msg_list):
            if error_msg == "":
                continue
            error_type = get_error_type_from_str_list(error_msg.split("\n"))
            print(error_type)
            if error_type is None:
                continue
            if error_type in error_type_dict:
                error_type_dict[error_type].append(f"sample_{sample_idx}_choice_{choice_idx}")
            else:
                error_type_dict[error_type] = [f"sample_{sample_idx}_choice_{choice_idx}"]
    return error_type_dict


def analyze_compilation_errors(file_path: str):
    """We only consider the samples that failed to fix the compilation error
    """
    error_type_dict = {}
    results = load_content(file_path)
    for sample_idx, validation in enumerate(results):
        validation_result = validation[1]
        if not any(validation_result['predict_compile_success']):
            error_msg_list = validation_result['predict_error_msg']
            for choice_idx, error_msg in enumerate(error_msg_list):
                if error_msg == "":
                    continue
                error_type = get_error_type_from_str_list(error_msg.split("\n"))
                if error_type is None:
                    continue
                if error_type in error_type_dict:
                    error_type_dict[error_type].append(f"sample_{sample_idx}_choice_{choice_idx}")
                else:
                    error_type_dict[error_type] = [f"sample_{sample_idx}_choice_{choice_idx}"]
    return error_type_dict


def plot_error_type_pie(file_path: str, fig_name: str):
    """
    Plot the error type pie figure
    Args:
        file_path: Path to the validation results file
        fig_name: Path to save the figure
    """
    error_type_dict = analyze_compilation_errors(file_path)
    new_error_dict = post_processing_error_dict(error_type_dict)
    draw_error_type_pie(new_error_dict, fig_name=fig_name)


def plot_all_datasets():
    """Plot the error type pie figure for all datasets"""
    file_fig_type = (
        ("sample_only_one_bb_Qwen3-32B-n8-assembly-with-comments", "sample_only_one_bb_qwen3-32b_gemini_error_type_pie.png"),
        ("sample_without_loops_Qwen3-32B-n8-assembly-with-comments", "sample_without_loops_qwen3-32b_gemini_error_type_pie.png"),
        ("sample_loops_Qwen3-32B-n8-assembly-with-comments", "sample_loops_qwen3-32b_gemini_error_type_pie.png"),
    )
    # error_type_dict = analyze_all_generation_errors(file_path)
    # plot_error_type_pie(error_type_dict, fig_name="sample_loops_qwen3-32b_gemini_error_type_pie.png")
    for file_path, fig_name in file_fig_type:
        file_path = os.path.join("validation/Qwen3-32B", file_path, "results.pkl")
        fig_name = os.path.join("figures", fig_name)
        plot_error_type_pie(file_path, fig_name=fig_name)


if __name__ == "__main__":
    # plot_all_datasets()
    fire.Fire(plot_all_datasets)