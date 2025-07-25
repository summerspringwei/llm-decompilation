import os
import pickle
import fire
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# from models.gemini.gemini_decompilation import evaluate_response


# def print_validation_results(results, dataset):
#     """We print the validation results for each sample and return the number of success samples.
    
#     Args:
#         results: List of validation results
#         dataset: List of dataset
#     Returns:
#         predict_compile_success_count: Number of predict compile success
#         predict_execution_success_count: Number of predict execution success
#         target_compile_success_count: Number of target compile success
#     """
#     for r in results:
#         print(r["idx"], r["predict_compile_success"], r["predict_execution_success"], r["target_compile_success"], r["target_execution_success"], dataset[r["idx"]]["llvm_ir"]["bb_count"])
#     predict_compile_success_count = sum(1 for result in results if result["predict_compile_success"])
#     predict_execution_success_count = sum(1 for result in results if result["predict_execution_success"])
#     target_compile_success_count = sum(1 for result in results if result["target_compile_success"])
#     target_execution_success_count = sum(1 for result in results if result["target_execution_success"])
#     print(f"Number of predict_compile_success: {predict_compile_success_count}")
#     print(f"Number of predict_execution_success: {predict_execution_success_count}")
#     print(f"Number of target_compile_success: {target_compile_success_count}")
#     print(f"Number of target_execution_success: {target_execution_success_count}")
#     return predict_compile_success_count, predict_execution_success_count, target_compile_success_count, target_execution_success_count


# def validate_first_trial(dataset, output_dir):
#     args = [
#         (pickle.load(open(os.path.join(output_dir, f"response_{idx}.pkl"), "rb")),
#             record, idx, output_dir)
#         for idx, record in enumerate(dataset)
#     ]
#     with Pool(processes=40) as pool:
#         results = pool.starmap(evaluate_response, args)
#     return print_validation_results(results, dataset)


# def validate_all_trials(dataset, output_dir, max_retry: int = 10):
#     process_chart = []
#     process_chart.append(validate_first_trial(dataset, output_dir, max_retry))
#     for retry in range(1, max_retry+1):
#         print("-"*100)
#         print(f"Validating retry {retry}")
#         args = []
#         for idx, record in enumerate(dataset):
#             pkl_file_path = os.path.join(output_dir, f"response_{idx}_retry_{retry}.pkl")
#             if os.path.exists(pkl_file_path):
#                 args.append((pickle.load(open(pkl_file_path, "rb")), record, idx, output_dir))
#         with Pool(processes=40) as pool:
#             results = pool.starmap(evaluate_response, args)
#         process_chart.append(print_validation_results(results, dataset))
#     return process_chart


def plot_validation_progress(process_chart):
    """Plot the validation progress over retries showing accumulated successes.
    
    Args:
        process_chart: List of tuples containing (predict_compile_success, predict_execution_success, 
                      target_compile_success, target_execution_success) for each retry
    """

    retries = range(len(process_chart))
    
    # Calculate cumulative maximums for compile and execution success
    predict_compile = np.maximum.accumulate([x[0] for x in process_chart])
    predict_execute = np.maximum.accumulate([x[1] for x in process_chart])
    target_compile = np.maximum.accumulate([x[2] for x in process_chart]) 
    target_execute = np.maximum.accumulate([x[3] for x in process_chart])
    print(predict_compile, predict_execute, target_compile, target_execute)
    plt.figure(figsize=(10, 6))
    plt.plot(retries, predict_compile, 'b-', label='Predict Compile Success', marker='o')
    plt.plot(retries, predict_execute, 'r-', label='Predict Execute Success', marker='s')
    plt.plot(retries, target_compile, 'g--', label='Target Compile Success', marker='^')
    plt.plot(retries, target_execute, 'y--', label='Target Execute Success', marker='v')

    plt.xlabel('Retry Number')
    plt.ylabel('Accumulated Success Count')
    plt.title('Validation Progress Over Retries')
    plt.legend()
    plt.grid(True)
    plt.xticks(retries)

    plt.tight_layout()
    plt.savefig('validation_progress.png')
    plt.close()


def plot_accumulated_success_figure(accumulated_compile_success_list, accumulated_execution_success_list, save_path):
    """
    Plot the accumulated success count over retries
    Args:
        compile_success_list: List of compile success count for each retry
        execution_success_list: List of execution success count for each retry
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(accumulated_compile_success_list, 'b-', label='Accumulated Compile Success', marker='o')
    plt.plot(accumulated_execution_success_list, 'r-', label='Accumulated Execution Success', marker='s')
    plt.xlabel('Retry Number')
    plt.ylabel('Accumulated Success Count')
    plt.title('Accumulated Success Count Over Retries')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def count_accumulated_success_count(output_dir = "validation/Qwen3-32B/sample_only_one_bb_Qwen3-32B-n8-assembly-with-comments",
                                   num_retry = 10) -> tuple[list, list]:
    """Count the accumulated success count over retries
    
    Args:
        output_dir: Path to the output directory
        num_retry: Number of retries
    Returns:
        accumulated_compile_success_list: List of accumulated compile success count
        accumulated_execution_success_list: List of accumulated execution success count
    """
    fix_evaluation_list = pickle.load(open(os.path.join(output_dir, "fix_evaluation_list.pkl"), "rb"))
    compile_success_list, execution_success_list = [], []
    compile_success_count, execution_success_count = 0, 0
    # If the first sample is empty, it means the first retry is successful
    for sample_evaluation in fix_evaluation_list:
        if sample_evaluation == {}:
            compile_success_count += 1
            execution_success_count += 1
    compile_success_list.append(compile_success_count)
    execution_success_list.append(execution_success_count)
    # For each retry, count the number of samples that are successful
    for count in range(num_retry):
        this_compile_success_count, this_execution_success_count = 0, 0
        # For each sample, count the number of samples that are successful
        for sample_evaluation in fix_evaluation_list:
            # Make sure don't exceed the length of the sample_evaluation
            if count < len(sample_evaluation):
                # If it's the first retry, count the number of samples that are successful
                if count == 0:
                    if any(sample_evaluation[count]["predict_compile_success"]):
                        this_compile_success_count += 1
                    if any(sample_evaluation[count]["predict_execution_success"]):
                        this_execution_success_count += 1
                else:
                    # If the previous retry is not successful, and the current retry is successful, count the number of samples that are successful
                    if not any(sample_evaluation[count-1]["predict_compile_success"]) and any(sample_evaluation[count]["predict_compile_success"]):
                        this_compile_success_count += 1
                    if any(sample_evaluation[count]["predict_execution_success"]):
                        this_execution_success_count += 1
        compile_success_list.append(this_compile_success_count)
        execution_success_list.append(this_execution_success_count)
    print(compile_success_list)
    print(execution_success_list)
    accumulated_compile_success_list = np.cumsum(compile_success_list)
    accumulated_execution_success_list = np.cumsum(execution_success_list)
    return accumulated_compile_success_list, accumulated_execution_success_list


def plot_accumulated_success_count(output_dir = "validation/Qwen3-32B/sample_only_one_bb_Qwen3-32B-n8-assembly-without-comments",
                                   num_retry = 10,
                                   save_path = "figures/accumulated_success_count.png"):
    """
    Plot the accumulated success count over retries
    Args:
        output_dir: Path to the output directory
        num_retry: Number of retries
        save_path: Path to save the figure
    """
    # 1. Count the accumulated success count
    accumulated_compile_success_list, accumulated_execution_success_list = count_accumulated_success_count(output_dir, num_retry)
    
    # 2. Plot the accumulated success count over retries
    plot_accumulated_success_figure(accumulated_compile_success_list, accumulated_execution_success_list, save_path)

    print("Final accumulated success count: ", accumulated_compile_success_list[-1], accumulated_execution_success_list[-1] )


def plot_all_datasets():
    """Plot the accumulated success count for all datasets"""
    datasets = [
        ("validation/Qwen3-32B/sample_only_one_bb_Qwen3-32B-n8-assembly-with-comments", "sample_only_one_bb_Qwen3-32B-n8-assembly-with-comments"),
        ("validation/Qwen3-32B/sample_loops_Qwen3-32B-n8-assembly-with-comments", "sample_loops_Qwen3-32B-n8-assembly-with-comments"),
        ("validation/Qwen3-32B/sample_without_loops_Qwen3-32B-n8-assembly-with-comments", "sample_without_loops_Qwen3-32B-n8-assembly-with-comments"),
        ("validation/Qwen3-32B/sample_without_loops_Qwen3-32B-n8-assembly-without-comments", "sample_without_loops_Qwen3-32B-n8-assembly-without-comments"),
    ]
    for output_dir, name_hint in datasets:
        plot_accumulated_success_count(output_dir, save_path=f"figures/{name_hint}-accumulated_success_count.pdf")


if __name__ == "__main__":
    # plot_all_datasets()
    fire.Fire(plot_accumulated_success_count)
    