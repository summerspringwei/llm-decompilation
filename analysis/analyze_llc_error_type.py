"""
First we need to run `bash get_all_error_predict.sh` to save all the error prediction results to a text file.

"""
import os
import logging
import fire
import matplotlib.pyplot as plt
from typing import Union


error_type_list = [
    "input module cannot be verified",
    "expected top-level entity",
    "use of undefined value",
    "label expected to be numbered",
    "instruction forward referenced with type 'label'",
    "alignment is not a power of two",
    "instruction expected to be numbered",
    "label expected to be numbered",
    "defined with type",
    "invalid getelementptr indices",
    "value doesn't match function result type",
    "instruction forward referenced with type",
    "redefinition of function",
    "expected value token",
    "is not a basic block",
    "multiple definition of local value named",
    "unable to create block numbered",
    "base element of getelementptr must be sized",
    "use of undefined comdat",
    "invalid shufflevector operands",
    "invalid use of function-local name",
    "expected instruction opcode"
]


def get_error_predict_list(file_path: str) -> list[str]:
    """Get all the file paths of error prediction results
    
    Parameters:
        file_path (str): the file path of error prediction results created by get_all_error_predict.sh
    
    Returns:
        list[str]: a list of file paths of error prediction results
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    error_predict_list = []
    for line in lines:
        error_predict_list.append(line.strip())
    return error_predict_list


def get_error_type_from_str_list(lines: list[str]) -> Union[str, None]:
    for line in lines:
            if "llc: error:" not in line:
                continue
            elif line.count("error:") == 1:
                if line.find("input module cannot be verified")>-1:
                    return "input module cannot be verified"
                else:
                    logging.error(f"error line: {line}")
            elif line.count("error:") == 2:
                error_type = line.split("error:")[-1].strip()
                return error_type
            else:
                logging.error(f"error line: {line}")
    return None


def get_error_type_from_file(file_path: str) -> Union[str, None]:
    """Get the error type from the error prediction result file

    Parameters:
        file_path (str): the file path of error prediction result `error_predict.error`
    
    Returns:
        str: the error type
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        error_type = get_error_type_from_str_list(lines)
    return error_type


def get_error_type(error_predict_list: list[str]) -> dict[str, list[str]]:
    """Classify each error prediction result into error types declared in `error_type_list`

    Parameters:
        error_predict_list (list): a list of error prediction results
    
    Returns:
        error_type_dict: a dictionary of error types and corresponding list of file paths
    """
    error_type_dict = {}
    for error_predict in error_predict_list:
        error_type = get_error_type_from_file(error_predict)
        if error_type is None:
            continue
        if error_type in error_type_dict:
            error_type_dict[error_type].append(error_predict)
        else:
            error_type_dict[error_type] = [error_predict,]
    return error_type_dict


def post_processing_error_dict(error_dict_count: dict[str, list[str]]) -> dict[str, list[str]]:
    """Post processing the error type dictionary by merging similar error types

    Parameters:
        error_dict_count (dict): a dictionary of error types and corresponding list of file paths
    
    Returns:
        dict: a dictionary of error types and corresponding list of file paths
    """
    new_error_dict = {}
    for k, v in error_dict_count.items():
        for error_type in error_type_list:
            if error_type in k:
                if error_type in new_error_dict:
                    new_error_dict[error_type].extend(v)
                else:
                    new_error_dict[error_type] = v
                break
    return new_error_dict


def draw_error_type_pie(error_dict_count: dict[str, list[str]], fig_name: str = "error_type_pie.png")->str:
    """Draw the pie chart of error types

    Parameters:
        error_dict_count (dict): a dictionary of error types and corresponding list of file paths
        fig_name (str): the name of the pie chart figure

    Returns:
        fig_path: the path to the pie chart figure
    """
    error_count_list = [(k, len(v)) for k,v in error_dict_count.items()]
    sum = 0
    for _, v in error_count_list:
        sum += v
    error_count_list.sort(key=lambda x: x[1], reverse=False)
    # Merge the error types with less than 5% of the total error count to `other` category
    other, idx = 0, 0
    for i, (_, v) in enumerate(error_count_list):
        if v / sum < 0.05:
            other += v
        else:
            idx = i
            break
    error_count_list = error_count_list[idx:]
    error_count_list.append(("other", other))

    labels = [k for k, _ in error_count_list]
    sizes = [v for _, v in error_count_list]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Distribution of Error Types")
    
    plt.savefig(fig_name)
    plt.clf()
    return fig_name


def analyze_error_type(file_path: str)->dict[str, list[str]]:
    """Read the error prediction results and analyze the error types
    
    Parameters:
        file_path (str): the file path of error prediction results created by get_all_error_predict.sh
    
    Returns:
        dict: a dictionary of error types and corresponding list of file paths
    """
    error_predict_list = get_error_predict_list(file_path)
    error_type_dict = get_error_type(error_predict_list)
    new_error_dict = post_processing_error_dict(error_type_dict)
    draw_error_type_pie(new_error_dict)
    return new_error_dict


def get_error_from_list_files(file_path_list: list[str])->dict[str, list[str]]:
    """Get the error types from a list of error prediction result files

    Parameters:
        file_path_list (list): a list of file paths of error prediction results
    
    Returns:
        dict: a dictionary of error types and corresponding list of file paths
    """
    all_error_dict = {}
    for file_path in file_path_list:
        error_dict = analyze_error_type(file_path)
        for error_type, error_file_list in error_dict.items():
            if error_type in all_error_dict:
                all_error_dict[error_type].extend(error_file_list)
            else:
                all_error_dict[error_type] = error_file_list
    draw_error_type_pie(all_error_dict)
    return all_error_dict


if __name__ == "__main__":
    fire.Fire(analyze_error_type)
