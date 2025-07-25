import os
import shutil
from datasets import load_from_disk, Dataset

def sample_dataset(dataset_path: str, output_path: str, num_samples: int = 100):
    """
    Samples a specified number of records from a Hugging Face Dataset loaded from disk
    and saves the sampled dataset to a new directory.

    Args:
        dataset_path: Path to the directory containing the saved dataset.
        output_path: Path to the directory where the sampled dataset will be saved.
        num_samples: Number of records to sample. Defaults to 100.
    
    Raises:
        FileNotFoundError: If the dataset directory doesn't exist.
        ValueError: If the number of samples is greater than the dataset size.
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        raise ValueError(f"Could not load dataset from disk: {e}")

    dataset_size = len(dataset)
    if num_samples > dataset_size:
      raise ValueError(f"Cannot sample {num_samples} from a dataset of size {dataset_size}.")

    sampled_dataset = dataset.shuffle(seed=42).select(range(num_samples))  # Shuffle for random sampling

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    try:
      sampled_dataset.save_to_disk(output_path)
      print(f"Sampled dataset saved to: {output_path}")
    except Exception as e:
      # If saving fails, attempt to clean up the directory
      if os.path.exists(output_path):
        shutil.rmtree(output_path)
      raise ValueError(f"Could not save sampled dataset: {e}")


# Example usage:
num_samples_to_take = 100 
dataset_dir = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff"  # Replace with the actual path to your dataset
output_dir = f"{dataset_dir}_sample_{num_samples_to_take}" # Replace with desired output directory

try:
    sample_dataset(dataset_dir, output_dir, num_samples_to_take)
except (FileNotFoundError, ValueError) as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")