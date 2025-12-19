import numpy as np
import os
from vllm import LLM
from utils.preprocessing_assembly import preprocessing_assembly

class Qwen3Embedding():
    def __init__(self, model_path, cuda_device_idx, pre_process_asm_code=True):
        self.model = self.load_embedding_model(model_path, cuda_device_idx)
        self.pre_process_asm_code = pre_process_asm_code


    def load_embedding_model(self, model_path="/data1/xiachunwei/Datasets/Models/Qwen3-Embedding-8B", device_idx=0) -> LLM:
        """Load the Qwen/Qwen3-Embedding-8B model using vLLM for generating embeddings."""
        # Setup CUDA environment
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)

        model = LLM(
            model=model_path,
            task="embed",
            gpu_memory_utilization=0.95,
            max_model_len=32000,
            enforce_eager=True
        )
        return model


    def get_exebench_embedding_batch(self, texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
        """Get embeddings for a list of texts using vLLM."""
        if self.pre_process_asm_code:
            texts = [preprocessing_assembly(text) for text in texts]
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.embed(
                batch_texts)
            embeddings.extend(batch_embeddings)
        return embeddings

