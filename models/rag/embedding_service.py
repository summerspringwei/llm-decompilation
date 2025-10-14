import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn

# Import your actual model loader
from models.rag.exebench_qdrant_base import load_embedding_model

parser = argparse.ArgumentParser(description="Run embedding service")
parser.add_argument("--embedding_model_path", type=str, default="/data1/xiachunwei/Datasets/Models/Qwen3-Embedding-8B", help="Path to the embedding model")
parser.add_argument("--device_idx", type=int, default=3, help="Device index")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
parser.add_argument("--port", type=int, default=8001, help="Port")
args = parser.parse_args()

embedding_model_path = args.embedding_model_path
device_idx = args.device_idx
embedding_model = load_embedding_model(embedding_model_path, device_idx=device_idx)

app = FastAPI()

# Request and Response Models
class EmbedRequest(BaseModel):
    texts: list[str]  # list of input texts

@app.post("/embed")
def get_embeddings(req: EmbedRequest):
    with torch.no_grad():
        # Expect embedding_model.emb() to accept a list of strings
        embeddings = embedding_model.embed(req.texts)

    # Convert each embedding to list form (for JSON serialization)
    # embeddings_as_list = [e.tolist() for e in embeddings]
    return {"embeddings": embeddings}

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)