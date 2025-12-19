import argparse
import torch
import uvicorn
from typing import Dict, List
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel

from contextlib import asynccontextmanager

from models.rag.embedding_model import Qwen3Embedding

parser = argparse.ArgumentParser(description="Run embedding service")
parser.add_argument(
    "--embedding_model_path",
    type=str,
    default="/data1/xiachunwei/Datasets/Models/Qwen3-Embedding-8B",
    help="Path to the embedding model")
parser.add_argument("--device_idx", type=int, default=3, help="Device index")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
parser.add_argument("--port", type=int, default=8001, help="Port")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--pre_process_asm_code",
    type=lambda x: (str(x).lower() == 'true'),
    default=True,
    help="Pre process assembly code (True/False)")
args = parser.parse_args()

embedding_model_path = args.embedding_model_path
device_idx = args.device_idx
batch_size = args.batch_size
embedding_model = None
pre_process_asm_code = args.pre_process_asm_code

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model
    embedding_model = Qwen3Embedding(
        embedding_model_path,
        device_idx,
        pre_process_asm_code
    )
    yield


# Request models
class EmbedRequest(BaseModel):
    asm_code: str

class BatchEmbedRequest(BaseModel):
    asm_code_list: List[str]

# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-Embedding-8B Embedding Service",
    description=
    "REST API for extracting embeddings from binary files using Qwen3-Embedding-8B",
    version="1.0.0",
    lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "service": "HermesSim Embedding Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/embed": "POST - Get embedding for a single exebench record",
            "/embed/batch":
            "POST - Get embeddings for a list of exebench records",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    if embedding_model is None:
        return {"status": "error", "message": "Embedding model not loaded"}
    return {"status": "ok"}


@app.post("/embed", response_model=Dict[str, List[List[float]]])
def get_embeddings(asm_code: str = Body(...)):
    """
    asm_code: str = Body(...) tells FastAPI to take the raw JSON 
    body and assign it to this variable.
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Use the batch method for consistency
    embeddings = embedding_model.get_exebench_embedding_batch([asm_code], batch_size=1)
    
    # Ensure the return key matches the client's expectation
    # If your client expects 'embedding', use that key instead of 'embeddings'
    return {"embeddings": embeddings}


@app.post("/embed/batch", response_model=Dict[str, List[List[float]]])
def get_embeddings_batch(asm_code_list: List[str] = Body(...)):
    """
    asm_code_list: List[str] = Body(...) expects a JSON list of strings.
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    embeddings = embedding_model.get_exebench_embedding_batch(
        asm_code_list, batch_size)
    return {"embeddings": embeddings}

if __name__ == "__main__":
    uvicorn.run("models.rag.embedding_service:app", host=args.host, port=args.port, reload=False)
