import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from datasets import load_from_disk
from models.rag.build_qdrant_database import search_similar_records

