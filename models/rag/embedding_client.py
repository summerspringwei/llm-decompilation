import requests

class RemoteEmbeddingModel:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def embed(self, texts: list[str]):
        response = requests.post(
            f"{self.url}/embed",
            json={"texts": texts}
        )
        response.raise_for_status()
        return response.json()["embeddings"]