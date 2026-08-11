import faiss
import numpy as np
import os

class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(self.dim)

    def build(self, embeddings: np.ndarray):
        faiss.normalize_L2(embeddings) # Normalize for inner product to be cosine similarity
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 100):
        # FAISS expects 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, filepath)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
            self.dim = self.index.d
            return True
        return False
