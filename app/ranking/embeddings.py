import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embeds a list of texts into a numpy array."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query string."""
        return self.embed_texts([text])[0]
