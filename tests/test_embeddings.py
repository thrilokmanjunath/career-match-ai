
from app.ranking.embeddings import EmbeddingService
def test_embedding_service():
    es = EmbeddingService()
    emb = es.embed_query("test")
    assert emb.shape == (es.dim,)
