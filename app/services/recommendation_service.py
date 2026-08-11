import time
import os
import numpy as np
from typing import List, Dict, Any

from ..data.providers import CSVProvider, JobDataProvider
from ..models.candidate import CandidateProfile
from ..ranking.embeddings import EmbeddingService
from ..ranking.vector_search import VectorIndex
from ..ranking.hybrid_ranker import HybridRanker
from ..ranking.explainer import Explainer

class RecommendationService:
    def __init__(self, data_provider: JobDataProvider, embed_service: EmbeddingService, ranker: HybridRanker, index_path: str = "models/faiss_index.bin", job_emb_path: str = "models/job_embeddings.npy"):
        self.data_provider = data_provider
        self.embed_service = embed_service
        self.ranker = ranker
        self.index_path = index_path
        self.job_emb_path = job_emb_path
        
        self.jobs = self.data_provider.load_jobs()
        self.vector_index = VectorIndex(self.embed_service.dim)
        
        self._initialize_index()

    def _initialize_index(self):
        # Create models dir if not exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        if self.vector_index.load(self.index_path) and os.path.exists(self.job_emb_path):
            self.job_embeddings = np.load(self.job_emb_path)
        else:
            texts = [job.searchable_text for job in self.jobs]
            self.job_embeddings = self.embed_service.embed_texts(texts)
            np.save(self.job_emb_path, self.job_embeddings)
            
            self.vector_index.build(self.job_embeddings.copy())
            self.vector_index.save(self.index_path)

    def recommend(self, candidate: CandidateProfile, top_k: int = 20) -> List[Dict[str, Any]]:
        # 1. Embed Candidate
        cand_text = f"{candidate.name} {' '.join(candidate.skills)} {' '.join(candidate.experience)}".lower()
        if not cand_text.strip():
            # Fallback if empty
            cand_text = "general candidate"
            
        cand_emb = self.embed_service.embed_query(cand_text)
        
        # 2. Vector Search (Filter top 100 semantically similar jobs to save time)
        search_k = min(100, len(self.jobs))
        if search_k == 0:
            return []
            
        distances, indices = self.vector_index.search(cand_emb, top_k=search_k)
        
        filtered_jobs = [self.jobs[idx] for idx in indices]
        
        # 3. Hybrid Ranking
        ranked_results = self.ranker.rank(candidate, filtered_jobs, distances)
        
        # 4. Generate Explanations for Top K
        final_results = []
        for res in ranked_results[:top_k]:
            res["explanation"] = Explainer.explain(res)
            final_results.append(res)
            
        return final_results
