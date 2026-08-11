import time
import math
import numpy as np
from typing import List, Dict, Any

class Evaluator:
    @staticmethod
    def precision_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if not recommended_ids or not relevant_ids:
            return 0.0
        top_k = recommended_ids[:k]
        hits = sum(1 for job_id in top_k if job_id in relevant_ids)
        return hits / k

    @staticmethod
    def recall_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        top_k = recommended_ids[:k]
        hits = sum(1 for job_id in top_k if job_id in relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(recommended_ids: List[str], relevant_ids: List[str]) -> float:
        for i, job_id in enumerate(recommended_ids):
            if job_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        dcg = 0.0
        idcg = 0.0
        top_k = recommended_ids[:k]
        
        for i, job_id in enumerate(top_k):
            if job_id in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)
                
        for i in range(min(k, len(relevant_ids))):
            idcg += 1.0 / math.log2(i + 2)
            
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def benchmark_latency(func, *args, **kwargs) -> float:
        """Returns execution time in seconds"""
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        return end - start
