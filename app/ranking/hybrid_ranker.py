import math
from typing import List, Dict, Any
from ..models.job import Job
from ..models.candidate import CandidateProfile
from .skill_gap import SkillGapAnalyzer

class HybridRanker:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "semantic": 0.40,
            "skills": 0.35,
            "experience": 0.15,
            "metadata": 0.10
        }

    def rank(
        self, 
        candidate: CandidateProfile, 
        jobs: List[Job], 
        semantic_scores: List[float]
    ) -> List[Dict[str, Any]]:
        
        results = []
        for job, sem_score in zip(jobs, semantic_scores):
            # 1. Semantic Score (Normalized from cosine sim, roughly 0 to 1, max 1)
            norm_sem = max(0.0, min(1.0, sem_score))
            
            # 2. Skill Compatibility
            gap_analysis = SkillGapAnalyzer.analyze(candidate.skills, job.skills)
            skill_score = gap_analysis["skill_coverage"]
            
            # 3. Experience Compatibility
            exp_score = self._calculate_exp_score(candidate.years_experience, job.description)
            
            # 4. Metadata Compatibility
            meta_score = 1.0 # Base
            # Example metadata logic: if remote preferred and job is remote (we default to True for candidate currently if we don't know, or we can check location)
            # For simplicity, if candidate has no specific metadata, assume 0.8
            
            # Final Score Calculation
            final_score = (
                self.weights["semantic"] * norm_sem +
                self.weights["skills"] * skill_score +
                self.weights["experience"] * exp_score +
                self.weights["metadata"] * meta_score
            )
            
            results.append({
                "job": job,
                "score": final_score,
                "components": {
                    "semantic": norm_sem,
                    "skills": skill_score,
                    "experience": exp_score,
                    "metadata": meta_score
                },
                "gap_analysis": gap_analysis
            })
            
        # Sort descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _calculate_exp_score(self, cand_exp: float, job_desc: str) -> float:
        # Heuristic to check if job needs more exp than candidate has
        if "senior" in job_desc.lower() and cand_exp < 5.0:
            return 0.2
        if "lead" in job_desc.lower() and cand_exp < 7.0:
            return 0.1
        if "entry" in job_desc.lower() and cand_exp > 3.0:
            return 0.5 # Overqualified penalty
        return 0.8 # Default good match
