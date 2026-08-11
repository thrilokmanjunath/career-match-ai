from typing import List, Dict, Any

class SkillGapAnalyzer:
    @staticmethod
    def analyze(candidate_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        cand_set = set([s.lower() for s in candidate_skills])
        job_set = set([s.lower() for s in job_skills])
        
        if not job_set:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "skill_coverage": 1.0 # If job has no skills, assume full coverage
            }

        matched = list(cand_set.intersection(job_set))
        missing = list(job_set.difference(cand_set))
        coverage = len(matched) / len(job_set)
        
        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "skill_coverage": coverage
        }
