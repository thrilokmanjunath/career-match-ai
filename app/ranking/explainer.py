from typing import Dict, Any

class Explainer:
    @staticmethod
    def explain(result: Dict[str, Any]) -> str:
        score = result["score"] * 100
        comp = result["components"]
        gap = result["gap_analysis"]
        
        reasons = []
        
        if comp["semantic"] > 0.8:
            reasons.append("+ Strong semantic alignment with job description")
        elif comp["semantic"] < 0.3:
            reasons.append("- Low semantic alignment with job role")
            
        if comp["skills"] > 0.7:
            reasons.append("+ Excellent skill match")
        elif comp["skills"] < 0.3:
            reasons.append("- Significant skill gaps")
            
        if comp["experience"] > 0.7:
            reasons.append("+ Experience level appears appropriate")
        else:
            reasons.append("- Experience mismatch (potentially over/under-qualified)")
            
        explanation = f"Match Score: {score:.1f}%\n\nWhy this job matches:\n"
        explanation += "\n".join(reasons)
        
        return explanation
