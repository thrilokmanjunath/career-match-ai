
from app.ranking.skill_gap import SkillGapAnalyzer
def test_skill_gap():
    res = SkillGapAnalyzer.analyze(["python", "java"], ["python", "c++"])
    assert "python" in res["matched_skills"]
    assert "c++" in res["missing_skills"]
