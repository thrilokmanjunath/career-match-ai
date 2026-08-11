
from app.ranking.hybrid_ranker import HybridRanker
from app.models.candidate import CandidateProfile
from app.models.job import Job
def test_ranker():
    ranker = HybridRanker()
    c = CandidateProfile(skills=["python"])
    j = Job(job_id="1", title="a", description="a", skills=["python"])
    res = ranker.rank(c, [j], [0.9])
    assert len(res) == 1
