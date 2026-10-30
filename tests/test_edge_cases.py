import pytest
import numpy as np
from app.ingestion.resume_parser import ResumeParser
from app.ingestion.extractor import ResumeExtractor
from app.ranking.embeddings import EmbeddingService
from app.ranking.hybrid_ranker import HybridRanker
from app.models.candidate import CandidateProfile
from app.models.job import Job

def test_empty_resume():
    """Verify empty text parses and extracts gracefully without crashing."""
    parsed = ResumeParser.parse(b"", "empty.txt")
    assert parsed == ""
    
    extractor = ResumeExtractor("skills")
    candidate = extractor.extract("")
    assert candidate.skills == []
    assert candidate.years_experience == 0


def test_special_characters_and_unicode():
    """Verify parser and extractor handle special characters, emojis, and non-ASCII."""
    special_text = "🚀 Senior Python Developer 💻 with C++ & C# skills! Contact: test@domain.org §¶•"
    parsed = ResumeParser.parse(special_text.encode("utf-8"), "special.txt")
    assert "Python" in parsed
    
    extractor = ResumeExtractor("skills")
    candidate = extractor.extract(parsed)
    assert isinstance(candidate.skills, list)
    assert candidate.email == "test@domain.org"


def test_very_long_resume():
    """Verify extractor handles very long documents (stress test)."""
    long_text = "Software Engineer with Python and SQL experience. " * 2000 # ~100k chars
    extractor = ResumeExtractor("skills")
    candidate = extractor.extract(long_text)
    assert "python" in candidate.skills or "sql" in candidate.skills


def test_hybrid_scoring_bounds():
    """Verify hybrid scoring remains within [0, 1] across extreme conditions."""
    ranker = HybridRanker()
    c = CandidateProfile(skills=["python", "docker"], years_experience=5)
    
    # Perfect match job
    j_perfect = Job(job_id="1", title="Python Senior", description="Requires 5 years Python experience", skills=["python", "docker"])
    res_perfect = ranker.rank(c, [j_perfect], [1.0])
    assert 0.0 <= res_perfect[0]["score"] <= 1.0
    
    # Zero match job
    j_zero = Job(job_id="2", title="Cobol", description="Ancient mainframe", skills=["cobol"])
    res_zero = ranker.rank(c, [j_zero], [0.0])
    assert 0.0 <= res_zero[0]["score"] <= 1.0
    assert res_perfect[0]["score"] > res_zero[0]["score"]


def test_embedding_service_dimension():
    """Verify embedding service produces expected normalized dimensions."""
    es = EmbeddingService()
    emb = es.embed_query("Python Machine Learning")
    assert emb.shape == (es.dim,)
    assert not np.isnan(emb).any()
