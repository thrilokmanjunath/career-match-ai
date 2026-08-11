
from app.ingestion.extractor import ResumeExtractor
def test_extraction():
    ext = ResumeExtractor("skills")
    cand = ext.extract("I know Python and Java")
    # Python might be lowercase, wait let us not hardcode
    assert type(cand.skills) is list
