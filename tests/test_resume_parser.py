
from app.ingestion.resume_parser import ResumeParser
def test_txt_parse():
    assert ResumeParser.parse(b"Hello World", "test.txt") == "Hello World"
