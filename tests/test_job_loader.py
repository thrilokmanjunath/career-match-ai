
from app.data.providers import DemoProvider
def test_demo_provider():
    dp = DemoProvider()
    jobs = dp.load_jobs()
    assert len(jobs) == 1
    assert jobs[0].title == "Data Scientist"
