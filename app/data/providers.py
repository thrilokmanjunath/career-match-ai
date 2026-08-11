import abc
import csv
import pandas as pd
from typing import List, Optional
from ..models.job import Job

class JobDataProvider(abc.ABC):
    @abc.abstractmethod
    def load_jobs(self) -> List[Job]:
        pass


class CSVProvider(JobDataProvider):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_jobs(self) -> List[Job]:
        jobs = []
        df = pd.read_csv(self.file_path)
        # Drop completely empty rows or rows without an id
        df = df.dropna(subset=['Job Id', 'position'])
        
        for _, row in df.iterrows():
            job_id = str(row.get('Job Id', ''))
            title = str(row.get('position', 'Unknown'))
            location = str(row.get('workplace', 'Unknown'))
            working_mode = str(row.get('working_mode', 'full-time'))
            salary = row.get('salary', None)
            duties = str(row.get('job_role_and_duties', ''))
            req_skills = str(row.get('requisite_skill', ''))
            
            # Extract skills heuristic
            skills_list = [s.strip().lower() for s in req_skills.split(',') if s.strip()]
            if not skills_list and req_skills:
                skills_list = [s.strip().lower() for s in req_skills.split(' ') if len(s.strip()) > 1]
                
            remote = 'remote' in working_mode.lower()
            
            # Naive salary parsing (e.g. 69500.0)
            salary_val = None
            try:
                if pd.notna(salary):
                    salary_val = float(salary)
            except ValueError:
                pass

            searchable_text = f"{title} {location} {working_mode} {duties} {req_skills}".lower()

            job = Job(
                job_id=job_id,
                title=title,
                location=location,
                description=duties,
                skills=skills_list,
                employment_type=working_mode,
                salary_min=salary_val,
                salary_max=salary_val,
                remote=remote,
                searchable_text=searchable_text
            )
            jobs.append(job)
        return jobs

class DemoProvider(JobDataProvider):
    def load_jobs(self) -> List[Job]:
        return [
            Job(
                job_id="demo1",
                title="Data Scientist",
                company="TechCorp",
                location="Remote",
                description="Looking for a data scientist to build machine learning models.",
                skills=["python", "machine learning", "sql", "pandas"],
                remote=True,
                searchable_text="data scientist remote machine learning python sql pandas"
            )
        ]
