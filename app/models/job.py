from pydantic import BaseModel, Field
from typing import List, Optional


class Job(BaseModel):
    job_id: str
    title: str
    company: str = "Unknown"
    location: str = "Unknown"
    description: str
    skills: List[str] = Field(default_factory=list)
    experience_level: str = "Unknown"
    employment_type: str = "full-time"
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    remote: bool = False
    
    # Text combining all relevant fields for semantic search
    searchable_text: str = ""
