from pydantic import BaseModel, Field
from typing import List


class CandidateProfile(BaseModel):
    name: str = "Unknown"
    email: str = "Unknown"
    skills: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    years_experience: float = 0.0
