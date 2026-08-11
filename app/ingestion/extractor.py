import re
import json
import os
import spacy
from typing import List, Dict, Any
from ..models.candidate import CandidateProfile

class ResumeExtractor:
    def __init__(self, taxonomy_dir: str = "skills"):
        # Load spaCy model (gracefully handle if not installed properly)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.taxonomy = self._load_taxonomy(taxonomy_dir)

    def _load_taxonomy(self, taxonomy_dir: str) -> Dict[str, str]:
        """Loads all JSON files in taxonomy_dir and creates a mapping from variation to canonical skill."""
        taxonomy = {}
        if not os.path.exists(taxonomy_dir):
            return taxonomy
            
        for filename in os.listdir(taxonomy_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(taxonomy_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        for canonical, variations in data.items():
                            for var in variations:
                                taxonomy[var.lower()] = canonical.lower()
                    except json.JSONDecodeError:
                        pass
        return taxonomy

    def extract(self, text: str) -> CandidateProfile:
        text_lower = text.lower()
        
        # 1. Extract Name (Naive approach: first line or using NER)
        doc = self.nlp(text)
        name = "Unknown"
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text
                break
        
        # 2. Extract Email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
        email = email_match.group(0) if email_match else "Unknown"

        # 3. Extract Skills based on Taxonomy
        skills_found = set()
        for variation, canonical in self.taxonomy.items():
            # word boundary match to avoid partial matches like "r" in "word"
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, text_lower):
                skills_found.add(canonical)
                
        # 4. Extract Years of Experience (Heuristic)
        # Look for patterns like "5 years of experience", "10+ years"
        years_exp = 0.0
        exp_matches = re.findall(r'(\d+)(?:\+)?\s*(?:years|yrs?)\s*(?:of)?\s*exp', text_lower)
        if exp_matches:
            try:
                years_exp = float(max([int(m) for m in exp_matches]))
            except ValueError:
                pass
                
        # 5. Extract Education (Heuristic)
        education = []
        edu_keywords = ['bachelor', 'master', 'phd', 'b.s.', 'm.s.', 'bsc', 'msc', 'b.a.', 'university', 'college']
        for line in text.split('\n'):
            line_lower = line.lower()
            if any(kw in line_lower for kw in edu_keywords) and len(line) < 100:
                education.append(line.strip())

        return CandidateProfile(
            name=name,
            email=email,
            skills=list(skills_found),
            education=education[:5], # Keep top 5 to avoid noise
            experience=[],
            certifications=[],
            years_experience=years_exp
        )
