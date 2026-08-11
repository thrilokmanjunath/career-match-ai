import sys
import os
# Add parent directory to path to allow absolute imports if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from app.ingestion.resume_parser import ResumeParser
from app.ingestion.extractor import ResumeExtractor
from app.data.providers import CSVProvider
from app.ranking.embeddings import EmbeddingService
from app.ranking.hybrid_ranker import HybridRanker
from app.services.recommendation_service import RecommendationService

st.set_page_config(page_title="CareerMatch AI", layout="wide")

@st.cache_resource
def load_system():
    data_provider = CSVProvider("data/raw/JobsFE.csv")
    embed_service = EmbeddingService()
    ranker = HybridRanker()
    service = RecommendationService(data_provider, embed_service, ranker)
    extractor = ResumeExtractor()
    return service, extractor

with st.spinner("Loading AI models and Job Database..."):
    recommendation_service, extractor = load_system()

st.title("CareerMatch AI")
st.markdown("### Intelligent Job Matching from Resumes")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    
    with st.spinner("Extracting resume content..."):
        try:
            raw_text = ResumeParser.parse(file_bytes, filename)
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            raw_text = ""
            
    if raw_text:
        with st.spinner("Analyzing candidate profile..."):
            candidate = extractor.extract(raw_text)
            
        st.subheader("Candidate Profile")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Extracted Skills:** {', '.join(candidate.skills) if candidate.skills else 'None detected'}")
            st.write(f"**Years of Experience:** {candidate.years_experience}")
        with col2:
            st.write("**Education Detected:**")
            if candidate.education:
                for edu in candidate.education:
                    st.write(f"- {edu}")
            else:
                st.write("None detected")
                
        st.markdown("---")
        
        with st.spinner("Ranking jobs..."):
            recommendations = recommendation_service.recommend(candidate, top_k=20)
            
        st.subheader("Recommended Positions")
        if not recommendations:
            st.warning("No matching jobs found.")
        else:
            for i, rec in enumerate(recommendations, 1):
                job = rec["job"]
                score = rec["score"] * 100
                with st.expander(f"{i}. {job.title} at {job.company} (Match: {score:.1f}%)"):
                    st.write(f"**Location:** {job.location} | **Type:** {job.employment_type} | **Remote:** {'Yes' if job.remote else 'No'}")
                    st.markdown("#### Explanation")
                    st.text(rec["explanation"])
                    
                    gap = rec["gap_analysis"]
                    st.markdown("#### Skill Analysis")
                    st.write(f"**Matched:** {', '.join(gap['matched_skills']) if gap['matched_skills'] else 'None'}")
                    st.write(f"**Missing:** {', '.join(gap['missing_skills']) if gap['missing_skills'] else 'None'}")
                    
                    st.markdown("#### Description")
                    st.write(job.description)
