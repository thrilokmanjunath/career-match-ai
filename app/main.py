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

st.set_page_config(
    page_title="CareerMatch AI — Intelligent Job Matching",
    page_icon="💼",
    layout="wide"
)

# ---------------------------------------------------------
# Preset Candidate Resumes for 1-Click Instant Testing
# ---------------------------------------------------------
PRESET_RESUMES = {
    "Select a preset resume...": "",
    "Python Backend Engineer": """Jane Doe
Email: jane.doe@example.com | Phone: (555) 234-5678 | San Francisco, CA

SUMMARY
Senior Python Backend Developer with 5 years of experience architecting high-throughput REST APIs, microservices, and distributed data pipelines.

TECHNICAL SKILLS
- Languages: Python, SQL, Bash
- Frameworks & Tools: FastAPI, Django, Flask, Docker, Kubernetes, Celery, Redis
- Databases: PostgreSQL, MySQL, MongoDB
- Cloud & CI/CD: AWS (S3, ECS, RDS), Git, GitHub Actions

EXPERIENCE
Senior Backend Engineer | DataCorp (2021 – Present)
- Designed and maintained scalable RESTful microservices processing 12M events daily using FastAPI and PostgreSQL.
- Reduced database query p95 latency by 42% through connection pooling and index optimization.
- Containerized legacy services with Docker and deployed multi-region clusters on AWS ECS.

EDUCATION
B.S. in Computer Science | UC Berkeley (2019)
""",
    "Data Scientist & ML Engineer": """Alex Smith
Email: alex.smith@example.com | New York, NY

SUMMARY
Data Scientist with 4 years of experience building predictive models, NLP classification pipelines, and recommendation systems in production environments.

TECHNICAL SKILLS
- Machine Learning: Scikit-learn, PyTorch, TensorFlow, XGBoost, HuggingFace
- Data & Analytics: Python, Pandas, NumPy, SQL, Spark, Tableau
- MLOps: MLflow, Docker, FastAPI, AWS SageMaker

EXPERIENCE
Machine Learning Engineer | FinTech Analytics (2022 – Present)
- Trained and deployed gradient boosted churn models achieving 91% F1-score across 500k customers.
- Implemented semantic search prototype with sentence transformers and Faiss vector search.
- Built automated model monitoring and drift detection alerts using MLflow.

EDUCATION
M.S. in Data Science | Columbia University (2022)
""",
    "Frontend React / Next.js Developer": """Morgan Lee
Email: morgan.lee@example.com | Seattle, WA

SUMMARY
Frontend Engineer with 3 years of experience crafting responsive, accessible, high-performance web applications using modern React, TypeScript, and Next.js.

TECHNICAL SKILLS
- Frontend: JavaScript (ES6+), TypeScript, React, Next.js, Redux, HTML5, CSS3, Tailwind CSS
- Testing & Tooling: Jest, React Testing Library, Vite, Webpack, Git

EXPERIENCE
Frontend Developer | WebSolutions (2023 – Present)
- Developed interactive web dashboards with Next.js App Router and Tailwind CSS serving 80k monthly active users.
- Improved Core Web Vitals (LCP reduced from 3.2s to 1.1s) through code splitting and image optimization.
- Implemented end-to-end component libraries and accessibility standards (WCAG AA).

EDUCATION
B.S. in Software Engineering | University of Washington (2023)
"""
}


@st.cache_resource(show_spinner=False)
def load_system():
    data_provider = CSVProvider("data/raw/JobsFE.csv")
    embed_service = EmbeddingService()
    ranker = HybridRanker()
    service = RecommendationService(data_provider, embed_service, ranker)
    extractor = ResumeExtractor()
    return service, extractor


# Header & Banner
st.title("💼 CareerMatch AI")
st.markdown("### Intelligent Job Matching & Skills Gap Analysis")

# Prominent Disclaimer Banner
st.warning("⚠️ **Demonstration tool** — Algorithmic recommendations are generated using hybrid lexical-semantic matching and should not be used as the sole basis for hiring decisions.")

with st.spinner("Initializing models and indexing job catalog..."):
    recommendation_service, extractor = load_system()

# Sidebar: Controls & Dataset Info
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Top Recommendations (K)", min_value=5, max_value=30, value=15, step=5)
    
    st.markdown("---")
    st.header("📊 Dataset Provenance")
    st.markdown("""
    - **Dataset**: `JobsFE.csv`
    - **Indexed Corpus**: 10,000 real-world job postings
    - **Features**: Roles, Requisite Skills, Locations, Workplace Modes, Salary
    - **Models**: `paraphrase-MiniLM-L6-v2` + Hybrid Ranker
    """)
    st.caption("🔒 Privacy Guarantee: Resumes are processed strictly in memory and are never persisted or transmitted to third-party endpoints.")


# Resume Input Tabs: Upload vs Paste vs Presets
st.markdown("#### Input Candidate Resume")
input_tab1, input_tab2, input_tab3 = st.tabs(["📄 Upload Resume File", "✍️ Paste Resume Text", "⚡ Select Demo Preset"])

resume_text = ""

with input_tab1:
    uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT resume", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        try:
            resume_text = ResumeParser.parse(file_bytes, filename)
            st.success(f"Parsed {len(resume_text):,} characters from `{filename}`")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

with input_tab2:
    pasted_text = st.text_area(
        "Paste plain-text resume here:",
        height=220,
        placeholder="Paste candidate work experience, technical skills, and education..."
    )
    if pasted_text.strip():
        resume_text = pasted_text.strip()

with input_tab3:
    selected_preset = st.selectbox("Choose a realistic candidate persona:", list(PRESET_RESUMES.keys()))
    if selected_preset != "Select a preset resume...":
        resume_text = PRESET_RESUMES[selected_preset]
        st.text_area("Preset Resume Preview:", value=resume_text, height=220, disabled=True)

# Main Processing Flow
if resume_text:
    with st.spinner("Extracting candidate profile and analyzing skills..."):
        candidate = extractor.extract(resume_text)
        
    st.markdown("---")
    st.subheader("👤 Extracted Candidate Profile")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**Candidate Name / Identifier:** {candidate.name}")
        st.write(f"**Contact / Email:** {candidate.email or 'Not provided'}")
        if candidate.skills:
            st.write("**Extracted Skills:**")
            badges = " ".join([f"`{s}`" for s in candidate.skills])
            st.markdown(badges)
        else:
            st.info("No standardized technical skills detected from taxonomy.")
            
    with col2:
        st.metric(label="Years of Experience", value=f"{candidate.years_experience} yrs")
        
    with col3:
        st.metric(label="Skill Count", value=len(candidate.skills))

    st.markdown("---")
    
    with st.spinner(f"Retrieving and hybrid-ranking top {top_k} matching jobs..."):
        recommendations = recommendation_service.recommend(candidate, top_k=top_k)
        
    st.subheader(f"🎯 Top {len(recommendations)} Job Matches")
    
    if not recommendations:
        st.warning("No matching jobs found in catalog for this profile.")
    else:
        for i, rec in enumerate(recommendations, 1):
            job = rec["job"]
            score = rec["score"] * 100
            gap = rec["gap_analysis"]
            
            # Color-coded match badge
            score_color = "🟢" if score >= 70 else ("🟡" if score >= 50 else "🟠")
            
            with st.expander(f"{score_color} #{i} — {job.title} @ {job.company} (Match Score: {score:.1f}%)"):
                col_info1, col_info2, col_info3 = st.columns(3)
                col_info1.write(f"📍 **Location:** {job.location}")
                col_info2.write(f"💼 **Mode:** {job.employment_type} ({'Remote' if job.remote else 'On-site'})")
                if job.salary:
                    col_info3.write(f"💰 **Salary:** ${job.salary:,.0f}/yr")
                else:
                    col_info3.write("💰 **Salary:** Not disclosed")
                    
                st.markdown("##### 🔍 Algorithmic Explanation")
                st.info(rec["explanation"])
                
                # Skills Breakdown
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown("**✅ Matched Requisite Skills:**")
                    if gap["matched_skills"]:
                        st.markdown(" ".join([f"`{s}`" for s in gap["matched_skills"]]))
                    else:
                        st.caption("No direct requisite skill overlaps detected.")
                        
                with col_m2:
                    st.markdown("**⚠️ Missing Skills Gap:**")
                    if gap["missing_skills"]:
                        st.markdown(" ".join([f"`{s}`" for s in gap["missing_skills"]]))
                    else:
                        st.caption("Candidate meets all requisite skill tags!")
                        
                # Actionable Resume Improvement Suggestion
                if gap["missing_skills"]:
                    st.markdown("##### 💡 Resume Improvement Recommendation")
                    top_missing = ", ".join(gap["missing_skills"][:3])
                    st.write(f"To increase your match probability for this position, consider highlighting projects or certifications demonstrating experience with **{top_missing}**.")
                    
                with st.expander("📄 Full Job Description"):
                    st.write(job.description)
else:
    st.info("👆 Please upload a resume file, paste resume text, or select a demo preset above to begin.")
