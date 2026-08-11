# CareerMatch AI

**Intelligent job matching from resumes**

## Overview
CareerMatch AI is an advanced, AI-driven job recommendation platform. Unlike simple keyword matchers, it uses a hybrid recommendation engine that combines semantic vector search with deterministic skill-gap analysis to provide highly relevant, explainable job matches.

## Problem
Traditional Applicant Tracking Systems (ATS) and job boards rely heavily on exact keyword matching, causing qualified candidates to be overlooked due to resume formatting or synonym usage. They also fail to explain *why* a job is a good fit, leaving candidates to guess what skills they might be missing.

## Solution
CareerMatch AI solves this by employing Natural Language Processing (NLP) and Sentence Transformers to semantically understand both the candidate's resume and the job descriptions. It extracts a structured candidate profile and ranks jobs using a multi-signal approach, providing a transparent breakdown of matched and missing skills.

## Architecture
- **Ingestion**: Supports PDF, DOCX, and TXT. Uses `PyMuPDF`, `python-docx`, and `spaCy` for NER and heuristic extraction.
- **Data Pipeline**: Modular `JobDataProvider` abstraction, parsing semi-structured job CSVs into strongly-typed Pydantic models.
- **ML Pipeline**: Uses `SentenceTransformer (paraphrase-MiniLM-L6-v2)` for semantic embeddings and `FAISS` for fast vector retrieval.
- **Recommendation Engine**: A configurable `HybridRanker` combining Semantic Similarity, Skill Compatibility, Experience, and Metadata.
- **UI**: A responsive, modern dashboard built with `Streamlit`.

## What Makes It Different
- **Skill-Gap Analysis**: For every recommended job, you see exactly which required skills you have and which you are missing.
- **Explainable Recommendations**: Transparent scoring. No "black box" AI.
- **Evaluation**: Built-in metrics (Precision, Recall, NDCG, MRR) for heuristic evaluation.
- **Career Intelligence**: A dashboard that aggregates your strongest skills and common missing skills across top jobs.

## Installation & Local Development

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd career-match-ai
   ```

2. **Setup Python Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python3 -m spacy download en_core_web_sm
   ```

3. **Run the Application**
   ```bash
   streamlit run app/main.py
   ```

## Docker Deployment
```bash
docker build -t career-match-ai .
docker run -p 8501:8501 career-match-ai
```
Alternatively, using Docker Compose:
```bash
docker compose up
```

## Evaluation Methodology
Due to the absence of a large-scale ground-truth labeled dataset mapping specific resumes to this job set, we have constructed a **heuristic/weak-label evaluation** module. It measures Precision@K, Recall@K, MRR, and NDCG using synthesized relevancy rules based on strict skill and experience thresholds.

## Limitations
- **Decision Support Only**: Recommendations are estimates and should not be used for automated rejection.
- **Parsing Errors**: Resume parsing is inherently noisy; the structured extraction might miss unconventionally formatted information.
- **Dataset Bias**: The underlying job dataset may reflect historical biases in wording or requirements.

## Future Roadmap
- Integration with live job board APIs (LinkedIn, Indeed).
- Finetuning the embedding model on domain-specific HR data.
- User accounts and saved job tracking.

## License
MIT License
