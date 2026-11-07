# 🧠 Smart Career AI — NLP, Resume ATS & Autonomous Agent Engine

> **Final Year Project (FYP) Core AI Microservice**  
> A high-performance Python & FastAPI microservice powered by **Custom Trained spaCy NER, Multi-Department Gazetteer, Scikit-Learn TF-IDF Vector Space, Groq LLM, and Google Gemini Flash** for intelligent resume parsing, multi-discipline ATS scoring, and department-tailored interview preparation.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-Custom_NER-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF_Vectors-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Groq](https://img.shields.io/badge/Groq-LPU_Inference-F55036?logo=groq&logoColor=white)](https://groq.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-Flash_API-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

---

## 📌 Overview

This repository contains the backend AI and Natural Language Processing (NLP) microservice that powers the **Smart Career Recruitment System**. It provides real-time resume file ingestion, custom entity extraction (`SKILL`, `DEGREE`, `JOB_TITLE`, `ORG`), deterministic 50-30-20 ATS score calculation, department-specific mock interview evaluation, and autonomous copilot workflows across **4 academic disciplines**:

1. **💻 Computer Science & Information Technology** (`React`, `Node.js`, `Python`, `Docker`, `AWS`, `MongoDB`, `PostgreSQL`)
2. **🩺 Healthcare, Pharmacy & Medical** (`Pharm-D`, `MBBS`, `Clinical Diagnostics`, `Pharmacology`, `Patient Care`, `CPR`, `EMR`)
3. **🏗️ Engineering (Mechanical, Civil, Electrical)** (`AutoCAD`, `SolidWorks`, `MATLAB`, `Finite Element Analysis (FEA)`, `ETABS`, `Six Sigma`)
4. **📊 Business, Finance & Accounting** (`Financial Modeling`, `Advanced Excel`, `Power BI`, `QuickBooks`, `GAAP/IFRS`, `Auditing`)

---

## 🌟 Core AI & NLP Capabilities

### 1. 📄 Multi-Format Resume Parser & 6-Section Diagnostic Audit
* Direct text extraction from **PDF** (`PyMuPDF / fitz`), **Word DOCX** (`python-docx`), and raw text.
* Segments and audits **6 fundamental ATS sections**: Contact Links, Core Technical Skills, Work & Internship Tenure, Academic Education & Degrees, Practical Projects, and ATS Readability.

### 2. 🏷️ Custom Trained spaCy NER + Multi-Department Gazetteer
* Fine-tuned on **1,410 real annotated resume records** and evaluated on **187 holdout test records** (Precision: **85.1%**, F1: **58.6%**).
* Extended with a single-source-of-truth `EntityRuler` gazetteer covering **113+ canonical skills**, **36 accredited degrees**, **32 job titles**, and **30 organizations**.

### 3. 🎯 Deterministic 50-30-20 Weighted ATS Scoring Engine
* **50% Skills Overlap:** Canonical alias normalization (`React.js` $\rightarrow$ `React`, `ms excel` $\rightarrow$ `Advanced Excel`) paired with **Scikit-Learn TF-IDF `(1, 2)` n-gram Cosine Similarity** as a fractional tie-breaker.
* **30% Experience & Tenure:** Evaluates verified years of experience (`3+ years`), active date ranges (`2022 - Present`), NER `JOB_TITLE`s, and quantifiable impact metrics.
* **20% Academic Education:** Verifies accredited degrees (`Pharm-D`, `MBBS`, `BDS`, `DPT`, `BS`, `B.E`, `BBA`, `MBA`, `MS`, `PhD`) via Custom NER and regex patterns.
  $$\text{Match Score} = (0.50 \times \text{Skills}) + (0.30 \times \text{Experience}) + (0.20 \times \text{Education})$$

### 4. ⚡ 3-Tier Cognitive AI Feedback & Department-Wise Interview Prep
* **Zero-Hang Fallback Chain:** Groq (`llama-3.3-70b-versatile`) $\rightarrow$ Google Gemini 1.5 Flash $\rightarrow$ Instant 0ms Deterministic Offline Expert Engine.
* **Domain-Tailored Mock Interviews:** Generates 10 role-specific questions and evaluates answers using domain-specific technical glossaries (`medical`, `engineering`, `business`, `it`) + TF-IDF semantic similarity.

---

## 📂 Repository Structure

```text
AI_features_python/
├── custom_nlp/
│   ├── gazetteer.py            # Single source of truth: 113+ skills, degrees, titles, orgs
│   └── resume_ner.py           # Custom spaCy NER wrapper + plausibility filters
├── models/smart_career_ner/    # Trained custom spaCy NER model & holdout training_metrics.json
├── data/                       # Processed train/test JSONL datasets (1,410 train / 187 test)
├── main.py                     # FastAPI microservice controller (15 registered routes)
├── resume_nlp_pipeline.py      # 7-step Hybrid NLP & 50-30-20 ATS Scoring Pipeline
├── train_resume_model.py       # Training & holdout evaluation script for custom spaCy NER
├── prepare_training_data.py    # Corpus standardization & multi-department data preparation
├── test_copilot_logic.py       # Verification suite forCandidate/Recruiter AI Copilot
├── n8n_ai_agent_workflow.json  # Exported n8n autonomous agent workflow definition
├── docker-compose.yml          # Optional container setup for n8n workflow engine
├── requirements.txt            # Python dependencies
└── .env.example                # Environment template for Groq & Gemini API keys
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check, 7 architecture layers, and active custom NER status. |
| `POST` | `/api/evaluate-resume-file` | Multipart upload (`.pdf`, `.docx`, `.txt`) for full ATS & NLP audit. |
| `POST` | `/api/evaluate-resume-text` | JSON payload with raw resume text + target job description. |
| `POST` | `/api/evaluate-candidate-profile` | Evaluates structured candidate profile & completeness percentage. |
| `GET` | `/api/nlp-model` | Returns custom spaCy NER metadata and holdout F1/Precision/Recall metrics. |
| `GET` | `/api/benchmark-departments` | Live cross-department benchmark across CS, Mechanical, Finance, and Medical. |
| `POST` | `/api/interview-prep/questions` | Generates 10 department-specific (`it`, `medical`, `engineering`, `business`) questions. |
| `POST` | `/api/interview-prep/evaluate` | Grades candidate interview answers using domain vocabulary + TF-IDF similarity. |
| `POST` | `/api/jobs/generate-description` | Employer AI job description, deliverables, and market skills generator. |
| `POST` | `/api/copilot/chat` | Dual-persona (Candidate / Recruiter RBAC) AI Copilot chat endpoint. |
| `POST` | `/webhook/career-agent` | Webhook endpoint for real-time AI Copilot and external agent triggers. |
| `GET` | `/docs` | Interactive OpenAPI Swagger UI documentation. |

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Msafwn/AI_features.git
cd AI_features

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure Environment Variables (Optional for LLM Layer)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
```env
# Groq API Configuration (Fast AI Agent Brain)
GROQ_API_KEY=your_groq_api_key_here
DEFAULT_MODEL=llama-3.3-70b-versatile

# Google Gemini API Configuration (Deep Resume Audit & Fallback)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```
*(Note: If API keys are omitted, the deterministic offline expert engine runs automatically with 0ms latency.)*

### 4. Run the FastAPI Server
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
API Documentation will be live at: **`http://localhost:8000/docs`**

---

## 📄 Academic Attribution & License

Engineered as part of the **Smart Career Role Match AI System** Final Year Project (FYP).  
Released under the **MIT License**.
