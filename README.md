# 🧠 Smart Career AI — NLP, Resume ATS & Autonomous Agent Engine

> **Final Year Project (FYP) Core AI Microservice**  
> A high-performance Python & FastAPI microservice powered by **Groq LLM, Google Gemini, Hugging Face Transformers, spaCy, and scikit-learn** for intelligent resume parsing, multi-discipline ATS scoring, and autonomous agent workflows.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org/)
[![Groq](https://img.shields.io/badge/Groq-LPU_Inference-F55036?logo=groq&logoColor=white)](https://groq.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-Flash_API-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Sentence_Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-n8n_Agent-2496ED?logo=docker&logoColor=white)](https://n8n.io/)

---

## 📌 Overview

This repository contains the backend AI and Natural Language Processing (NLP) microservices that power the **Smart Career Recruitment System**. It provides real-time resume file ingestion, deep semantic candidate evaluation, ATS score calculation, and autonomous workflow coordination across **4 academic disciplines**:

1. **💻 Computer Science & Information Technology** (MERN, Python, Cloud, DevOps, AI)
2. **🩺 Medical & Healthcare** (Clinical Pharmacology, Patient Care, Diagnostics, HL7)
3. **🏗️ Engineering & Construction** (Civil, AutoCAD, Structural Analysis, ETABS, BIM)
4. **📊 Business & Finance** (Financial Modeling, DCF, Risk Analysis, IFRS Compliance)

---

## 🌟 Core AI & NLP Capabilities

### 1. 📄 Multi-Format Resume Parser
* Direct text and structural extraction from **PDF** (`PyMuPDF / fitz`) and **Word DOCX** (`python-docx`).
* Normalizes bullet points, section headings, education credentials, and chronological work experience.

### 2. 🎯 Hybrid Multi-Stage ATS Scoring Engine
Combines fast statistical token matching with deep dense semantic vector embeddings:
* **TF-IDF + Cosine Similarity:** Computes exact keyword match frequencies against required vs nice-to-have skills.
* **Dense Semantic Embeddings:** Evaluates conceptual similarity using Hugging Face `all-MiniLM-L6-v2` dense embeddings.
* **Official 50-30-20 Weighted Formula:**
  $$\text{Match Score} = (0.50 \times \text{Skills}) + (0.30 \times \text{Experience}) + (0.20 \times \text{Education})$$

### 3. ⚡ Generative LLM Audit (Groq + Gemini)
* Dual-LLM fallback architecture (*Groq ultra-fast LPU inference + Google Gemini Flash deep review*).
* Generates quantified candidate strengths, red-flag risks, and targeted technical interview questions.

### 4. 🤖 Autonomous n8n Workflow Integration
* Self-hosted `docker-compose.yml` deploying an **n8n agent container** (`port 5678`).
* Coordinates background tasks, dynamic candidate notification queues, and Google Sheets synchronization via webhooks.

---

## 📂 Repository Structure

```text
AI_features_python/
├── docker-compose.yml          # Container configuration for autonomous n8n agent
├── train_resume_model.py       # Train custom spaCy NER (SKILL, DEGREE, JOB_TITLE, ORG)
├── custom_nlp/                 # Gazetteer, EntityRuler, inference wrapper
├── models/smart_career_ner/    # Saved trained model (created after training)
├── main.py                     # FastAPI server & ATS endpoint controllers
├── resume_nlp_pipeline.py      # Core NLP engine: spaCy, TF-IDF, PyMuPDF, sentence-transformers
├── requirements.txt            # Python dependencies (FastAPI, Groq, PyMuPDF, scikit-learn)
├── n8n_ai_agent_workflow.json  # Exported n8n visual AI agent workflow definition
├── GOOGLE_SHEETS_SCHEMA.md     # Data contract documentation for live sheet sync
├── .env.example                # Template for Groq & Gemini API keys
└── shared_files/               # Mount point for Docker-shared candidate files
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check, engine uptime, and active AI model names. |
| `POST` | `/api/evaluate-resume-file` | Multipart upload (PDF/DOCX) + job criteria for deep ATS scoring. |
| `POST` | `/api/evaluate-resume-text` | JSON payload with resume raw text + job description for instant matching. |
| `GET` | `/api/nlp-model` | Custom spaCy NER status, labels, and last training F1/precision/recall. |
| `GET` | `/api/benchmark-departments` | Sample evaluation payloads across IT, Healthcare, Engineering, and Finance. |
| `GET` | `/docs` | Interactive OpenAPI Swagger UI documentation. |

---

## ⚡ Quick Start & Installation

### Prerequisites
* **Python:** 3.10 or higher
* **Docker Desktop:** (Optional, required only for n8n agent container)
* **API Keys:** Free API keys from [Groq Console](https://console.groq.com) and [Google AI Studio](https://aistudio.google.com).

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Msafwn/AI_features.git
cd AI_features

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Train the custom resume NER (EntityRuler + statistical NER)
python train_resume_model.py
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your API keys in `.env`:
```env
# Groq API Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here
DEFAULT_MODEL=qwen/qwen3.8-27b

# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest
```

### 4. Run the FastAPI Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
API Documentation will be live at: **`http://localhost:8000/docs`**

### 5. (Optional) Run Autonomous Agent (n8n)
```bash
docker compose up -d
```
Access the n8n Workflow canvas at: **`http://localhost:5678`**

---

## 📄 Academic Attribution & License

Engineered as part of the **Smart Career Role Match AI System** Final Year Project (FYP).  
Released under the **MIT License**.
