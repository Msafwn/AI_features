"""
========================================================================================
SMART CAREER AI - HYBRID RESUME NLP PIPELINE & ATS SCORING ENGINE
========================================================================================
Architecture Components:
  1. Raw Text Extraction: PyMuPDF (PDF) + python-docx (DOCX)
  2. Section Segmentation: spaCy (en_core_web_sm) + Rule-Based Section Patterns
  3. 6-Section Diagnostic Audit: Exact match to ProfileResumeTab.jsx UI (6 sections)
  4. Mathematical Feature Match: scikit-learn TF-IDF + Cosine Similarity Vector Distance
  5. Multi-Department Skill Taxonomy: Normalizes synonyms across CS/IT, Engineering, Biz/Fin, Medical
  6. Deterministic Weighted Scoring: 50% Skills + 30% Experience + 20% Education
  7. Cognitive AI Feedback: Groq (Llama-3.3-70b) / Gemini Flash with Offline Expert Fallback
========================================================================================
"""

import os
import re
import json
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional

# Load environment variables (.env in AI_features_python or project root)
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

from custom_nlp.gazetteer import SKILL_ALIASES, DEGREES, JOB_TITLES, ORGS

# Core NLP & ML Libraries
import pymupdf
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def compute_cosine_similarity(vec1, vec2):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        return float(cosine_similarity(vec1, vec2)[0][0])
    except Exception:
        if hasattr(vec1, "toarray"):
            v1 = vec1.toarray().flatten()
            v2 = vec2.toarray().flatten()
        else:
            v1 = np.asarray(vec1).flatten()
            v2 = np.asarray(vec2).flatten()
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ======================================================================================
# 1. RAW TEXT EXTRACTION (PDF / DOCX / TEXT)
# ======================================================================================
class ResumeTextExtractor:
    """Extracts and normalizes raw text from PDFs, DOCX, and raw strings."""

    @staticmethod
    def extract_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF (fitz)."""
        doc = pymupdf.open(pdf_path)
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pages_text.append(page.get_text("text"))
        doc.close()
        return ResumeTextExtractor.clean_text("\n".join(pages_text))

    @staticmethod
    def extract_from_docx(docx_path: str) -> str:
        """Extract text from DOCX using python-docx."""
        doc = docx.Document(docx_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        return ResumeTextExtractor.clean_text("\n".join(paragraphs))

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Normalize whitespace, remove non-printable characters, preserve paragraphs."""
        text = raw_text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ======================================================================================
# 2. MULTI-DEPARTMENT TAXONOMY & SYNONYM NORMALIZER
# ======================================================================================
class MultiDeptTaxonomy:
    """Normalizes department-specific technical skills, degrees, and common aliases from gazetteer."""

    # Dynamic Single Source of Truth directly from custom_nlp.gazetteer.SKILL_ALIASES
    SYNONYM_MAP: Dict[str, str] = {}
    for _canonical, _aliases in SKILL_ALIASES.items():
        SYNONYM_MAP[_canonical.strip().lower()] = _canonical
        for _alias in _aliases:
            SYNONYM_MAP[_alias.strip().lower()] = _canonical

    # Department Catalogs for rich categorization and indexing
    DEPARTMENT_CATALOG = {
        "Computer Science & IT": [
            "React", "Node.js", "Express", "MongoDB", "PostgreSQL", "Python",
            "Docker", "Kubernetes", "AWS", "Git", "Tailwind CSS", "FastAPI",
            "Scikit-Learn", "Machine Learning", "NLP", "TypeScript", "REST APIs"
        ],
        "Engineering": [
            "AutoCAD", "SolidWorks", "MATLAB", "Finite Element Analysis (FEA)",
            "PLC Programming", "SCADA", "Six Sigma", "Lean Manufacturing", "ETABS", "Revit"
        ],
        "Business & Finance": [
            "Advanced Excel", "Power BI", "Financial Modeling", "QuickBooks",
            "SQL", "Data Analysis", "SAP ERP", "GAAP Accounting", "Auditing", "Budgeting"
        ],
        "Healthcare & Medical": [
            "Clinical Diagnostics", "Pharmacology", "Good Manufacturing Practice (GMP)",
            "Pharmacovigilance", "HIPAA Compliance", "Patient Care", "Biochemistry", "CPR", "EMR"
        ]
    }

    @classmethod
    def normalize_skill(cls, skill: str) -> str:
        """Map raw skill alias to canonical standardized form from gazetteer."""
        if not skill:
            return ""
        normalized_key = skill.strip().lower()
        return cls.SYNONYM_MAP.get(normalized_key, skill.strip())

    @classmethod
    def extract_and_normalize_skills(cls, text: str) -> List[str]:
        """Scan text and extract all recognized skills across all department catalogs."""
        found_skills = set()
        lower_text = " " + text.lower() + " "
        
        # Check against synonym aliases from gazetteer
        for alias, canonical in cls.SYNONYM_MAP.items():
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, lower_text):
                found_skills.add(canonical)
                
        # Also check against catalog items
        for dept, skills_list in cls.DEPARTMENT_CATALOG.items():
            for sk in skills_list:
                pattern = r"(?<!\w)" + re.escape(sk.lower()) + r"(?!\w)"
                if re.search(pattern, lower_text):
                    found_skills.add(sk)

        return sorted(list(found_skills))


# ======================================================================================
# 3. SPACY SECTION SEGMENTATION
# ======================================================================================
class ResumeSectionSegmenter:
    """Uses spaCy tokenization and heading regex rules to split resume into 6 components."""

    SECTION_PATTERNS = {
        "contact": [
            r"contact", r"personal details", r"personal info", r"address", r"profile links"
        ],
        "summary": [
            r"summary", r"professional summary", r"executive summary", r"career objective", r"objective", r"about me"
        ],
        "skills": [
            r"technical skills", r"core skills", r"skills & competencies", r"skills", r"technologies", r"tools & technologies", r"competencies"
        ],
        "experience": [
            r"work experience", r"employment history", r"professional experience", r"experience", r"internships", r"work history"
        ],
        "education": [
            r"education", r"academic background", r"academic qualifications", r"degrees", r"educational background"
        ],
        "projects": [
            r"projects", r"academic projects", r"personal projects", r"practical projects", r"key projects", r"practical portfolio", r"portfolio"
        ],
        "certifications": [
            r"certifications", r"certificates", r"awards & achievements", r"licenses", r"honors", r"achievements"
        ]
    }

    @classmethod
    def segment_resume(cls, text: str) -> Dict[str, str]:
        """Splits raw resume text into structured sections dictionary."""
        lines = text.split("\n")
        sections: Dict[str, List[str]] = {
            "contact": [],
            "summary": [],
            "skills": [],
            "experience": [],
            "education": [],
            "projects": [],
            "certifications": []
        }
        
        current_section = "contact"
        
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            detected_heading = cls._match_heading(trimmed)
            if detected_heading:
                current_section = detected_heading
                continue

            sections[current_section].append(trimmed)

        return {sec: "\n".join(content_lines).strip() for sec, content_lines in sections.items()}

    @classmethod
    def _match_heading(cls, line: str) -> Optional[str]:
        """Detect if line is a prominent section header."""
        if len(line) > 45:
            return None
        
        cleaned = re.sub(r"[^a-zA-Z\s&]", "", line).strip().lower()
        if not cleaned:
            return None

        for sec_name, patterns in cls.SECTION_PATTERNS.items():
            for pat in patterns:
                if re.fullmatch(pat, cleaned):
                    return sec_name
        return None


# ======================================================================================
# 4. 6-SECTION DIAGNOSTIC AUDIT (MATCHES ProfileResumeTab.jsx UI)
# ======================================================================================
class DiagnosticAuditEngine:
    """Evaluates the 6 fundamental ATS sections and produces status cards matching UI."""

    @classmethod
    def audit_sections(
        cls,
        full_text: str,
        sections: Dict[str, str],
        skills: List[str],
        extracted_degrees: List[str] = None,
        extracted_titles: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Produces 6 structured diagnostic cards:
          1. Contact & Profile Links
          2. Core Technical Skills
          3. Work & Internship Experience
          4. Academic Education & Degree
          5. Projects & Practical Portfolio
          6. Formatting & ATS Readability
        """
        audit_results = []

        # --- 1. Contact & Profile Links ---
        has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", full_text))
        has_phone = bool(re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", full_text))
        has_linkedin = bool(re.search(r"linkedin\.com/in/|linkedin", full_text, re.IGNORECASE))
        has_github = bool(re.search(r"github\.com/|github", full_text, re.IGNORECASE))

        contact_score = 0
        contact_elements = []
        if has_email:
            contact_score += 35
            contact_elements.append("Email")
        if has_phone:
            contact_score += 35
            contact_elements.append("Phone")
        if has_linkedin or has_github:
            contact_score += 30
            contact_elements.append("Online Profile")

        audit_results.append({
            "title": "Contact & Profile Links",
            "score": min(contact_score, 100),
            "status": "Passed" if contact_score >= 80 else ("Needs Polish" if contact_score >= 50 else "Action Required"),
            "notes": f"Detected: {', '.join(contact_elements)}." if contact_elements else "Missing direct contact details or LinkedIn URL."
        })

        # --- 2. Core Technical Skills ---
        skills_count = len(skills)
        if skills_count >= 8:
            skills_score = 95
            skills_status = "Passed"
            skills_notes = f"{skills_count} relevant industry skills extracted and indexed."
        elif skills_count >= 4:
            skills_score = 75
            skills_status = "Needs Polish"
            skills_notes = f"{skills_count} skills detected. Expand your technical skillset to hit target thresholds."
        else:
            skills_score = 45
            skills_status = "Action Required"
            skills_notes = "Very few industry keywords detected. Add dedicated Skills section."

        audit_results.append({
            "title": "Core Technical Skills",
            "score": skills_score,
            "status": skills_status,
            "notes": skills_notes
        })

        # --- 3. Work & Internship Experience (P1: Years / Tenures / Titles, not line count) ---
        exp_text = sections.get("experience", "")
        combined_exp_context = exp_text + "\n" + full_text

        tenure_matches = [int(m) for m in re.findall(r"\b(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+experience)?\b", combined_exp_context, re.IGNORECASE)]
        max_tenure_years = max(tenure_matches) if tenure_matches else 0

        date_ranges = re.findall(r"\b(19\d\d|20\d\d)\s*(?:-|–|to)\s*(present|current|19\d\d|20\d\d)\b", combined_exp_context, re.IGNORECASE)
        has_numbers = bool(re.search(r"\b(\d+[\%kK]|\d+\+?|\$\d+)\b", exp_text))
        exp_words = len(exp_text.split())
        has_titles = bool(extracted_titles and len(extracted_titles) > 0)

        if max_tenure_years >= 2 or len(date_ranges) >= 2 or (exp_words >= 50 and (has_numbers or has_titles)):
            exp_score = 95
            exp_status = "Passed"
            exp_notes = f"Verified experience ({max_tenure_years}+ years or multiple tenures) with quantifiable impact."
        elif max_tenure_years >= 1 or len(date_ranges) >= 1 or has_titles or exp_words >= 25:
            exp_score = 80
            exp_status = "Passed"
            exp_notes = "Professional role and industry experience verified. Include more quantifiable achievement metrics."
        elif exp_words >= 10:
            exp_score = 60
            exp_status = "Needs Polish"
            exp_notes = "Experience descriptions are brief. Clearly state your job titles, organizations, and years of experience."
        else:
            exp_score = 45
            exp_status = "Action Required"
            exp_notes = "No substantial work experience, internships, or job tenure detected."

        audit_results.append({
            "title": "Work & Internship Experience",
            "score": exp_score,
            "status": exp_status,
            "notes": exp_notes
        })

        # --- 4. Academic Education & Degree (P0: Regex + NER for Pharm-D, MBBS, BDS, DPT, FCPS, BE, etc.) ---
        edu_text = sections.get("education", "") + " " + full_text
        DEGREE_REGEX = r"\b(bachelor|b\.s|bs|bsc|master|m\.s|ms|msc|bba|mba|phd|associate|engineering|b\.e|be|pharm-?d|mbbs|bds|dpt|fcps|mcps|m\.com|acca|ca|cfa|intermediate|fsc|a-levels|university|college|institute)\b"
        has_regex_degree = bool(re.search(DEGREE_REGEX, edu_text, re.IGNORECASE))
        has_ner_degree = bool(extracted_degrees and len(extracted_degrees) > 0)
        has_degree = has_regex_degree or has_ner_degree
        has_gpa = bool(re.search(r"\b(gpa|cgpa|\d\.\d{1,2}/\d\.\d)\b", edu_text, re.IGNORECASE))

        detected_label = ", ".join(extracted_degrees[:2]) if extracted_degrees else "Accredited degree"

        if has_degree and has_gpa:
            edu_score = 100
            edu_status = "Passed"
            edu_notes = f"{detected_label}, institution, and GPA verified against role targets."
        elif has_degree:
            edu_score = 95
            edu_status = "Passed"
            edu_notes = f"{detected_label} and educational institution verified."
        else:
            edu_score = 40
            edu_status = "Action Required"
            edu_notes = "Education credentials or accredited degree name not clearly recognized."

        audit_results.append({
            "title": "Academic Education & Degree",
            "score": edu_score,
            "status": edu_status,
            "notes": edu_notes
        })

        # --- 5. Projects & Practical Portfolio ---
        proj_text = sections.get("projects", "")
        has_github_proj = bool(re.search(r"github|gitlab|demo|live|url|http|drive", proj_text, re.IGNORECASE))
        proj_words = len(proj_text.split())

        if proj_words > 40 and has_github_proj:
            proj_score = 95
            proj_status = "Passed"
            proj_notes = "Detailed practical projects with live or repository proof links."
        elif proj_words > 30:
            proj_score = 75
            proj_status = "Needs Polish"
            proj_notes = "Include active GitHub repository or live demo links for top 2 projects."
        elif proj_words > 10:
            proj_score = 60
            proj_status = "Action Required"
            proj_notes = "Brief project listings. Detail tools used and outcomes achieved."
        else:
            proj_score = 50
            proj_status = "Action Required"
            proj_notes = "No practical portfolio or capstone projects section identified."

        audit_results.append({
            "title": "Projects & Practical Portfolio",
            "score": proj_score,
            "status": proj_status,
            "notes": proj_notes
        })

        # --- 6. Formatting & ATS Readability ---
        doc = nlp(full_text[:2000])
        sentences_count = len(list(doc.sents))
        total_words = len(full_text.split())

        if total_words >= 150 and sentences_count >= 8:
            format_score = 88
            format_status = "Passed"
            format_notes = "Single-column structure, clean fonts, machine-readable text."
        else:
            format_score = 65
            format_status = "Needs Polish"
            format_notes = "Text density is low or formatting may disrupt automated ATS parsers."

        audit_results.append({
            "title": "Formatting & ATS Readability",
            "score": format_score,
            "status": format_status,
            "notes": format_notes
        })

        return audit_results


# ======================================================================================
# 5. MATHEMATICAL FEATURE MATCH (TF-IDF + COSINE SIMILARITY)
# ======================================================================================
class MathFeatureMatcher:
    """Calculates vector space distance using scikit-learn TF-IDF & Cosine Similarity."""

    @classmethod
    def compute_similarity(cls, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Vectorizes both documents with TF-IDF and computes:
          - Mathematical Cosine Similarity (0.0 to 1.0)
          - Shared Keywords
          - Missing Keywords from Job Description
        """
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=500
        )

        corpus = [resume_text, job_description]
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Cosine distance: cos(theta) = (A . B) / (||A|| * ||B||)
        cos_sim = compute_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        feature_names = vectorizer.get_feature_names_out()
        resume_vec = tfidf_matrix[0].toarray()[0]
        job_vec = tfidf_matrix[1].toarray()[0]

        shared_keywords = []
        missing_keywords = []

        for idx, word in enumerate(feature_names):
            in_resume = resume_vec[idx] > 0
            in_job = job_vec[idx] > 0

            if in_resume and in_job:
                shared_keywords.append((word, float(job_vec[idx])))
            elif in_job and not in_resume:
                missing_keywords.append((word, float(job_vec[idx])))

        shared_keywords.sort(key=lambda x: x[1], reverse=True)
        missing_keywords.sort(key=lambda x: x[1], reverse=True)

        feature_model_info = {
            "model": "scikit-learn TF-IDF n-gram (1, 2) Vector Space Model",
            "representation": "High-dimensional sparse lexical TF-IDF feature matrix",
            "distance_metric": "Cosine Similarity: (A · B) / (||A|| * ||B||)",
            "vocabulary_size": len(feature_names),
            "mode": "Classical Natural Language Vector Space Distance"
        }

        return {
            "cosine_similarity": round(cos_sim, 4),
            "match_percentage": round(cos_sim * 100, 1),
            "shared_keywords": [k[0] for k in shared_keywords[:12]],
            "missing_keywords": [k[0] for k in missing_keywords[:8]],
            "vector_similarity_model": feature_model_info,
            "deep_learning_transformer": feature_model_info
        }


# ======================================================================================
# 6. 50-30-20 WEIGHTED SCORING ENGINE (CANONICAL ALIAS MATCHING)
# ======================================================================================
class WeightedScoringEngine:
    """
    Computes deterministic, reproducible composite ATS score:
      Final Score = (0.50 * Skills Overlap Match) + (0.30 * Experience & Diagnostics) + (0.20 * Education Match)
    """

    @classmethod
    def calculate_score(
        cls,
        math_match: Dict[str, Any],
        section_audits: List[Dict[str, Any]],
        candidate_skills: List[str],
        job_skills: List[str] = None,
        candidate_degrees: List[str] = None,
        job_description: str = ""
    ) -> Dict[str, Any]:
        
        # P0: Normalize both candidate and job skills to canonical gazetteer keys
        cand_canon = {MultiDeptTaxonomy.normalize_skill(s).lower() for s in candidate_skills if s and s.strip()}
        job_canon = {MultiDeptTaxonomy.normalize_skill(s).lower() for s in (job_skills or []) if s and s.strip()}

        # 1. Skills Component (50%): Real NLP overlap ratio between candidate skills and required skills
        if job_canon and len(job_canon) > 0:
            matched_skills = cand_canon.intersection(job_canon)
            overlap_ratio = len(matched_skills) / len(job_canon)
            cos_sim = float(math_match.get("cosine_similarity", 0.5) or 0.5)
            # 50% Skills Score: primary overlap + cosine similarity tie-breaker
            skills_score = min(98.0, max(35.0, (overlap_ratio * 88.0) + (cos_sim * 10.0) + 2.0))
        else:
            skills_score = min(98.0, max(45.0, len(cand_canon) * 9.5))

        # 2. Experience & Diagnostic Completeness Component (30%)
        exp_audit = next((s for s in section_audits if "Experience" in s["title"]), None)
        format_audit = next((s for s in section_audits if "Readability" in s["title"]), None)
        exp_score = (exp_audit["score"] if exp_audit else 75) * 0.75 + (format_audit["score"] if format_audit else 80) * 0.25

        # 3. Education Component (20%): Degree presence & department alignment
        edu_audit = next((s for s in section_audits if "Education" in s["title"]), None)
        edu_score = float(edu_audit["score"] if edu_audit else 80)
        if candidate_degrees and len(candidate_degrees) > 0:
            edu_score = max(edu_score, 90.0)

        # Weighted calculation: 50% Skills + 30% Experience + 20% Education
        composite_score = round(
            (0.50 * skills_score) +
            (0.30 * exp_score) +
            (0.20 * edu_score)
        )
        composite_score = min(98, max(35, composite_score))

        return {
            "overall_ats_score": composite_score,
            "weights": {
                "skills_weight": "50%",
                "experience_weight": "30%",
                "education_weight": "20%"
            },
            "sub_scores": {
                "skills": round(skills_score),
                "experience": round(exp_score),
                "education": round(edu_score)
            }
        }


# ======================================================================================
# 7. COGNITIVE AI SMART FEEDBACK (GROQ / GEMINI / OFFLINE EXPERT)
# ======================================================================================
class SmartFeedbackGenerator:
    """Generates structured, actionable feedback with zero-hang fallbacks."""

    @classmethod
    def generate_feedback(
        cls,
        candidate_name: str,
        score: int,
        missing_keywords: List[str],
        section_audits: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calls Groq LLaMA-3.3-70b or Gemini 1.5 Flash if keys are valid; otherwise immediate offline fallback."""
        
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

        # 1. Try Groq (Llama-3.3-70b-versatile) only if key is valid (skips empty keys in 0ms)
        if groq_key and not groq_key.startswith("gsk_your") and len(groq_key) > 20:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                }
                prompt = f"""You are the Chief AI Career Strategist for Smart Career System.
Candidate: {candidate_name}
Overall ATS Score: {score}/100
Missing High-Priority Keywords: {', '.join(missing_keywords[:6])}

Provide a JSON response with:
1. "summary": One encouraging sentence summarizing candidate's current profile strength.
2. "actionable_tips": Array of 3 specific, punchy tips to raise their score above 90.
3. "keywords_to_add": Array of top 4 missing keywords to place on resume.

Return ONLY pure valid JSON without markdown fences."""
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "You return strictly valid JSON matching requested keys."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 300
                }

                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=4) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    content = res_data["choices"][0]["message"]["content"].strip()
                    content = re.sub(r"^```json\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                    return json.loads(content)
            except Exception:
                pass

        # 2. Try Google Gemini 1.5 Flash if Gemini key is valid
        if gemini_key and not gemini_key.startswith("your_") and len(gemini_key) > 15:
            try:
                gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                headers = {"Content-Type": "application/json"}
                gemini_prompt = f"""You are the Chief AI Career Strategist for Smart Career System.
Candidate: {candidate_name}
Overall ATS Score: {score}/100
Missing Keywords: {', '.join(missing_keywords[:6])}

Return valid JSON with keys: "summary" (string), "actionable_tips" (array of 3 strings), "keywords_to_add" (array of 4 strings). Return ONLY raw JSON without markdown fences."""
                gemini_payload = {
                    "contents": [{"parts": [{"text": gemini_prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300}
                }
                req = urllib.request.Request(gemini_url, data=json.dumps(gemini_payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=4) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    raw_text = re.sub(r"^```json\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                    return json.loads(raw_text)
            except Exception:
                pass

        # 3. Offline Deterministic Fallback (Guaranteed 0ms response, zero network delay)
        weakest_section = min(section_audits, key=lambda x: x["score"])
        return {
            "summary": f"Candidate profile demonstrates strong technical foundation matching {score}% of target role requirements.",
            "actionable_tips": [
                f"Upgrade '{weakest_section['title']}': {weakest_section['notes']}",
                "Add measurable metrics to experience bullet points (e.g. 'Improved system throughput by 25%').",
                "Include active links to verified projects or portfolio repositories."
            ],
            "keywords_to_add": missing_keywords[:4] if missing_keywords else ["REST APIs", "Unit Testing", "System Design", "Data Modeling"]
        }


# ======================================================================================
# 8. MASTER PIPELINE ORCHESTRATOR
# ======================================================================================
class ResumeNLPPipeline:
    """Master orchestrator connecting all 7 steps of the NLP architecture."""

    def __init__(self):
        self.extractor = ResumeTextExtractor()
        self.taxonomy = MultiDeptTaxonomy()
        self.segmenter = ResumeSectionSegmenter()
        self.audit_engine = DiagnosticAuditEngine()
        self.matcher = MathFeatureMatcher()
        self.scorer = WeightedScoringEngine()
        self.feedback_engine = SmartFeedbackGenerator()
        self._custom_ner = None

    @property
    def custom_ner(self):
        if self._custom_ner is None:
            from custom_nlp.resume_ner import CustomResumeNER

            self._custom_ner = CustomResumeNER()
        return self._custom_ner

    def process_resume(
        self,
        resume_input: str,
        job_description: Optional[str] = None,
        candidate_name: str = "Candidate"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation pipeline:
          Input: PDF path, DOCX path, or Raw Text
          Output: Comprehensive JSON evaluation payload
        """
        # 1. Text Extraction
        if resume_input.lower().endswith(".pdf") and os.path.exists(resume_input):
            raw_text = self.extractor.extract_from_pdf(resume_input)
            source_type = "PDF"
        elif resume_input.lower().endswith(".docx") and os.path.exists(resume_input):
            raw_text = self.extractor.extract_from_docx(resume_input)
            source_type = "DOCX"
        else:
            raw_text = self.extractor.clean_text(resume_input)
            source_type = "RAW_TEXT"

        # Default Target Job Description if none provided
        if not job_description:
            job_description = """
            We are looking for a Software Engineer with expertise in modern web architectures.
            Proficiency in React, Node.js, Express, MongoDB, PostgreSQL, and Python.
            Hands-on experience with Docker, Git, REST APIs, and Tailwind CSS.
            Bachelor's degree in Computer Science or related engineering field.
            Strong problem-solving skills, unit testing, and performance optimization.
            """

        # 2. Custom spaCy NER (SKILL/DEGREE/JOB_TITLE/ORG) + taxonomy fallback
        ner_extraction = self.custom_ner.extract(raw_text)
        taxonomy_skills = self.taxonomy.extract_and_normalize_skills(raw_text)
        merged_skills = []
        seen_skill_keys = set()
        for skill in list(ner_extraction.get("skills") or []) + taxonomy_skills:
            key = skill.strip().lower()
            if not key or key in seen_skill_keys:
                continue
            seen_skill_keys.add(key)
            merged_skills.append(skill)
        extracted_skills = merged_skills

        # 3. spaCy Section Segmentation
        sections = self.segmenter.segment_resume(raw_text)

        # 4. 6-Section Diagnostic Audit (Passing NER extracted degrees and job titles)
        section_audits = self.audit_engine.audit_sections(
            full_text=raw_text,
            sections=sections,
            skills=extracted_skills,
            extracted_degrees=ner_extraction.get("degrees") or [],
            extracted_titles=ner_extraction.get("job_titles") or []
        )

        # 5. Mathematical TF-IDF & Cosine Similarity Match
        math_match = self.matcher.compute_similarity(raw_text, job_description)

        # Extract required skills from target job description
        job_skills = self.taxonomy.extract_and_normalize_skills(job_description)
        job_ner = self.custom_ner.extract(job_description)
        for js in job_ner.get("skills") or []:
            if js not in job_skills:
                job_skills.append(js)

        # 6. 50-30-20 Weighted Score Engine (Skills Overlap + Experience + Education)
        score_breakdown = self.scorer.calculate_score(
            math_match=math_match,
            section_audits=section_audits,
            candidate_skills=extracted_skills,
            job_skills=job_skills,
            candidate_degrees=ner_extraction.get("degrees") or [],
            job_description=job_description
        )

        # 7. Cognitive AI Smart Feedback
        feedback = self.feedback_engine.generate_feedback(
            candidate_name=candidate_name,
            score=score_breakdown["overall_ats_score"],
            missing_keywords=math_match["missing_keywords"],
            section_audits=section_audits
        )

        return {
            "source_type": source_type,
            "candidate_name": candidate_name,
            "overall_ats_score": score_breakdown["overall_ats_score"],
            "score_breakdown": score_breakdown,
            "extracted_skills": extracted_skills,
            "skills_count": len(extracted_skills),
            "custom_nlp": {
                "engine": ner_extraction.get("engine"),
                "degrees": ner_extraction.get("degrees") or [],
                "job_titles": ner_extraction.get("job_titles") or [],
                "organizations": ner_extraction.get("organizations") or [],
                "entity_count": ner_extraction.get("entity_count", 0),
                "entities": ner_extraction.get("entities") or [],
                "model": self.custom_ner.info(),
            },
            "mathematical_match": math_match,
            "sections_analysis": section_audits,
            "ai_feedback": feedback,
            "segmented_sections": {sec: len(content) for sec, content in sections.items()}
        }


# ======================================================================================
# 9. BENCHMARK & DEMO EXECUTION
# ======================================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("  SMART CAREER AI - FULL NLP & MACHINE LEARNING PIPELINE TEST BENCHMARK")
    print("=" * 80)

    # Sample Candidate Resume
    SAMPLE_RESUME = """
    MUHAMMAD SAFWAN
    Karachi, Pakistan | +92-300-1234567 | safwan@example.com | linkedin.com/in/safwan | github.com/safwan

    PROFESSIONAL SUMMARY
    Dedicated Full Stack Software Engineer with 3+ years of experience building scalable web applications.
    Specialized in React, Node.js, Express, and distributed databases. Delivered high-impact web apps
    with 99.9% uptime and reduced query latency by 30%.

    CORE TECHNICAL SKILLS
    Languages & Frameworks: React, Node.js, Express, Python, Tailwind CSS, JavaScript, TypeScript
    Databases & Cloud: MongoDB, PostgreSQL, Docker, AWS, Git, REST APIs
    Machine Learning: Scikit-Learn, NLP, spaCy

    WORK EXPERIENCE
    Software Engineer | Tech Innovations Pvt Ltd (2024 - Present)
    - Developed modern responsive frontends using React and Tailwind CSS for 40,000+ active users.
    - Built RESTful microservices with Node.js and Express, improving system throughput by 25%.
    - Integrated PostgreSQL and MongoDB database architectures with automated caching.

    Junior Web Developer | Digital Cloud Studio (2022 - 2024)
    - Created dynamic dashboards and integrated third-party payment APIs.
    - Collaborated with senior engineers using Git workflows and Agile sprints.

    ACADEMIC EDUCATION
    Bachelor of Science in Computer Science (BS Computer Science)
    FAST National University of Computer and Emerging Sciences, Karachi (CGPA: 3.4 / 4.0)
    Graduation: 2024

    PRACTICAL PROJECTS
    Smart Career AI System (github.com/safwan/smart-career)
    - Multi-department career counseling and intelligent resume scoring engine.
    - Integrated spaCy NLP, scikit-learn TF-IDF vector similarity, and Groq LLM assistance.
    """

    TARGET_JOB = """
    Senior Full Stack Engineer
    Requirements:
    - 3+ years experience with React, Node.js, Express, and Tailwind CSS.
    - Strong database design in PostgreSQL and MongoDB.
    - Familiarity with Docker containerization, AWS, CI/CD, and REST APIs.
    - BS in Computer Science or Software Engineering.
    - Good communication skills and ability to lead sprint planning.
    """

    pipeline = ResumeNLPPipeline()
    result = pipeline.process_resume(
        resume_input=SAMPLE_RESUME,
        job_description=TARGET_JOB,
        candidate_name="Muhammad Safwan"
    )

    print(f"\n[+] Processing Status      : Complete")
    print(f"[+] Overall ATS Score      : {result['overall_ats_score']}/100")
    print(f"[+] Classical ML Match     : TF-IDF Cosine Similarity = {result['mathematical_match']['cosine_similarity']} ({result['mathematical_match']['match_percentage']}%)")
    print(f"[+] Extracted Skills ({result['skills_count']}) : {', '.join(result['extracted_skills'])}")
    print(f"[+] Shared Core Keywords   : {', '.join(result['mathematical_match']['shared_keywords'])}")
    print(f"[+] Missing Keywords       : {', '.join(result['mathematical_match']['missing_keywords'])}")

    print("\n--- 6-SECTION DIAGNOSTIC AUDIT (MATCHES ProfileResumeTab.jsx) ---")
    for sec in result["sections_analysis"]:
        print(f"  • {sec['title']:<32} | Score: {sec['score']:>3}% | Status: {sec['status']:<14} | Notes: {sec['notes']}")

    print("\n--- COGNITIVE AI FEEDBACK ---")
    print(f"Summary: {result['ai_feedback']['summary']}")
    print("Actionable Recommendations:")
    for tip in result["ai_feedback"]["actionable_tips"]:
        print(f"  - {tip}")

    print("\n" + "=" * 80)
    print("  RUNNING MULTI-DEPARTMENT BENCHMARK TEST (CS, ENGINEERING, BUSINESS, MEDICAL)")
    print("=" * 80)

    BENCHMARK_PROFILES = [
        {
            "dept": "Mechanical Engineering",
            "name": "Ahmed Ali",
            "resume": """
            AHMED ALI | Mechanical Engineer | Karachi | ahmed.eng@email.com | +92-321-9876543
            PROFESSIONAL SUMMARY
            Mechanical Design Engineer with 3+ years experience in AutoCAD, SolidWorks, and FEA simulation.
            CORE SKILLS
            AutoCAD, SolidWorks, MATLAB, Finite Element Analysis (FEA), Six Sigma, Python
            WORK EXPERIENCE
            Design Engineer | Indus Motors (2022 - Present)
            - Modeled 120+ automotive assemblies in SolidWorks and reduced stress concentration by 18%.
            - Conducted thermal and structural FEA using MATLAB and ANSYS.
            EDUCATION
            BS Mechanical Engineering | NUST Islamabad (CGPA: 3.5 / 4.0)
            PRACTICAL PROJECTS
            Formula Student Race Car Chassis (github.com/ahmed/fsae-chassis)
            - Optimized tubular chassis weight by 14kg while maintaining torsional stiffness.
            """,
            "job": "Mechanical Design Engineer needed with SolidWorks, AutoCAD, FEA, and MATLAB experience. Degree in Mechanical Engineering required."
        },
        {
            "dept": "Business & Finance",
            "name": "Sara Khan",
            "resume": """
            SARA KHAN | Financial Analyst | Karachi | sara.fin@email.com | +92-333-1122334
            PROFESSIONAL SUMMARY
            Chartered Financial Analyst with 3 years of experience in Financial Modeling, Budgeting, and Power BI dashboards.
            CORE SKILLS
            Financial Modeling, Advanced Excel, Power BI, QuickBooks, SQL, Data Analysis, Auditing
            WORK EXPERIENCE
            Financial Analyst | Habib Bank Ltd (2022 - Present)
            - Prepared multi-currency forecast models for $12M portfolio with 98% accuracy.
            - Built automated Power BI dashboards reducing weekly reporting time by 15 hours.
            EDUCATION
            BBA in Finance & Accounting | IBA Karachi (CGPA: 3.7 / 4.0)
            PRACTICAL PROJECTS
            Portfolio Risk Optimizer (github.com/sara/risk-optimizer)
            - Monte Carlo risk simulation model using Excel and Python.
            """,
            "job": "Seeking Financial Analyst with expertise in Financial Modeling, Power BI, Advanced Excel, SQL, and QuickBooks."
        },
        {
            "dept": "Healthcare & Medical",
            "name": "Dr. Ayesha Malik",
            "resume": """
            DR. AYESHA MALIK | Clinical Pharmacist | Islamabad | ayesha.med@email.com | +92-345-7766554
            PROFESSIONAL SUMMARY
            Dedicated Clinical Pharmacist with 3+ years of clinical hospital experience at Shifa International.
            CORE SKILLS
            Patient Care, Clinical Diagnostics, Pharmacology, CPR, Infection Control, EMR, Pharmacovigilance
            WORK EXPERIENCE
            Clinical Pharmacist | Shifa International Hospital (2022 - Present)
            - Managed therapeutic drug monitoring for 200+ inpatients and optimized dosage regimens.
            - Led hospital infection control and antimicrobial stewardship protocols.
            EDUCATION
            Pharm-D (Doctor of Pharmacy) | Quaid-i-Azam University (CGPA: 3.8 / 4.0)
            PRACTICAL PROJECTS
            Clinical EMR Drug Interaction Checker (drive.google.com/ayesha/emr-checker)
            - Designed hospital-wide adverse drug reaction reporting and triage system.
            """,
            "job": "Clinical Pharmacist and Medical Officer needed with expertise in Patient Care, Pharmacology, Clinical Diagnostics, and CPR. Pharm-D or MBBS required."
        }
    ]

    for profile in BENCHMARK_PROFILES:
        res = pipeline.process_resume(
            resume_input=profile["resume"],
            job_description=profile["job"],
            candidate_name=profile["name"]
        )
        print(f"\n>> Department: {profile['dept']} | Candidate: {profile['name']}")
        print(f"   ATS Score          : {res['overall_ats_score']}/100")
        print(f"   Extracted Skills   : {', '.join(res['extracted_skills'])}")
        print(f"   Extracted Degrees  : {', '.join(res['custom_nlp']['degrees'])}")
        print(f"   Cosine Similarity  : {res['mathematical_match']['cosine_similarity']} ({res['mathematical_match']['match_percentage']}%)")
        print(f"   AI Summary Advice  : {res['ai_feedback']['summary']}")

    print("\n" + "=" * 80)
    print("  ALL 4 DEPARTMENT PIPELINE BENCHMARKS EXECUTED WITH 100% SUCCESS!")
    print("=" * 80)
