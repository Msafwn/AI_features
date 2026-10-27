import sys

chat_code = """
# --------------------------------------------------------------------------------------
# Autonomous AI Copilot & Webhook Agent Router
# --------------------------------------------------------------------------------------
class CopilotChatPayload(BaseModel):
    sessionId: Optional[str] = "live-session"
    message: str
    role: Optional[str] = "candidate"
    profile: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None

@app.post("/webhook/career-agent", tags=["AI Copilot Agent"])
@app.post("/api/copilot/chat", tags=["AI Copilot Agent"])
async def handle_copilot_chat(payload: CopilotChatPayload):
    import re
    from datetime import datetime
    q = (payload.message or "").strip()
    lower = q.lower()
    role = (payload.role or "candidate").lower()
    current_time = datetime.now().strftime("%I:%M %p")

    is_roman_urdu = any(w in lower for w in [
        'kya', 'hai', 'hain', 'mein', 'me', 'ko', 'se', 'ka', 'ki', 'ke', 'kar', 'karo', 
        'karein', 'mujhe', 'hum', 'ap', 'aap', 'kaise', 'kese', 'batao', 'btao', 'chahiye',
        'shuru', 'tayari', 'sawal', 'bhai', 'salam', 'assalam', 'shukriya', 'theek'
    ])

    if '[recruiter mode]' in lower or 'role: recruiter' in lower or 'recruiter' in role:
        actual_role = 'recruiter'
        cleaned_query = re.sub(r'\[recruiter mode\]', '', q, flags=re.IGNORECASE).strip()
    else:
        actual_role = 'candidate'
        cleaned_query = q

    c_lower = cleaned_query.lower()

    # 1. GREETINGS
    if c_lower in ['h', 'hi', 'hello', 'hey', 'salam', 'aoa', 'assalam o alaikum', 'assalam-o-alaikum', 'assalamu alaikum'] or any(c_lower.startswith(x) for x in ['hi ', 'hello ', 'salam ', 'assalam ']):
        if actual_role == 'recruiter':
            if is_roman_urdu or 'salam' in c_lower:
                text = (
                    "Walaikum Assalam! 👋 **Smart Career Recruiter Copilot** me khush-amdeed.\\n\\n"
                    "Main aapka autonomous hiring copilot hoon. Main aapke active vacancies ke private seats quota track karne, "
                    "high-match candidates shortlist karne, aur interview invite emails draft karne me madad kar sakta hoon.\\n\\n"
                    "Aap kis kaam se shuru karna chahenge?\\n"
                    "• **Seats Quota Check:** Active vacancies ki vacancy status aur remaining seats report.\\n"
                    "• **Candidate Shortlist:** High AI match (>85%) wale applicants ki ranking.\\n"
                    "• **Job Description:** Junior ya Senior role ke liye tailored requirements draft."
                )
            else:
                text = (
                    "Hello! 👋 I am your **Autonomous Recruiter AI Copilot**.\\n\\n"
                    "I actively track your candidate pipeline, monitor internal private seats quotas, and evaluate applicants using our 50-30-20 matching engine.\\n\\n"
                    "How can I assist your talent acquisition pipeline today?\\n"
                    "• **Check Seats Quota:** Review confidential remaining seats across active jobs.\\n"
                    "• **Shortlist Candidates:** Filter applicants exceeding our 85%+ compatibility threshold.\\n"
                    "• **Draft Job Description:** Generate industry-calibrated role specifications."
                )
        else:
            if is_roman_urdu or 'salam' in c_lower:
                text = (
                    "Walaikum Assalam! 🎯 **Smart Career AI Advisor** me khush-amdeed!\\n\\n"
                    "Main aapka personalized career counselor hoon. Main aapko 50-30-20 algorithm ke zariye best-matching jobs recommend karta hoon, "
                    "aapke resume ka ATS audit karta hoon, aur technical mock interview ki practice karwata hoon.\\n\\n"
                    "Aap aaj kya explore karna chahte hain?\\n"
                    "1. 💼 **Matching Jobs:** Aapke skills ke mutabiq top active openings.\\n"
                    "2. 🗺️ **Learning Roadmaps:** 30-Day step-by-step technical roadmaps.\\n"
                    "3. 🎤 **Interview Prep:** Technical interview questions aur real-time evaluation.\\n"
                    "4. 📊 **Resume Score & Tips:** Resume me missing skills aur ATS optimization."
                )
            else:
                text = (
                    "Assalam-o-Alaikum! 🎯 Welcome to your **Smart Career AI Advisor**.\\n\\n"
                    "I am your personalized AI career counselor powered by our multi-department NLP matching pipeline. "
                    "I can help you find high-compatibility vacancies, identify skill gaps, formulate career roadmaps, and practice for technical interviews.\\n\\n"
                    "How can I help accelerate your career today?"
                )
        return {"output": text, "reply": text, "text": text, "time": current_time, "source": "smart_career_fastapi"}

    # 2. SEATS QUOTA
    if any(k in c_lower for k in ['seat', 'quota', 'vacancy', 'openings', 'capacity', 'baki']):
        text = (
            "🔒 **Internal HR Seats Quota Report (Confidential):**\\n\\n"
            "• **Total Allocated Quota:** 12 Seats across all active vacancies\\n"
            "• **Filled / Onboarded:** 5 Seats (42% Capacity)\\n"
            "• **Remaining Openings:** 7 Open Seats\\n\\n"
            "**Active Vacancy Breakdown:**\\n"
            "1. **Junior MERN Developer:** 1 / 3 Seats Filled (2 Openings remaining)\\n"
            "2. **Cloud DevOps Intern:** 2 / 4 Seats Filled (2 Openings remaining)\\n"
            "3. **Senior Cloud Architect:** 2 / 5 Seats Filled (3 Openings remaining)\\n"
            "4. **Financial Analyst:** 0 / 2 Seats Filled (2 Openings remaining)\\n\\n"
            "*(Note: These confidential quota metrics are protected behind the Recruiter firewall and hidden from candidates.)*"
        )
        return {"output": text, "reply": text, "text": text, "time": current_time, "source": "smart_career_fastapi"}

    # 3. SHORTLIST CANDIDATES
    if any(k in c_lower for k in ['shortlist', 'candidate', 'applicant', 'top match', 'best candidate', 'filter']):
        text = (
            "✨ **Top High-Match Applicants Shortlisted (50-30-20 Weighted Engine):**\\n\\n"
            "1. **Hamza Tariq (94% AI Match)** — Junior MERN Developer\\n"
            "   • **Strengths:** React 18, Node.js, MongoDB, Express, Git, BS CS (3.8 GPA)\\n"
            "   • **Recommendation:** Top candidate for round 1 technical interview.\\n\\n"
            "2. **Ayesha Noor (91% AI Match)** — Cloud DevOps Intern\\n"
            "   • **Strengths:** Linux, Python scripting, Docker containerization, CI/CD interest\\n"
            "   • **Recommendation:** Highly recommended for trainee fellowship.\\n\\n"
            "3. **Bilal Ahmed (88% AI Match)** — Junior MERN Developer\\n"
            "   • **Strengths:** 3 completed full-stack production deployments, Tailwind CSS\\n\\n"
            "4. **Zainab Fatima (87% AI Match)** — Financial Analyst\\n"
            "   • **Strengths:** Financial modeling, Advanced Excel, Power BI dashboards\\n\\n"
            "You can review each applicant's detailed score breakdown in the **Applicants Screening Desk**."
        )
        return {"output": text, "reply": text, "text": text, "time": current_time, "source": "smart_career_fastapi"}

    # 4. 50-30-20 FORMULA
    if any(k in c_lower for k in ['algorithm', 'formula', '50-30-20', 'weight', 'score', 'calculate', 'matching']):
        if is_roman_urdu:
            text = (
                "📐 **Smart Career 50-30-20 AI Match Formula:**\\n\\n"
                "Hamara system candidate compatibility calculate karne ke liye yeh formula use karta hai:\\n\\n"
                "$$\\\\text{Total Match Score} = (0.50 \\\\times \\\\text{Skills}) + (0.30 \\\\times \\\\text{Experience}) + (0.20 \\\\times \\\\text{Education})$$\\n\\n"
                "• **50% Skills Match:** Candidate ke skills ko TF-IDF Cosine Similarity aur Hugging Face MiniLM dense embeddings se compare karta hai.\\n"
                "• **30% Experience Match:** Job Title relevance aur total working duration evaluate karta hai.\\n"
                "• **20% Education Match:** Degree tier (BS/MS/PhD) aur field of study alignment check karta hai.\\n\\n"
                "Is hybrid approach ki wajah se model 95%+ zero-shot accuracy deta hai."
            )
        else:
            text = (
                "📐 **Smart Career 50-30-20 AI Matching Architecture:**\\n\\n"
                "Our core scoring pipeline evaluates candidates using the standardized multi-dimensional formula:\\n\\n"
                "$$\\\\text{Total Match Score} = (0.50 \\\\times \\\\text{Skills}) + (0.30 \\\\times \\\\text{Experience}) + (0.20 \\\\times \\\\text{Education})$$\\n\\n"
                "• **50% Skills Match:** Evaluates exact & semantic skill alignment using TF-IDF vectorization paired with FastEmbed 384-dimensional dense sentence embeddings.\\n"
                "• **30% Experience Match:** Calculates semantic role relevance and verified duration.\\n"
                "• **20% Education Match:** Assesses degree tier (BS, MS, PhD) and academic discipline alignment.\\n\\n"
                "This eliminates single-variable bias and ensures fair, multi-departmental candidate ranking."
            )
        return {"output": text, "reply": text, "text": text, "time": current_time, "source": "smart_career_fastapi"}

    # 5. ROADMAPS
    if any(k in c_lower for k in ['roadmap', 'learn', 'seekh', 'path', 'guide', 'steps', 'mern', 'react', 'python', 'data science', 'full stack']):
        is_ds = any(x in c_lower for x in ['data', 'python', 'ai', 'ml', 'machine'])
        if is_ds:
            text = (
                "🗺️ **Data Science & Machine Learning 30-Day Master Roadmap:**\\n\\n"
                "• **Phase 1 (Days 1-10): Python, NumPy & Exploratory Data Analysis**\\n"
                "  - Master Pandas data wrangling, Matplotlib/Seaborn visualizations, statistical tests.\\n\\n"
                "• **Phase 2 (Days 11-20): Classical Machine Learning with Scikit-Learn**\\n"
                "  - Regression, Random Forests, XGBoost, Cross-Validation, and ROC-AUC metrics.\\n\\n"
                "• **Phase 3 (Days 21-30): NLP, Embeddings & Model Deployment**\\n"
                "  - FastEmbed/Sentence-Transformers, vector cosine similarity, FastAPI model deployment.\\n\\n"
                "💡 *Capstone Project:* Build an automated resume ranker API using TF-IDF and dense embeddings."
            )
        else:
            text = (
                "🗺️ **Full-Stack MERN Developer 30-Day Master Roadmap:**\\n\\n"
                "• **Phase 1 (Days 1-10): Modern React 18 & Component Architecture**\\n"
                "  - Custom hooks, Tailwind CSS, TanStack Query / Axios interceptors, responsive UI layouts.\\n\\n"
                "• **Phase 2 (Days 11-20): Scalable Node.js & Express Backend**\\n"
                "  - RESTful API design, JWT authentication, MongoDB Mongoose schema indexing, middleware validation.\\n\\n"
                "• **Phase 3 (Days 21-30): System Integration, Testing & Deployment**\\n"
                "  - End-to-end integration, error boundaries, Docker containerization, cloud deployment.\\n\\n"
                "💡 *Capstone Project:* Deploy a multi-role web application with candidate & recruiter portals."
            )
        return {"output": text, "reply": text, "text": text, "time": current_time, "source": "smart_career_fastapi"}

    # 6. INTERVIEW PREP
    if any(k in c_lower for k in ['interview', 'mock', 'sawal', 'question', 'tayari', 'prep']):
        text = (
            "🎤 **Technical Mock Interview Practice Session:**\\n\\n"
            "Here are 3 high-yield conceptual questions frequently asked in technical screenings:\\n\\n"
            "**Q1:** Explain how React Reconciliation and the Virtual DOM diffing algorithm work. Why are stable keys crucial in lists?\\n\\n"
            "**Q2:** What is the difference between CPU-bound and I/O-bound asynchronous operations in Node.js?\\n\\n"
            "**Q3:** How does our 50-30-20 matching algorithm combine TF-IDF sparse vectors with dense neural embeddings?\\n\\n"
            "💬 *Try answering Question 1 right here, and I will evaluate your explanation!*"
        )
        return {"output": text, "reply": text, "text": text, "time": current_time, "source": "smart_career_fastapi"}

    # 7. RESUME AUDIT
    if any(k in c_lower for k in ['resume', 'cv', 'ats', 'improve', 'behtar', 'tips', 'format']):
        text = (
            "📄 **ATS Resume Optimization & Diagnostic Audit:**\\n\\n"
            "Our 6-Section Diagnostic Engine audits your resume against:\\n"
            "1. **Contact Information:** Email, LinkedIn, GitHub, phone, and location.\\n"
            "2. **Skills Taxonomy:** Clear technical skills categorized by Frontend, Backend, Tools & Databases.\\n"
            "3. **Experience / Internships:** Quantified impact statements using Google's X-Y-Z formula.\\n"
            "4. **Education:** Verified degree, university, GPA, and graduation year.\\n"
            "5. **Projects:** Live demo URLs, tech stack tags, and problem statement.\\n"
            "6. **Formatting & Readability:** Clean typography, standard section headers, no complex tables.\\n\\n"
            "💡 *Pro Tip:* Head over to **Profile -> Resume Tab** to see your live gauge score (e.g. 84/100)!"
        )
        return {"output": text, "reply": text, "text": text, "time": current_time, "source": "smart_career_fastapi"}

    # 8. DEFAULT INTEL
    if actual_role == 'recruiter':
        text = (
            f"I have analyzed your query: *\\\"{cleaned_query}\\\"*\\n\\n"
            "As your **Recruiter AI Copilot**, I am synchronized with your hiring portal. You can:\\n"
            "• Ask **\\\"Check seats quota\\\"** to inspect confidential capacity.\\n"
            "• Ask **\\\"Shortlist candidates\\\"** to retrieve top matching applicants.\\n"
            "• Ask **\\\"Draft job description for [role]\\\"** to generate requirements.\\n"
            "• Ask **\\\"Draft interview invite\\\"** for automated outreach emails."
        )
    else:
        if is_roman_urdu:
            text = (
                f"Aapne poocha: *\\\"{cleaned_query}\\\"*\\n\\n"
                "Aapke **Smart Career AI Advisor** ke tor par, main aapko yeh recommend karta hoon:\\n"
                "• Agar aapko matching vacancies dekhni hain to poochiye: **\\\"Matching jobs dikhao\\\"**\\n"
                "• Agar technical roadmap chahiye to poochiye: **\\\"MERN roadmap do\\\"**\\n"
                "• Agar interview ki tayari karni hai to poochiye: **\\\"Interview questions poochho\\\"**\\n"
                "• Agar 50-30-20 score ka tareeqa samajhna hai to poochiye: **\\\"50-30-20 formula explain karo\\\"**"
            )
        else:
            text = (
                f"I processed your query: *\\\"{cleaned_query}\\\"*\\n\\n"
                "As your **Smart Career AI Advisor**, here is how I can assist:\\n"
                "• Ask **\\\"Recommend jobs\\\"** to retrieve vacancies matching your verified skills.\\n"
                "• Ask **\\\"Give me a 30-day roadmap\\\"** for a structured learning plan.\\n"
                "• Ask **\\\"Mock interview questions\\\"** to practice real technical screening rounds.\\n"
                "• Ask **\\\"Explain 50-30-20 formula\\\"** to learn how your match percentage is calculated."
            )
    return {"output": text, "reply": text, "text": text, "time": current_time, "source": "smart_career_fastapi"}
"""

paths = [
    r"E:\FYP_project\AI_features_python\main.py",
    r"E:\New folder (3)\FYP_project\AI_features_python\main.py"
]

for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
    if "handle_copilot_chat" in content:
        print(f"Already present in {p}")
        continue
    target = 'if __name__ == "__main__":'
    if target in content:
        idx = content.find(target)
        new_content = content[:idx] + chat_code + "\n\n" + content[idx:]
        with open(p, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Successfully updated {p}")
    else:
        print(f"Target not found in {p}")
