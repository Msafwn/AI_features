# 📊 SMART CAREER AI - GOOGLE SHEETS SCHEMA & CONFIGURATION

Yeh Google Sheet aapke AI Agent ke liye **Data Source (Jobs)**, **Applicant Tracker**, aur **Memory (Logs)** ka kaam karegi.

---

## 📑 Sheet 1: Jobs_Directory (Active Vacancies & Seats Quota)
Is tab me company ki active jobs aur unka confidential seats quota store hoga:

| job_id | job_title | department | required_skills | min_experience_years | allocated_seats | filled_seats | status |
|---|---|---|---|---|---|---|---|
| JOB-01 | Junior MERN Developer | IT & Software | React, Node.js, Express, MongoDB, Git | 1 | 3 | 1 | Active |
| JOB-02 | Financial Analyst | Finance & Accounts | Financial Modeling, Excel, QuickBooks, Auditing | 2 | 2 | 1 | Active |
| JOB-03 | Clinical Research Associate | Healthcare | Clinical Trials, GCP, Patient Care, Pharmacology | 1 | 4 | 2 | Active |
| JOB-04 | Digital Marketing Specialist | Marketing & Sales | SEO, Google Ads, Content Strategy, Analytics | 2 | 3 | 0 | Active |

---

## 📑 Sheet 2: Candidate_Applications (ATS Score & Matches)
Jab bhi koi candidate resume upload karega, AI Agent uska parsed record yahan add karega:

| applicant_id | name | email | department | skills_extracted | match_percentage | ats_score | applied_job_id | status |
|---|---|---|---|---|---|---|---|---|
| APP-101 | Hamza Tariq | hamza@example.com | IT & Software | React, Node.js, MongoDB, Git, Tailwind | 94% | 88/100 | JOB-01 | Shortlisted |
| APP-102 | Sara Khan | sara@example.com | Finance & Accounts | QuickBooks, Excel, Financial Analysis | 86% | 84/100 | JOB-02 | Under Review |

---

## 📑 Sheet 3: Agent_Chat_Logs (AI Persistent Memory)
Candidate aur Recruiter ke chat interactions ka record taake agent ko pichli baatein yaad rahein:

| session_id | user_type | user_query | agent_response | timestamp |
|---|---|---|---|---|
| SESS-501 | Candidate | Mere skills ke mutabiq best job konsi hai? | Aapke skills React aur Node hain, Junior MERN Developer (94% match) best hai. | 2026-10-11 15:45 |
| SESS-502 | Recruiter | Junior MERN Developer ke seats status batao. | 1/3 seats filled hain, 2 seats abhi open hain. | 2026-10-11 15:46 |
