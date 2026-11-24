import sys, os

sys.path.insert(0, os.path.abspath("AI_features_python"))
from resume_nlp_pipeline import ResumeNLPPipeline

def run_senior_vs_junior_test():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("================================================================================")
    print("  RUNNING SENIOR VS JUNIOR REACT DISCRIMINATION TEST (VIVA SPECIAL)")
    print("================================================================================")
    pipeline = ResumeNLPPipeline()

    senior_react_job = (
        "Senior React Architect & Full-Stack Engineer. Required 5+ years of production experience. "
        "Must possess deep expertise in React, TypeScript, Redux Toolkit, Next.js, Node.js, GraphQL, "
        "Microfrontends, Webpack, Jest, Performance Optimization, and Distributed System Design. "
        "BS or MS in Computer Science or Software Engineering required."
    )

    junior_resume = (
        "Junior React Developer. Recent graduate (BS Computer Science, 2026). "
        "8 months internship experience. Skills: HTML, CSS, JavaScript, basic React, useState, useEffect, "
        "Tailwind CSS, Git. Built personal portfolio and simple weather app."
    )

    senior_resume = (
        "Senior Lead React Architect with 6+ years of production engineering experience. "
        "Proven track record scaling enterprise SaaS web applications. "
        "Core competencies: React 18, TypeScript, Redux Toolkit, Next.js, Node.js, Express, Microfrontends, "
        "GraphQL, Jest unit testing, Webpack bundling, Docker, Web Performance Optimization, and System Design. "
        "BS Software Engineering. Led team of 7 developers in agile microservices architecture."
    )

    print("\nEvaluating Target Job:")
    print("  Job Title: Senior React Architect & Full-Stack Engineer (5+ Yrs Req)")
    print("--------------------------------------------------------------------------------")

    junior_result = pipeline.process_resume(junior_resume, senior_react_job, "Junior Candidate")
    senior_result = pipeline.process_resume(senior_resume, senior_react_job, "Senior Candidate")

    j_score = junior_result["overall_ats_score"]
    s_score = senior_result["overall_ats_score"]

    j_subs = junior_result["score_breakdown"]["sub_scores"]
    s_subs = senior_result["score_breakdown"]["sub_scores"]

    print(f"\n📊 [1] Junior React Developer Results:")
    print(f"  • Overall Match Score : {j_score}/100")
    print(f"  • Skills Sub-score    : {j_subs['skills']}/100")
    print(f"  • Experience Alignment: {j_subs['experience']}/100")
    print(f"  • Education Relevance : {j_subs['education']}/100")
    print(f"  • Matched Skills      : {junior_result.get('matched_skills', [])}")
    print(f"  • Missing Skills      : {junior_result.get('missing_skills', [])}")

    print(f"\n📊 [2] Senior React Architect Results:")
    print(f"  • Overall Match Score : {s_score}/100")
    print(f"  • Skills Sub-score    : {s_subs['skills']}/100")
    print(f"  • Experience Alignment: {s_subs['experience']}/100")
    print(f"  • Education Relevance : {s_subs['education']}/100")
    print(f"  • Matched Skills      : {senior_result.get('matched_skills', [])}")
    print(f"  • Missing Skills      : {senior_result.get('missing_skills', [])}")

    delta = s_score - j_score
    print(f"\n🎯 DISCRIMINATION DELTA: +{delta} Points in favor of Senior Architect!")

    # Assertions
    assert s_score >= 80, f"Expected Senior score >= 80, got {s_score}"
    assert j_score < s_score, f"Expected Senior to beat Junior, but got {s_score} vs {j_score}"
    assert delta >= 25, f"Expected at least 25 points discrimination gap between Junior and Senior, got {delta}"
    print(f"✅ PASSED: System clearly discriminates seniority level with a massive {delta}% gap!")

    print("\n================================================================================")
    print("  SENIOR VS JUNIOR DISCRIMINATION TEST PASSED WITH FLYING COLORS!")
    print("================================================================================")

if __name__ == "__main__":
    run_senior_vs_junior_test()
