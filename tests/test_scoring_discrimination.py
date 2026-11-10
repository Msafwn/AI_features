import sys, os

sys.path.insert(0, os.path.abspath("AI_features_python"))
from resume_nlp_pipeline import ResumeNLPPipeline

def run_suite():
    print("================================================================================")
    print("  RUNNING SCORING DISCRIMINATION & DOMAIN ALIGNMENT TEST SUITE")
    print("================================================================================")
    pipeline = ResumeNLPPipeline()

    resumes = {
        "IT": "Software Engineer with 3+ years experience. Skills: React, Node.js, Express, MongoDB, PostgreSQL, Docker, Git. BS Computer Science.",
        "MED": "Clinical Pharmacist with 3 years hospital experience. Skills: Patient Care, Clinical Diagnostics, Pharmacology, CPR, Infection Control, EMR. Pharm-D degree.",
        "ENG": "Mechanical Engineer with 3+ years experience. Skills: AutoCAD, SolidWorks, MATLAB, Finite Element Analysis (FEA), Six Sigma. BS Mechanical Engineering.",
        "FIN": "Financial Analyst with 3 years experience. Skills: Financial Modeling, Advanced Excel, Power BI, QuickBooks, SQL, Auditing. BBA in Finance.",
    }
    jobs = {
        "IT": "Software Engineer with React, Node.js, Express, databases. BS CS required.",
        "MED": "Clinical Pharmacist required. Skills: Patient Care, Pharmacology, Clinical Diagnostics, CPR. Pharm-D or MBBS required.",
        "ENG": "Mechanical Engineer needed with SolidWorks, AutoCAD, FEA, MATLAB. BS Engineering.",
        "FIN": "Financial Analyst required. Skills: Financial Modeling, Power BI, Advanced Excel, SQL, QuickBooks. BBA or MBA.",
    }

    # 1. Matching evaluations
    print("\n[TEST 1] Department-Matching Alignment (Target >= 80%):")
    for dept in ["IT", "MED", "ENG", "FIN"]:
        res = pipeline.process_resume(resumes[dept], jobs[dept], "Candidate")
        score = res["overall_ats_score"]
        print(f"  • {dept} Resume vs {dept} Job: {score}/100 -> {'PASS' if score >= 80 else 'FAIL'}")
        assert score >= 80, f"Expected {dept} score >= 80, got {score}"

    # 2. Cross-domain mismatch evaluations
    print("\n[TEST 2] Cross-Department Discrimination (Target <= 35%):")
    med_on_it = pipeline.process_resume(resumes["MED"], jobs["IT"], "Doctor Candidate")
    it_score = med_on_it["overall_ats_score"]
    sub = med_on_it["score_breakdown"]["sub_scores"]
    print(f"  • Medical Resume vs IT Job: {it_score}/100 (Sub-scores: {sub})")
    assert it_score <= 35, f"Expected medical on IT job <= 35, got {it_score}"
    assert sub["skills"] <= 20, f"Expected skills <= 20, got {sub['skills']}"
    assert sub["education"] <= 50, f"Expected education <= 50, got {sub['education']}"
    print("  -> PASS: Medical candidate correctly rejected/scored low (25%) on IT software role!")

    # 3. Empty resume evaluation
    print("\n[TEST 3] Irrelevant / Empty Resume:")
    empty_res = pipeline.process_resume("Hello my name is Ali. I like cricket.", jobs["IT"], "Ali")
    print(f"  • Empty/Casual Text: {empty_res['overall_ats_score']}/100")
    assert empty_res["overall_ats_score"] <= 30
    print("  -> PASS: Non-resume text correctly penalized!")

    print("\n================================================================================")
    print("  ALL TESTS PASSED SUCCESSFULLY!")
    print("================================================================================")

if __name__ == "__main__":
    run_suite()
