"""
Multi-department gazetteer for Smart Career custom NLP.

Labels:
  SKILL      - tools, frameworks, clinical/engineering/finance competencies
  DEGREE     - academic credentials
  JOB_TITLE  - role titles found on resumes
  ORG        - universities and employers (Pakistan + global)
"""

import re
from typing import Dict, List, Tuple

LABELS = ("SKILL", "DEGREE", "JOB_TITLE", "ORG")

# Canonical skill -> aliases (lowercase). Keep aliases unique across map.
SKILL_ALIASES: Dict[str, List[str]] = {
    # -------------------------------------------------------------
    # Computer Science & Information Technology
    # -------------------------------------------------------------
    "Python": ["python", "python3", "py"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "React": ["react", "react.js", "reactjs"],
    "Node.js": ["node.js", "nodejs", "node"],
    "Express": ["express", "express.js", "expressjs"],
    "MongoDB": ["mongodb", "mongo", "mongoose"],
    "PostgreSQL": ["postgresql", "postgres", "psql"],
    "MySQL": ["mysql"],
    "SQL": ["sql"],
    "FastAPI": ["fastapi"],
    "Django": ["django"],
    "Flask": ["flask"],
    "Tailwind CSS": ["tailwind css", "tailwindcss", "tailwind"],
    "Docker": ["docker", "docker-compose"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Google Cloud": ["google cloud", "gcp", "google cloud platform"],
    "Azure": ["azure", "microsoft azure"],
    "Git": ["git", "github", "gitlab"],
    "REST APIs": ["rest apis", "rest api", "restful apis", "rest"],
    "GraphQL": ["graphql"],
    "Redux": ["redux", "redux toolkit"],
    "Next.js": ["next.js", "nextjs"],
    "Flutter": ["flutter", "dart"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "spaCy": ["spacy"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow"],
    "Scikit-Learn": ["scikit-learn", "sklearn", "scikit learn"],
    "Hugging Face": ["hugging face", "huggingface", "transformers"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration"],
    "Linux": ["linux", "ubuntu"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "Cyber Security": ["cyber security", "cybersecurity", "infosec"],
    "Penetration Testing": ["penetration testing", "pen testing", "ethical hacking"],

    # -------------------------------------------------------------
    # Engineering (Civil, Mechanical, Electrical, Chemical)
    # -------------------------------------------------------------
    "AutoCAD": ["autocad", "auto cad"],
    "SolidWorks": ["solidworks", "solid works"],
    "MATLAB": ["matlab"],
    "ETABS": ["etabs"],
    "Revit": ["revit", "autodesk revit"],
    "BIM": ["bim", "building information modeling"],
    "STAAD": ["staad", "staad.pro", "staad pro"],
    "Civil 3D": ["civil 3d", "civil3d"],
    "Finite Element Analysis (FEA)": ["fea", "finite element analysis"],
    "Ansys": ["ansys", "ansys workbench"],
    "PLC Programming": ["plc", "plc programming"],
    "SCADA": ["scada"],
    "Six Sigma": ["six sigma"],
    "Lean Manufacturing": ["lean manufacturing", "lean six sigma"],
    "Thermodynamics": ["thermodynamics"],
    "Structural Analysis": ["structural analysis"],
    "Primavera": ["primavera", "primavera p6"],
    "Circuit Design": ["circuit design", "schematic design"],
    "PCB Design": ["pcb design", "altium", "eagle pcb"],
    "Power Systems": ["power systems", "power distribution"],
    "Microcontrollers": ["microcontrollers", "pic microcontroller"],
    "Arduino": ["arduino"],
    "Embedded Systems": ["embedded systems", "embedded c"],
    "Surveying": ["surveying", "land surveying", "total station"],
    "Geotechnical Engineering": ["geotechnical engineering", "soil mechanics"],
    "Concrete Technology": ["concrete technology", "rcc design"],
    "CNC Machining": ["cnc machining", "cnc programming"],
    "HVAC Design": ["hvac", "hvac design"],
    "Fluid Mechanics": ["fluid mechanics", "cfd"],

    # -------------------------------------------------------------
    # Business, Finance, Accounting & Marketing
    # -------------------------------------------------------------
    "Advanced Excel": ["advanced excel", "ms excel", "microsoft excel", "excel"],
    "Power BI": ["power bi", "powerbi"],
    "Financial Modeling": ["financial modeling", "financial modelling", "dcf"],
    "QuickBooks": ["quickbooks"],
    "SAP ERP": ["sap", "sap erp"],
    "Data Analysis": ["data analysis", "data analytics"],
    "GAAP Accounting": ["gaap", "gaap accounting"],
    "IFRS": ["ifrs", "international financial reporting standards"],
    "Auditing": ["auditing", "internal audit", "external audit"],
    "Budgeting": ["budgeting", "financial forecasting", "variance analysis"],
    "Tableau": ["tableau"],
    "Risk Analysis": ["risk analysis", "risk management"],
    "Taxation": ["taxation", "income tax", "sales tax", "tax compliance"],
    "Corporate Finance": ["corporate finance"],
    "Portfolio Management": ["portfolio management"],
    "Cost Accounting": ["cost accounting"],
    "Financial Reporting": ["financial reporting"],
    "Supply Chain Management": ["supply chain", "supply chain management", "logistics"],
    "Digital Marketing": ["digital marketing", "social media marketing"],
    "SEO": ["seo", "search engine optimization"],
    "Google Analytics": ["google analytics"],
    "CRM": ["crm", "salesforce", "hubspot"],
    "Talent Acquisition": ["talent acquisition", "recruitment", "human resources"],

    # -------------------------------------------------------------
    # Healthcare, Pharmacy, Nursing & Medical Sciences
    # -------------------------------------------------------------
    "Clinical Trials Management": ["clinical trials", "clinical trial"],
    "Good Manufacturing Practice (GMP)": ["gmp", "good manufacturing practice"],
    "Pharmacovigilance": ["pharmacovigilance"],
    "HIPAA Compliance": ["hipaa", "hipaa compliance"],
    "Patient Care": ["patient care", "nursing care"],
    "Pharmacology": ["pharmacology", "clinical pharmacology"],
    "HL7": ["hl7", "hl7 fhir", "fhir"],
    "EMR": ["emr", "ehr", "electronic medical records"],
    "Biochemistry": ["biochemistry"],
    "Diagnostics": ["diagnostics", "clinical diagnostics"],
    "Pathology": ["pathology"],
    "Phlebotomy": ["phlebotomy"],
    "Medical Coding": ["medical coding", "icd-10"],
    "CPR": ["cpr", "cardiopulmonary resuscitation", "bls", "acls"],
    "Intensive Care": ["icu", "intensive care unit", "critical care"],
    "Infection Control": ["infection control"],
    "Medical Terminology": ["medical terminology"],
    "Anatomy": ["human anatomy", "anatomy"],
    "Medical Imaging": ["medical imaging", "radiology", "ultrasound", "mri", "x-ray"],
}

DEGREES: List[str] = [
    # Computer Science & IT
    "BS Computer Science",
    "Bachelor of Science in Computer Science",
    "Bachelor of Science",
    "BSc Computer Science",
    "BS Software Engineering",
    "Bachelor of Software Engineering",
    "BS Information Technology",
    "Bachelors in Information Technology",
    "MS Computer Science",
    "Master of Science in Computer Science",
    "MSc Data Science",
    "PhD Computer Science",

    # Business & Finance
    "BBA",
    "Bachelor of Business Administration",
    "MBA",
    "Master of Business Administration",
    "BS Accounting and Finance",
    "ACCA",
    "CA",
    "CFA",
    "M.Com",
    "BS Marketing",
    "MS Finance",

    # Engineering
    "B.E Civil Engineering",
    "BE Civil Engineering",
    "BS Civil Engineering",
    "Bachelor of Engineering",
    "BSc Mechanical Engineering",
    "BS Mechanical Engineering",
    "BS Electrical Engineering",
    "BE Electrical Engineering",
    "MS Structural Engineering",
    "BS Mechatronics",
    "BE Chemical Engineering",

    # Healthcare & Medical
    "MBBS",
    "BDS",
    "Pharm-D",
    "Doctor of Pharmacy",
    "BS Nursing",
    "DPT",
    "Doctor of Physical Therapy",
    "BS Medical Lab Technology",
    "FCPS",
    "MCPS",
    "MPhil Pharmacology",

    # General / Secondary
    "Intermediate",
    "FSc Pre-Medical",
    "FSc Pre-Engineering",
    "A-Levels",
    "ICS",
]

JOB_TITLES: List[str] = [
    # Computer Science & IT
    "Software Engineer",
    "Full Stack Developer",
    "Full Stack Engineer",
    "Frontend Developer",
    "Backend Developer",
    "MERN Stack Developer",
    "Data Analyst",
    "Machine Learning Engineer",
    "DevOps Engineer",
    "QA Engineer",
    "Mobile App Developer",
    "Cyber Security Analyst",
    "Junior Software Engineer",
    "Senior Software Engineer",

    # Engineering
    "Civil Engineer",
    "Structural Engineer",
    "Site Engineer",
    "Project Engineer",
    "Mechanical Design Engineer",
    "HVAC Engineer",
    "Maintenance Engineer",
    "Electrical Engineer",
    "Power Systems Engineer",
    "Quality Control Engineer",

    # Business & Finance
    "Financial Analyst",
    "Investment Analyst",
    "Accountant",
    "Audit Associate",
    "Business Analyst",
    "Tax Associate",
    "HR Officer",
    "Marketing Manager",
    "Operations Executive",
    "Supply Chain Specialist",
    "Credit Analyst",

    # Healthcare & Medical
    "Clinical Pharmacist",
    "Medical Officer",
    "Staff Nurse",
    "Lab Technician",
    "Physician",
    "Resident Medical Officer",
    "Physical Therapist",
    "Radiologist",
    "Quality Assurance Officer",

    # General
    "Intern",
    "Teaching Assistant",
]

ORGS: List[str] = [
    # Universities
    "FAST National University",
    "NUST",
    "LUMS",
    "NED University",
    "University of Karachi",
    "Aga Khan University",
    "Dow University of Health Sciences",
    "COMSATS",
    "IBA Karachi",
    "UET Lahore",
    "King Edward Medical University",
    "GIKI",
    "PIEAS",
    "University of the Punjab",

    # Tech & Engineering Firms
    "Techlogix",
    "Systems Limited",
    "NETSOL",
    "10Pearls",
    "Afiniti",
    "Careem",
    "Daraz",
    "NESPAK",
    "Descon Engineering",
    "Engro Corporation",
    "Habib Construction Services",

    # Financial & Corporate
    "HBL",
    "Meezan Bank",
    "UBL",
    "Bank Alfalah",
    "KPMG",
    "PwC",
    "EY",
    "Deloitte",
    "Unilever",

    # Healthcare Organizations
    "Shifa International Hospital",
    "Aga Khan Hospital",
    "Shaukat Khanum Hospital",
    "Indus Hospital",
    "GlaxoSmithKline",
    "Getz Pharma",
    "Searle Company",

    # Global Tech
    "Google",
    "Microsoft",
    "Amazon",
    "Meta",
]


def _unique_lower(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip()
        if not key:
            continue
        low = key.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(key)
    return out


def all_skill_phrases() -> List[Tuple[str, str]]:
    """Return (canonical, alias_or_canonical) pairs, longest aliases first."""
    pairs: List[Tuple[str, str]] = []
    for canonical, aliases in SKILL_ALIASES.items():
        phrases = _unique_lower([canonical] + list(aliases))
        for phrase in phrases:
            pairs.append((canonical, phrase))
    pairs.sort(key=lambda x: len(x[1]), reverse=True)
    return pairs


def _phrase_to_token_pattern(phrase: str) -> List[dict]:
    """Split phrases so Node.js / CI/CD still match spaCy's tokenizer."""
    pieces = re.findall(r"[\w+#]+|[^\s\w]", phrase.strip())
    return [{"LOWER": piece.lower()} for piece in pieces if piece.strip()]


def entity_ruler_patterns() -> List[dict]:
    """spaCy EntityRuler phrase patterns (case-insensitive via LOWER tokens)."""
    patterns: List[dict] = []

    def add_phrase(label: str, phrase: str) -> None:
        token_pattern = _phrase_to_token_pattern(phrase)
        if not token_pattern:
            return
        patterns.append({"label": label, "pattern": token_pattern})

    for _canonical, alias in all_skill_phrases():
        add_phrase("SKILL", alias)

    for degree in _unique_lower(DEGREES):
        add_phrase("DEGREE", degree)

    for title in _unique_lower(JOB_TITLES):
        add_phrase("JOB_TITLE", title)

    for org in _unique_lower(ORGS):
        add_phrase("ORG", org)

    return patterns
