"""
Data preparation pipeline for Smart Career Multi-Department Resume NER.

1. Downloads real human-annotated resume dataset (310 CVs) and holdout test set (51 CVs).
2. Maps annotations to Smart Career 4 core labels (SKILL, DEGREE, JOB_TITLE, ORG).
3. Strips XML headers and aligns entity boundaries cleanly.
4. Generates balanced multi-department synthetic examples for Engineering, Healthcare,
   and Business/Finance to ensure all departments are thoroughly represented.
5. Saves train_data.jsonl and test_data.jsonl to data/processed/.
"""

import io
import json
import random
import re
import string
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from custom_nlp.gazetteer import DEGREES, JOB_TITLES, ORGS, SKILL_ALIASES, all_skill_phrases

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TRAIN_FILE = PROCESSED_DIR / "train_data.jsonl"
TEST_FILE = PROCESSED_DIR / "test_data.jsonl"

LABEL_MAPPING = {
    "tech tools": "SKILL",
    "job specific skills": "SKILL",
    "soft skills": "SKILL",
    "skills": "SKILL",
    "designation": "JOB_TITLE",
    "job title": "JOB_TITLE",
    "degree": "DEGREE",
    "companies worked at": "ORG",
    "college name": "ORG",
    "university": "ORG",
    "company": "ORG",
}


def parse_labeled_xml(xml_content: str) -> Dict[str, any]:
    """Extract clean plain text and character-aligned entity offsets from annotated XML."""
    # Strip XML header and cv wrappers
    xml_content = re.sub(r'<\?xml[^>]*\?>', '', xml_content)
    xml_content = re.sub(r'</?cv>', '', xml_content).strip()

    pattern = re.compile(r'<label\s+type="([^"]+)">([^<]+)</label>')
    text_parts = []
    raw_entities = []
    last_end = 0

    for m in pattern.finditer(xml_content):
        # Text preceding this label
        text_parts.append(xml_content[last_end:m.start()])
        raw_label = m.group(1).strip().lower()
        span_text = m.group(2)
        target_label = LABEL_MAPPING.get(raw_label)

        if target_label:
            start_char = len("".join(text_parts))
            # Clean punctuation and whitespaces from boundaries
            cleaned_span = span_text.strip(" \t\r\n,.;:()[]{}'\"")
            
            if len(cleaned_span) >= 2:
                # Find exact position within span_text
                offset_in_span = span_text.find(cleaned_span)
                if offset_in_span >= 0:
                    s = start_char + offset_in_span
                    e = s + len(cleaned_span)
                    raw_entities.append((s, e, target_label, cleaned_span))

        text_parts.append(span_text)
        last_end = m.end()

    text_parts.append(xml_content[last_end:])
    plain_text = "".join(text_parts)

    # Normalize XML escape entities
    plain_text = plain_text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

    # Remove overlapping entities and ensure strict character match
    cleaned_entities = []
    raw_entities.sort(key=lambda x: x[0])
    for s, e, lab, original_text in raw_entities:
        if s >= e or s < 0 or e > len(plain_text):
            continue
        if plain_text[s:e] != original_text:
            continue
        overlap = False
        for ex_s, ex_e, _ in cleaned_entities:
            if not (e <= ex_s or s >= ex_e):
                overlap = True
                break
        if not overlap:
            cleaned_entities.append((s, e, lab))

    return {"text": plain_text, "entities": cleaned_entities}


def fetch_annotated_resumes(zip_url: str) -> List[Dict]:
    """Download and parse XML resumes from remote zip."""
    print(f"[data] Fetching {zip_url}...")
    req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    zf = zipfile.ZipFile(io.BytesIO(resp.read()))

    records = []
    for fname in zf.namelist():
        if not fname.endswith(".xml"):
            continue
        content = zf.read(fname).decode("utf-8", errors="ignore")
        record = parse_labeled_xml(content)
        if record["text"].strip() and record["entities"]:
            records.append(record)
    return records


def generate_multi_dept_examples(n: int, seed: int = 42) -> List[Dict]:
    """Synthesize high-quality domain sentences across Engineering, Healthcare, Finance, and CS."""
    rng = random.Random(seed)
    results = []

    dept_templates = [
        # Engineering (Civil, Mechanical, Electrical, Chemical)
        ("engineering", [
            "Experienced in structural modeling using {skill1} and {skill2} for commercial infrastructure.",
            "Graduated with {degree} from {org} specializing in finite element analysis and structural safety.",
            "Serving as {title} at {org}, responsible for designing MEP layouts with {skill1} and {skill2}.",
            "Applied {skill1}, {skill2}, and {skill3} in industrial automation and plant maintenance.",
            "Led infrastructure design using {skill1} and site supervision with {skill2} at {org}."
        ]),
        # Healthcare & Medical
        ("healthcare", [
            "Completed {degree} from {org} with clinical rotations in internal medicine and surgery.",
            "Certified {title} at {org} adhering strictly to {skill1} and {skill2} protocols.",
            "Extensive experience in {skill1}, {skill2}, and {skill3} during hospital ward rounds.",
            "Specialized in pharmacotherapy, patient monitoring, and {skill1} compliance at {org}.",
            "Conducted clinical research and diagnostic procedures including {skill1} and {skill2} at {org}."
        ]),
        # Business & Finance
        ("business", [
            "Holds a {degree} from {org} with concentration in corporate finance and audit standards.",
            "Employed as {title} at {org}, performing financial statement analysis using {skill1} and {skill2}.",
            "Demonstrated proficiency in {skill1}, {skill2}, and {skill3} for executive decision dashboards.",
            "Managed internal controls, tax filings, and budgeting reconciliations compliant with {skill1}.",
            "Prepared statutory financial statements according to {skill1} and audited ledgers at {org}."
        ]),
        # CS & IT
        ("it", [
            "Full stack engineer skilled in {skill1}, {skill2}, and {skill3} building cloud native systems.",
            "Awarded {degree} by {org} with honors in distributed computing and algorithms.",
            "Working as {title} at {org} maintaining continuous deployment pipelines using {skill1} and {skill2}."
        ])
    ]

    for _ in range(n):
        dept, templates = rng.choice(dept_templates)
        tmpl = rng.choice(templates)
        
        degree = rng.choice(DEGREES)
        org = rng.choice(ORGS)
        title = rng.choice(JOB_TITLES)
        
        skill1_cand = rng.choice(all_skill_phrases())[1]
        skill2_cand = rng.choice(all_skill_phrases())[1]
        skill3_cand = rng.choice(all_skill_phrases())[1]

        text = tmpl.format(
            degree=degree,
            org=org,
            title=title,
            skill1=skill1_cand,
            skill2=skill2_cand,
            skill3=skill3_cand
        )

        ents: List[Tuple[int, int, str]] = []
        def _add_span(phrase: str, label: str):
            idx = text.find(phrase)
            if idx >= 0:
                s = idx
                e = s + len(phrase)
                overlap = False
                for ex_s, ex_e, _ in ents:
                    if not (e <= ex_s or s >= ex_e):
                        overlap = True
                        break
                if not overlap:
                    ents.append((s, e, label))

        if "{degree}" in tmpl: _add_span(degree, "DEGREE")
        if "{org}" in tmpl: _add_span(org, "ORG")
        if "{title}" in tmpl: _add_span(title, "JOB_TITLE")
        if "{skill1}" in tmpl: _add_span(skill1_cand, "SKILL")
        if "{skill2}" in tmpl: _add_span(skill2_cand, "SKILL")
        if "{skill3}" in tmpl: _add_span(skill3_cand, "SKILL")

        ents.sort(key=lambda x: x[0])
        results.append({"text": text, "entities": ents})

    return results


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch real annotated CVs
    train_url = "https://raw.githubusercontent.com/dotin-inc/resume-dataset-NER-annotations/master/545_cvs_train_v2.zip"
    test_url = "https://raw.githubusercontent.com/dotin-inc/resume-dataset-NER-annotations/master/set_aside_test_v2_50cvs.zip"

    real_train_cvs = fetch_annotated_resumes(train_url)
    real_test_cvs = fetch_annotated_resumes(test_url)

    print(f"[data] Parsed {len(real_train_cvs)} clean real training resumes and {len(real_test_cvs)} holdout test resumes.")

    # 2. Add multi-department balance
    print("[data] Generating multi-department balanced examples (Engineering, Healthcare, Business, CS)...")
    synthetic_train = generate_multi_dept_examples(n=1200, seed=101)
    synthetic_test = generate_multi_dept_examples(n=150, seed=202)

    all_train = real_train_cvs + synthetic_train
    all_test = real_test_cvs + synthetic_test

    random.seed(42)
    random.shuffle(all_train)
    random.shuffle(all_test)

    # 3. Save datasets
    with TRAIN_FILE.open("w", encoding="utf-8") as fh:
        for r in all_train:
            fh.write(json.dumps(r) + "\n")

    with TEST_FILE.open("w", encoding="utf-8") as fh:
        for r in all_test:
            fh.write(json.dumps(r) + "\n")

    # 4. Summary counts
    label_counts = {"SKILL": 0, "DEGREE": 0, "JOB_TITLE": 0, "ORG": 0}
    for r in all_train:
        for _, _, lab in r["entities"]:
            if lab in label_counts:
                label_counts[lab] += 1

    print("\n[data] Preparation Complete!")
    print(f"  Train Set Size: {len(all_train)} records -> {TRAIN_FILE.name}")
    print(f"  Test Set Size:  {len(all_test)} records -> {TEST_FILE.name}")
    print("  Entity Distribution in Training Corpus:")
    for k, v in label_counts.items():
        print(f"    {k:12}: {v:,} annotations")


if __name__ == "__main__":
    main()
