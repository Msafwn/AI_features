"""
Inference wrapper for the trained Smart Career resume NER model.

Loads models/smart_career_ner if present. Falls back to a live EntityRuler
on en_core_web_sm so the API still extracts SKILL/DEGREE/JOB_TITLE/ORG
before the first training run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import spacy
from spacy.language import Language

from custom_nlp.gazetteer import (
    DEGREES,
    JOB_TITLES,
    LABELS,
    ORGS,
    SKILL_ALIASES,
    all_skill_phrases,
    entity_ruler_patterns,
)

_KNOWN_TITLES = {title.lower() for title in JOB_TITLES}
_KNOWN_DEGREES = {d.lower() for d in DEGREES}
_KNOWN_ORGS = {o.lower() for o in ORGS}
_KNOWN_SKILL_PHRASES = {alias.lower() for _, alias in all_skill_phrases()} | {c.lower() for c in SKILL_ALIASES}

_SKILL_STOPWORDS = {
    "proficient", "experienced", "skilled", "responsible", "strong", "knowledge",
    "working", "understanding", "hands-on", "proven", "ability", "excellent",
    "familiar", "expertise", "skills", "technical", "including", "experience",
    "background", "good", "year", "years", "high-performance", "daily", "requests",
    "developed", "engineered", "built", "designed", "cgpa", "email", "phone", "summary",
    "description", "objective", "overview", "details", "contact", "curriculum"
}

_ORG_BLACKLIST = {
    "phone", "email", "contact", "address", "summary", "profile", "experience",
    "education", "skills", "projects", "references", "pakistan", "islamabad",
    "lahore", "karachi", "rawalpindi", "peshawar", "quetta", "multan", "candidate"
}

_DEGREE_INDICATORS = (
    "bs", "b.s", "ms", "m.s", "bachelor", "master", "bba", "mba", "phd",
    "associate", "degree", "diploma", "engineering", "matric", "fsc",
    "intermediate", "computer science", "software engineering", "b.sc", "m.sc",
    "doctorate", "postgraduate", "undergraduate", "b.e", "m.e"
)


def _is_plausible_job_title(surface: str) -> bool:
    if not surface or any(c in surface for c in ("\n", "\r", "\t", "@", "http:", "https:")):
        return False
    low = surface.strip().lower()
    if len(low) < 3 or len(low) > 60:
        return False
    if low in _KNOWN_TITLES:
        return True
    return any(known in low or low in known for known in _KNOWN_TITLES if len(low) >= 6)


def _is_plausible_skill(surface: str) -> bool:
    if not surface or any(c in surface for c in ("\n", "\r", "\t", "@", "http:", "https:", "/", "\\")):
        return False
    s = surface.strip()
    low = s.lower()
    if len(s) < 2 or len(s) > 35:
        return False
    if not any(c.isalpha() for c in s):
        return False
    words = s.split()
    if len(words) > 4:
        return False
    if low in _SKILL_STOPWORDS:
        return False
    if any(token in _SKILL_STOPWORDS for token in words if len(token) > 3):
        return False
    if low in _KNOWN_ORGS or low in _KNOWN_DEGREES or low in _KNOWN_TITLES:
        return False
    if any(edu_term in low for edu_term in ("university", "institute", "nuces", "college", "school", "campus", "fast")):
        return False
    return True


def _is_plausible_degree(surface: str) -> bool:
    if not surface or any(c in surface for c in ("\n", "\r", "\t", "@", "http:", "https:")):
        return False
    s = surface.strip()
    low = s.lower()
    if len(s) < 2 or len(s) > 60:
        return False
    if any(phrase in low for phrase in ("daily requests", "experience building", "scalable web", "years of experience")):
        return False
    if low in _KNOWN_DEGREES:
        return True
    return any(indicator in low for indicator in _DEGREE_INDICATORS)


def _is_plausible_org(surface: str) -> bool:
    if not surface or any(c in surface for c in ("\n", "\r", "\t", "@", "http:", "https:")):
        return False
    s = surface.strip()
    low = s.lower()
    if len(s) < 3 or len(s) > 60:
        return False
    if low in _ORG_BLACKLIST:
        return False
    if low in _KNOWN_ORGS:
        return True
    org_keywords = (
        "university", "college", "institute", "tech", "software", "solutions",
        "systems", "ltd", "pvt", "inc", "corp", "company", "hospital", "bank"
    )
    return any(kw in low for kw in org_keywords)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "smart_career_ner"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

_ALIAS_TO_CANONICAL = {}
for canonical, aliases in SKILL_ALIASES.items():
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


def normalize_skill(raw: str) -> str:
    return _ALIAS_TO_CANONICAL.get(raw.strip().lower(), raw.strip())


def attach_entity_ruler(nlp: Language, name: str = "smart_career_ruler") -> Language:
    if name in nlp.pipe_names:
        return nlp
    before = "ner" if "ner" in nlp.pipe_names else None
    ruler = nlp.add_pipe(
        "entity_ruler",
        name=name,
        before=before,
        config={"overwrite_ents": False},
    )
    ruler.add_patterns(entity_ruler_patterns())
    return nlp


def load_base_spacy() -> Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as download_spacy_model

        download_spacy_model("en_core_web_sm")
        return spacy.load("en_core_web_sm")


class CustomResumeNER:
    def __init__(self) -> None:
        self.model_dir = MODEL_DIR
        self.trained = MODEL_DIR.exists() and (MODEL_DIR / "meta.json").exists()
        self.metrics: Dict[str, Any] = {}
        if METRICS_PATH.exists():
            try:
                self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.metrics = {}

        if self.trained:
            self.nlp = spacy.load(str(MODEL_DIR))
            self.source = "trained_smart_career_ner"
        else:
            self.nlp = load_base_spacy()
            attach_entity_ruler(self.nlp)
            self.source = "entity_ruler_fallback"

        if "smart_career_ruler" not in self.nlp.pipe_names:
            attach_entity_ruler(self.nlp)

    def info(self) -> Dict[str, Any]:
        return {
            "loaded": True,
            "source": self.source,
            "trained": self.trained,
            "model_path": str(MODEL_DIR) if self.trained else None,
            "labels": list(LABELS),
            "pipe_names": list(self.nlp.pipe_names),
            "metrics": self.metrics,
            "note": (
                "Custom spaCy NER trained on multi-department resume sentences "
                "with EntityRuler gazetteer for SKILL, DEGREE, JOB_TITLE, ORG."
                if self.trained
                else "Trained model not found. Using EntityRuler gazetteer. Run: python train_resume_model.py"
            ),
        }

    def extract(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "skills": [],
                "degrees": [],
                "job_titles": [],
                "organizations": [],
                "entities": [],
                "engine": self.source,
            }

        doc = self.nlp(text[:200000])
        skills: List[str] = []
        degrees: List[str] = []
        titles: List[str] = []
        orgs: List[str] = []
        entities: List[Dict[str, Any]] = []
        seen_skills = set()
        seen_degrees = set()
        seen_titles = set()
        seen_orgs = set()

        for ent in doc.ents:
            if ent.label_ not in LABELS:
                continue
            surface = ent.text.strip()
            if not surface:
                continue
            if ent.label_ == "JOB_TITLE" and not _is_plausible_job_title(surface):
                continue
            if ent.label_ == "SKILL" and not _is_plausible_skill(surface):
                continue
            if ent.label_ == "DEGREE" and not _is_plausible_degree(surface):
                continue
            if ent.label_ == "ORG" and not _is_plausible_org(surface):
                continue

            entities.append(
                {
                    "text": surface,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
            if ent.label_ == "SKILL":
                canonical = normalize_skill(surface)
                key = canonical.lower()
                if key not in seen_skills:
                    seen_skills.add(key)
                    skills.append(canonical)
            elif ent.label_ == "DEGREE":
                key = surface.lower()
                if key not in seen_degrees:
                    seen_degrees.add(key)
                    degrees.append(surface)
            elif ent.label_ == "JOB_TITLE":
                key = surface.lower()
                if key not in seen_titles:
                    seen_titles.add(key)
                    titles.append(surface)
            elif ent.label_ == "ORG":
                key = surface.lower()
                if key not in seen_orgs:
                    seen_orgs.add(key)
                    orgs.append(surface)

        return {
            "skills": skills,
            "degrees": degrees,
            "job_titles": titles,
            "organizations": orgs,
            "entities": entities[:80],
            "entity_count": len(entities),
            "engine": self.source,
        }
