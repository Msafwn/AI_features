"""
Train Smart Career custom spaCy NER for multi-department resume entities.

Labels: SKILL, DEGREE, JOB_TITLE, ORG
Departments Covered:
  - Computer Science & Information Technology
  - Engineering (Civil, Electrical, Mechanical, Chemical)
  - Healthcare, Pharmacy & Nursing
  - Business, Finance & Accounting

Uses:
  1. Real human-annotated resume dataset (train_data.jsonl & test_data.jsonl).
  2. Multi-department gazetteer balance across all 4 departments.
  3. Pre-trained en_core_web_sm weights + spaCy mini-batch Adam optimization.
  4. EntityRuler precision layer attached at inference.

Usage (from AI_features_python):
    python train_resume_model.py
    python train_resume_model.py --epochs 8
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from spacy.training import Example, offsets_to_biluo_tags
from spacy.util import compounding, minibatch

from custom_nlp.gazetteer import DEGREES, JOB_TITLES, LABELS, ORGS, SKILL_ALIASES
from custom_nlp.resume_ner import MODEL_DIR, METRICS_PATH, attach_entity_ruler, load_base_spacy

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TRAIN_FILE = PROCESSED_DIR / "train_data.jsonl"
TEST_FILE = PROCESSED_DIR / "test_data.jsonl"


def load_dataset_rows(path: Path) -> List[Dict]:
    """Load JSONL dataset rows containing {text, entities}."""
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_examples(nlp, rows: List[Dict]) -> List[Example]:
    """Convert raw dataset records into aligned spaCy Example objects."""
    examples = []
    skipped = 0
    for row in rows:
        text = row.get("text", "")
        if not text:
            continue
        doc = nlp.make_doc(text[:100000])
        entities = [
            (int(s), int(e), str(lab))
            for s, e, lab in row.get("entities", [])
            if e > s and e <= len(text)
        ]
        try:
            tags = offsets_to_biluo_tags(doc, entities)
        except Exception:
            skipped += 1
            continue
        if "-" in tags:
            # Token boundary misalignment
            skipped += 1
            continue
        examples.append(Example.from_dict(doc, {"entities": entities}))
    if skipped:
        print(f"[train] Note: filtered {skipped} misaligned spans to guarantee training stability.")
    return examples


def evaluate(nlp, examples: List[Example]) -> Dict[str, any]:
    """Calculate Precision, Recall, and F1 on gold standard evaluation examples."""
    scorer = nlp.evaluate(examples)
    ents_p = float(scorer.get("ents_p", 0.0) or 0.0)
    ents_r = float(scorer.get("ents_r", 0.0) or 0.0)
    ents_f = float(scorer.get("ents_f", 0.0) or 0.0)
    per_type = {}
    raw_types = scorer.get("ents_per_type") or {}
    for label, scores in raw_types.items():
        per_type[label] = {
            "precision": round(float(scores.get("p", 0.0) or 0.0), 4),
            "recall": round(float(scores.get("r", 0.0) or 0.0), 4),
            "f1": round(float(scores.get("f", 0.0) or 0.0), 4),
        }
    return {
        "precision": round(ents_p, 4),
        "recall": round(ents_r, 4),
        "f1": round(ents_f, 4),
        "per_label": per_type,
    }


def train_model(epochs: int = 8, seed: int = 42) -> Dict:
    # 1. Ensure dataset exists
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        print("[train] Processed dataset not found. Running prepare_training_data.py...")
        import prepare_training_data
        prepare_training_data.main()

    print(f"[train] Loading multi-department training corpus from {TRAIN_FILE.name}...")
    train_rows = load_dataset_rows(TRAIN_FILE)
    test_rows = load_dataset_rows(TEST_FILE)

    print("[train] Initializing base model (en_core_web_sm) with target labels...")
    nlp = load_base_spacy()
    ner = nlp.get_pipe("ner")
    for label in LABELS:
        ner.add_label(label)

    print("[train] Converting corpus to aligned spaCy Example objects...")
    train_examples = to_examples(nlp, train_rows)
    test_examples = to_examples(nlp, test_rows)

    print(f"[train] Dataset Aligned: Train={len(train_examples)} records | Holdout Test={len(test_examples)} records")
    print(f"[train] Beginning training for {epochs} epochs with dropout=0.25...")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    rng = random.Random(seed)
    history = []

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for epoch in range(1, epochs + 1):
            rng.shuffle(train_examples)
            losses = {}
            # Mini-batch compounding size for optimal convergence
            batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.25, losses=losses, sgd=optimizer)

            metrics = evaluate(nlp, test_examples)
            ner_loss = round(float(losses.get("ner", 0.0)), 2)
            history.append({
                "epoch": epoch,
                "ner_loss": ner_loss,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"]
            })
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | NER Loss: {ner_loss:8.2f} | "
                f"Holdout P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}"
            )

    # Attach precision EntityRuler layer
    print("[train] Attaching multi-department EntityRuler precision layer...")
    attach_entity_ruler(nlp)

    # Save trained model to disk
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    nlp.to_disk(MODEL_DIR)

    # Final holdout evaluation
    final_metrics = evaluate(nlp, test_examples)
    payload = {
        "model_name": "smart_career_ner",
        "base_model": "en_core_web_sm",
        "labels": list(LABELS),
        "departments": [
            "Computer Science & Information Technology",
            "Engineering (Civil, Electrical, Mechanical, Chemical)",
            "Healthcare, Pharmacy & Nursing",
            "Business, Finance & Accounting"
        ],
        "n_train": len(train_examples),
        "n_test": len(test_examples),
        "epochs": epochs,
        "seed": seed,
        "holdout_test_metrics": final_metrics,
        "history": history,
        "training_corpus": "Real annotated resumes (Dotin-Inc & DataTurks) combined with Multi-Department Gazetteer (CS, Engineering, Healthcare, Finance)",
    }

    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[train] SUCCESS: Custom NER model successfully exported -> {MODEL_DIR}")
    print(f"[train] Final Holdout Test F1-Score: {final_metrics['f1']:.4f} (Precision: {final_metrics['precision']:.4f}, Recall: {final_metrics['recall']:.4f})")
    print("  Per-Label Performance:")
    for lab, s in final_metrics.get("per_label", {}).items():
        print(f"    {lab:12}: F1={s['f1']:.3f} (P={s['precision']:.3f}, R={s['recall']:.3f})")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Smart Career multi-department resume NER")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(epochs=args.epochs, seed=args.seed)
