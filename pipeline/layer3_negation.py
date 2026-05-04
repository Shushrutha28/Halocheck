"""
layer3_negation.py — Negation Flip + Uncertainty Loss Detector
Week 1-2 / Person 3

Detects:
  1. Negation present in source but absent from summary (e.g. "no chest pain" → "chest pain")
  2. Uncertainty language present in source but absent from summary
     (e.g. "possible pneumonia" → "pneumonia" — certainty inflation)

Reads:   data/corpus.json
Writes:  cache/layer3.json

Run: python pipeline/layer3_negation.py
Requires: pip install medspacy
          python -m spacy download en_core_web_sm  (medspaCy dependency)
"""

import json
import re
from pathlib import Path
import medspacy
from medspacy.ner import TargetRule

# ── Config ────────────────────────────────────────────────────────────────────

CORPUS_PATH = Path("data/corpus.json")
OUTPUT_PATH = Path("cache/layer3.json")

# Negation cue words (single tokens only — multi-word handled separately)
NEGATION_CUES = {
    "no", "not", "without", "denies", "denied", "non",
    "never", "neither", "absent", "free",
}

NEGATION_PHRASES = [
    "no evidence of", "negative for", "absence of",
    "rules out", "ruled out",
]

# Uncertainty cues
UNCERTAINTY_CUES = [
    "possible", "possibly", "probable", "probably", "likely", "may",
    "might", "could", "suspected", "questionable", "consistent with",
    "suggests", "concerning for", "cannot exclude", "rule out",
]

# Clinical stopwords — single-word matches on these are ignored
# (they appear in nearly every note and cause massive FP rate)
CLINICAL_STOPWORDS = {
    "the", "a", "an", "is", "was", "of", "in", "with", "and", "or",
    "no", "not", "non", "her", "his", "the", "this", "that", "she",
    "he", "they", "it", "any", "all", "some", "other", "been", "be",
    "has", "have", "had", "are", "were", "will", "would", "could",
    "patient", "history", "noted", "reported", "known", "given",
    "noted", "noted", "also", "well", "including", "including",
}

# Minimum words required in a negated entity to consider it meaningful
MIN_ENTITY_WORDS = 2

# Negation window (words after cue to capture)
NEGATION_WINDOW = 4

# ── Setup ─────────────────────────────────────────────────────────────────────

print("Loading medspaCy pipeline...")
nlp = medspacy.load()
print("medspaCy loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_negated_spans(text: str) -> list[str]:
    """
    Extract meaningful negated spans from text.
    Returns list of multi-word clinical phrases that are negated.
    Only returns spans with >= MIN_ENTITY_WORDS non-stopword tokens.
    """
    text_lower = text.lower()
    words = re.findall(r"\b[\w\-]+\b", text_lower)
    spans = []

    # Single-token negation cues
    for i, word in enumerate(words):
        if word not in NEGATION_CUES:
            continue
        window_end = min(len(words), i + 1 + NEGATION_WINDOW)
        span_words = words[i + 1:window_end]
        # Filter stopwords to check meaningfulness
        meaningful = [w for w in span_words if w not in CLINICAL_STOPWORDS and len(w) > 2]
        if len(meaningful) >= MIN_ENTITY_WORDS:
            spans.append(" ".join(span_words[:3]))  # keep first 3 words as key

    # Multi-word negation phrases
    for phrase in NEGATION_PHRASES:
        for m in re.finditer(re.escape(phrase) + r"\s+([\w\s\-]{3,40})", text_lower):
            span = m.group(1).strip()
            meaningful = [w for w in span.split() if w not in CLINICAL_STOPWORDS and len(w) > 2]
            if len(meaningful) >= MIN_ENTITY_WORDS:
                spans.append(span[:30])

    return spans


def is_positively_asserted(span: str, summary_lower: str,
                           summary_negated_spans: list[str]) -> bool:
    """
    Returns True only if the span is POSITIVELY ASSERTED in the summary
    (present without negation). Requires multi-word phrase match, not
    just single-word presence.
    """
    span_words = [w for w in span.split() if w not in CLINICAL_STOPWORDS and len(w) > 2]
    if len(span_words) < MIN_ENTITY_WORDS:
        return False

    # Check if ≥2 meaningful words from the span appear in the summary
    words_found = sum(1 for w in span_words if re.search(r'\b' + re.escape(w) + r'\b', summary_lower))
    if words_found < MIN_ENTITY_WORDS:
        return False

    # Check it's not negated in the summary
    for negated in summary_negated_spans:
        neg_words = [w for w in negated.split() if w not in CLINICAL_STOPWORDS and len(w) > 2]
        overlap = sum(1 for w in span_words if w in neg_words)
        if overlap >= 1:
            return False   # same entity is also negated in summary — not a flip

    return True


def extract_uncertain_spans(text: str) -> list[str]:
    """Extract clinical phrases qualified with uncertainty language."""
    spans = []
    text_lower = text.lower()
    for cue in UNCERTAINTY_CUES:
        for match in re.finditer(r"\b" + re.escape(cue) + r"\b\s+([\w][\w\s\-]{5,40})",
                                 text_lower):
            span = match.group(1).strip()
            meaningful = [w for w in span.split() if w not in CLINICAL_STOPWORDS and len(w) > 2]
            if len(meaningful) >= 1:
                spans.append(span[:30])
    return spans


def make_entity_id(entity_text: str, note_id: str, flag_type: str) -> str:
    clean = re.sub(r"[^a-z0-9_]", "_", entity_text[:20].lower())
    return f"{note_id}_{flag_type}_{clean}"


# ── Main processing ───────────────────────────────────────────────────────────

def process_note(note: dict, det_counter: list) -> list[dict]:
    """
    Compare negations and uncertainty language between source and summary.
    Returns list of detection dicts.
    """
    detections = []

    source_text = note.get("source_text") or note.get("text", "")
    test_text   = note.get("test_summary") or note.get("clean_summary") or note.get("text", "")

    source_lower  = source_text.lower()
    summary_lower = test_text.lower()

    source_negated   = extract_negated_spans(source_text)
    summary_negated  = extract_negated_spans(test_text)
    source_uncertain = extract_uncertain_spans(source_text)
    summary_uncertain = extract_uncertain_spans(test_text)

    # ── Negation flip detection ───────────────────────────────────────────────
    # Only flag if: negated in source AND positively asserted in summary
    seen_spans = set()
    for span in source_negated:
        if span in seen_spans:
            continue
        seen_spans.add(span)

        if is_positively_asserted(span, summary_lower, summary_negated):
            det_counter[0] += 1
            detections.append({
                "detection_id": f"det_{det_counter[0]:04d}",
                "entity_id":    make_entity_id(span, note["note_id"], "negflip"),
                "detected_by":  ["layer3_negation"],
                "type":         "negation_flip",
                "flagged_text": span,
                "severity":     "Critical",
                "confidence":   0.75,
            })

    # ── Uncertainty loss detection ─────────────────────────────────────────
    # DISABLED: maps to diagnosis_fabrication which Layer 1 handles well.
    # Re-enable in a future sprint with a dedicated uncertainty injection type.

    return detections


def main():
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    print(f"Processing {len(corpus)} notes through negation detector...")

    det_counter = [0]
    results = []

    for i, note in enumerate(corpus):
        print(f"  [{i+1}/{len(corpus)}] {note['note_id']}")
        detections = process_note(note, det_counter)
        results.append({
            "note_id":    note["note_id"],
            "detections": detections,
        })

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    total_flags   = sum(len(r["detections"]) for r in results)
    notes_flagged = sum(1 for r in results if r["detections"])
    print(f"\n✓ Layer 3 complete → {OUTPUT_PATH}")
    print(f"  {notes_flagged}/{len(corpus)} notes flagged | {total_flags} total detections")
    print(f"\nNext step: run pipeline/layer4_severity.py to merge and score")


if __name__ == "__main__":
    main()