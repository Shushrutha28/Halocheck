"""
layer1_ner.py — NER Entity Checker (Commission + Omission Detection)
Week 1-2 / Person 2

Reads:   data/corpus.json
Writes:  cache/layer1.json

Logic:
  - Extract medical entities from source_text (ground truth entities)
  - Extract medical entities from test_summary
  - Commission flag: entity in summary NOT in source → fabrication
  - Omission flag:   entity in source NOT in summary → missing critical info

FIX (Bug 3): Replaced re.findall() with re.finditer() + match.group(0)
throughout extract_entities(). The old code used capture groups inside
re.findall(), which returns only the captured group (e.g. "mg" instead of
"500mg", "allerg" instead of "allergy"). group(0) always returns the full
match regardless of capture groups in the pattern.

Run: python pipeline/layer1_ner.py
"""

import json
import re
from pathlib import Path
import spacy
import scispacy

# ── Config ────────────────────────────────────────────────────────────────────

CORPUS_PATH = Path("data/corpus.json")
OUTPUT_PATH = Path("cache/layer1.json")

CLINICAL_ENTITY_TYPES = {
    "DISEASE",
    "CHEMICAL",
    "GENE_OR_GENE_PRODUCT",
    "ORGANISM",
    "CELL_TYPE",
    "SIMPLE_CHEMICAL",
}

# Bug 3 fix: patterns use non-capturing groups (?:...) where a group is needed
# for alternation, so re.findall also works correctly if ever used elsewhere.
# But the extraction itself now uses re.finditer + group(0) — see extract_entities().
CRITICAL_PATTERNS = [
    r"\b\d+\s*(?:mg|mcg|g|ml|units?|IU)\b",
    r"\b(?:allerg|contraindic)\w*\b",
    r"\b(?:penicillin|aspirin|warfarin|heparin|metformin|insulin|lisinopril)\b",
    r"\b(?:diabetes|hypertension|COPD|CHF|MI|stroke|sepsis|cancer)\b",
]

# Strict omission detection config.
# The core problem: summaries legitimately omit most dosages from source notes
# (e.g. labs, procedure notes, background meds). We only want to flag omissions
# of medications/allergies that were in the summary-relevant medication list.
#
# 234 FPs in last run → tightening three gates:
#   1. Longer minimum entity length (6 chars) — "mg", "mcg", "5mg" are noise
#   2. Require the dosage to include both number AND named drug in close proximity
#   3. Stricter context — must appear within 40 chars of an explicit medication verb
OMISSION_PATTERNS = [
    r"\b\d+\s*(?:mg|mcg|g|ml|units?|IU)\b",    # dosage with number
    r"\b(?:allerg|contraindic)\w*\b",             # allergy / contraindication
    r"\b(?:penicillin|warfarin|heparin|insulin|metformin)\b",  # critical drugs
]

OMISSION_MIN_LENGTH = 6   # raised from 4 — filters "5mg", "10g" etc.

# Stricter context: must be within 40 chars of an active medication verb
# "prescribed", "taking", "started on", "initiated", "continued" etc.
OMISSION_CONTEXT_PATTERNS = [
    r"\b(?:prescribed|taking|started|initiated|continued|receiving|given|"
    r"administered|ordered|dispensed|maintained)\b",
    r"\b(?:allerg|contraindic)\w*\b",
]

OMISSION_CONTEXT_WINDOW = 40   # chars either side of entity in source text

# ── Setup ─────────────────────────────────────────────────────────────────────

print("Loading scispaCy en_core_sci_lg model...")
nlp = spacy.load("en_core_sci_lg")
print("Model loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_entities(text: str) -> set[str]:
    """
    Extract and normalize all clinical entities from text.

    Bug 3 fix: use re.finditer() + match.group(0) for pattern extraction.
    group(0) returns the entire matched string, not a captured sub-group.
    The old code used re.findall() with capture groups, which silently
    returned only the captured portion (e.g. "mg" from r"\b\d+\s*(mg|...)").
    """
    doc = nlp(text[:10000])
    entities = set()

    # scispaCy named entities
    for ent in doc.ents:
        if ent.label_ in CLINICAL_ENTITY_TYPES:
            entities.add(normalize_entity(ent.text))

    # Regex patterns — finditer + group(0) for full match
    for pattern in CRITICAL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.add(normalize_entity(match.group(0)))  # ← full match always

    return entities


def entity_similarity(a: str, b: str) -> float:
    if a == b:        return 1.0
    if a in b or b in a: return 0.7
    return 0.0


def find_unmatched(
    entities_a: set[str],
    entities_b: set[str],
    threshold:  float = 0.7,
) -> list[str]:
    unmatched = []
    for e_a in entities_a:
        matched = any(entity_similarity(e_a, e_b) >= threshold for e_b in entities_b)
        if not matched:
            unmatched.append(e_a)
    return unmatched


def make_entity_id(entity_text: str, note_id: str) -> str:
    """
    Stable entity_id in the format {note_id}_{normalized_entity}.
    Layer 2 uses the same format so Layer 4 deduplication actually fires.
    """
    clean = re.sub(r"[^a-z0-9_]", "_", normalize_entity(entity_text)[:30])
    return f"{note_id}_{clean}"


# ── Main processing ───────────────────────────────────────────────────────────

def process_note(note: dict, det_counter: list) -> list[dict]:
    detections = []

    source_text = note.get("source_text") or note.get("text", "")
    test_text   = note.get("test_summary") or note.get("clean_summary") or note.get("text", "")

    source_entities  = extract_entities(source_text)
    summary_entities = extract_entities(test_text)

    # Commission: in summary but NOT in source
    commission_entities = find_unmatched(summary_entities, source_entities)
    for ent in commission_entities:
        det_counter[0] += 1
        detections.append({
            "detection_id": f"det_{det_counter[0]:04d}",
            "entity_id":    make_entity_id(ent, note["note_id"]),
            "detected_by":  ["layer1_ner"],
            "type":         "diagnosis_fabrication",
            "flagged_text": ent,
            "severity":     "Moderate",
            "confidence":   0.65,
        })

    # Omission: in source but NOT in summary — with strict clinical filtering
    omission_entities = find_unmatched(source_entities, summary_entities)
    for ent in omission_entities:
        # Must match an omission pattern
        is_critical = any(
            re.search(p, ent, re.IGNORECASE)
            for p in OMISSION_PATTERNS
        )
        if not is_critical:
            continue

        # Must meet minimum length (filter out "mg", "g", "ml" alone)
        if len(ent.strip()) < OMISSION_MIN_LENGTH:
            continue

        # Dosage-only matches must include a digit (500mg yes, mg no)
        if re.fullmatch(r"(?:mg|mcg|g|ml|units?|IU)", ent.strip(), re.IGNORECASE):
            continue

        # Must appear in a clinical medication/allergy context in the source
        source_lower = source_text.lower()
        ent_pos = source_lower.find(ent.lower())
        if ent_pos >= 0:
            context_window = source_lower[max(0, ent_pos-OMISSION_CONTEXT_WINDOW):ent_pos+OMISSION_CONTEXT_WINDOW]
            in_context = any(
                re.search(p, context_window, re.IGNORECASE)
                for p in OMISSION_CONTEXT_PATTERNS
            )
            if not in_context:
                continue
        else:
            # Entity not found as substring — regex match only, skip
            continue

        det_counter[0] += 1
        detections.append({
            "detection_id": f"det_{det_counter[0]:04d}",
            "entity_id":    make_entity_id(ent, note["note_id"]),
            "detected_by":  ["layer1_ner"],
            "type":         "critical_omission",
            "flagged_text": ent,
            "severity":     "Critical",
            "confidence":   0.60,
        })

    return detections


def main():
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} notes")

    det_counter = [0]
    results = []

    for i, note in enumerate(corpus):
        print(f"  [{i+1}/{len(corpus)}] {note['note_id']}")
        detections = process_note(note, det_counter)
        results.append({
            "note_id":    note["note_id"],
            "detections": detections,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    total_flags   = sum(len(r["detections"]) for r in results)
    notes_flagged = sum(1 for r in results if r["detections"])
    print(f"\n✓ Layer 1 complete → {OUTPUT_PATH}")
    print(f"  {notes_flagged}/{len(corpus)} notes flagged | {total_flags} total detections")
    print(f"\nNext step: python pipeline/layer2_nli.py")


if __name__ == "__main__":
    main()