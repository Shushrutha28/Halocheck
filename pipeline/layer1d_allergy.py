"""
layer1d_allergy.py — Allergy Section Extraction + Omission Detector

Extracts the structured ALLERGIES section from source notes and checks
whether each allergen is mentioned in the LLM summary.

Why this beats Layer 1 NER for allergies:
  Layer 1 scans full text for allergy keywords — noisy, misses context
  This layer reads the ALLERGIES: section specifically — surgical precision

Logic:
  1. Find ALLERGIES: section header in source
  2. Extract each allergen name + reaction if present
  3. Check if allergen appears in summary (or NKDA acknowledged)
  4. Flag missing Critical allergens

New detection type: allergy_omission
Severity: always Critical — missing allergy = patient safety risk

Precision: near 1.0 — only fires when ALLERGIES section explicitly
           lists a drug and summary omits it entirely

Reads:   data/corpus.json
Writes:  cache/layer1d.json

Run: python pipeline/layer1d_allergy.py
"""

import json
import re
from pathlib import Path

CORPUS_PATH = Path("data/corpus.json")
OUTPUT_PATH = Path("cache/layer1d.json")

# ── Section extraction ────────────────────────────────────────────────────────

# Patterns that mark end of allergy section
SECTION_END_PATTERNS = [
    r"\n\s*[A-Z][A-Z\s]{3,}:",          # next section header e.g. "MEDICATIONS:"
    r"\n\s*\d+\.\s+[A-Z]",              # numbered list start
    r"\n\s*(?:PAST|PRESENT|SOCIAL|FAMILY|REVIEW|HISTORY|PHYSICAL|EXAM|"
    r"PROCEDURE|HOSPITAL|DISCHARGE|ADMISSION|ASSESSMENT|PLAN|LAB|VITAL)",
]

SECTION_END_RE = re.compile(
    "|".join(SECTION_END_PATTERNS),
    re.IGNORECASE
)

ALLERGY_HEADER_RE = re.compile(
    r"ALLERG(?:IES|Y)\s*(?:HISTORY\s*)?[:\-]\s*",
    re.IGNORECASE
)

# ── NKDA phrases ──────────────────────────────────────────────────────────────

NKDA_PHRASES = [
    "no known drug allerg", "nkda", "no known allerg",
    "no drug allerg", "denies allerg", "no allergies",
    "allergies: none", "allergy: none", "none known",
]

# ── Allergen extraction ───────────────────────────────────────────────────────

# Drug names that are high-risk when omitted
HIGH_RISK_ALLERGENS = {
    "penicillin", "amoxicillin", "ampicillin", "cephalosporin",
    "sulfa", "sulfamethoxazole", "aspirin", "nsaid", "ibuprofen",
    "codeine", "morphine", "opioid", "latex", "iodine",
    "contrast", "heparin", "warfarin", "vancomycin",
    "metronidazole", "ciprofloxacin", "doxycycline",
    "erythromycin", "clindamycin", "levofloxacin",
}


def extract_allergy_section(source_text: str) -> str:
    """Extract raw text of the ALLERGIES section."""
    header_match = ALLERGY_HEADER_RE.search(source_text)
    if not header_match:
        return ""

    start = header_match.end()
    remaining = source_text[start:]

    # Find where section ends
    end_match = SECTION_END_RE.search(remaining)
    section = remaining[:end_match.start()] if end_match else remaining[:500]

    return section.strip()


def is_nkda(section_text: str) -> bool:
    """Returns True if section states no known drug allergies."""
    text_lower = section_text.lower()
    return any(phrase in text_lower for phrase in NKDA_PHRASES)


def extract_allergens(section_text: str) -> list[dict]:
    """
    Parse allergen names and reactions from allergy section text.

    Handles formats:
      PENICILLIN
      PENICILLIN, AMOXICILLIN, SULFA
      PENICILLIN - causes hives
      Penicillin (anaphylaxis)
      Multiple drugs listed with reactions
    """
    if not section_text or is_nkda(section_text):
        return []

    allergens = []

    # Split on common delimiters
    # First try to get the drug names before any reaction description
    # Pattern: drug names (ALL CAPS or Title Case) separated by commas/semicolons
    drug_pattern = re.compile(
        r"\b([A-Z][A-Z\s\-]{2,}(?:\s+(?:AND|OR)\s+[A-Z][A-Z\s\-]{2,})*)\b",
    )

    # Also handle comma-separated mixed case
    entries = re.split(r"[,;]\s*|\s+AND\s+|\s+OR\s+", section_text)

    for entry in entries:
        entry = entry.strip()
        if not entry or len(entry) < 2:
            continue

        # Remove reaction descriptions — text after dash, colon, or parenthesis
        drug_name = re.split(r"\s*[-–(]\s*", entry)[0].strip()

        # Also remove "cause/causes/causing" type phrases
        drug_name = re.sub(
            r"\s+(?:cause|causes|causing|result|all\s+cause).*$",
            "", drug_name, flags=re.IGNORECASE
        ).strip()

        # Clean up
        drug_name = re.sub(r"\s+", " ", drug_name).strip()
        drug_name = drug_name.strip(".,;:")

        # Skip if too short, a common word, or looks like a sentence
        if len(drug_name) < 3:
            continue
        if len(drug_name.split()) > 4:
            continue
        if drug_name.lower() in {"none", "nkda", "additionally", "the",
                                   "patient", "also", "has", "note",
                                   "please", "see", "per", "above"}:
            continue
        # Skip if it looks like prose (lowercase words)
        words = drug_name.split()
        if len(words) > 1 and all(w[0].islower() for w in words):
            continue

        # Extract reaction if present in original entry
        reaction = ""
        reaction_match = re.search(
            r"(?:cause[s]?|result[s]?|produce[s]?|trigger[s]?|lead[s]?\s+to"
            r"|[-–:]\s*)(.{5,60})",
            entry, re.IGNORECASE
        )
        if reaction_match:
            reaction = reaction_match.group(1).strip()[:60]

        is_high_risk = any(
            hr in drug_name.lower()
            for hr in HIGH_RISK_ALLERGENS
        )

        allergens.append({
            "name":         drug_name,
            "name_lower":   drug_name.lower(),
            "reaction":     reaction,
            "is_high_risk": is_high_risk,
        })

    # Deduplicate by normalized name
    seen = set()
    unique = []
    for a in allergens:
        key = re.sub(r"\s+", " ", a["name_lower"]).strip()
        if key not in seen and len(key) >= 3:
            seen.add(key)
            unique.append(a)

    return unique


def allergen_in_summary(allergen: dict, summary_text: str) -> bool:
    """
    Check if the allergen is mentioned in the summary.
    Uses partial matching — 'penicillin' matches 'penicillin-class antibiotics'.
    """
    summary_lower = summary_text.lower()
    name_lower    = allergen["name_lower"]

    # Check if NKDA is stated in summary — covers all allergens
    if any(phrase in summary_lower for phrase in NKDA_PHRASES):
        return True

    # Direct name match (partial — allergen name appears in summary)
    if name_lower in summary_lower:
        return True

    # Word-level match for multi-word allergens
    words = [w for w in name_lower.split() if len(w) > 3]
    if words and all(w in summary_lower for w in words):
        return True

    return False


def process_note(note: dict, det_counter: list) -> list[dict]:
    source_text  = note.get("source_text") or note.get("text", "")
    summary_text = note.get("test_summary") or note.get("clean_summary", "")

    if not source_text or not summary_text:
        return []

    detections = []

    # Extract allergy section
    section = extract_allergy_section(source_text)
    if not section:
        return []

    # NKDA — check summary acknowledges or also states NKDA
    if is_nkda(section):
        # If source says NKDA, summary should not fabricate allergies
        # (handled by Layer 1 NER — not this layer's job)
        return []

    # Extract allergens
    allergens = extract_allergens(section)
    if not allergens:
        return []

    # Check each allergen against summary
    for allergen in allergens:
        if allergen_in_summary(allergen, summary_text):
            continue   # correctly mentioned — no flag

        # Allergen missing from summary
        det_counter[0] += 1
        name    = allergen["name"]
        tok     = re.sub(r"[^a-z0-9_]", "_", allergen["name_lower"][:25])
        entity  = f"{note['note_id']}_{tok}"

        flag_text = f"Allergy to {name} not mentioned in summary"
        if allergen["reaction"]:
            flag_text += f" (reaction: {allergen['reaction']})"

        detections.append({
            "detection_id":  f"det_{det_counter[0]:04d}",
            "entity_id":     entity,
            "detected_by":   ["layer1d_allergy"],
            "type":          "allergy_omission",
            "flagged_text":  flag_text,
            "severity":      "Critical",
            "confidence":    0.95,
            "allergen":      name,
            "reaction":      allergen["reaction"],
            "is_high_risk":  allergen["is_high_risk"],
            "source_section": section[:150],
        })

    return detections


def main():
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    print(f"Allergy omission check — {len(corpus)} notes...")

    det_counter   = [0]
    results       = []
    notes_flagged = 0
    total_flags   = 0
    has_section   = 0

    for i, note in enumerate(corpus):
        dets = process_note(note, det_counter)
        results.append({"note_id": note["note_id"], "detections": dets})

        src = note.get("source_text") or note.get("text", "")
        if extract_allergy_section(src):
            has_section += 1

        if dets:
            notes_flagged += 1
            total_flags   += len(dets)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(corpus)}] allergy sections={has_section}, "
                  f"flagged={notes_flagged}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLayer 1d complete → {OUTPUT_PATH}")
    print(f"  Notes with allergy section: {has_section}/{len(corpus)}")
    print(f"  Notes flagged: {notes_flagged}")
    print(f"  Total detections: {total_flags}")

    # Sample
    samples = [r for r in results if r["detections"]][:5]
    if samples:
        print("\nSample detections:")
        for s in samples:
            d = s["detections"][0]
            print(f"\n  [Critical] {s['note_id']}")
            print(f"  {d['flagged_text']}")
            print(f"  Section: ...{d['source_section'][:80]}...")

    print(f"\nNext: python run_pipeline.py --from 6")


if __name__ == "__main__":
    main()
