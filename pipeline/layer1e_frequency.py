"""
layer1e_frequency.py — Dosing Frequency Error Detector

Extracts medication dosing frequencies from source notes and checks
whether the summary states a different frequency for the same drug.

Same dose, different frequency is clinically significant:
  metformin 500mg twice daily → metformin 500mg daily  (halved dose)
  warfarin 5mg daily → warfarin 5mg twice daily         (doubled — dangerous)
  insulin 20 units nightly → insulin 20 units twice daily

Logic:
  1. Extract (drug, dose, frequency) triples from source
  2. Extract same from summary
  3. Match by drug+dose, compare frequency
  4. Numeric frequency mismatch → flag

New detection type: dosing_frequency_error
Severity: Critical for anticoagulants/insulin/narrow-TI drugs, Moderate otherwise

Reads:   data/corpus.json
Writes:  cache/layer1e.json

Run: python pipeline/layer1e_frequency.py
"""

import json
import re
from pathlib import Path

CORPUS_PATH = Path("data/corpus.json")
OUTPUT_PATH = Path("cache/layer1e.json")

# ── Frequency normalisation ───────────────────────────────────────────────────
# Map every frequency expression to a numeric doses-per-day value

FREQ_TO_DAILY = [
    # Latin abbreviations
    (r"\bq\.?d\.?\b",                          1.0),
    (r"\bb\.?i\.?d\.?\b",                      2.0),
    (r"\bt\.?i\.?d\.?\b",                      3.0),
    (r"\bq\.?i\.?d\.?\b",                      4.0),
    (r"\bq\.?h\.?s\.?\b",                      1.0),   # at bedtime
    # q.Nh patterns
    (r"\bq\.?\s*(\d+)\s*h(?:ours?)?\b",        None),  # special: 24/N
    # English — per day
    (r"\bonce\s+(?:a\s+|per\s+)?day\b",        1.0),
    (r"\bonce\s+daily\b",                      1.0),
    (r"\bdaily\b",                             1.0),
    (r"\bevery\s+day\b",                       1.0),
    (r"\bevery\s+morning\b",                   1.0),
    (r"\bevery\s+night(?:ly)?\b",              1.0),
    (r"\btwice\s+(?:a\s+|per\s+)?day\b",       2.0),
    (r"\btwice\s+daily\b",                     2.0),
    (r"\b2\s*(?:times?\s+)?(?:a\s+|per\s+)?day\b", 2.0),
    (r"\bthree\s+times\s+(?:a\s+|per\s+)?day\b", 3.0),
    (r"\b3\s*(?:times?\s+)?(?:a\s+|per\s+)?day\b", 3.0),
    (r"\bfour\s+times\s+(?:a\s+|per\s+)?day\b", 4.0),
    (r"\b4\s*(?:times?\s+)?(?:a\s+|per\s+)?day\b", 4.0),
    # q.Nh in English
    (r"\bevery\s+(\d+)\s*hours?\b",            None),  # special: 24/N
    # Weekly
    (r"\bonce\s+(?:a\s+|per\s+)?week\b",       1/7),
    (r"\bweekly\b",                            1/7),
    (r"\btwice\s+(?:a\s+|per\s+)?week\b",      2/7),
    # PRN / as needed — not a fixed frequency, skip
    (r"\bp\.?r\.?n\.?\b",                      None),
    (r"\bas\s+needed\b",                       None),
]

FREQ_RE = re.compile(
    r"\b(?:q\.?\d+\s*h(?:ours?)?|qd|bid|tid|qid|qhs|"
    r"once\s+(?:a\s+|per\s+)?day|once\s+daily|daily|"
    r"every\s+day|every\s+morning|every\s+night(?:ly)?|"
    r"twice\s+(?:a\s+|per\s+)?day|twice\s+daily|"
    r"(?:two|2|three|3|four|4)\s+times?\s+(?:a\s+|per\s+)?day|"
    r"b\.?i\.?d\.?|t\.?i\.?d\.?|q\.?i\.?d\.?|q\.?h\.?s\.?|"
    r"every\s+\d+\s*hours?|once\s+a\s+week|weekly|"
    r"twice\s+(?:a\s+|per\s+)?week|"
    r"per\s+day|times\s+(?:a|per)\s+day)\b",
    re.IGNORECASE
)


def freq_to_number(freq_str: str) -> float | None:
    """Convert frequency string to doses per day. Returns None for PRN."""
    s = freq_str.lower().strip()
    for pattern, value in FREQ_TO_DAILY:
        m = re.match(pattern, s, re.IGNORECASE)
        if m:
            if value is None:
                # q.Nh or every N hours
                groups = m.groups()
                if groups:
                    try:
                        n = float(groups[0])
                        return round(24.0 / n, 3)
                    except (ValueError, ZeroDivisionError):
                        return None
                return None  # PRN
            return value
    return None


def normalize_drug(name: str) -> str:
    name = name.lower().strip()
    aliases = {
        "lasix": "furosemide", "coumadin": "warfarin",
        "synthroid": "levothyroxine", "glucophage": "metformin",
        "lopressor": "metoprolol", "zocor": "simvastatin",
        "prinivil": "lisinopril", "norvasc": "amlodipine",
    }
    for brand, generic in aliases.items():
        if brand in name:
            return generic
    return name


# High-risk drugs where frequency errors are Critical
HIGH_RISK_DRUGS = {
    "warfarin", "coumadin", "heparin", "enoxaparin", "insulin",
    "digoxin", "lithium", "phenytoin", "carbamazepine", "valproate",
    "methotrexate", "tacrolimus", "cyclosporine", "amiodarone",
    "levothyroxine", "prednisone", "prednisolone",
}

# Drug name patterns (same as layer1b)
DRUG_PATTERN = re.compile(
    r"\b(metformin|warfarin|lisinopril|atorvastatin|amlodipine|"
    r"furosemide|lasix|aspirin|insulin|levothyroxine|synthroid|"
    r"prednisone|prednisolone|heparin|enoxaparin|digoxin|metoprolol|"
    r"carvedilol|losartan|sertraline|omeprazole|ciprofloxacin|"
    r"vancomycin|amoxicillin|azithromycin|clopidogrel|apixaban|"
    r"rivaroxaban|atenolol|simvastatin|rosuvastatin|pravastatin|"
    r"doxycycline|keflex|cephalexin|motrin|ibuprofen|naproxen|"
    r"tramadol|oxycodone|morphine|gabapentin|pregabalin|"
    r"potassium|colace|docusate|senna)\b",
    re.IGNORECASE
)


def extract_drug_frequencies(text: str) -> list[dict]:
    """
    Extract (drug, dose, frequency, doses_per_day) from text.
    Looks for drug name within 80 chars of a frequency term.
    """
    results = []
    text_lower = text.lower()

    for freq_match in FREQ_RE.finditer(text_lower):
        freq_str  = freq_match.group(0)
        freq_dpd  = freq_to_number(freq_str)

        if freq_dpd is None:
            continue  # PRN or unparseable

        pos = freq_match.start()

        # Look for drug name within 80 chars before the frequency
        window_start = max(0, pos - 80)
        window       = text_lower[window_start:pos + 20]

        drug_match = None
        for dm in DRUG_PATTERN.finditer(window):
            drug_match = dm  # take last drug found before frequency

        if not drug_match:
            continue

        drug_name = normalize_drug(drug_match.group(0))

        # Look for dose within window
        dose_match = re.search(
            r"(\d+\.?\d*)\s*(mg|mcg|g|units?|IU|mEq)",
            window, re.IGNORECASE
        )
        dose_str = dose_match.group(0) if dose_match else ""

        # Context
        ctx_start = max(0, pos - 60)
        ctx_end   = min(len(text), pos + 60)
        context   = text[ctx_start:ctx_end].strip()

        results.append({
            "drug":          drug_name,
            "dose":          dose_str,
            "freq_str":      freq_str,
            "freq_dpd":      freq_dpd,
            "context":       context,
            "pos":           pos,
        })

    return results


def compare_frequencies(source_text: str, summary_text: str,
                        note_id: str, det_counter: list) -> list[dict]:
    src_freqs  = extract_drug_frequencies(source_text)
    summ_freqs = extract_drug_frequencies(summary_text)

    if not src_freqs or not summ_freqs:
        return []

    # Index by drug name
    src_by_drug  = {}
    for item in src_freqs:
        src_by_drug.setdefault(item["drug"], []).append(item)

    summ_by_drug = {}
    for item in summ_freqs:
        summ_by_drug.setdefault(item["drug"], []).append(item)

    detections = []
    seen_pairs = set()

    for drug, summ_items in summ_by_drug.items():
        if drug not in src_by_drug:
            continue

        # Only check if drug appears once in each (avoid trending)
        if len(src_by_drug[drug]) != 1:
            continue

        src_item = src_by_drug[drug][0]

        for si in summ_items:
            pair = (drug, src_item["freq_dpd"], si["freq_dpd"])
            if pair in seen_pairs:
                continue

            if abs(src_item["freq_dpd"] - si["freq_dpd"]) < 0.01:
                continue   # same frequency

            seen_pairs.add(pair)

            # Determine severity
            is_high_risk = any(hr in drug for hr in HIGH_RISK_DRUGS)
            severity     = "Critical" if is_high_risk else "Moderate"

            det_counter[0] += 1
            tok = re.sub(r"[^a-z0-9_]", "_", f"{drug}_freq"[:28])

            # Human-readable description
            src_freq_desc  = f"{src_item['freq_dpd']:.0f}x/day ({src_item['freq_str']})"
            summ_freq_desc = f"{si['freq_dpd']:.0f}x/day ({si['freq_str']})"
            flag_text = (
                f"{drug.title()} frequency: "
                f"source={src_freq_desc} → summary={summ_freq_desc}"
            )

            detections.append({
                "detection_id":     f"det_{det_counter[0]:04d}",
                "entity_id":        f"{note_id}_{tok}",
                "detected_by":      ["layer1e_frequency"],
                "type":             "dosing_frequency_error",
                "flagged_text":     flag_text,
                "severity":         severity,
                "confidence":       0.90,
                "drug":             drug,
                "dose":             src_item["dose"],
                "source_freq":      src_item["freq_str"],
                "summary_freq":     si["freq_str"],
                "source_dpd":       src_item["freq_dpd"],
                "summary_dpd":      si["freq_dpd"],
                "source_context":   src_item["context"][:120],
                "summary_context":  si["context"][:120],
            })

    return detections


def main():
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    print(f"Dosing frequency check — {len(corpus)} notes...")

    det_counter   = [0]
    results       = []
    notes_flagged = 0
    total_flags   = 0

    for i, note in enumerate(corpus):
        src  = note.get("source_text") or note.get("text", "")
        summ = note.get("test_summary") or note.get("clean_summary", "")

        dets = compare_frequencies(src, summ, note["note_id"], det_counter)
        results.append({"note_id": note["note_id"], "detections": dets})

        if dets:
            notes_flagged += 1
            total_flags   += len(dets)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(corpus)}] flagged: {notes_flagged} notes")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLayer 1e complete → {OUTPUT_PATH}")
    print(f"  {notes_flagged}/{len(corpus)} notes with frequency mismatches")
    print(f"  {total_flags} total detections")

    samples = [r for r in results if r["detections"]][:5]
    if samples:
        print("\nSample detections:")
        for s in samples:
            d = s["detections"][0]
            print(f"\n  [{d['severity']}] {s['note_id']}")
            print(f"  {d['flagged_text']}")
            print(f"  Src: ...{d['source_context'][:80]}...")
            print(f"  Sum: ...{d['summary_context'][:80]}...")

    print(f"\nNext: python run_pipeline.py --from 6")


if __name__ == "__main__":
    main()
