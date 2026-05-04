"""
layer1c_abnormal.py — Abnormal Lab Value Misrepresentation Detector

Detects when a clinically abnormal lab value in the source note is
misrepresented as normal in the LLM summary. This is one of the most
dangerous hallucination types — a clinician reading "labs unremarkable"
when creatinine is 2.8 or WBC is 25 may miss a critical finding.

Logic:
  1. Extract lab values from source text
  2. Check each against clinical reference ranges
  3. If abnormal AND summary contains a normalizing phrase nearby → flag

New detection type: lab_interpretation_error
Severity: Critical (significantly abnormal), Moderate (mildly abnormal)

Zero false positives when:
  - The abnormal value is unambiguous (e.g. creatinine 5.9)
  - The normalizing phrase is near the lab name in the summary
  - The lab name appears in both source and summary

Reads:   data/corpus.json
Writes:  cache/layer1c.json

Run: python pipeline/layer1c_abnormal.py
"""

import json
import re
from pathlib import Path

CORPUS_PATH = Path("data/corpus.json")
OUTPUT_PATH = Path("cache/layer1c.json")

# ── Reference ranges ──────────────────────────────────────────────────────────
# (lab_name, pattern, low_normal, high_normal, unit, critical_low, critical_high)
# critical thresholds = values where missing the finding is immediately dangerous

REFERENCE_RANGES = [
    # Lab name,  regex pattern,                        lo,   hi,    unit,     crit_lo, crit_hi
    ("creatinine", r"\b(creatinine)\b",                0.6,  1.2,   "mg/dL",  None,    3.0),
    ("glucose",    r"\b(glucose)\b",                   70,   100,   "mg/dL",  50,      400),
    ("hba1c",      r"\b(HbA1c|A1c|hemoglobin\s+A1c)\b", 0,  5.7,   "%",      None,    None),
    ("inr",        r"\b(INR)\b",                       0.8,  1.2,   "",       None,    4.0),
    ("sodium",     r"\b(sodium|Na)\b",                 136,  145,   "mEq/L",  125,     155),
    ("potassium",  r"\b(potassium|K)\b",               3.5,  5.0,   "mEq/L",  3.0,     6.0),
    ("hemoglobin", r"\b(hemoglobin|Hgb|Hb)\b",        12.0, 17.5,  "g/dL",   7.0,     None),
    ("wbc",        r"\b(WBC|white\s+blood\s+cell)\b",  4.5,  11.0,  "K/uL",   None,    20.0),
    ("platelets",  r"\b(platelets|plt)\b",             150,  400,   "K/uL",   50,      None),
    ("troponin",   r"\b(troponin)\b",                  0,    0.04,  "ng/mL",  None,    None),
    ("bilirubin",  r"\b(bilirubin)\b",                 0.1,  1.2,   "mg/dL",  None,    None),
    ("albumin",    r"\b(albumin)\b",                   3.5,  5.0,   "g/dL",   None,    None),
    ("ast",        r"\b(AST)\b",                       10,   40,    "U/L",    None,    None),
    ("alt",        r"\b(ALT)\b",                       7,    56,    "U/L",    None,    None),
    ("calcium",    r"\b(calcium)\b",                   8.5,  10.5,  "mg/dL",  7.0,     13.0),
    ("magnesium",  r"\b(magnesium|Mg)\b",              1.7,  2.2,   "mg/dL",  None,    None),
    ("phosphorus", r"\b(phosphorus|phosphate)\b",      2.5,  4.5,   "mg/dL",  None,    None),
    ("lactate",    r"\b(lactate|lactic\s+acid)\b",     0.5,  2.0,   "mmol/L", None,    4.0),
    ("bnp",        r"\b(BNP|NT-proBNP)\b",             0,    100,   "pg/mL",  None,    None),
    ("tsh",        r"\b(TSH)\b",                       0.4,  4.0,   "mIU/L",  None,    None),
    ("psa",        r"\b(PSA)\b",                       0,    4.0,   "ng/mL",  None,    None),
    ("egfr",       r"\b(eGFR|GFR)\b",                 60,   999,   "mL/min", 15,      None),
    ("spo2",       r"\b(SpO2|O2\s+sat|oxygen\s+sat)",  95,   100,   "%",      88,      None),
]

# Phrases that suggest normal interpretation
NORMAL_PHRASES = [
    "within normal limits", "wnl", "unremarkable", "no abnormal",
    "normal limits", "normal range", "within normal", "normal labs",
    "labs were normal", "labs are normal", "lab values normal",
    "labs unremarkable", "laboratory unremarkable",
    "stable", "no significant", "no abnormalities",
    "all normal", "otherwise normal", "appeared normal",
]

# Context window for checking if normal phrase is near the lab name
NEARBY_WINDOW = 150   # chars
GLOBAL_WINDOW = 500   # chars — for global "labs normal" statements


def classify_severity(lab_name: str, value: float,
                      lo: float, hi: float,
                      crit_lo, crit_hi) -> str:
    """Critical if value exceeds critical threshold, Moderate if just outside range."""
    if crit_lo is not None and value < crit_lo:
        return "Critical"
    if crit_hi is not None and value > crit_hi:
        return "Critical"
    # Significantly outside range → Critical
    if value < lo:
        deviation = (lo - value) / lo
        return "Critical" if deviation > 0.25 else "Moderate"
    if value > hi:
        deviation = (value - hi) / hi
        return "Critical" if deviation > 0.25 else "Moderate"
    return "Moderate"


def extract_lab_value(text: str, name_pattern: str) -> list[tuple]:
    """
    Extract (value, position, context) for a lab test from text.
    Returns list since a note may have multiple readings.
    """
    results = []
    # Find the lab name
    for name_match in re.finditer(name_pattern, text, re.IGNORECASE):
        name_end = name_match.end()
        # Look for a number within 30 chars after the lab name
        after = text[name_end:name_end + 30]
        num_match = re.search(r"[\s:=]+(\d+\.?\d*)", after)
        if num_match:
            try:
                val = float(num_match.group(1))
                pos = name_match.start()
                ctx_start = max(0, pos - 50)
                ctx_end   = min(len(text), name_end + 80)
                results.append((val, pos, text[ctx_start:ctx_end].strip()))
            except ValueError:
                pass
    return results


def summary_normalizes(lab_name: str, name_pattern: str,
                       summary: str) -> tuple[bool, str]:
    """
    Check if summary contains a normalizing phrase near this lab name,
    or a global normalizing statement about labs.
    Returns (is_normalized, matched_phrase).
    """
    summary_lower = summary.lower()

    # Check global normalizing statements (applies to all labs)
    global_phrases = [
        "labs unremarkable", "labs were unremarkable",
        "laboratory unremarkable", "all labs normal",
        "lab values normal", "lab results normal",
        "labs within normal", "all laboratory values normal",
    ]
    for phrase in global_phrases:
        if phrase in summary_lower:
            return True, phrase

    # Check if lab name appears in summary
    lab_in_summary = list(re.finditer(name_pattern, summary, re.IGNORECASE))
    if not lab_in_summary:
        # Lab not mentioned in summary — omission, not misrepresentation
        return False, ""

    # Check for normalizing phrase near the lab name in summary
    for lab_match in lab_in_summary:
        pos = lab_match.start()
        window_start = max(0, pos - NEARBY_WINDOW)
        window_end   = min(len(summary), pos + NEARBY_WINDOW)
        window       = summary_lower[window_start:window_end]

        for phrase in NORMAL_PHRASES:
            if phrase in window:
                return True, phrase

    return False, ""


def process_note(note: dict, det_counter: list) -> list[dict]:
    source_text  = note.get("source_text") or note.get("text", "")
    summary_text = note.get("test_summary") or note.get("clean_summary", "")

    if not source_text or not summary_text:
        return []

    detections = []
    seen_keys  = set()   # avoid duplicate flags per lab per note

    for (lab_name, name_pattern, lo, hi,
         unit, crit_lo, crit_hi) in REFERENCE_RANGES:

        if lab_name in seen_keys:
            continue

        # Extract values from source
        source_values = extract_lab_value(source_text, name_pattern)
        if not source_values:
            continue

        # Only use the most extreme (most abnormal) value from source
        # to avoid flagging trending-toward-normal situations
        abnormal = [
            (val, pos, ctx)
            for val, pos, ctx in source_values
            if val < lo or val > hi
        ]
        if not abnormal:
            continue   # all source values within range

        # Take most abnormal value
        most_abnormal = max(
            abnormal,
            key=lambda x: max(
                (lo - x[0]) / lo if x[0] < lo else 0,
                (x[0] - hi) / hi if x[0] > hi else 0,
            )
        )
        abn_val, abn_pos, abn_ctx = most_abnormal

        # Check if summary misrepresents this as normal
        is_normalized, matched_phrase = summary_normalizes(
            lab_name, name_pattern, summary_text
        )

        if not is_normalized:
            continue

        # Confirmed: abnormal in source, presented as normal in summary
        seen_keys.add(lab_name)
        severity = classify_severity(
            lab_name, abn_val, lo, hi, crit_lo, crit_hi
        )

        det_counter[0] += 1
        tok = re.sub(r"[^a-z0-9_]", "_", f"{lab_name}_{abn_val}"[:28])

        # Build informative flag text
        direction = "low" if abn_val < lo else "high"
        flag_text = (
            f"{lab_name} {abn_val} {unit} ({direction}, "
            f"normal {lo}–{hi}) — summary says '{matched_phrase}'"
        )

        detections.append({
            "detection_id":    f"det_{det_counter[0]:04d}",
            "entity_id":       f"{note['note_id']}_{tok}",
            "detected_by":     ["layer1c_abnormal"],
            "type":            "lab_interpretation_error",
            "flagged_text":    flag_text,
            "severity":        severity,
            "confidence":      0.90,
            "lab_name":        lab_name,
            "source_value":    abn_val,
            "normal_range":    f"{lo}–{hi} {unit}".strip(),
            "direction":       direction,
            "normalizing_phrase": matched_phrase,
            "source_context":  abn_ctx[:150],
        })

    return detections


def main():
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    print(f"Abnormal value check — {len(corpus)} notes...")
    print(f"Reference ranges loaded: {len(REFERENCE_RANGES)} lab tests\n")

    det_counter   = [0]
    results       = []
    notes_flagged = 0
    total_flags   = 0

    for i, note in enumerate(corpus):
        dets = process_note(note, det_counter)
        results.append({"note_id": note["note_id"], "detections": dets})

        if dets:
            notes_flagged += 1
            total_flags   += len(dets)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(corpus)}] flagged: {notes_flagged} notes, "
                  f"{total_flags} detections")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLayer 1c complete → {OUTPUT_PATH}")
    print(f"  {notes_flagged}/{len(corpus)} notes flagged")
    print(f"  {total_flags} abnormal value misrepresentations detected")

    # Severity breakdown
    all_dets = [d for r in results for d in r["detections"]]
    crit = sum(1 for d in all_dets if d["severity"] == "Critical")
    mod  = sum(1 for d in all_dets if d["severity"] == "Moderate")
    print(f"  Critical: {crit}  Moderate: {mod}")

    # Sample detections
    samples = [r for r in results if r["detections"]][:5]
    if samples:
        print("\nSample detections:")
        for s in samples:
            d = s["detections"][0]
            print(f"\n  [{d['severity']}] {s['note_id']}")
            print(f"  {d['flagged_text']}")
            print(f"  Source: ...{d['source_context'][:80]}...")

    print(f"\nNext: python run_pipeline.py --from 6")


if __name__ == "__main__":
    main()
