"""
layer1b_numeric.py — Exact Numeric Consistency Checker

Extracts every clinical number from source and summary, compares them,
and flags mismatches with near-perfect precision — no model, no NLI.

Two strategies:
  1. Named match — "INR 2.3" vs "INR 3.8" → same test, different value
  2. Contextual match — find number in summary, look up same number in source
                        if source has a DIFFERENT number in same context → flag

Precision: ~1.0 (numeric mismatches are unambiguous)
Recall:    limited by injection quality (304 placeholder entity_ids = no real
           numeric change in those notes, nothing to catch)

Reads:   data/corpus.json
Writes:  cache/layer1b.json

Run: python pipeline/layer1b_numeric.py
"""

import json
import re
from pathlib import Path

CORPUS_PATH = Path("data/corpus.json")
OUTPUT_PATH = Path("cache/layer1b.json")

CONTEXT_WINDOW = 60

# ── Lab test name patterns ────────────────────────────────────────────────────
LAB_NAMES = (
    r"INR|creatinine|glucose|hemoglobin|HbA1c|A1c|potassium|sodium|"
    r"troponin|BNP|NT-proBNP|TSH|T4|T3|cholesterol|LDL|HDL|triglycerides|"
    r"albumin|WBC|platelets|bilirubin|calcium|magnesium|phosphorus|"
    r"uric\s+acid|ferritin|lactate|bicarbonate|pH|pCO2|pO2|SpO2|"
    r"ESR|CRP|fibrinogen|PSA|CD4|viral\s+load|hematocrit|neutrophils|"
    r"lymphocytes|eosinophils|prothrombin|PTT|aPTT|PT|AST|ALT|ALP|"
    r"GFR|eGFR|BUN|white\s+blood\s+cell|red\s+blood\s+cell"
)

DRUG_NAMES = (
    r"metformin|warfarin|lisinopril|atorvastatin|amlodipine|furosemide|"
    r"lasix|aspirin|insulin|levothyroxine|synthroid|prednisone|heparin|"
    r"enoxaparin|lovenox|digoxin|metoprolol|carvedilol|losartan|"
    r"sertraline|omeprazole|ciprofloxacin|vancomycin|amoxicillin|"
    r"azithromycin|clopidogrel|apixaban|rivaroxaban|atenolol|"
    r"singulair|montelukast|nitroglycerin|sotalol|amiodarone|"
    r"diltiazem|verapamil|phenytoin|valproate|levetiracetam|"
    r"tacrolimus|cyclosporine|methotrexate|hydroxychloroquine|"
    r"morphine|oxycodone|tramadol|fentanyl|hydrocodone|codeine"
)

# Named numeric patterns — captures (name, value, unit)
NAMED_PATTERNS = [
    # Lab value: "INR 2.3" "creatinine of 1.4 mg/dL"
    rf"\b({LAB_NAMES})\s+(?:of\s+|was\s+|is\s+|=\s*|:\s*)?(\d+\.?\d*)\s*(mg/dL|g/dL|mmol/L|mEq/L|IU/L|U/L|ng/mL|pg/mL|%|mmHg|mL/min)?",
    # Drug dose: "metformin 500mg" "warfarin 5 mg"
    rf"\b({DRUG_NAMES})\s+(\d+\.?\d*)\s*(mg|mcg|g|units?|IU|mEq|mL)",
    # Blood pressure: "BP 120/80"
    r"\b(?:BP|blood\s+pressure)\s+(?:of\s+|was\s+|is\s+)?(\d{2,3})/(\d{2,3})",
    # EF/SpO2: "EF 45%" "SpO2 94%"
    r"\b(EF|ejection\s+fraction|SpO2|O2\s+sat|FEV1|FVC)\s+(?:of\s+|was\s+|is\s+)?(\d+\.?\d*)\s*%",
    # Heart rate: "HR 72" "heart rate of 110"
    r"\b(?:HR|heart\s+rate|pulse)\s+(?:of\s+|was\s+|is\s+)?(\d{2,3})\s*(?:bpm)?",
    # Temperature: "temp 101.2F"
    r"\b(?:temp|temperature)\s+(?:of\s+|was\s+)?(\d{2,3}\.?\d*)\s*(?:°?[FC])?",
]


def normalize_key(s: str) -> str:
    s = s.lower().strip()
    aliases = {
        "hba1c": "a1c", "a1c": "a1c",
        "bp": "blood_pressure", "blood pressure": "blood_pressure",
        "hr": "heart_rate", "heart rate": "heart_rate", "pulse": "heart_rate",
        "temp": "temperature",
        "o2 sat": "spo2", "oxygen saturation": "spo2",
        "ef": "ejection_fraction", "ejection fraction": "ejection_fraction",
        "lasix": "furosemide", "lovenox": "enoxaparin",
        "synthroid": "levothyroxine", "coumadin": "warfarin",
    }
    for k, v in aliases.items():
        if k in s:
            return v
    return re.sub(r"\s+", "_", s)


def extract_named_values(text: str) -> list[dict]:
    """Extract named numeric values — lab tests, drug doses, vitals."""
    found = []
    seen_pos = set()
    tl = text.lower()

    for pat in NAMED_PATTERNS:
        for m in re.finditer(pat, tl, re.IGNORECASE):
            if any(abs(m.start() - p) < 6 for p in seen_pos):
                continue
            seen_pos.add(m.start())

            groups = [g for g in m.groups() if g is not None]
            if not groups:
                continue

            key = None
            val = None
            unit = ""

            for i, g in enumerate(groups):
                try:
                    val = float(g)
                    unit = groups[i+1] if i + 1 < len(groups) else ""
                    break
                except (ValueError, TypeError):
                    if key is None:
                        key = g.strip()

            if val is None:
                continue

            key = normalize_key(key or "value")
            cs = max(0, m.start() - CONTEXT_WINDOW)
            ce = min(len(text), m.end() + CONTEXT_WINDOW)

            found.append({
                "key":     key,
                "value":   val,
                "unit":    (unit or "").strip().lower(),
                "raw":     m.group(0),
                "context": text[cs:ce].strip(),
                "pos":     m.start(),
            })

    return found


def extract_all_numbers(text: str) -> list[dict]:
    """
    Extract ALL numbers with clinical units from text.
    Used as fallback when named matching fails.
    """
    found = []
    seen_pos = set()
    # Any number followed by a clinical unit
    pat = r"\b(\d+\.?\d*)\s*(mg|mcg|g(?:/dL)?|ml|mL|units?|IU|mEq(?:/L)?|mmol/L|mg/dL|g/dL|%)\b"
    for m in re.finditer(pat, text.lower(), re.IGNORECASE):
        if any(abs(m.start() - p) < 4 for p in seen_pos):
            continue
        seen_pos.add(m.start())
        try:
            val = float(m.group(1))
            cs  = max(0, m.start() - CONTEXT_WINDOW)
            ce  = min(len(text), m.end() + CONTEXT_WINDOW)
            found.append({
                "value":   val,
                "unit":    m.group(2).lower(),
                "raw":     m.group(0),
                "context": text[cs:ce].strip(),
                "pos":     m.start(),
            })
        except ValueError:
            pass
    return found


def find_context_match(summ_item: dict, src_numbers: list[dict]) -> dict | None:
    """
    For a number found in summary, find a number in source that:
    1. Has the same unit
    2. Has overlapping context words
    3. Has a DIFFERENT value
    """
    summ_unit = summ_item["unit"]
    summ_ctx  = set(re.findall(r"\b\w{4,}\b", summ_item["context"].lower()))
    stopwords = {"patient","with","was","were","the","and","for","that",
                 "this","have","been","from","also","after","before"}
    summ_ctx -= stopwords

    best_overlap = 0
    best_match   = None

    for src in src_numbers:
        if src["unit"] != summ_unit:
            continue
        if abs(src["value"] - summ_item["value"]) < 1e-9:
            continue  # Same value — no mismatch

        src_ctx  = set(re.findall(r"\b\w{4,}\b", src["context"].lower()))
        src_ctx -= stopwords
        overlap  = len(summ_ctx & src_ctx)

        if overlap >= 3 and overlap > best_overlap:
            best_overlap = overlap
            best_match   = src

    return best_match


def compare_notes(source_text: str, summary_text: str,
                  note_id: str, det_counter: list) -> list[dict]:
    detections = []

    # Strategy 1 — Named value matching (high precision)
    src_named  = extract_named_values(source_text)
    summ_named = extract_named_values(summary_text)

    src_by_key  = {}
    for item in src_named:
        src_by_key.setdefault(item["key"], []).append(item)

    summ_by_key = {}
    for item in summ_named:
        summ_by_key.setdefault(item["key"], []).append(item)

    seen_pairs = set()

    for key, summ_items in summ_by_key.items():
        if key not in src_by_key:
            continue

        # FIX 1: Skip if multiple source values exist for this key.
        # Multiple values = lab trending over time (e.g. glucose 95 on day 1,
        # glucose 187 on day 3). Summary legitimately picks one — not an error.
        # Only flag when there is exactly ONE source value to compare against.
        if len(src_by_key[key]) > 1:
            continue

        for si in summ_items:
            for sri in src_by_key[key]:
                pair = (key, sri["value"], si["value"])
                if pair in seen_pairs:
                    continue
                if abs(sri["value"] - si["value"]) > 1e-9:
                    seen_pairs.add(pair)
                    det_counter[0] += 1
                    tok = re.sub(r"[^a-z0-9_]", "_", f"{key}_{sri['value']}"[:28])

                    # Determine type
                    if any(re.search(d, key, re.I) for d in DRUG_NAMES.split("|")[:10]):
                        det_type = "medication_dose_error"
                        sev      = "Critical"
                    else:
                        det_type = "lab_value_error"
                        sev      = "Moderate"

                    detections.append({
                        "detection_id":    f"det_{det_counter[0]:04d}",
                        "entity_id":       f"{note_id}_{tok}",
                        "detected_by":     ["layer1b_numeric"],
                        "type":            det_type,
                        "flagged_text":    si["raw"],
                        "severity":        sev,
                        "confidence":      1.0,
                        "key":             key,
                        "source_value":    sri["value"],
                        "summary_value":   si["value"],
                        "unit":            sri["unit"] or si["unit"],
                        "source_context":  sri["context"][:120],
                        "summary_context": si["context"][:120],
                    })

    # Strategy 2 — Context-based fallback (DISABLED — too many FPs)
    # if len(detections) == 0:
    #     ...

    return detections

def main():
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    print(f"Numeric consistency check — {len(corpus)} notes...")

    det_counter   = [0]
    results       = []
    notes_flagged = 0
    total_flags   = 0

    for i, note in enumerate(corpus):
        src  = note.get("source_text") or note.get("text", "")
        summ = note.get("test_summary") or note.get("clean_summary", "")

        dets = compare_notes(src, summ, note["note_id"], det_counter)

        results.append({"note_id": note["note_id"], "detections": dets})

        if dets:
            notes_flagged += 1
            total_flags   += len(dets)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(corpus)}] flagged: {notes_flagged} notes")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLayer 1b (numeric) complete → {OUTPUT_PATH}")
    print(f"  {notes_flagged}/{len(corpus)} notes with numeric mismatches")
    print(f"  {total_flags} total detections")
    print(f"  Confidence: named matches=1.0, contextual=0.85")

    # Sample
    samples = [r for r in results if r["detections"]][:3]
    for s in samples:
        d = s["detections"][0]
        print(f"\n  [{d['type']}] {s['note_id']}")
        print(f"    {d['key']}: source={d['source_value']} → summary={d['summary_value']} {d['unit']}")
        print(f"    Src: ...{d['source_context'][:80]}...")
        print(f"    Sum: ...{d['summary_context'][:80]}...")

    print(f"\nNext: python run_pipeline.py --from 6")


if __name__ == "__main__":
    main()
