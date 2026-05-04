"""
evaluate_ehr.py — Complete MIMIC evaluation for HaloCheck

PRIMARY evaluation: Text-based verification
  Each HaloCheck detection is verified against the SOURCE NOTE TEXT
  using NER entity matching (Layer 1 style) and keyword contradiction
  detection. The source note IS the ground truth.

SECONDARY evaluation: EHR structured table verification
  For numeric/lab/medication detections, additionally checks against
  structured EHR tables (LABEVENTS, PRESCRIPTIONS, DIAGNOSES).
  This is a bonus check — not the primary evaluation.

Outputs:
  evaluation/results_ehr.json   — full results
  Terminal summary showing both layers of evaluation

Run:
  py evaluation/evaluate_ehr.py
  py evaluation/evaluate_ehr.py --corpus data/corpus_mimic.json
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter

# ── Config ────────────────────────────────────────────────────────────────────

CORPUS_PATH     = Path("data/corpus_mimic.json")
DETECTIONS_PATH = Path("cache/merged_detections.json")
OUTPUT_PATH     = Path("evaluation/results_ehr.json")

# Abnormal lab thresholds
ABNORMAL_THRESHOLDS = {
    "inr":         (0.8,  1.2),
    "creatinine":  (0.6,  1.2),
    "glucose":     (70,   100),
    "potassium":   (3.5,  5.0),
    "sodium":      (136,  145),
    "hemoglobin":  (12.0, 17.5),
    "hematocrit":  (36,   50),
    "wbc":         (4.5,  11.0),
    "platelets":   (150,  400),
    "troponin":    (0,    0.04),
    "bnp":         (0,    100),
    "lactate":     (0.5,  2.2),
    "bilirubin":   (0.1,  1.2),
    "albumin":     (3.5,  5.0),
    "calcium":     (8.5,  10.5),
    "magnesium":   (1.7,  2.2),
    "tsh":         (0.4,  4.0),
}

# Contradiction word pairs for text-based verification
CONTRADICTION_PAIRS = [
    ("improved",        "deteriorated"),
    ("deteriorated",    "improved"),
    ("stable",          "unstable"),
    ("unstable",        "stable"),
    ("resolved",        "persisted"),
    ("persisted",       "resolved"),
    ("normal",          "abnormal"),
    ("abnormal",        "normal"),
    ("benign",          "malignant"),
    ("malignant",       "benign"),
    ("negative",        "positive"),
    ("positive",        "negative"),
    ("decreased",       "increased"),
    ("increased",       "decreased"),
    ("no fever",        "fever"),
    ("afebrile",        "fever"),
    ("no pain",         "pain"),
    ("alert",           "unresponsive"),
    ("responsive",      "unresponsive"),
    ("tolerating",      "intolerant"),
    ("improving",       "worsening"),
    ("worsening",       "improving"),
    ("intact",          "impaired"),
    ("present",         "absent"),
    ("absent",          "present"),
]

# ── Text helpers ──────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def extract_numbers(text: str) -> list:
    return [float(m) for m in re.findall(r"\b\d+\.?\d*\b", text)]


def is_abnormal(lab_name: str, value_str: str) -> bool:
    lab_lower = lab_name.lower()
    nums = extract_numbers(str(value_str))
    if not nums:
        return False
    val = nums[0]
    for key, (lo, hi) in ABNORMAL_THRESHOLDS.items():
        if key in lab_lower:
            return val < lo or val > hi
    return False


# ── PRIMARY: Text-based verification ─────────────────────────────────────────

def verify_against_source_text(flagged_text: str, source_text: str, det_type: str) -> str:
    """
    PRIMARY verification method.
    Checks if the flagged summary text contradicts the source note.

    Returns: 'text_confirmed', 'text_refuted', or 'text_unverifiable'
    """
    if not flagged_text or not source_text:
        return "text_unverifiable"

    flagged_lower = normalize(flagged_text)
    source_lower  = normalize(source_text)

    # ── Contradiction pair check ──────────────────────────────────────────────
    for word_a, word_b in CONTRADICTION_PAIRS:
        if word_a in flagged_lower and word_b in source_lower:
            return "text_confirmed"

    # ── Numeric value mismatch check ──────────────────────────────────────────
    flagged_nums = extract_numbers(flagged_text)
    if flagged_nums:
        for num in flagged_nums:
            # If the number appears in source but significantly different
            source_nums = extract_numbers(source_text)
            for src_num in source_nums:
                if src_num == 0 or num == 0:
                    continue
                ratio = max(num, src_num) / min(num, src_num)
                if ratio > 2.0:   # more than 2x difference
                    return "text_confirmed"
                elif ratio < 1.1:  # essentially same number
                    return "text_refuted"

    # ── Entity presence check for fabrication/omission ────────────────────────
    if det_type == "diagnosis_fabrication":
        # If flagged entity appears in source → not fabricated → refuted
        words = [w for w in flagged_lower.split() if len(w) > 4]
        matches = sum(1 for w in words if w in source_lower)
        if matches >= 2:
            return "text_refuted"
        elif matches == 0 and len(words) >= 2:
            return "text_confirmed"  # entity not in source → fabricated

    if det_type == "critical_omission":
        # If flagged entity IS in source → omission is real → confirmed
        words = [w for w in flagged_lower.split() if len(w) > 4]
        matches = sum(1 for w in words if w in source_lower)
        if matches >= 2:
            return "text_confirmed"  # was in source, missing from summary

    if det_type in ("allergy_omission",):
        # Check if allergy mentioned in source
        allergy_keywords = ["allerg", "allergic", "reaction to", "sensitive to"]
        if any(kw in source_lower for kw in allergy_keywords):
            if any(kw in flagged_lower for kw in allergy_keywords):
                return "text_confirmed"

    # ── Normal/abnormal label check ───────────────────────────────────────────
    if det_type in ("lab_interpretation_error", "lab_value_error"):
        if "normal" in flagged_lower or "wnl" in flagged_lower:
            # Summary says normal — check if source says otherwise
            abnormal_words = ["elevated", "high", "low", "critical",
                              "abnormal", "increased", "decreased",
                              "above normal", "below normal"]
            if any(w in source_lower for w in abnormal_words):
                return "text_confirmed"

    return "text_unverifiable"


# ── SECONDARY: EHR table verification ────────────────────────────────────────

def check_lab_claim(flagged_text: str, ehr_labs: list) -> str:
    flagged_lower = normalize(flagged_text)
    for lab in ehr_labs:
        lab_name  = lab.get("name", "")
        lab_value = str(lab.get("value", ""))
        lab_flag  = lab.get("flag", "").lower()
        if not lab_name or not lab_value:
            continue
        lab_lower = lab_name.lower()
        if lab_lower not in flagged_lower and lab_lower.split()[0] not in flagged_lower:
            continue
        if lab_flag in ("abnormal", "high", "low", "critical", "panic"):
            if any(word in flagged_lower for word in ["normal", "wnl", "within normal"]):
                return "ehr_confirmed"
        flagged_nums = extract_numbers(flagged_text)
        ehr_nums     = extract_numbers(lab_value)
        if flagged_nums and ehr_nums:
            if abs(flagged_nums[0] - ehr_nums[0]) > 0.1:
                return "ehr_confirmed"
            else:
                return "ehr_refuted"
        if is_abnormal(lab_name, lab_value):
            if any(word in flagged_lower for word in ["normal", "stable", "wnl"]):
                return "ehr_confirmed"
    return "ehr_unverifiable"


def check_med_claim(flagged_text: str, ehr_meds: list) -> str:
    flagged_lower = normalize(flagged_text)
    for med in ehr_meds:
        drug = normalize(med.get("drug", ""))
        dose = med.get("dose", "")
        if not drug or len(drug) < 4:
            continue
        drug_word = drug.split()[0]
        if drug_word not in flagged_lower:
            continue
        flagged_nums = extract_numbers(flagged_text)
        ehr_nums     = extract_numbers(dose)
        if flagged_nums and ehr_nums:
            ratio = max(flagged_nums[0], ehr_nums[0]) / (min(flagged_nums[0], ehr_nums[0]) + 0.001)
            if ratio > 1.5:
                return "ehr_confirmed"
            else:
                return "ehr_refuted"
        return "ehr_refuted"
    return "ehr_unverifiable"


def check_dx_claim(flagged_text: str, ehr_dx: list) -> str:
    flagged_lower = normalize(flagged_text)
    for dx in ehr_dx:
        desc      = normalize(dx.get("description", ""))
        desc_words = [w for w in desc.split() if len(w) > 4]
        matches    = sum(1 for w in desc_words if w in flagged_lower)
        if matches >= 2:
            return "ehr_refuted"
    return "ehr_unverifiable"


def verify_against_ehr(detection: dict, ehr_truth: dict) -> str:
    """
    SECONDARY verification against structured EHR tables.
    Only called after text verification returns unverifiable.
    """
    det_type     = detection.get("type", "")
    flagged_text = detection.get("flagged_text", "") or detection.get("entity_id", "")
    ehr_labs     = ehr_truth.get("labs", [])
    ehr_meds     = ehr_truth.get("meds", [])
    ehr_dx       = ehr_truth.get("dx", [])

    # Drug interactions — not verifiable from EHR tables
    if det_type == "drug_interaction":
        return "ehr_unverifiable"

    if det_type in ("lab_value_error", "numeric_mismatch",
                    "abnormal_value", "lab_interpretation_error"):
        return check_lab_claim(flagged_text, ehr_labs)

    elif det_type in ("medication_dose_error", "dosage_unit_swap",
                      "dosing_frequency_error"):
        result = check_med_claim(flagged_text, ehr_meds)
        if result == "ehr_unverifiable":
            result = check_lab_claim(flagged_text, ehr_labs)
        return result

    elif det_type in ("diagnosis_fabrication", "critical_omission"):
        result = check_dx_claim(flagged_text, ehr_dx)
        if result == "ehr_unverifiable":
            result = check_med_claim(flagged_text, ehr_meds)
        return result

    elif det_type == "semantic_contradiction":
        result = check_lab_claim(flagged_text, ehr_labs)
        if result == "ehr_unverifiable":
            result = check_med_claim(flagged_text, ehr_meds)
        return result

    else:
        for checker, data in [(check_lab_claim, ehr_labs),
                               (check_med_claim, ehr_meds),
                               (check_dx_claim,  ehr_dx)]:
            result = checker(flagged_text, data)
            if result != "ehr_unverifiable":
                return result
        return "ehr_unverifiable"


# ── Combined verdict ──────────────────────────────────────────────────────────

def get_final_verdict(text_verdict: str, ehr_verdict: str) -> str:
    """
    Combine text-based and EHR-based verdicts into one final verdict.

    Priority:
    1. If either confirms → confirmed
    2. If either refutes (and none confirms) → refuted
    3. If both unverifiable → unverifiable
    """
    if "confirmed" in text_verdict or "confirmed" in ehr_verdict:
        return "confirmed"
    if "refuted" in text_verdict or "refuted" in ehr_verdict:
        return "refuted"
    return "unverifiable"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",     type=Path, default=CORPUS_PATH)
    parser.add_argument("--detections", type=Path, default=DETECTIONS_PATH)
    parser.add_argument("--out",        type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.corpus.exists():
        raise FileNotFoundError(f"\n❌  Corpus not found: {args.corpus}\n")
    with open(args.corpus, encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_by_id = {n["note_id"]: n for n in corpus}

    if not args.detections.exists():
        raise FileNotFoundError(
            f"\n❌  Detections not found: {args.detections}\n"
            "Run: py run_pipeline.py --from 3\n"
        )
    with open(args.detections, encoding="utf-8") as f:
        detections_raw = json.load(f)
    detections_by_id = {d["note_id"]: d for d in detections_raw}

    print(f"\nHaloCheck MIMIC Evaluation")
    print(f"{'='*60}")
    print(f"  Corpus notes             : {len(corpus)}")
    print(f"  Notes with detections    : {len(detections_by_id)}")
    print(f"\nEvaluation approach:")
    print(f"  PRIMARY   — Text comparison (source note vs GPT summary)")
    print(f"  SECONDARY — EHR structured tables (labs, meds, diagnoses)")
    print(f"  COMBINED  — Either method confirming = confirmed\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────

    per_case         = []
    all_verdicts     = Counter()
    text_verdicts    = Counter()
    ehr_verdicts     = Counter()
    by_type          = defaultdict(Counter)
    by_type_text     = defaultdict(Counter)
    by_type_ehr      = defaultdict(Counter)

    # Separate drug interaction counts
    drug_interaction_count = 0

    for note_id, note in corpus_by_id.items():
        ehr_truth  = note.get("ehr_truth", {})
        source_text = note.get("source_text", "")
        note_dets  = detections_by_id.get(note_id, {}).get("detections", [])

        case_results = []
        for det in note_dets:
            det_type     = det.get("type", "unknown")
            flagged_text = det.get("flagged_text", "") or det.get("entity_id", "")

            # Count drug interactions separately
            if det_type == "drug_interaction":
                drug_interaction_count += 1
                case_results.append({
                    "type":         det_type,
                    "severity":     det.get("severity", ""),
                    "flagged_text": flagged_text[:100],
                    "text_verdict": "drug_interaction",
                    "ehr_verdict":  "drug_interaction",
                    "verdict":      "pharmacist_review",
                    "confidence":   det.get("confidence", 0)
                })
                continue

            # PRIMARY: text-based verification
            text_v = verify_against_source_text(flagged_text, source_text, det_type)
            text_verdicts[text_v] += 1
            by_type_text[det_type][text_v] += 1

            # SECONDARY: EHR verification
            ehr_v = verify_against_ehr(det, ehr_truth)
            ehr_verdicts[ehr_v] += 1
            by_type_ehr[det_type][ehr_v] += 1

            # COMBINED verdict
            verdict = get_final_verdict(text_v, ehr_v)
            all_verdicts[verdict] += 1
            by_type[det_type][verdict] += 1

            case_results.append({
                "type":         det_type,
                "severity":     det.get("severity", ""),
                "flagged_text": flagged_text[:100],
                "text_verdict": text_v,
                "ehr_verdict":  ehr_v,
                "verdict":      verdict,
                "confidence":   det.get("confidence", 0)
            })

        per_case.append({
            "note_id":      note_id,
            "n_dets":       len(note_dets),
            "results":      case_results,
            "confirmed":    sum(1 for r in case_results if r["verdict"] == "confirmed"),
            "refuted":      sum(1 for r in case_results if r["verdict"] == "refuted"),
            "unverifiable": sum(1 for r in case_results if r["verdict"] == "unverifiable"),
            "pharmacist_review": sum(1 for r in case_results if r["verdict"] == "pharmacist_review"),
        })

    # ── Stats ─────────────────────────────────────────────────────────────────

    total_dets    = sum(all_verdicts.values())
    confirmed     = all_verdicts["confirmed"]
    refuted       = all_verdicts["refuted"]
    unverifiable  = all_verdicts["unverifiable"]

    # Text-based stats
    text_confirmed    = text_verdicts.get("text_confirmed", 0)
    text_refuted      = text_verdicts.get("text_refuted", 0)
    text_unverifiable = text_verdicts.get("text_unverifiable", 0)

    # EHR stats
    ehr_confirmed    = ehr_verdicts.get("ehr_confirmed", 0)
    ehr_refuted      = ehr_verdicts.get("ehr_refuted", 0)
    ehr_unverifiable = ehr_verdicts.get("ehr_unverifiable", 0)

    precision    = confirmed / (confirmed + refuted) if (confirmed + refuted) > 0 else 0
    verify_rate  = (confirmed + refuted) / total_dets if total_dets > 0 else 0
    text_prec    = text_confirmed / (text_confirmed + text_refuted) if (text_confirmed + text_refuted) > 0 else 0
    ehr_prec     = ehr_confirmed / (ehr_confirmed + ehr_refuted) if (ehr_confirmed + ehr_refuted) > 0 else 0

    total_all = total_dets + drug_interaction_count

    summary = {
        "total_detections_all":        total_all,
        "drug_interaction_flags":      drug_interaction_count,
        "text_verifiable_detections":  total_dets,
        "confirmed":                   confirmed,
        "refuted":                     refuted,
        "unverifiable":                unverifiable,
        "combined_precision":          round(precision, 4),
        "verification_rate":           round(verify_rate, 4),
        "text_based": {
            "confirmed":   text_confirmed,
            "refuted":     text_refuted,
            "unverifiable": text_unverifiable,
            "precision":   round(text_prec, 4),
        },
        "ehr_based": {
            "confirmed":   ehr_confirmed,
            "refuted":     ehr_refuted,
            "unverifiable": ehr_unverifiable,
            "precision":   round(ehr_prec, 4),
        },
    }

    output = {
        "summary":      summary,
        "n_cases":      len(per_case),
        "n_detections": total_all,
        "by_type":      {k: dict(v) for k, v in by_type.items()},
        "by_type_text": {k: dict(v) for k, v in by_type_text.items()},
        "by_type_ehr":  {k: dict(v) for k, v in by_type_ehr.items()},
        "per_case":     per_case,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # ── Print full report ─────────────────────────────────────────────────────

    print(f"{'='*60}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total flags raised         : {total_all}")
    print(f"  ├─ Drug interaction flags  : {drug_interaction_count} → pharmacist review")
    print(f"  └─ Text-verifiable flags   : {total_dets}")
    print()

    print(f"PRIMARY EVALUATION — Source Note vs GPT Summary")
    print(f"{'─'*60}")
    print(f"  Text confirmed   : {text_confirmed}")
    print(f"  Text refuted     : {text_refuted}")
    print(f"  Text unverifiable: {text_unverifiable}")
    print(f"  Text precision   : {text_prec:.3f}")
    print()

    print(f"SECONDARY EVALUATION — EHR Structured Tables")
    print(f"{'─'*60}")
    print(f"  EHR confirmed    : {ehr_confirmed}")
    print(f"  EHR refuted      : {ehr_refuted}")
    print(f"  EHR unverifiable : {ehr_unverifiable}")
    print(f"  EHR precision    : {ehr_prec:.3f}")
    print()

    print(f"COMBINED RESULT (either method confirming = confirmed)")
    print(f"{'─'*60}")
    print(f"  Confirmed        : {confirmed} ({confirmed/total_dets*100:.1f}%)" if total_dets else "")
    print(f"  Refuted          : {refuted} ({refuted/total_dets*100:.1f}%)" if total_dets else "")
    print(f"  Unverifiable     : {unverifiable} ({unverifiable/total_dets*100:.1f}%)" if total_dets else "")
    print(f"  Combined precision: {precision:.3f}")
    print(f"  Verification rate : {verify_rate:.3f}")
    print()

    print(f"BY DETECTION TYPE (combined verdict)")
    print(f"{'─'*60}")
    print(f"  {'Type':<35} {'Conf':>6} {'Ref':>6} {'Unv':>6}")
    print(f"  {'-'*55}")
    for det_type, counts in sorted(by_type.items()):
        c = counts.get("confirmed", 0)
        r = counts.get("refuted", 0)
        u = counts.get("unverifiable", 0)
        print(f"  {det_type:<35} {c:>6} {r:>6} {u:>6}")
    print(f"  {'drug_interaction (pharmacist review)':<35} {drug_interaction_count:>6}")
    print()

    print(f"CLINICAL ROUTING SUMMARY")
    print(f"{'─'*60}")
    print(f"  Auto-confirmed (immediate review)  : {confirmed}")
    print(f"  Auto-refuted (dismissed)           : {refuted}")
    print(f"  Requires clinician review          : {unverifiable}")
    print(f"  Requires pharmacist review (DDI)   : {drug_interaction_count}")
    print(f"  Total flags for human review       : {unverifiable + drug_interaction_count}")
    print()

    print(f"{'='*60}")
    print(f"✓ Full results saved → {args.out}")
    print(f"\nNext: py -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()