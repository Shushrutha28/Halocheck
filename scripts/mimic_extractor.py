"""
mimic_extractor.py — MIMIC-III corpus builder for HaloCheck

Reads MIMIC-III CSV files and builds a corpus.json with:
  - source_text: the full discharge summary
  - ehr_truth:   structured ground truth from EHR tables
    - labs:       lab test results (itemid, name, value, unit, flag)
    - meds:       medication orders (drug, dose, route, frequency)
    - dx:         diagnosis codes (icd9, description)
    - demographics: age, gender

Supports both:
  - MIMIC-III demo dataset (~100 patients, free, no PhysioNet auth)
  - MIMIC-III full dataset (59,652 notes, requires PhysioNet credentials)

Demo download:
  https://physionet.org/content/mimiciii-demo/1.4/

Full dataset:
  https://physionet.org/content/mimiciii/1.4/

Required MIMIC files:
  NOTEEVENTS.csv      — discharge summaries
  LABEVENTS.csv       — lab results
  PRESCRIPTIONS.csv   — medication orders
  DIAGNOSES_ICD.csv   — diagnosis codes
  D_ICD_DIAGNOSES.csv — ICD9 code descriptions
  D_LABITEMS.csv      — lab item descriptions
  ADMISSIONS.csv      — admission demographics

Run:
  python scripts/mimic_extractor.py
  python scripts/mimic_extractor.py --mimic_dir /path/to/mimic --max 100
"""

import csv
import json
import argparse
import os
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MIMIC_DIR = Path("data/mimic")
OUTPUT_PATH       = Path("data/corpus_mimic.json")
MAX_NOTES         = None   # None = all notes, set to 100 for quick test

# Lab items to extract (most clinically relevant)
KEY_LAB_ITEMS = {
    "INR", "PT", "PTT", "Creatinine", "BUN", "Glucose", "Sodium",
    "Potassium", "Chloride", "Bicarbonate", "Hemoglobin", "Hematocrit",
    "WBC", "Platelets", "Troponin", "BNP", "Lactate", "pH", "pO2", "pCO2",
    "ALT", "AST", "Bilirubin", "Albumin", "Calcium", "Magnesium",
    "Phosphate", "TSH", "HbA1c", "PSA", "CRP", "ESR", "D-dimer",
    "Fibrinogen", "LDH", "Lipase", "Amylase", "GFR"
}


def load_csv(path: Path, encoding: str = "utf-8") -> list:
    """Load a CSV file and return list of dicts."""
    if not path.exists():
        print(f"  [warn] file not found: {path}")
        return []
    rows = []
    try:
        with open(path, encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"  [error] reading {path}: {e}")
    return rows


def build_lab_lookup(mimic_dir: Path) -> dict:
    """Build subject_id → list of lab results. Streams file row by row — no full load."""
    print("  Loading lab items dictionary...")
    d_labitems = load_csv(mimic_dir / "D_LABITEMS.csv")
    item_names = {row["ITEMID"]: row["LABEL"] for row in d_labitems}

    # Build a set of key item IDs to filter — much faster than string matching per row
    key_item_ids = set()
    for itemid, label in item_names.items():
        if any(key.lower() in label.lower() for key in KEY_LAB_ITEMS):
            key_item_ids.add(itemid)
    print(f"  Filtering to {len(key_item_ids)} key lab item IDs")

    lab_path = mimic_dir / "LABEVENTS.csv"
    if not lab_path.exists():
        print(f"  [warn] LABEVENTS.csv not found")
        return {}

    print("  Streaming lab events (this file has 27M rows — streaming to avoid memory issues)...")
    labs_by_subject = defaultdict(list)
    seen_per_subject = defaultdict(set)  # deduplicate per subject+itemid
    row_count = 0

    with open(lab_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            if row_count % 1_000_000 == 0:
                print(f"    ...processed {row_count:,} rows, {len(labs_by_subject)} subjects so far")

            itemid = row.get("ITEMID", "")
            if itemid not in key_item_ids:
                continue

            subject_id = row.get("SUBJECT_ID", "")
            value      = row.get("VALUE", "").strip()
            valuenum   = row.get("VALUENUM", "").strip()
            valueuom   = row.get("VALUEUOM", "").strip()
            flag       = row.get("FLAG", "").strip()
            label      = item_names.get(itemid, itemid)

            if not value and not valuenum:
                continue

            # Keep only first occurrence per subject+item to avoid huge lists
            dedup_key = f"{subject_id}_{itemid}"
            if dedup_key in seen_per_subject[subject_id]:
                continue
            seen_per_subject[subject_id].add(dedup_key)

            labs_by_subject[subject_id].append({
                "itemid": itemid,
                "name":   label,
                "value":  valuenum if valuenum else value,
                "unit":   valueuom,
                "flag":   flag.lower() if flag else "normal"
            })

    print(f"  Streamed {row_count:,} rows → labs for {len(labs_by_subject)} subjects")
    return labs_by_subject


def build_med_lookup(mimic_dir: Path) -> dict:
    """Build subject_id → list of medications."""
    print("  Loading prescriptions...")
    prescriptions = load_csv(mimic_dir / "PRESCRIPTIONS.csv")

    meds_by_subject = defaultdict(list)
    seen = defaultdict(set)   # deduplicate per subject

    for row in prescriptions:
        subject_id = row.get("SUBJECT_ID", "")
        drug       = row.get("DRUG", "").strip()
        dose_val   = row.get("DOSE_VAL_RX", "").strip()
        dose_unit  = row.get("DOSE_UNIT_RX", "").strip()
        route      = row.get("ROUTE", "").strip()
        frequency  = row.get("FREQUENCY", "").strip() or row.get("FREQ", "").strip()

        if not drug:
            continue

        key = f"{drug}_{dose_val}_{dose_unit}"
        if key in seen[subject_id]:
            continue
        seen[subject_id].add(key)

        meds_by_subject[subject_id].append({
            "drug":      drug,
            "dose":      f"{dose_val} {dose_unit}".strip(),
            "route":     route,
            "frequency": frequency
        })

    print(f"  Loaded meds for {len(meds_by_subject)} subjects")
    return meds_by_subject


def build_dx_lookup(mimic_dir: Path) -> dict:
    """Build subject_id → list of diagnoses."""
    print("  Loading ICD9 descriptions...")
    d_icd = load_csv(mimic_dir / "D_ICD_DIAGNOSES.csv")
    icd_desc = {row["ICD9_CODE"]: row["LONG_TITLE"] for row in d_icd}

    print("  Loading diagnoses...")
    diagnoses = load_csv(mimic_dir / "DIAGNOSES_ICD.csv")

    dx_by_subject = defaultdict(list)
    seen = defaultdict(set)

    for row in diagnoses:
        subject_id = row.get("SUBJECT_ID", "")
        icd9       = row.get("ICD9_CODE", "").strip()
        desc       = icd_desc.get(icd9, icd9)
        seq        = row.get("SEQ_NUM", "99")

        if icd9 in seen[subject_id]:
            continue
        seen[subject_id].add(icd9)

        dx_by_subject[subject_id].append({
            "icd9":        icd9,
            "description": desc,
            "seq":         int(seq) if seq.isdigit() else 99
        })

    # Sort by sequence number (primary diagnosis first)
    for subject_id in dx_by_subject:
        dx_by_subject[subject_id].sort(key=lambda x: x["seq"])

    print(f"  Loaded diagnoses for {len(dx_by_subject)} subjects")
    return dx_by_subject


def build_demographics_lookup(mimic_dir: Path) -> dict:
    """Build subject_id → demographics."""
    print("  Loading admissions...")
    admissions = load_csv(mimic_dir / "ADMISSIONS.csv")

    demo_by_subject = {}
    for row in admissions:
        subject_id = row.get("SUBJECT_ID", "")
        if subject_id not in demo_by_subject:
            demo_by_subject[subject_id] = {
                "gender":            row.get("GENDER", ""),
                "admission_type":    row.get("ADMISSION_TYPE", ""),
                "insurance":         row.get("INSURANCE", ""),
                "marital_status":    row.get("MARITAL_STATUS", ""),
                "ethnicity":         row.get("ETHNICITY", ""),
            }

    print(f"  Loaded demographics for {len(demo_by_subject)} subjects")
    return demo_by_subject


def extract_notes(mimic_dir: Path, max_notes: int = None) -> list:
    """Extract discharge summaries from NOTEEVENTS."""
    print("  Loading discharge summaries...")
    noteevents = load_csv(mimic_dir / "NOTEEVENTS.csv")

    notes = []
    for row in noteevents:
        if row.get("CATEGORY", "").lower() != "discharge summary":
            continue
        if row.get("ISERROR", "0") == "1":
            continue

        text = row.get("TEXT", "").strip()
        if len(text) < 200:   # skip very short notes
            continue

        notes.append({
            "subject_id": row.get("SUBJECT_ID", ""),
            "hadm_id":    row.get("HADM_ID", ""),
            "note_id":    f"mimic_{row.get('ROW_ID', '')}",
            "text":       text
        })

        if max_notes and len(notes) >= max_notes:
            break

    print(f"  Found {len(notes)} discharge summaries")
    return notes


def main():
    parser = argparse.ArgumentParser(description="MIMIC-III corpus extractor for HaloCheck")
    parser.add_argument("--mimic_dir", type=Path, default=DEFAULT_MIMIC_DIR,
                        help=f"Path to MIMIC-III CSV files (default: {DEFAULT_MIMIC_DIR})")
    parser.add_argument("--max", type=int, default=MAX_NOTES,
                        help="Maximum number of notes to process (default: all)")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH,
                        help=f"Output path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    mimic_dir = args.mimic_dir
    max_notes = args.max
    out_path  = args.out

    print(f"\nMIMIC-III Corpus Extractor")
    print(f"  MIMIC directory : {mimic_dir}")
    print(f"  Max notes       : {max_notes or 'all'}")
    print(f"  Output          : {out_path}\n")

    if not mimic_dir.exists():
        raise FileNotFoundError(
            f"\n❌  MIMIC directory not found: {mimic_dir}\n"
            "Download MIMIC-III demo (free) from:\n"
            "  https://physionet.org/content/mimiciii-demo/1.4/\n"
            "Place CSV files in: data/mimic/\n"
        )

    # Check required files
    required = ["NOTEEVENTS.csv", "LABEVENTS.csv", "PRESCRIPTIONS.csv",
                "DIAGNOSES_ICD.csv", "D_ICD_DIAGNOSES.csv", "D_LABITEMS.csv"]
    missing = [f for f in required if not (mimic_dir / f).exists()]
    if missing:
        print(f"⚠️  Missing files: {missing}")
        print("   Continuing with available files...\n")

    print("Building lookup tables...")
    labs_lookup  = build_lab_lookup(mimic_dir)
    meds_lookup  = build_med_lookup(mimic_dir)
    dx_lookup    = build_dx_lookup(mimic_dir)
    demo_lookup  = build_demographics_lookup(mimic_dir)

    print("\nExtracting discharge summaries...")
    notes = extract_notes(mimic_dir, max_notes)

    print(f"\nBuilding corpus with EHR ground truth...")
    corpus = []
    for i, note in enumerate(notes):
        subject_id = note["subject_id"]

        corpus.append({
            "note_id":      note["note_id"],
            "subject_id":   subject_id,
            "hadm_id":      note["hadm_id"],
            "source_text":  note["text"],
            "test_summary": "",   # filled by generate_handoff.py
            "is_injected":  False,
            "injections":   [],
            "detections":   [],
            "hhem_score":   None,
            "ehr_truth": {
                "labs":         labs_lookup.get(subject_id, [])[:20],   # top 20 labs
                "meds":         meds_lookup.get(subject_id, [])[:20],   # top 20 meds
                "dx":           dx_lookup.get(subject_id, [])[:10],     # top 10 diagnoses
                "demographics": demo_lookup.get(subject_id, {})
            }
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(notes)} notes...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    # Stats
    notes_with_labs  = sum(1 for n in corpus if n["ehr_truth"]["labs"])
    notes_with_meds  = sum(1 for n in corpus if n["ehr_truth"]["meds"])
    notes_with_dx    = sum(1 for n in corpus if n["ehr_truth"]["dx"])

    print(f"\n{'='*50}")
    print(f"✓ Corpus saved → {out_path}")
    print(f"  Total notes     : {len(corpus)}")
    print(f"  With lab data   : {notes_with_labs}")
    print(f"  With med data   : {notes_with_meds}")
    print(f"  With dx data    : {notes_with_dx}")
    print(f"\nNext: python scripts/generate_handoff.py")


if __name__ == "__main__":
    main()