"""
layer4_severity.py — Deduplicator + Severity Scorer
Week 2 / Person 4

Merges all layer outputs into unified detections per note.
Deduplicates by entity_id (multi-layer flags on same entity = 1 detection).
Assigns final severity tiers using SNOMED-mapped categories.
Writes merged detections back into corpus.json.

Reads:   data/corpus.json
         cache/layer1.json
         cache/layer2.json
         cache/layer3.json
Writes:  cache/merged_detections.json
         data/corpus_final.json  ← corpus with all detections populated

Run: python pipeline/layer4_severity.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

CORPUS_PATH    = Path("data/corpus.json")
LAYER1_PATH    = Path("cache/layer1.json")
LAYER2_PATH    = Path("cache/layer2.json")
LAYER3_PATH    = Path("cache/layer3.json")
LAYER2B_PATH   = Path("cache/layer2b.json")   # RAG DDI layer
LAYER1B_PATH   = Path("cache/layer1b.json")   # Numeric consistency layer
LAYER1C_PATH   = Path("cache/layer1c.json")   # Abnormal value layer
LAYER1D_PATH   = Path("cache/layer1d.json")   # Allergy omission layer
LAYER1E_PATH   = Path("cache/layer1e.json")   # Dosing frequency layer
MERGED_PATH    = Path("cache/merged_detections.json")
CORPUS_OUT     = Path("data/corpus_final.json")

# ── Severity mapping ──────────────────────────────────────────────────────────
#
# Maps hallucination type + entity category → severity tier
# Based on clinical risk taxonomy (SNOMED CT severity concepts)

SEVERITY_RULES = {
    # (type_keyword, entity_keyword) → severity
    ("medication_dose_error",  None):           "Critical",
    ("dosage_unit_swap",       None):           "Critical",
    ("negation_flip",          None):           "Critical",
    ("critical_omission",      None):           "Critical",
    ("diagnosis_fabrication",  "allerg"):       "Critical",
    ("diagnosis_fabrication",  "contraindic"):  "Critical",
    ("diagnosis_fabrication",  "penicillin"):   "Critical",
    ("diagnosis_fabrication",  "warfarin"):     "Critical",
    ("diagnosis_fabrication",  "heparin"):      "Critical",
    ("diagnosis_fabrication",  None):           "Moderate",
    ("lab_value_error",        None):           "Moderate",
    ("timeline_inversion",     None):           "Moderate",
    ("drug_interaction", "Critical"):           "Critical",
    ("drug_interaction", "Moderate"):           "Moderate",
    ("drug_interaction", None):                 "Critical",
    ("dosing_frequency_error", None):           "Moderate",
    (None,                     "critical"):     "Critical",
    (None,                     "allerg"):       "Critical",
    (None,                     None):           "Moderate",   # fallback
}


def assign_severity(detection: dict) -> str:
    """Apply severity rules to a detection. Returns 'Critical', 'Moderate', or 'Minor'."""
    det_type  = detection.get("type", "").lower()
    entity    = detection.get("entity_id", "").lower() + " " + detection.get("flagged_text", "").lower()

    for (type_kw, entity_kw), severity in SEVERITY_RULES.items():
        type_match   = (type_kw is None) or (type_kw in det_type)
        entity_match = (entity_kw is None) or (entity_kw in entity)
        if type_match and entity_match:
            return severity

    return "Moderate"  # default if no rules match


# ── Confidence fusion ─────────────────────────────────────────────────────────

def fuse_confidence(detections: list[dict]) -> float:
    """
    When multiple layers flag the same entity, fuse confidence scores.
    Uses: 1 - product of (1 - p_i) — noisy-OR combination.
    """
    if not detections:
        return 0.0
    scores = [d.get("confidence", 0.5) for d in detections]
    combined = 1.0 - 1.0
    product = 1.0
    for s in scores:
        product *= (1.0 - s)
    return round(1.0 - product, 4)


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(detections: list[dict]) -> list[dict]:
    """
    Merge detections that share the same entity_id.
    Multi-layer detections: combine detected_by, fuse confidence, apply severity.
    """
    grouped = defaultdict(list)
    for d in detections:
        grouped[d["entity_id"]].append(d)

    merged = []
    for entity_id, group in grouped.items():
        # Combine layers
        all_layers = []
        for d in group:
            all_layers.extend(d.get("detected_by", []))
        all_layers = sorted(set(all_layers))

        # Pick the detection with highest confidence as base
        base = max(group, key=lambda d: d.get("confidence", 0))

        merged_det = {
            "detection_id": base["detection_id"],
            "entity_id":    entity_id,
            "detected_by":  all_layers,
            "type":         base["type"],
            "flagged_text": base["flagged_text"],
            "severity":     assign_severity(base),
            "confidence":   fuse_confidence(group),
        }
        merged.append(merged_det)

    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load corpus and all layer outputs
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    def load_cache(path: Path) -> dict:
        """Load cache file → dict keyed by note_id."""
        if not path.exists():
            print(f"  [warn] {path} not found — skipping this layer")
            return {}
        with open(path) as f:
            data = json.load(f)
        return {r["note_id"]: r["detections"] for r in data}

    l1  = load_cache(LAYER1_PATH)
    l2  = load_cache(LAYER2_PATH)
    l3  = load_cache(LAYER3_PATH)
    l2b = load_cache(LAYER2B_PATH)
    l1b = load_cache(LAYER1B_PATH)
    l1c = load_cache(LAYER1C_PATH)
    l1d = load_cache(LAYER1D_PATH)
    l1e = load_cache(LAYER1E_PATH)

    print(f"Loaded: L1={len(l1)} L1b={len(l1b)} L1c={len(l1c)} L1d={len(l1d)} L1e={len(l1e)} L2={len(l2)} L2b={len(l2b)}")

    all_merged = []

    for note in corpus:
        nid = note["note_id"]

        # Collect all raw detections for this note across layers
        raw_detections = (
            l1.get(nid, []) +
            l2.get(nid, []) +
            l3.get(nid, []) +
            l2b.get(nid, []) +   # RAG DDI detections
            l1b.get(nid, []) +    # Numeric consistency detections
            l1c.get(nid, []) +      # Abnormal value detections
            l1d.get(nid, []) +       # Allergy omission detections
            l1e.get(nid, [])        # Dosing frequency detections
        )

        # Deduplicate + assign final severity
        merged = deduplicate(raw_detections)

        # Write back into corpus
        note["detections"] = merged

        all_merged.append({
            "note_id":    nid,
            "detections": merged,
        })

    # Save merged detections cache
    MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MERGED_PATH, "w") as f:
        json.dump(all_merged, f, indent=2)

    # Save updated corpus
    with open(CORPUS_OUT, "w") as f:
        json.dump(corpus, f, indent=2)

    total_det = sum(len(n["detections"]) for n in corpus)
    flagged   = sum(1 for n in corpus if n["detections"])
    critical  = sum(
        1 for n in corpus for d in n["detections"] if d["severity"] == "Critical"
    )
    moderate  = sum(
        1 for n in corpus for d in n["detections"] if d["severity"] == "Moderate"
    )
    minor     = sum(
        1 for n in corpus for d in n["detections"] if d["severity"] == "Minor"
    )

    print(f"\n✓ Layer 4 complete")
    print(f"  Merged detections → {MERGED_PATH}")
    print(f"  Final corpus      → {CORPUS_OUT}")
    print(f"\n  {flagged}/{len(corpus)} notes flagged | {total_det} total merged detections")
    print(f"  Severity: Critical={critical} | Moderate={moderate} | Minor={minor}")
    print(f"\nNext step: run pipeline/hhem_baseline.py")


if __name__ == "__main__":
    main()
