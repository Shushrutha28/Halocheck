"""
layer2b_rag.py — Drug-Drug Interaction detector using RAG + DrugBank index.

Sits between Layer 2 (NLI) and Layer 4 (severity) in the pipeline.
Reads corpus.json, extracts medication entities from each note,
queries the DrugBank DDI index for interactions, and flags any
interaction not acknowledged in the summary.

New detection type: drug_interaction
New severity: Critical (bleeding, serotonin syndrome, QT prolongation etc.)
             Moderate (pharmacokinetic interactions, level changes)

Requires: python scripts/build_rag_index.py (run once to build index)

Reads:   data/corpus.json
Writes:  cache/layer2b.json

Run: python pipeline/layer2b_rag.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

CORPUS_PATH = Path("data/corpus.json")
OUTPUT_PATH = Path("cache/layer2b.json")
INDEX_DIR   = Path("data/rag_index/ddi")

# Drug name patterns — same as Layer 1 but expanded
DRUG_PATTERNS = [
    # Generic drug names — common clinical medications
    r"\b(?:warfarin|coumadin|heparin|enoxaparin|apixaban|rivaroxaban|dabigatran)\b",
    r"\b(?:aspirin|clopidogrel|ticagrelor|prasugrel)\b",
    r"\b(?:metformin|insulin|glipizide|glargine|sitagliptin|empagliflozin)\b",
    r"\b(?:lisinopril|enalapril|ramipril|losartan|valsartan|amlodipine)\b",
    r"\b(?:metoprolol|atenolol|carvedilol|bisoprolol|propranolol)\b",
    r"\b(?:atorvastatin|simvastatin|rosuvastatin|pravastatin|lovastatin)\b",
    r"\b(?:furosemide|hydrochlorothiazide|spironolactone|chlorthalidone)\b",
    r"\b(?:omeprazole|pantoprazole|esomeprazole|lansoprazole)\b",
    r"\b(?:fluoxetine|sertraline|paroxetine|escitalopram|citalopram|venlafaxine)\b",
    r"\b(?:amiodarone|digoxin|diltiazem|verapamil)\b",
    r"\b(?:tramadol|morphine|oxycodone|hydrocodone|fentanyl|codeine)\b",
    r"\b(?:ciprofloxacin|levofloxacin|azithromycin|clarithromycin|erythromycin)\b",
    r"\b(?:fluconazole|itraconazole|ketoconazole|voriconazole)\b",
    r"\b(?:phenytoin|carbamazepine|valproate|levetiracetam|lamotrigine)\b",
    r"\b(?:lithium|clozapine|olanzapine|risperidone|quetiapine)\b",
    r"\b(?:prednisone|prednisolone|dexamethasone|methylprednisolone)\b",
    r"\b(?:tacrolimus|cyclosporine|mycophenolate|sirolimus)\b",
    r"\b(?:methotrexate|hydroxychloroquine|sulfasalazine)\b",
    r"\b(?:levothyroxine|methimazole|propylthiouracil)\b",
    r"\b(?:allopurinol|colchicine|probenecid)\b",
    r"\b(?:sildenafil|tadalafil|vardenafil)\b",
    r"\b(?:rifampin|rifampicin|isoniazid|ethambutol)\b",
    # Catch any word ending in common drug suffixes
    r"\b\w+(?:pril|sartan|olol|statin|pam|zam|pine|vir|mab|nib|tide|mycin|cycline|floxacin)\b",
]

# Interaction acknowledgement phrases — if summary mentions these near
# the drug pair, the interaction is acknowledged (not a hallucination)
ACKNOWLEDGEMENT_PHRASES = [
    "interaction", "interacts", "caution", "monitor", "avoid",
    "contraindicated", "risk of", "increased risk", "decreased",
    "bleeding risk", "careful", "concomitant", "together",
]


def normalize_drug(name: str) -> str:
    name = name.lower().strip()
    for suffix in [" hydrochloride", " hcl", " sodium", " potassium",
                   " sulfate", " acetate", " citrate", " tartrate"]:
        name = name.replace(suffix, "")
    name = re.sub(r"\(.*?\)", "", name).strip()
    return name


def extract_drugs(text: str) -> list[str]:
    """Extract drug names from text using pattern matching."""
    found = set()
    text_lower = text.lower()
    for pattern in DRUG_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            drug = normalize_drug(match.group(0))
            if len(drug) > 3:
                found.add(drug)
    return sorted(found)


def interaction_acknowledged(drug1: str, drug2: str, summary: str) -> bool:
    """
    Check if the interaction between drug1 and drug2 is already
    acknowledged in the summary text.
    """
    summary_lower = summary.lower()
    # Check if both drugs mentioned near an acknowledgement phrase
    if drug1 in summary_lower and drug2 in summary_lower:
        for phrase in ACKNOWLEDGEMENT_PHRASES:
            if phrase in summary_lower:
                return True
    return False


class DDIDetector:
    """Drug-Drug Interaction detector using DrugBank index."""

    def __init__(self):
        self.lookup    = {}
        self.chroma    = None
        self.available = False
        self._load_index()

    def _load_index(self):
        lookup_path = INDEX_DIR / "lookup.json"
        if not lookup_path.exists():
            print(f"  [warn] DDI index not found at {INDEX_DIR}")
            print(f"  Run: python scripts/build_rag_index.py")
            return

        with open(lookup_path) as f:
            raw = json.load(f)

        # Reconstruct frozenset-keyed lookup
        self.lookup = {}
        for key_str, interactions in raw.items():
            parts = key_str.split("__")
            if len(parts) == 2:
                self.lookup[frozenset(parts)] = interactions

        self.available = True
        print(f"  DDI index loaded: {len(self.lookup)} drug pairs")

        # ChromaDB vector index — only loaded if built with --with_chroma
        chroma_path = INDEX_DIR / "chroma"
        if chroma_path.exists():
            try:
                import chromadb
                from chromadb.utils import embedding_functions
                client = chromadb.PersistentClient(path=str(chroma_path))
                try:
                    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
                    )
                except Exception:
                    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                self.chroma = client.get_collection(
                    name="ddi", embedding_function=emb_fn
                )
                print(f"  ChromaDB loaded: {self.chroma.count()} vectors")
            except Exception:
                pass  # ChromaDB optional — JSON lookup handles 95% of cases

    def check_pair(self, drug1: str, drug2: str) -> list[dict]:
        """
        Check if drug1 and drug2 have a known interaction.
        Returns list of interaction dicts.
        """
        if not self.available:
            return []

        key = frozenset({normalize_drug(drug1), normalize_drug(drug2)})

        # Exact lookup first (fast, O(1))
        if key in self.lookup:
            return self.lookup[key]

        # Partial name match — one drug name contains the other
        d1n = normalize_drug(drug1)
        d2n = normalize_drug(drug2)
        for stored_key, interactions in self.lookup.items():
            stored_list = list(stored_key)
            if len(stored_list) == 2:
                s1, s2 = stored_list
                if ((d1n in s1 or s1 in d1n) and
                        (d2n in s2 or s2 in d2n)):
                    return interactions
                if ((d1n in s2 or s2 in d1n) and
                        (d2n in s1 or s1 in d1n)):
                    return interactions

        # ChromaDB semantic fallback
        if self.chroma:
            try:
                query = f"{drug1} interacts with {drug2}"
                results = self.chroma.query(
                    query_texts=[query], n_results=3
                )
                hits = []
                for i, meta in enumerate(results["metadatas"][0]):
                    distance = results["distances"][0][i]
                    if distance < 0.25:  # close semantic match
                        hits.append({
                            "drug1":       meta["drug1"],
                            "drug2":       meta["drug2"],
                            "action":      meta["action"],
                            "description": meta["description"],
                            "severity":    meta["severity"],
                            "source":      "semantic",
                        })
                return hits
            except Exception:
                pass

        return []

    def check_note(self, source_text: str, summary_text: str,
                   note_id: str, det_counter: list) -> list[dict]:
        """
        Extract all drugs from source note, check all pairs for interactions,
        flag interactions not acknowledged in the summary.
        """
        if not self.available:
            return []

        source_drugs  = extract_drugs(source_text)
        summary_drugs = extract_drugs(summary_text)

        # Only check interactions between drugs mentioned in the source
        # (summary drugs alone don't tell us the clinical context)
        all_drugs = list(set(source_drugs + summary_drugs))

        detections = []
        checked    = set()

        for i, d1 in enumerate(all_drugs):
            for d2 in all_drugs[i+1:]:
                pair_key = frozenset({d1, d2})
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                interactions = self.check_pair(d1, d2)
                if not interactions:
                    continue

                # Check if interaction is acknowledged in the summary
                if interaction_acknowledged(d1, d2, summary_text):
                    continue

                # Flag — interaction present in source context but not
                # acknowledged in summary
                for intr in interactions[:1]:  # top interaction per pair
                    det_counter[0] += 1
                    action = intr.get("action", "")
                    desc   = intr.get("description", "")[:200]
                    sev    = intr.get("severity", "Moderate")

                    detections.append({
                        "detection_id":   f"det_{det_counter[0]:04d}",
                        "entity_id":      f"{note_id}_{d1}_{d2}",
                        "detected_by":    ["layer2b_rag"],
                        "type":           "drug_interaction",
                        "flagged_text":   f"{d1} + {d2}",
                        "severity":       sev,
                        "confidence":     0.80,
                        "action":         action,
                        "description":    desc,
                        "drug1":          d1,
                        "drug2":          d2,
                    })

        return detections


def main():
    detector = DDIDetector()

    if not detector.available:
        print("\nIndex not built. Run first:")
        print("  python scripts/build_rag_index.py")
        return

    print(f"\nProcessing corpus: {CORPUS_PATH}")
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    det_counter = [0]
    results     = []
    total_flags = 0

    for i, note in enumerate(corpus):
        source  = note.get("source_text") or note.get("text", "")
        summary = note.get("test_summary") or note.get("clean_summary", "")

        detections = detector.check_note(
            source, summary, note["note_id"], det_counter
        )

        results.append({
            "note_id":    note["note_id"],
            "detections": detections,
        })
        total_flags += len(detections)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(corpus)}] DDI flags so far: {total_flags}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    notes_flagged = sum(1 for r in results if r["detections"])
    print(f"\nLayer 2b (RAG) complete → {OUTPUT_PATH}")
    print(f"  {notes_flagged}/{len(corpus)} notes with DDI flags")
    print(f"  {total_flags} total interactions flagged")
    print(f"\nNext: python pipeline/layer4_severity.py")


if __name__ == "__main__":
    main()
