"""
build_rag_index.py — Build ChromaDB DDI index from DrugBank XML.

Expected file: data/drugbank/drugbank.xml  (or full_database.xml)
               The full DrugBank XML export from Kaggle.

Output: data/rag_index/ddi/   ← ChromaDB persistent index
                                  lookup.json (fast exact match)
                                  interactions.json (full data)
                                  metadata.json

Usage:
  python scripts/build_rag_index.py              # fast — JSON lookup only (~2 min)
  python scripts/build_rag_index.py --with_chroma  # also build vector index (slow)
  python scripts/build_rag_index.py --xml path/to/drugbank.xml
  python scripts/build_rag_index.py --test

Run once. layer2b_rag.py uses the index at inference time.
"""

import json
import re
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

DRUGBANK_DIR = Path("data/drugbank")
INDEX_DIR    = Path("data/rag_index/ddi")
NS           = {"db": "http://www.drugbank.ca"}

# Severity classification from description text
CRITICAL_KEYWORDS = [
    "bleeding", "hemorrhage", "haemorrhage", "QT prolongation",
    "cardiac arrest", "serotonin syndrome", "hypertensive crisis",
    "respiratory depression", "agranulocytosis", "anaphylaxis",
    "seizure", "nephrotoxicity", "hepatotoxicity", "rhabdomyolysis",
    "torsades", "arrhythmia", "anticoagulant activities",
    "risk or severity of bleeding", "severe", "fatal",
]

MODERATE_KEYWORDS = [
    "serum concentration", "increased when", "decreased when",
    "metabolism", "absorption", "excretion", "efficacy",
    "hypotension", "bradycardia", "sedation", "inhibit",
]


def classify_severity(description: str) -> str:
    desc_lower = description.lower()
    for kw in CRITICAL_KEYWORDS:
        if kw.lower() in desc_lower:
            return "Critical"
    for kw in MODERATE_KEYWORDS:
        if kw.lower() in desc_lower:
            return "Moderate"
    return "Moderate"


def normalize_drug(name: str) -> str:
    name = name.lower().strip()
    for suffix in [" hydrochloride", " hcl", " sodium", " potassium",
                   " sulfate", " acetate", " citrate", " tartrate",
                   " maleate", " mesylate", " monohydrate"]:
        name = name.replace(suffix, "")
    name = re.sub(r"\(.*?\)", "", name).strip()
    return name


def find_xml_file(drugbank_dir: Path) -> Path:
    """Find the DrugBank XML file — tries common filenames."""
    candidates = [
        "drugbank.xml",
        "full_database.xml",
        "drugbank_full_database.xml",
        "drugbank_all_full_database.xml",
    ]
    for name in candidates:
        p = drugbank_dir / name
        if p.exists():
            return p

    # Try any XML in the directory
    xmls = list(drugbank_dir.glob("*.xml"))
    if xmls:
        return xmls[0]

    raise FileNotFoundError(
        f"No DrugBank XML found in {drugbank_dir}/\n"
        f"Download from: https://www.kaggle.com/datasets/sergeguillemart/drugbank\n"
        f"Expected: {drugbank_dir}/drugbank.xml"
    )


def parse_drugbank_xml(xml_path: Path) -> list[dict]:
    """
    Parse DrugBank XML and extract all drug-drug interactions.

    XML structure per drug:
      <drug>
        <name>DrugName</name>
        <drug-interactions>
          <drug-interaction>
            <name>OtherDrug</name>
            <description>The risk of X increases...</description>
          </drug-interaction>
        </drug-interactions>
      </drug>
    """
    print(f"Parsing {xml_path} ({xml_path.stat().st_size / 1e6:.1f} MB)...")
    print("  This may take 1-2 minutes for the full database...")

    # Use iterparse for memory efficiency on large files
    interactions = []
    drug_count   = 0
    intr_count   = 0

    context = ET.iterparse(str(xml_path), events=("start", "end"))
    context = iter(context)

    current_drug_name = None
    current_drug_id   = None
    in_drug           = False
    in_interactions   = False
    current_intr      = {}

    # Namespace prefix
    def tag(local):
        return f"{{{NS['db']}}}{local}"

    for event, elem in context:
        if event == "start":
            if elem.tag == tag("drug") and elem.get("type") in ("small molecule", "biotech"):
                in_drug = True
                current_drug_name = None
                current_drug_id   = None

            elif in_drug and elem.tag == tag("drug-interactions"):
                in_interactions = True

            elif in_interactions and elem.tag == tag("drug-interaction"):
                current_intr = {}

        elif event == "end":
            if not in_drug:
                continue

            # Capture drug name and primary ID
            if elem.tag == tag("name") and current_drug_name is None:
                current_drug_name = (elem.text or "").strip()

            elif (elem.tag == tag("drugbank-id") and
                  elem.get("primary") == "true" and
                  current_drug_id is None):
                current_drug_id = (elem.text or "").strip()

            # Inside a drug-interaction element
            elif in_interactions:
                if elem.tag == tag("name"):
                    current_intr["name"] = (elem.text or "").strip()
                elif elem.tag == tag("description"):
                    current_intr["description"] = (elem.text or "").strip()
                elif elem.tag == tag("drugbank-id") and "name" not in current_intr:
                    current_intr["id"] = (elem.text or "").strip()

                elif elem.tag == tag("drug-interaction"):
                    # Complete interaction record
                    intr_name = current_intr.get("name", "")
                    intr_desc = current_intr.get("description", "")

                    if current_drug_name and intr_name and intr_desc:
                        d1_norm = normalize_drug(current_drug_name)
                        d2_norm = normalize_drug(intr_name)
                        severity = classify_severity(intr_desc)

                        interactions.append({
                            "id":          f"{d1_norm}__{d2_norm}",
                            "drug1":       current_drug_name,
                            "drug2":       intr_name,
                            "drug1_norm":  d1_norm,
                            "drug2_norm":  d2_norm,
                            "action":      "",  # DrugBank XML uses description
                            "description": intr_desc,
                            "severity":    severity,
                            "doc_text":    f"{current_drug_name} interacts with {intr_name}. {intr_desc}",
                        })
                        intr_count += 1

                    current_intr = {}

                elif elem.tag == tag("drug-interactions"):
                    in_interactions = False

            elif elem.tag == tag("drug"):
                # Drug element complete
                in_drug = False
                drug_count += 1
                current_drug_name = None
                current_drug_id   = None

                # Free memory — critical for large XML
                elem.clear()

                if drug_count % 1000 == 0:
                    print(f"  Parsed {drug_count} drugs, {intr_count} interactions...")

    print(f"  ✓ Parsed {drug_count} drugs → {intr_count} interactions")
    return interactions


def build_lookup_index(interactions: list[dict]) -> dict:
    """
    Build fast dict lookup: frozenset({drug1_norm, drug2_norm}) → list.
    Symmetric — warfarin+aspirin == aspirin+warfarin.
    """
    lookup = defaultdict(list)
    for intr in interactions:
        key = frozenset({intr["drug1_norm"], intr["drug2_norm"]})
        lookup[key].append(intr)

    # Deduplicate within each pair — keep highest severity
    deduped = {}
    for key, intrs in lookup.items():
        # Sort: Critical first, then by description length (richer = better)
        sorted_intrs = sorted(
            intrs,
            key=lambda x: (0 if x["severity"] == "Critical" else 1, -len(x["description"]))
        )
        deduped[key] = sorted_intrs

    return deduped


def build_chroma_index(interactions: list[dict]) -> None:
    """Build ChromaDB vector index for fuzzy drug name matching."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("\n  [warn] chromadb not installed — skipping vector index")
        print("  Run: pip install chromadb sentence-transformers")
        print("  Exact lookup index still works without it.")
        return

    print("\nBuilding ChromaDB vector index...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))

    try:
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="pritamdeka/S-PubMedBert-MS-MARCO"
        )
        print("  Using PubMedBERT embeddings")
    except Exception:
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        print("  Using MiniLM embeddings (fallback)")

    try:
        client.delete_collection("ddi")
    except Exception:
        pass

    collection = client.create_collection(
        name="ddi",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Deduplicate by ID before indexing
    seen_ids = set()
    docs, metas, ids = [], [], []
    BATCH = 500

    for i, intr in enumerate(interactions):
        uid = intr["id"][:490]
        if uid in seen_ids:
            continue
        seen_ids.add(uid)

        docs.append(intr["doc_text"][:1000])
        metas.append({
            "drug1":       intr["drug1"][:100],
            "drug2":       intr["drug2"][:100],
            "drug1_norm":  intr["drug1_norm"][:100],
            "drug2_norm":  intr["drug2_norm"][:100],
            "severity":    intr["severity"],
            "description": intr["description"][:500],
        })
        ids.append(uid)

        if len(docs) == BATCH:
            collection.upsert(documents=docs, metadatas=metas, ids=ids)
            print(f"  Indexed {i + 1}/{len(interactions)}")
            docs, metas, ids = [], [], []

    if docs:
        collection.upsert(documents=docs, metadatas=metas, ids=ids)

    print(f"  ✓ ChromaDB: {collection.count()} vectors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml",       default=None,
                        help="Path to DrugBank XML file")
    parser.add_argument("--drugbank_dir", default=str(DRUGBANK_DIR))
    parser.add_argument("--test",      action="store_true")
    parser.add_argument("--no_chroma",    action="store_true", default=True,
                        help="Skip ChromaDB vector index (default: True — JSON lookup is sufficient)")
    parser.add_argument("--with_chroma",  action="store_true",
                        help="Also build ChromaDB vector index (slow, optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("  HaloCheck — Building DDI RAG Index from DrugBank XML")
    print("=" * 60)

    # Find XML
    if args.xml:
        xml_path = Path(args.xml)
    else:
        xml_path = find_xml_file(Path(args.drugbank_dir))

    print(f"  XML: {xml_path}")
    print(f"  Output: {INDEX_DIR}\n")

    # Parse
    interactions = parse_drugbank_xml(xml_path)

    # Lookup index
    print("\nBuilding lookup index...")
    lookup = build_lookup_index(interactions)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Serialize lookup (frozenset → sorted list string key)
    serializable = {
        "__".join(sorted(k)): v
        for k, v in lookup.items()
    }
    with open(INDEX_DIR / "lookup.json", "w") as f:
        json.dump(serializable, f)
    print(f"  ✓ Lookup index: {len(lookup)} drug pairs")

    # Save full interactions
    with open(INDEX_DIR / "interactions.json", "w") as f:
        json.dump(interactions, f, indent=2)

    # Severity stats
    crit = sum(1 for i in interactions if i["severity"] == "Critical")
    mod  = sum(1 for i in interactions if i["severity"] == "Moderate")

    # Metadata
    meta = {
        "source":              str(xml_path),
        "total_interactions":  len(interactions),
        "unique_drug_pairs":   len(lookup),
        "critical":            crit,
        "moderate":            mod,
    }
    with open(INDEX_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Vector index
    if args.with_chroma:
        build_chroma_index(interactions)
    else:
        print("\nSkipped ChromaDB vector index (use --with_chroma to build it)")
        print("JSON lookup index is sufficient for exact + partial drug name matching.")

    print(f"\n{'='*60}")
    print(f"  INDEX COMPLETE")
    print(f"  Interactions: {len(interactions):,}")
    print(f"  Unique pairs: {len(lookup):,}")
    print(f"  Critical:     {crit:,}")
    print(f"  Moderate:     {mod:,}")
    print(f"  Location:     {INDEX_DIR}")
    print(f"{'='*60}")

    if args.test:
        print("\n--- Test Queries ---")
        import sys
        sys.path.insert(0, "pipeline")
        from layer2b_rag import DDIDetector
        detector = DDIDetector()
        tests = [
            ("warfarin",    "aspirin"),
            ("lepirudin",   "apixaban"),
            ("simvastatin", "amiodarone"),
            ("fluoxetine",  "tramadol"),
            ("metformin",   "lisinopril"),
        ]
        for d1, d2 in tests:
            results = detector.check_pair(d1, d2)
            if results:
                r = results[0]
                print(f"  [{r['severity']}] {d1} + {d2}")
                print(f"    → {r['description'][:100]}...")
            else:
                print(f"  [none]  {d1} + {d2}")


if __name__ == "__main__":
    main()
