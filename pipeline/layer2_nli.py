"""
layer2_nli.py — NLI Fact Grounder (MedNLI-tuned Bio_ClinicalBERT)

UPGRADE: Replaced general-purpose DeBERTa with Bio_ClinicalBERT
fine-tuned on MedNLI (11,232 clinical NLI pairs from MIMIC-III).

Model priority:
  1. models/mednli_clinicalbert/  — fine-tuned on MedNLI (preferred)
  2. cross-encoder/nli-deberta-v3-small — fallback if not trained yet

To train:  python scripts/train_mednli.py --mednli_dir data/mednli/

Reads:   data/corpus.json
Writes:  cache/layer2.json
Run:     python pipeline/layer2_nli.py
"""

import json
import re
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CORPUS_PATH             = Path("data/corpus.json")
OUTPUT_PATH             = Path("cache/layer2.json")
FINETUNED_MODEL         = Path("models/mednli_clinicalbert")
FALLBACK_MODEL          = "cross-encoder/nli-deberta-v3-small"
CONTRADICTION_THRESHOLD = 0.50   # sweep confirmed 0.50 is optimal on entity-level eval
TOP_K_SOURCE_SENTENCES  = 3
MAX_SOURCE_SENTENCES    = 40
NOTES_LIMIT             = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CPU mode")


def load_model():
    if FINETUNED_MODEL.exists() and (FINETUNED_MODEL / "config.json").exists():
        print(f"Loading fine-tuned MedNLI model: {FINETUNED_MODEL}")
        tok   = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL)
        model.to(device).eval()
        meta_path = FINETUNED_MODEL / "training_metadata.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            print(f"  Test acc: {meta.get('test_acc')} | Dev acc: {meta.get('best_dev_acc')}")
        return tok, model, "mednli_clinicalbert", True
    else:
        print(f"Fine-tuned model not found — falling back to {FALLBACK_MODEL}")
        print("  Run: python scripts/train_mednli.py to train the clinical model")
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline("text-classification", model=FALLBACK_MODEL,
                           device=0 if torch.cuda.is_available() else -1, top_k=None)
        return pipe, None, "deberta_fallback", False


tokenizer, model, model_name, is_finetuned = load_model()
print(f"Model ready: {model_name}")


def get_score_finetuned(premise, hypothesis):
    enc = tokenizer(premise, hypothesis, max_length=256, truncation=True,
                    padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).squeeze()
    return probs[2].item()  # index 2 = contradiction


def get_score_fallback(pipe, premise, hypothesis):
    result = pipe(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)
    scores = {item["label"].upper(): item["score"] for item in result[0]}
    for key in ["CONTRADICTION", "CONTRADICTS", "LABEL_2", "LABEL_0"]:
        if key in scores:
            return scores[key]
    return 0.0


def nli_score(premise, hypothesis):
    if is_finetuned:
        return get_score_finetuned(premise, hypothesis)
    return get_score_fallback(tokenizer, premise, hypothesis)


def normalize_entity(text):
    return re.sub(r"\s+", " ", text.lower().strip())


def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 15]


def keyword_overlap(s1, s2):
    stopwords = {"the","and","was","were","with","that","this","have","has","had",
                 "been","from","they","their","which","also","when","after","before",
                 "patient","clinical","history"}
    w1 = {w.lower() for w in re.findall(r"\b\w{4,}\b", s1) if w.lower() not in stopwords}
    w2 = {w.lower() for w in re.findall(r"\b\w{4,}\b", s2) if w.lower() not in stopwords}
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


ENTITY_PATTERNS = [
    # Dosages — highest priority, most specific
    r"\b\d+\s*(?:mg|mcg|g|ml|units?|IU)\b",
    # Common high-risk drug names
    r"\b(?:metformin|lisinopril|warfarin|insulin|aspirin|heparin|prednisone|"
    r"amoxicillin|atorvastatin|amlodipine|furosemide|levothyroxine|carvedilol|"
    r"omeprazole|sertraline|losartan|metoprolol|clopidogrel|tamoxifen)\b",
    # Diagnoses
    r"\b(?:diabetes|hypertension|COPD|pneumonia|sepsis|cancer|MI|stroke|"
    r"CHF|appendicitis|fracture|infection|asthma|atrial|lupus|arthritis)\b",
    # Lab names
    r"\b(?:creatinine|glucose|hemoglobin|potassium|sodium|troponin|"
    r"bilirubin|INR|BNP|HbA1c|WBC|platelets|albumin|cholesterol|TSH)\b",
    # FIX: outcome/status words — used in semantic_contradiction injections
    # Injection entity_id is derived from these words (e.g. "deteriorated",
    # "malignant"). Without this pattern, Layer 2 falls back to first-3-words
    # and can never match the injection entity_id.
    r"\b(?:deteriorated|improved|worsened|resolved|persisted|stabilized|"
    r"malignant|benign|unstable|stable|increased|decreased|acute|chronic|"
    r"positive|negative|controlled|uncontrolled|responded|failed|tolerated|"
    r"progressed|regressed|recurred|remission|exacerbated|recovered)\b",
    # FIX: temporal words — used in timeline_inversion injections
    # Injection flips before/after relationship; entity_id uses temporal anchor.
    r"\b(?:before|after|prior|following|preceded|subsequent|initiated|"
    r"started|began|since|until|within|during|previously|thereafter)\b",
]


def extract_entity(sentence):
    for pat in ENTITY_PATTERNS:
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            return normalize_entity(m.group(0))
    stopwords = {"the","and","was","were","with","that","this","have","has",
                 "had","been","from","they","their","also"}
    words = [w for w in sentence.split() if len(w) > 3 and w.lower() not in stopwords]
    return normalize_entity("_".join(words[:3])) if words else "unknown"


def make_entity_id(sentence, note_id):
    entity = extract_entity(sentence)
    clean  = re.sub(r"[^a-z0-9_]", "_", entity[:30])
    return f"{note_id}_{clean}"


def process_note(note, det_counter):
    detections  = []
    source_text = note.get("source_text") or note.get("text", "")
    test_text   = note.get("test_summary") or note.get("clean_summary") or note.get("text", "")
    source_sents  = split_sentences(source_text)[:MAX_SOURCE_SENTENCES]
    summary_sents = split_sentences(test_text)

    if not source_sents or not summary_sents:
        return detections

    seen = set()
    for summary_sent in summary_sents:
        if summary_sent in seen:
            continue
        scored = sorted([(keyword_overlap(summary_sent, s), s) for s in source_sents],
                        key=lambda x: x[0], reverse=True)
        top_src = [s for _, s in scored[:TOP_K_SOURCE_SENTENCES]]

        best_score, best_premise = 0.0, ""
        for src in top_src:
            s = nli_score(src, summary_sent)
            if s > best_score:
                best_score, best_premise = s, src

        if best_score >= CONTRADICTION_THRESHOLD:
            seen.add(summary_sent)
            det_counter[0] += 1
            detections.append({
                "detection_id":   f"det_{det_counter[0]:04d}",
                "entity_id":      make_entity_id(summary_sent, note["note_id"]),
                "detected_by":    ["layer2_nli"],
                "model":          model_name,
                "type":           "semantic_contradiction",
                "flagged_text":   summary_sent,
                "severity":       "Moderate",
                "confidence":     round(best_score, 4),
                "matched_source": best_premise[:120],
            })
    return detections


def main():
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    if NOTES_LIMIT:
        corpus = corpus[:NOTES_LIMIT]
        print(f"[DEV MODE] {NOTES_LIMIT} notes")

    print(f"\nProcessing {len(corpus)} notes | model={model_name} | threshold={CONTRADICTION_THRESHOLD}")

    det_counter = [0]
    results     = []
    start       = time.time()

    for i, note in enumerate(corpus):
        elapsed = time.time() - start
        eta = (elapsed / (i+1)) * (len(corpus)-i-1) if i > 0 else 0
        print(f"  [{i+1}/{len(corpus)}] {note['note_id']}  ETA: {eta/60:.1f}m")
        detections = process_note(note, det_counter)
        results.append({"note_id": note["note_id"], "detections": detections})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    total_flags   = sum(len(r["detections"]) for r in results)
    notes_flagged = sum(1 for r in results if r["detections"])
    print(f"\nLayer 2 complete -> {OUTPUT_PATH}")
    print(f"  {notes_flagged}/{len(corpus)} notes flagged | {total_flags} total detections")
    print(f"  Time: {(time.time()-start)/60:.1f}m")
    print(f"\nNext: python scripts/noop.py")


if __name__ == "__main__":
    main()
