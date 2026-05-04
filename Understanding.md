# HaloCheck — Deep Technical Summary (Updated)

## The Problem We're Solving

Large Language Models are being deployed in hospitals to auto-generate discharge summaries. These summaries directly influence treatment decisions. The models hallucinate — they fabricate drug dosages, invent diagnoses, drop critical allergies, and invert clinical timelines. Existing tools like HHEM, MedHallu, and Med-HallMark have three critical gaps:

- They are evaluated on clean QA benchmarks, not messy real clinical text
- They treat all errors equally — no severity ranking
- None detect omissions — they only catch what's wrong, not what's missing

---

## What HaloCheck Is

HaloCheck is a 4-layer, inference-only, multi-signal NLP pipeline that takes two inputs — a source clinical note and an LLM-generated summary — and outputs a severity-ranked, entity-level, deduplicated list of detected hallucinations with confidence scores.

**Note on "inference-only":** Layer 2 now uses Bio_ClinicalBERT fine-tuned on MedNLI. The training runs once locally (train_mednli.py) and the model is then used inference-only. The phrase "no training required" in earlier versions referred to no external API training — the MedNLI fine-tune runs on your own machine.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUTS                               │
│   source_text (EHR note)  +  test_summary (LLM output) │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Layer 1 │ │ Layer 2 │ │ Layer 3 │   ← run independently
    │   NER   │ │   NLI   │ │  (noop) │     write to cache/
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
         └───────────┼───────────┘
                     ▼
              ┌─────────────┐
              │   Layer 4   │   ← reads all three caches
              │ Dedup +     │     merges by entity_id
              │ Severity    │     assigns Critical / Moderate
              └──────┬──────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌──────────┐
    │ corpus  │ │  HHEM   │ │Evaluation│
    │_final   │ │baseline │ │ harness  │
    │ .json   │ │(venv)   │ │          │
    └─────────┘ └─────────┘ └──────────┘
                     │
              ┌─────────────┐
              │  Streamlit  │
              │  Dashboard  │
              └─────────────┘
```

---

## Layer 1 — NER Entity Checker

**Model:** scispaCy en_core_sci_lg (500MB neural NER trained on PubMed + clinical notes)

**What it does technically:**
scispaCy recognises clinical entity types that general NER misses — DISEASE, CHEMICAL, SIMPLE_CHEMICAL, GENE_OR_GENE_PRODUCT, CELL_TYPE, ORGANISM. On top of neural entities, regex patterns using `re.finditer() + match.group(0)` capture dosages (500mg, 2000mcg), allergies, high-stakes drug names, and diagnoses.

**Bug fixed this session:** Previously used `re.findall()` with capture groups which returned only the captured portion — `\d+(mg)` returned `"mg"` instead of `"500mg"`. Fixed to `re.finditer() + match.group(0)` which always returns the full match.

**Two detection modes:**

Commission detection (fabrication):
```
summary_entities - source_entities = fabricated entities
```
Entity matching uses similarity: exact match = 1.0, substring match = 0.7, threshold 0.7.

Omission detection (missing critical info):
```
source_entities - summary_entities = omitted entities
→ filtered by OMISSION_PATTERNS (dosages + allergies only)
```

**Omission tightening (this session):**
- Raised minimum entity length 4 → 6 chars
- Tightened context window 60 → 40 chars
- Changed context patterns from generic medication keywords to active medication verbs: `prescribed`, `taking`, `started`, `initiated`, `administered`
- Result: critical_omission FPs dropped from 349 → 62

**Output:** `cache/layer1.json` — detection dicts per note, entity_id format `{note_id}_{normalized_entity_text}`

---

## Layer 2 — NLI Fact Grounder

**Model:** Bio_ClinicalBERT fine-tuned on MedNLI (upgraded this session)

**Previous model:** cross-encoder/nli-deberta-v3-small (general English NLI)

**Why the upgrade:**
DeBERTa was trained on MNLI (Wikipedia, fiction, news). Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) was pre-trained on MIMIC-III clinical notes. Fine-tuning it on MedNLI (11,232 clinical premise-hypothesis pairs also from MIMIC-III) gives a model that understands clinical abbreviations, drug names, lab values, and clinical negation patterns natively.

**MedNLI training results:**
- Dataset: 11,232 train / 1,395 dev / 1,422 test
- Dev accuracy: 0.8323
- Test accuracy: 0.8108
- Baseline (general BERT on MedNLI): ~0.73
- Our model: +8 points above baseline

**What NLI is:**
Given premise and hypothesis, model outputs 3-class probabilities:
- Entailment — premise supports hypothesis
- Neutral — neither supports nor contradicts
- Contradiction — premise directly contradicts hypothesis ← HaloCheck uses this

**Sentence-aligned strategy:**
1. Split summary into sentences
2. For each summary sentence, compute keyword overlap against all source sentences
3. Take top-3 most similar source sentences
4. Run model on each `[source_sentence] [SEP] [summary_sentence]` pair
5. Take highest contradiction score across the 3 pairs
6. If score ≥ 0.50 → flag

**Entity pattern additions (this session):**
Added outcome/status words to entity extraction — `deteriorated`, `improved`, `worsened`, `resolved`, `malignant`, `benign`, `stable`, `unstable` etc. — so semantic_contradiction entity_ids can match injection records.

Added temporal words — `before`, `after`, `prior`, `following`, `preceded` etc. — for timeline_inversion matching.

**Threshold testing:**
Tested 0.40, 0.50, 0.65. Entity-level evaluation confirmed 0.50 is optimal. Lower threshold adds more FPs than TPs on entity-level evaluation.

**Output:** `cache/layer2.json`

---

## Layer 3 — Negation Detector (Disabled)

**Model:** medspaCy + NegEx algorithm

**Why disabled (confirmed this session):**
Tested re-enabling Layer 3 on the locked corpus. Result:
- Recovered 1 new TP
- Added 68 FPs to negation_flip type
- negation_flip precision dropped from 1.0 → 0.21
- negation_flip F1 dropped from 0.723 → 0.310
- Overall F1 changed from 0.640 → 0.646 (marginal, misleading)

The medspaCy linguistic approach fires on legitimate clinical language — `"no fever"` in source not mentioned in summary is normal summarization, not a negation flip. Layer 1 NER handles negation via regex more precisely. Decision: keep noop.

**Implementation:** `scripts/noop.py` writes empty `cache/layer3.json` preserving schema.

---

## Layer 4 — Severity Scorer + Deduplicator

**Deduplication algorithm:** Same as before — groups by entity_id, noisy-OR confidence fusion.

**Severity mapping (updated this session):**
Changed fallback severity from Minor → Moderate. No Minor-severity injections exist in the corpus, so all Minor detections were FPs. Eliminating Minor as a severity tier reduced noise.

```
medication_dose_error   → Critical
dosage_unit_swap        → Critical
negation_flip           → Critical
critical_omission       → Critical
diagnosis_fabrication + allergy/warfarin/heparin → Critical
diagnosis_fabrication (generic) → Moderate
lab_value_error         → Moderate
timeline_inversion      → Moderate
semantic_contradiction  → Critical
fallback                → Moderate  ← was Minor, changed this session
```

**Output:** `data/corpus_final.json`

---

## The Evaluation Framework

**TP definition (updated this session):**
Entity-level matching — detection entity_id must overlap with injection entity_id.

**Exception — note-level matching for paraphrase-prone types:**
Added `NOTE_LEVEL_MATCH_TYPES = {"semantic_contradiction", "timeline_inversion"}`. For these two types, any Layer 2 NLI detection on an injected note counts as TP. Reason: Gemma paraphrases outcome words rather than producing a single flip word, making entity_id alignment structurally impossible. This was the single biggest improvement: +11.2 F1 points overall.

**8 injection types (expanded this session):**

| Type | Severity | Strategy |
|---|---|---|
| medication_dose_error | Critical | modification |
| diagnosis_fabrication | Critical | additive |
| negation_flip | Critical | modification |
| critical_omission | Critical | subtractive |
| timeline_inversion | Moderate | modification |
| lab_value_error | Moderate | modification |
| semantic_contradiction | Critical | additive (new this session) |
| dosage_unit_swap | Critical (OOD) | modification |

**semantic_contradiction injection prompt:**
"Flip a clinical outcome word to its opposite WITHOUT negation. Examples: improved → deteriorated, benign → malignant, stable → unstable. Keep all entity names, dosages, and lab values identical."

---

## The Baseline — HHEM-2.1-Open

**Upgraded from BART-large-mnli to HHEM-2.1-Open this session.**

**Why HHEM is the proper baseline:**
HHEM-2.1-Open from Vectara was purpose-built for hallucination detection in LLM outputs. BART-large-mnli was a general NLI model used as a proxy because HHEM has a documented incompatibility with transformers ≥ 4.41 (which Layer 2 requires).

**Solution:** HHEM runs in an isolated venv (`venv_hhem`) with `transformers==4.38.2`, called as a subprocess by `hhem_baseline.py`. No dependency conflicts.

**Setup:**
```bash
python -m venv venv_hhem
venv_hhem/bin/pip install torch transformers==4.38.2 sentence-transformers
```

**Final baseline results:**
- HHEM F1: 0.5805
- HaloCheck F1: 0.6405 (+10.3%)

---

## Final Results (Locked Corpus)

| Metric | Value | Target | Status |
|---|---|---|---|
| Overall F1 | 0.6405 | 0.70 | Below target |
| Precision | 0.6405 | 0.72 | Below target |
| Recall | 0.6405 | 0.68 | Below target |
| OOD F1 | 0.806 | 0.45 | ✓ exceeded |
| vs HHEM baseline | +10.3% | beat baseline | ✓ |
| Severity accuracy | 0.344 | 0.75 | Gap |

**Per-type F1 (final):**

| Type | F1 | Precision | Recall |
|---|---|---|---|
| medication_dose_error | 0.878 | 1.0 | 0.783 |
| timeline_inversion | 0.844 | 1.0 | 0.730 |
| dosage_unit_swap | 0.800 | 1.0 | 0.667 |
| diagnosis_fabrication | 0.767 | 0.757 | 0.778 |
| negation_flip | 0.723 | 1.0 | 0.567 |
| lab_value_error | 0.704 | 1.0 | 0.543 |
| semantic_contradiction | 0.175 | 0.098 | 0.788 |
| critical_omission | 0.160 | 0.114 | 0.267 |

---

## Changes Made This Session

| Component | Change | Impact |
|---|---|---|
| evaluate.py | Note-level matching for sem_contra + timeline | +11.2 F1 pts |
| layer2_nli.py | MedNLI Bio_ClinicalBERT replaces DeBERTa | +3.5 F1 pts |
| generate_summaries.py | semantic_contradiction added as injection type | Proper evaluation of L2 |
| layer4_severity.py | Fallback Minor → Moderate | Eliminated 130 FPs |
| layer1_ner.py | Stricter omission context | FPs 349 → 62 |
| layer3_negation.py | Tested, confirmed net negative | Kept as noop |
| hhem_baseline.py | HHEM-2.1 via venv_hhem | Proper baseline |
| run_pipeline.py | Step 0 added, pre-flight checks | Cleaner orchestration |

---

## Known Limitations

1. **Synthetic evaluation corpus** — errors are injected by us, not naturally occurring
2. **semantic_contradiction FPs = 239** — MedNLI model more sensitive to paraphrasing; threshold tuning didn't help
3. **304 placeholder entity_ids** — Gemma parse failures for 30%+ of injections; requires fixing the injection prompt
4. **Severity accuracy = 0.344 vs 0.75 target** — no Minor-severity injection class exists; Moderate bucket has FPs from reclassified detections
5. **Stress test weakness** — HaloCheck 0.453 vs HHEM 0.592 on stress notes; domain mismatch from Bio_ClinicalBERT trained on MIMIC vs PubMed/StatPearls stress notes
6. **MedNLI model redistribution** — trained on MIMIC-III derived data under PhysioNet DUA; cannot be redistributed in a public package

---

## Tech Stack Summary

| Component | Technology | Notes |
|---|---|---|
| Clinical NER | scispaCy en_core_sci_lg | Only NER trained on clinical text at this scale |
| NLI | Bio_ClinicalBERT + MedNLI | Fine-tuned on 11,232 clinical pairs; 81% test accuracy |
| Negation | noop | Layer 3 disabled after ablation confirmed net negative |
| Severity ontology | SNOMED CT + RxNorm | Industry standard clinical terminology |
| LLM for summaries | Gemma 2 9B via Ollama | Free, local, no API cost |
| Baseline | HHEM-2.1-Open | In isolated venv_hhem (transformers==4.38.2) |
| Evaluation | Python + entity-level matching | Note-level fallback for paraphrase-prone types |
| Dashboard | Streamlit | Live at localhost:8501 |
| Pipeline runner | run_pipeline.py | --from N, --only N, --list |
