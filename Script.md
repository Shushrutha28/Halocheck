# HaloCheck — Meeting Script & Prepared Answers

**Format:** Kadhiravan opens (2–3 min). Professor rotates around the room — every person needs answers to all four questions. 2h15min window = deep conversation, not a polished demo.

---

## The Four Questions — Everyone Needs These

**1. Team status — where are you right now?**
**2. Where are you going next — what does it take to finish?**
**3. Difficulties the team is having**
**4. Your individual role — what are you doing, what have you done**

---

## Opening — Kadhiravan (2–3 min)

"HaloCheck is a 4-layer NLP pipeline that takes a source clinical note and an LLM-generated summary and detects hallucinations — fabricated diagnoses, wrong medication doses, dropped allergies, inverted timelines. It ranks every detected error as Critical or Moderate severity and returns entity-level flags with confidence scores.

We have a fully working end-to-end pipeline. The evaluation harness is complete. Our current overall F1 is 0.640, which beats our HHEM-2.1 baseline at 0.581. Six out of eight injection types have F1 above 0.70, with medication dose errors at 0.878, timeline inversions at 0.844, and dosage unit swaps at 0.800. The two weak types — semantic contradiction and critical omission — have documented root causes we can explain.

The system runs entirely locally with no API cost, which matters for clinical deployment where patient data cannot leave the hospital network."

---

## Everyone's Prepared Answers

---

### Kadhiravan — Project Lead & Data Pipeline

**Team status:**
"We have a fully working end-to-end pipeline. The scraper collected 1,100+ clinical notes across 4 sources. Gemma 2 9B generates summaries via Ollama and injects controlled hallucinations. All 4 layers run. The Streamlit dashboard is live. Evaluation harness is complete with entity-level TP matching. Final overall F1 is 0.640, beating the HHEM-2.1 baseline at 0.581."

**Where next / what it takes to finish:**
"Two remaining items. First, fix the 304 placeholder entity_ids — about 30% of injections have fallback IDs because Gemma sometimes fails to return structured JSON. This means those injections can never be matched as TPs in evaluation, artificially lowering recall. The fix is a simpler, more constrained injection prompt. Second, the 4-page ACM paper is the final deliverable — methodology, results, and ablation sections are the remaining writing."

**Difficulties:**
"The biggest challenge was corpus stability — we kept re-running generate_summaries while tuning the pipeline, which re-injected different errors each run and made F1 comparisons meaningless. We eventually locked the corpus and only re-ran from Layer 3 or later. That discipline brought stability to our measurements.

The second challenge was ground truth alignment — injection records had placeholder entity_ids that couldn't match detection entity_ids, so evaluation was doing note-level binary matching instead of entity-level. We identified this and fixed it in both the injection pipeline and the evaluator."

**Your role:**
"I'm coordinating and leading the data pipeline. I built the scraper that collected the corpus across MTSamples, PubMed, StatPearls, and MedQA. I built and maintained generate_summaries.py — the Gemma injection pipeline across 8 hallucination types. I added semantic_contradiction as a proper injection type this session. I also did the corpus audit and false positive diagnosis work, and I coordinate sprints and integration across the team."

---

### Shushrutha — NLP Engineer & UI

**Team status:**
"The pipeline is fully functional end to end. Layer 1 NER and the Streamlit dashboard are both complete and integrated. Final F1 is 0.640."

**Where next / what it takes to finish:**
"Layer 1 is stable — it only needs re-running if generate_summaries changes. The dashboard is complete. My remaining contribution is the presentation materials and the system architecture section of the paper."

**Difficulties:**
"Layer 1 had a subtle regex bug that took a while to diagnose. We were using re.findall with capture groups, which returns only the captured portion — so the pattern `\d+(mg)` was returning just 'mg' instead of '500mg'. That meant entity IDs were essentially useless strings. Fixed by switching to re.finditer and match.group(0) which always returns the full matched text.

The second challenge in Layer 1 was omission over-detection — we were flagging 349 false positive critical omissions because the context filter was too broad. Any dosage in the source not in the summary got flagged, even legitimate summarization choices. We tightened the context to only flag dosages within 40 characters of active medication verbs like 'prescribed' or 'initiated'. That brought FPs from 349 down to 62."

**Your role:**
"I built Layer 1 — the NER entity checker using scispaCy's en_core_sci_lg model. It extracts clinical entities from both source and summary and flags fabrications and omissions with entity-level IDs. I also built the Streamlit dashboard — three-page UI with note inspector, metrics view, and corpus browser. And I designed the JSON schema that all pipeline layers communicate through."

---

### Ritvik — NLP Engineer — Inference

**Team status:**
"Layer 2 is complete with the MedNLI upgrade. We fine-tuned Bio_ClinicalBERT on 11,232 clinical NLI pairs from MIMIC-III. Test accuracy on MedNLI is 81.1%, which is 8 points above what general BERT achieves on the same dataset. The ablation study confirmed Layer 3 is net negative and should stay disabled."

**Where next / what it takes to finish:**
"Layer 2 is stable on the locked corpus. If generate_summaries is re-run to fix placeholder entity_ids, Layer 2 needs a one-time re-run (~40 min). My remaining work is the system architecture section of the paper — specifically explaining the sentence-aligned NLI strategy and the MedNLI upgrade."

**Difficulties:**
"The main challenge with Layer 2 was entity_id format mismatch. Layer 1 generates IDs like mts_001_metformin_500mg. Layer 2 was generating IDs like mts_001_nli_2_a_54_year_old — it was falling back to first-3-words of the sentence when no clinical entity pattern matched. They could never align so Layer 4 deduplication produced zero cross-layer merges.

Fixed by adding outcome words and temporal words to the entity extraction patterns — deteriorated, improved, stable, unstable for semantic_contradiction; before, after, prior, following for timeline_inversion. That gave Layer 2 the vocabulary to produce matching IDs.

The second challenge was testing Layer 3 re-enablement. Enabling medspaCy negation detection recovered 1 TP but added 68 FPs — negation_flip F1 dropped from 0.723 to 0.310. The fundamental problem is clinical source notes are full of negations that summaries legitimately omit. medspaCy can't distinguish a real negation flip from normal summarization."

**Your role:**
"I built Layer 2 — the NLI fact grounder. I ran the MedNLI fine-tuning on Bio_ClinicalBERT. I implemented the sentence-aligned NLI strategy — split summary into sentences, find top-3 similar source sentences by keyword overlap, run the model on each pair, take highest contradiction score. I also ran the ablation study that confirmed Layer 3 was net negative, and I did the threshold sweep that confirmed 0.50 is optimal for entity-level evaluation."

---

### Vaibhav — Systems & Evaluation

**Team status:**
"Layer 4, the HHEM baseline, and the evaluation harness are all complete. The evaluation framework was significantly upgraded this session — we moved from note-level TP matching to entity-level TP matching, which is meaningfully more rigorous. Final F1 is 0.640 vs HHEM baseline of 0.581."

**Where next / what it takes to finish:**
"The evaluation harness is ready — it just needs clean ground truth data from the fixed generate_summaries run. Once placeholder entity_ids are resolved, I re-run evaluate.py and we have final clean numbers. I am also writing the results and ablation tables section of the paper."

**Difficulties:**
"The evaluation had a fundamental issue — the original evaluate.py was doing note-level binary matching. Any detection on an injected note counted as a true positive, even if it flagged the completely wrong entity. That was inflating our F1 artificially.

We rewrote the evaluator to require entity_id overlap between injection record and detection record. More honest but more demanding. We also added a smart exception: for semantic_contradiction and timeline_inversion, Gemma paraphrases the injected change rather than producing a single flip word, so entity_id alignment is structurally impossible. For those two types, any Layer 2 detection on an injected note counts as TP. This note-level fallback for paraphrase-prone types was the single biggest improvement of the session — +11.2 F1 points.

The HHEM baseline also required significant engineering. HHEM-2.1-Open has a documented incompatibility with transformers ≥ 4.41, which Layer 2 requires. We solved it by running HHEM in an isolated venv with transformers==4.38.2 and calling it as a subprocess. Proper dependency isolation."

**Your role:**
"I built Layer 4 — deduplication and severity scoring. It groups detections by entity_id, merges cross-layer detections using noisy-OR confidence fusion, and assigns Critical or Moderate severity based on SNOMED CT and RxNorm category mapping. I built the HHEM baseline infrastructure — the isolated venv, the hhem_inference.py sidecar, and the hhem_baseline.py orchestrator. I wrote evaluate.py and ran the threshold sweep confirming 0.50 is optimal."

---

### Priyansh — Research & Documentation

**Team status:**
"The research paper is in progress. The demo script is ready. Corpus audit and injection quality analysis informed several pipeline fixes this session. Final F1 is 0.640 with detailed per-type breakdowns."

**Where next / what it takes to finish:**
"The paper needs final clean numbers once placeholder entity_ids are fixed and the pipeline re-runs. Introduction, related work, methodology, and limitations can be drafted now. Results and ablation tables get filled in last. The paper is a 4-page ACM format — introduction, related work, architecture, results, ablation, limitations, conclusion."

**Difficulties:**
"The injection quality audit revealed Gemma was sometimes returning the summary unchanged when asked to inject an error — the JSON output was parseable but original_value and injected_value were identical or empty. Those notes were labeled injected but were identical to the clean summary, silently scoring as false negatives with no way to detect them.

We added structured JSON output requirements to the injection prompt and added parse failure tracking — fallback IDs are now prefixed with 'fallback_' so evaluate.py can identify them and apply note-level matching as a fallback instead of silently dropping them as FNs.

Also did the limitations research — the ground truth gap question, the PhysioNet DUA redistribution issue, and the covariate shift framing for the paper's limitations section."

**Your role:**
"I handle research and documentation. I ran check_injection_quality.py and audit_corpus.py to measure injection success rates and schema compliance. I'm writing the 4-page ACM paper — introduction, related work, limitations, and conclusion. I prepared the demo script and the presentation materials. I also researched the deployment gap — the difference between synthetic evaluation and real-world deployment — and wrote the framing for the limitations section."

---

## Three Things Everyone Should Know Cold

**1. What is HaloCheck in one sentence?**
"HaloCheck is a 4-layer inference-only NLP pipeline that takes a source clinical note and an LLM-generated summary, detects hallucinations using NER and NLI, and ranks every detected error by clinical severity — Critical or Moderate — with entity-level confidence scores."

**2. What is the honest limitation?**
"It is a research prototype evaluated on synthetic controlled data. We injected the errors ourselves to have ground truth labels. In real deployment there is no ground truth — the clinician verifies each flag. Our precision is 0.640, which needs to reach 0.85+ before this is production-viable. The path forward is a pilot phase where clinician accept/reject decisions build a real labeled dataset."

**3. What is left to finish?**
"Fix the 304 placeholder entity_ids by improving the injection prompt in generate_summaries.py. Re-run the pipeline one final time to get clean metrics. Write the 4-page ACM paper."

---

## Key Numbers to Know

| Number | Value |
|---|---|
| Overall F1 | 0.640 |
| Precision | 0.640 |
| Recall | 0.640 |
| vs HHEM baseline | +10.3% (0.640 vs 0.581) |
| MedNLI training accuracy | 81.1% test, 83.2% dev |
| Medication dose error F1 | 0.878 |
| Timeline inversion F1 | 0.844 |
| OOD F1 | 0.806 |
| Corpus size | 990 total, 336 test |
| Injection types | 8 |
| Placeholder entity_ids | 304 (30% of injections) |
| Biggest single improvement | +11.2 pts (evaluate.py note-level matching) |
