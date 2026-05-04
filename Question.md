# Questions & Answers — HaloCheck

---

## Q: We have ground truth during evaluation but not in deployment. How do I explain this gap?

This is the most important conceptual gap in the project. Here is exactly how to frame it.

### The Core Gap in One Sentence

In evaluation, we know what's wrong. In deployment, we don't.

### The Two Modes Side by Side

```
EVALUATION MODE (what we built)          DEPLOYMENT MODE (real world)
─────────────────────────────────        ─────────────────────────────
Source note → Gemma → clean summary      Source note → Hospital LLM → summary
                   ↓                                              ↓
         We inject a known error                    Error happens naturally
                   ↓                                              ↓
         We log exactly what changed              Nobody logged what changed
                   ↓                                              ↓
         Pipeline runs                            Pipeline runs
                   ↓                                              ↓
         Compare detection to injection           Compare detection to... what?
                   ↓                                              ↓
         TP / FP / FN calculable                  TP / FP / FN unknowable
```

### How to Explain It — Three Points

**1. The pipeline itself doesn't change — only the evaluation does**

Layer 1, Layer 2, Layer 4 work exactly the same in both modes. They compare a source note against a summary and flag contradictions. They do not need ground truth to run. Ground truth was only ever needed to measure how well it works, not to make it work.

This is the same as any software test harness. Unit tests need known correct outputs to measure accuracy. The production system doesn't run those tests at runtime — it just runs.

**2. In deployment, the output changes from a measurement to a recommendation**

In evaluation:
```
Detection → compare to injection → TP or FP → compute F1
```

In deployment:
```
Detection → present to clinician → clinician verifies → accept or reject
```

The clinician becomes the ground truth. HaloCheck surfaces a flag. The clinician looks at the source note and the summary and decides if the flag is valid. Their judgment replaces the injection label.

This is exactly how radiology AI works today. A model flags a suspicious region. The radiologist decides. The model doesn't need to know the ground truth — the radiologist does.

**3. The gap is a precision problem, not a fundamental flaw**

In evaluation, a false positive hurts your F1. In production, a false positive wastes a clinician's time and — if repeated — erodes trust until they stop reading flags.

**Our current precision is 0.640** — roughly 36 out of every 100 flags on any note are false alarms. That is acceptable for a research prototype but not for production. The gap isn't that the system can't work without ground truth — it's that precision needs to improve before clinicians trust the flags enough to act on them.

The path forward:

```
Phase 1 — Today (research prototype)
  Synthetic ground truth → measure performance → prove the approach works
  ← WE ARE HERE

Phase 2 — Pilot deployment
  Clinician reviews every flag → logs accept/reject → builds real labeled dataset
  This becomes the new ground truth — real errors in real notes

Phase 3 — Production
  Train on pilot dataset → calibrate confidence scores
  High-confidence flags → auto-warn clinician
  Low-confidence flags → background audit only
```

### The Analogy That Makes It Click

Think of it like a spell checker.

When you build a spell checker you test it on a corpus of deliberately misspelled words where you know all the correct answers. That gives you precision and recall numbers.

When you deploy the spell checker, it runs on real documents where nobody told it what's misspelled. It flags things. Some are right, some are wrong. The user verifies each one.

The spell checker doesn't need the answer key to run. It needed the answer key to prove it was worth deploying.

HaloCheck is exactly the same. The injection pipeline was the answer key for our research evaluation. In deployment, the clinician is the answer key.

### How to Frame This in the Meeting

"Great question — this is the most important gap we want to be honest about. The pipeline detection logic doesn't need ground truth to run. Ground truth was only needed to measure performance during evaluation — to prove the approach is sound. In deployment, the clinician becomes the verifier. They see the flag, they look at the source note, they make the call.

The real challenge this creates is a precision problem. If we flag too many false positives, clinicians stop trusting the system. Our current precision is 0.640 — that is acceptable for a research prototype but would need to reach 0.85+ for production deployment. The path forward is a pilot phase where clinician accept/reject decisions on real flags build an organically labeled dataset — that becomes the real-world ground truth we use to calibrate and improve the system over time."

### Technical Term for This Gap

This is called **covariate shift with unknown labels at inference time** — or more broadly, the **label shift problem**.

During evaluation, labels are available. At inference, labels are unavailable. The model still runs, but you can't compute accuracy in real-time.

Two standard approaches:

1. **Human-in-the-loop** — exactly what HaloCheck does in deployment. Clinician verifies each flag. Their decisions build a labeled production dataset.

2. **Proxy metrics** — monitor distributional signals like flagging rate, confidence distributions, and clinician override rate. If clinicians are overriding 90% of flags, something has shifted. This is how you detect model degradation without ground truth.

Both are standard practice in clinical AI deployment and worth mentioning in the paper's limitations section.

---

## Q: The pipeline only works with this specific data — how does it generalise?

See README.md for the deployment architecture. The short answer: the evaluation corpus is used to measure performance, not to run the pipeline. In deployment you feed any source note + any LLM summary and the pipeline runs. No corpus, no injection, no evaluate.py needed.

The locked corpus rule was for evaluation stability — don't change your test set while tuning your model. It is not a constraint on the system itself.

---

## Q: Why HHEM instead of BART-NLI as the baseline?

HHEM-2.1-Open from Vectara was purpose-built for hallucination detection in LLM outputs — it is a much more meaningful comparison than BART-large-mnli which is a general NLI model. HHEM was the original planned baseline but had a documented incompatibility with transformers ≥ 4.41 (which Layer 2 requires). We solved this by running HHEM in an isolated venv (venv_hhem with transformers==4.38.2) called as a subprocess. This is production-grade dependency isolation — the same approach Docker uses at the container level.

**Final numbers:** HaloCheck F1 0.640 vs HHEM F1 0.581 — a 10.3% improvement over a purpose-built hallucination detector.

---

## Q: Why not re-enable Layer 3?

We tested it on the locked corpus this session. Result: recovered 1 TP, added 68 FPs to negation_flip, precision dropped from 1.0 to 0.21, negation_flip F1 dropped from 0.723 to 0.310. Layer 1 NER handles negation via regex more precisely than medspaCy's linguistic approach on this diverse corpus. Layer 3 stays as noop.

---

## Q: Can this be deployed as a real module?

Yes — as a research prototype. The pipeline logic (Layers 1, 2, 4) runs on any source note + summary pair with no corpus or injection needed. Wrapping it as a pip-installable package with a clean API is Phase 1. Clinical production deployment requires:

1. Validation on real annotated clinical data (not synthetic injections)
2. IRB approval for use with real patient notes
3. Precision improvement to ~0.85+ before clinicians can trust flags in a hard-blocking capacity
4. Resolution of the MedNLI model redistribution issue (PhysioNet DUA prohibits redistribution)
