# HaloCheck

**Severity-aware hallucination detection for LLM-generated clinical summaries.**

HaloCheck is a 4-layer inference-only NLP pipeline that detects and ranks hallucinations in LLM-generated clinical notes. It runs on CPU with no API cost, provides entity-level flags with severity rankings (Critical / Moderate), and outperforms HHEM-2.1-Open (F1 0.640 vs 0.581) on synthetic clinical evaluation data.

**Team Fearsome Five — CS595 Medical Informatics AI, IIT Chicago, Spring 2026**

---

## Architecture

```
Source Note + LLM Summary
        ↓
Layer 1 — NER entity checker       (scispaCy en_core_sci_lg)
        ↓
Layer 2 — NLI fact grounder        (Bio_ClinicalBERT fine-tuned on MedNLI)
        ↓
Layer 3 — Negation detector        (disabled — noop)
        ↓
Layer 4 — Severity + dedup         (SNOMED CT / RxNorm severity mapping)
        ↓
Structured hallucination flags with severity and confidence scores
```

---

## Results

| Metric | HaloCheck | HHEM-2.1 baseline |
|---|---|---|
| Overall F1 | **0.640** | 0.581 |
| Medication dose error F1 | **0.878** | — |
| Timeline inversion F1 | **0.844** | — |
| Dosage unit swap F1 | **0.800** | — |
| OOD F1 | **0.806** | — |

---

## Requirements

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended for GPU training)
- Ollama (for corpus generation only — not required for inference)
- GPU optional but recommended for MedNLI training (~25 min on GPU vs ~3 hours on CPU)

---

## Installation

### macOS

```bash
# 1. Clone the repository
git clone https://github.com/your-org/halocheck.git
cd halocheck

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch (CPU version — change to CUDA if you have a GPU)
pip install torch torchvision torchaudio

# 4. Install core dependencies
pip install transformers==4.41.0 \
            sentence-transformers \
            spacy \
            scispacy \
            medspacy \
            requests \
            beautifulsoup4 \
            datasets \
            ollama \
            streamlit \
            tqdm

# 5. Install scispaCy clinical NLP model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# 6. Install medspaCy dependency
python -m spacy download en_core_web_sm

# 7. Create required directories
mkdir -p data/raw data/mednli cache models evaluation

# 8. Set up HHEM isolated environment (for proper baseline — optional)
python -m venv venv_hhem
venv_hhem/bin/pip install torch transformers==4.38.2 sentence-transformers
```

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/your-org/halocheck.git
cd halocheck

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install PyTorch
# CPU only:
pip install torch torchvision torchaudio
# GPU (CUDA 12.1 — adjust for your CUDA version):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install core dependencies
pip install transformers==4.41.0 `
            sentence-transformers `
            spacy `
            scispacy `
            medspacy `
            requests `
            beautifulsoup4 `
            datasets `
            ollama `
            streamlit `
            tqdm

# 5. Install scispaCy clinical NLP model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# 6. Install medspaCy dependency
python -m spacy download en_core_web_sm

# 7. Create required directories
mkdir data\raw, data\mednli, cache, models, evaluation

# 8. Set up HHEM isolated environment (for proper baseline — optional)
python -m venv venv_hhem
venv_hhem\Scripts\pip install torch transformers==4.38.2 sentence-transformers
```

---

## Project Structure

```
halocheck/
├── scripts/
│   ├── scraper_v2.py              # Multi-source corpus scraper
│   ├── filter_corpus_v2.py        # Corpus cleaning and filtering
│   ├── generate_summaries.py      # LLM summary generation + hallucination injection
│   ├── train_mednli.py            # Fine-tune Bio_ClinicalBERT on MedNLI
│   └── noop.py                    # Empty Layer 3 cache (Layer 3 disabled)
├── pipeline/
│   ├── layer1_ner.py              # NER entity checker (scispaCy)
│   ├── layer2_nli.py              # NLI fact grounder (MedNLI Bio_ClinicalBERT)
│   ├── layer3_negation.py         # Negation detector (disabled — net negative on v2)
│   ├── layer4_severity.py         # Severity mapping + deduplication
│   ├── hhem_baseline.py           # HHEM-2.1-Open baseline runner
│   └── hhem_inference.py          # HHEM sidecar (runs in venv_hhem)
├── evaluation/
│   └── evaluate.py                # Evaluation harness → results.json
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── data/
│   ├── raw/                       # Raw scraped notes
│   ├── mednli/                    # MedNLI dataset (from PhysioNet)
│   └── corpus.json                # Generated corpus with injections
├── cache/                         # Layer output caches
├── models/
│   └── mednli_clinicalbert/       # Fine-tuned model (after training)
└── run_pipeline.py                # Master pipeline runner
```

---

## MedNLI Dataset (Required for Layer 2 Upgrade)

The default Layer 2 uses DeBERTa (general NLI). For the full clinical NLI upgrade, obtain MedNLI from PhysioNet:

1. Create an account at https://physionet.org/register/
2. Complete CITI training at https://www.citiprogram.org — select "Data or Specimens Only Research"
3. Submit credentialing request with your CITI completion certificate
4. Request access to MedNLI v1.0.0 at https://physionet.org/content/mednli/1.0.0/
5. Download and place the three files in `data/mednli/`:

```
data/mednli/mli_train_v1.jsonl   (11,232 pairs)
data/mednli/mli_dev_v1.jsonl     (1,395 pairs)
data/mednli/mli_test_v1.jsonl    (1,422 pairs)
```

---

## Ollama Setup (Required for Corpus Generation Only)

Corpus generation uses Gemma 2 9B via Ollama. This is only needed for Step 2 — not for inference on existing data.

### macOS

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve

# Pull Gemma 2 9B (~5.4 GB download)
ollama pull gemma2:9b

# Verify
ollama list
```

### Windows

```powershell
# Download installer from: https://ollama.com/download/windows
# Run the installer, then open a new terminal:

# Pull Gemma 2 9B (~5.4 GB download)
ollama pull gemma2:9b

# Verify
ollama list
```

---

## Running the Pipeline

### Check current status

```bash
python run_pipeline.py --list
```

### Full pipeline from scratch (first time)

```bash
# Step 0 — Train MedNLI model (once only)
# ~25 min on GPU, ~3 hours on CPU
# Skip this if you want to use DeBERTa fallback
python run_pipeline.py --only 0

# Steps 1 through 8
python run_pipeline.py --from 1
```

### Skip MedNLI training (uses DeBERTa fallback for Layer 2)

```bash
python run_pipeline.py --from 1
```

### Resuming after a change

```bash
# Changed layer1_ner.py → re-run from Layer 1
python run_pipeline.py --from 3

# Changed layer2_nli.py → re-run from Layer 2
python run_pipeline.py --from 4

# Changed layer4_severity.py → re-run from Layer 4
python run_pipeline.py --from 6

# Changed evaluate.py only → re-run evaluation only (1 min)
python run_pipeline.py --from 8
```

> **Important:** Never re-run `--from 2` unless you intentionally changed `generate_summaries.py`. Step 2 takes ~2 hours and regenerates the entire corpus with new random injections, making all previous results non-comparable.

### Pipeline steps reference

| Step | Name | Script | Time | Notes |
|---|---|---|---|---|
| 0 | train_mednli | `scripts/train_mednli.py` | ~25 min GPU | Once only. Needs MedNLI files. |
| 1 | filter_corpus | `scripts/filter_corpus_v2.py` | ~1 min | Cleans raw scraped notes |
| 2 | generate | `scripts/generate_summaries.py` | ~2 hours | Needs Ollama + Gemma running |
| 3 | layer1 | `pipeline/layer1_ner.py` | ~5 min | scispaCy NER |
| 4 | layer2 | `pipeline/layer2_nli.py` | ~40 min | MedNLI model or DeBERTa fallback |
| 5 | layer3 | `scripts/noop.py` | <1 min | Disabled — net negative on v2 |
| 6 | layer4 | `pipeline/layer4_severity.py` | ~2 min | Severity mapping + dedup |
| 7 | hhem | `pipeline/hhem_baseline.py` | ~15 min | Needs venv_hhem for HHEM-2.1 |
| 8 | evaluate | `evaluation/evaluate.py` | ~1 min | Writes evaluation/results.json |

---

## Dashboard

```bash
# macOS
streamlit run dashboard/app.py

# Windows
streamlit run dashboard\app.py
```

Opens at http://localhost:8501

---

## Troubleshooting

### scispaCy model installation fails

```bash
# Install scispacy first, then the model wheel
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

### Ollama: connection refused

```bash
# Make sure Ollama is running before executing Step 2
# macOS
ollama serve

# Windows — check if Ollama is in the system tray, or start manually
ollama serve

# Check Gemma model is available
ollama list
# Should show: gemma2:9b
```

### Layer 2 using DeBERTa instead of MedNLI model

```bash
# macOS
ls models/mednli_clinicalbert/config.json

# Windows
dir models\mednli_clinicalbert\config.json

# If missing, run training
python run_pipeline.py --only 0
```

### HHEM falling back to BART-NLI

```bash
# macOS — check venv_hhem exists
ls venv_hhem/bin/python

# Windows
dir venv_hhem\Scripts\python.exe

# If missing, create and install
python -m venv venv_hhem

# macOS
venv_hhem/bin/pip install torch transformers==4.38.2 sentence-transformers

# Windows
venv_hhem\Scripts\pip install torch transformers==4.38.2 sentence-transformers
```

### GPU not detected

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If False, reinstall PyTorch with the correct CUDA version for your system
# See: https://pytorch.org/get-started/locally/
```

### Windows: long path errors during pip install

```powershell
# Run PowerShell as Administrator
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1
# Restart terminal after applying
```

### medspaCy install fails due to spacy version conflict

```bash
pip install "medspacy>=1.0.0" "spacy>=3.4,<4.0"
python -m spacy download en_core_web_sm
```

### Step 2 hangs or Gemma produces empty output

```bash
# Test Ollama is responsive
ollama run gemma2:9b "Hello, respond with one word."

# Check Ollama logs
# macOS
cat ~/.ollama/logs/server.log | tail -20

# Windows
type %USERPROFILE%\.ollama\logs\server.log
```

---

## Limitations

- Evaluated on synthetically injected hallucinations. Real-world LLM hallucinations may differ in pattern and frequency.
- MedNLI fine-tuned model is trained on MIMIC-III derived data under a PhysioNet data use agreement and cannot be redistributed.
- `semantic_contradiction` detection has high false positive rate (precision 0.098) due to legitimate clinical paraphrasing triggering NLI contradiction scores.
- Severity accuracy (0.344) is below target — no Minor-severity injection class exists in the current corpus.
- Not validated for clinical deployment. Research prototype only.

---

## Citation

```
HaloCheck: Severity-Aware Hallucination Detection for LLM-Generated Clinical Summaries
Kadhiravan Gopal, Shushrutha Rami Reddy, Ritvik Narayan Shetty,
Vaibhav Pitambar Patil, Priyansh Salian
CS595 Medical Informatics AI, Illinois Institute of Technology, Spring 2026
```
