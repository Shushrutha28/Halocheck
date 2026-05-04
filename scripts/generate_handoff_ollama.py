"""
generate_handoff.py — Generate handoff summaries from MIMIC-III notes using Gemma 2 9B

Takes real MIMIC-III discharge summaries and asks Gemma 2 9B (via Ollama)
to generate a shorter handoff summary. No error injection — Gemma hallucinates
naturally. The natural hallucinations are then caught by HaloCheck and verified
against the EHR ground truth tables (labs, meds, diagnoses).

Run:
  python scripts/generate_handoff.py
  python scripts/generate_handoff.py --max 50   # quick test with 50 notes
  python scripts/generate_handoff.py --model gemma2:9b  # explicit model

Requires:
  - data/corpus_mimic.json  (from mimic_extractor.py)
  - Ollama running with Gemma 2 9B pulled:
      ollama serve
      ollama pull gemma2:9b
"""

import json
import time
import argparse
import requests
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_PATH  = Path("data/corpus_mimic.json")
OUTPUT_PATH = Path("data/corpus_mimic.json")   # overwrite with summaries added

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma2:latest"  # use latest tag to automatically get updates (e.g. new Gemma 2.1)
MAX_RETRIES   = 3
RETRY_WAIT    = 5    # seconds between retries on failure

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical documentation assistant helping with hospital handoffs.
Read the discharge summary provided and write a concise handoff summary for the incoming care team.

Requirements:
- 4-8 sentences maximum
- Include: primary diagnosis, key medications with doses, critical lab values, allergies if mentioned
- Be faithful to the source — do not add information not in the original note
- Write in clinical shorthand style appropriate for physician handoffs"""

USER_PROMPT = """Write a handoff summary for the following discharge note:

{source_text}

Handoff Summary:"""


# ── Ollama client ─────────────────────────────────────────────────────────────

def check_ollama(model: str) -> bool:
    """Verify Ollama is running and the model is available."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        available = [m["name"] for m in r.json().get("models", [])]
        # Check if model is available (handles tags like gemma2:9b or gemma2)
        for m in available:
            if model.split(":")[0] in m:
                return True
        print(f"\n⚠️  Model '{model}' not found in Ollama.")
        print(f"   Available: {available}")
        print(f"   Run: ollama pull {model}")
        return False
    except requests.exceptions.ConnectionError:
        print("\n❌  Ollama not running. Start it with: ollama serve")
        return False


import ollama

def generate_summary(source_text: str, model: str) -> str:
    """
    Use Ollama chat API (same as your working script)
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(source_text=source_text[:3000])
                },
            ],
            options={
                "temperature": 0.3,
                "num_predict": 500,
            }
        )

        return response["message"]["content"].strip()

    except Exception as e:
        print(f"    [error] {str(e)[:80]}")
        return ""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate handoff summaries from MIMIC-III notes using Gemma 2 9B (Ollama)"
    )
    parser.add_argument("--in",    dest="input",  type=Path,  default=INPUT_PATH)
    parser.add_argument("--out",   dest="output", type=Path,  default=OUTPUT_PATH)
    parser.add_argument("--max",   dest="max",    type=int,   default=None,
                        help="Max notes to process (default: all)")
    parser.add_argument("--model", dest="model",  type=str,   default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip notes that already have summaries (default: True)")
    args = parser.parse_args()

    print(f"\nHaloCheck — MIMIC Handoff Generator")
    print(f"  Model  : {args.model} (via Ollama)")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")

    # Check Ollama is up
    if not check_ollama(args.model):
        print("\nSetup:")
        print("  1. Install Ollama: https://ollama.com/download")
        print("  2. ollama serve")
        print(f"  3. ollama pull {args.model}")
        return

    print(f"\n✓ Ollama + {args.model} ready\n")

    # Load corpus
    if not args.input.exists():
        raise FileNotFoundError(
            f"\n❌  Input not found: {args.input}\n"
            "Run scripts/mimic_extractor.py first.\n"
        )
    with open(args.input, encoding="utf-8") as f:
        corpus = json.load(f)

    if args.max:
        corpus = corpus[:args.max]

    # Skip notes that already have summaries (resume support)
    to_process = [n for n in corpus if not n.get("test_summary", "").strip()]
    already    = len(corpus) - len(to_process)

    print(f"Loaded {len(corpus)} notes")
    print(f"  Already have summaries : {already}")
    print(f"  Need to generate       : {len(to_process)}")
    print(f"  Cost                   : $0.00 (local Ollama)\n")

    if len(to_process) == 0:
        print("All notes already have summaries. Use --resume=false to regenerate.")
        return

    # Estimate time
    # Gemma 2 9B: ~20-40s per note on CPU, ~5-10s on GPU
    est_min = len(to_process) * 30 / 60
    print(f"  Estimated time: ~{est_min:.0f} min on CPU, ~{est_min/4:.0f} min on GPU")
    print(f"  (Tip: run overnight for large batches)\n")

    failed    = 0
    t_start   = __import__("time").time()

    for i, note in enumerate(to_process):
        elapsed  = __import__("time").time() - t_start
        eta      = (elapsed / (i + 1)) * (len(to_process) - i - 1) if i > 0 else 0
        eta_min  = eta / 60

        print(f"  [{i+1:>3}/{len(to_process)}] {note['note_id']}"
              f"  ETA: {eta_min:.1f}m", end=" ", flush=True)

        summary = generate_summary(note["source_text"], args.model)

        if not summary:
            # Fallback — use first 300 chars of source
            summary = note["source_text"][:300]
            failed += 1
            print("→ FAILED (fallback used)")
        else:
            note_len = len(summary.split())
            print(f"→ {note_len} words")

        note["test_summary"] = summary
        note["generator"]    = args.model   # track which model generated this

        # Save incrementally every 10 notes so progress isn't lost
        if (i + 1) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(corpus, f, indent=2)
            print(f"  [checkpoint] saved {i+1} notes")

    # Final save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    total_time = (__import__("time").time() - t_start) / 60
    print(f"\n{'='*50}")
    print(f"✓ Saved → {args.output}")
    print(f"  Summaries generated : {len(to_process) - failed}")
    print(f"  Fallbacks used      : {failed}")
    print(f"  Total time          : {total_time:.1f} min")
    print(f"  Avg per note        : {total_time * 60 / max(len(to_process), 1):.1f}s")
    print(f"\nNext steps:")
    print(f"  python run_pipeline.py --from 3")
    print(f"  python evaluation/evaluate_ehr.py")


if __name__ == "__main__":
    main()
