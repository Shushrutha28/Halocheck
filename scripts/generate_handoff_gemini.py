"""
generate_handoff.py — Generate handoff summaries from MIMIC-III notes using Google Gemini

Takes real MIMIC-III discharge summaries and asks Gemini to generate a shorter
handoff summary. No error injection — Gemini hallucinates naturally.
The natural hallucinations are then caught by HaloCheck and verified against
the EHR ground truth tables (labs, meds, diagnoses).

Run:
  python scripts/generate_handoff.py
  python scripts/generate_handoff.py --max 50
  python scripts/generate_handoff.py --model gemini-2.5-flash

Setup:
  pip install google-genai
  Set your API key below or as environment variable GEMINI_API_KEY
  Get a free API key at: https://aistudio.google.com/apikey

Supported models (as of 2025):
  gemini-2.5-flash   ← recommended — fast, cheap, good quality
  gemini-2.5-pro     ← higher quality, slower, more expensive
  gemini-2.0-flash   ← fastest, most economical
"""

import json
import time
import os
import argparse
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import APIError

# ── API Key ───────────────────────────────────────────────────────────────────
# Option 1: hardcode here (not recommended for shared code)
GEMINI_API_KEY = ""   # paste your key here

# Option 2: set environment variable GEMINI_API_KEY (recommended)
# export GEMINI_API_KEY="your-key-here"  (Mac/Linux)
# $env:GEMINI_API_KEY="your-key-here"    (Windows PowerShell)

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_PATH    = Path("data/corpus_mimic.json")
OUTPUT_PATH   = Path("data/corpus_mimic.json")   # overwrite with summaries added
DEFAULT_MODEL = "gemini-2.5-flash"

MAX_RETRIES   = 5
RETRY_WAIT    = 15   # seconds between retries on rate limit

# Approximate cost per note (input ~800 tokens + output ~200 tokens)
# gemini-2.5-flash: $0.000075/1K input + $0.0003/1K output
COST_PER_NOTE = {
    "gemini-2.5-flash": 0.00012,
    "gemini-2.5-pro":   0.00225,
    "gemini-2.0-flash": 0.000075,
}

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant helping with hospital handoffs. "
    "Read the discharge summary provided and write a concise handoff summary "
    "for the incoming care team.\n\n"
    "Requirements:\n"
    "- 4-8 sentences maximum\n"
    "- Include: primary diagnosis, key medications with doses, critical lab "
    "values, allergies if mentioned\n"
    "- Be faithful to the source — do not add information not in the original note\n"
    "- Write in clinical shorthand style appropriate for physician handoffs"
)

USER_PROMPT = (
    "Write a handoff summary for the following discharge note:\n\n"
    "{source_text}\n\n"
    "Handoff Summary:"
)


# ── Gemini client ─────────────────────────────────────────────────────────────

def build_client() -> genai.Client:
    """Build Gemini client from API key."""
    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "\n❌  No Gemini API key found.\n"
            "  Option 1: Set GEMINI_API_KEY in this file\n"
            "  Option 2: Set environment variable:\n"
            "    $env:GEMINI_API_KEY='your-key'  (PowerShell)\n"
            "    export GEMINI_API_KEY='your-key'  (Mac/Linux)\n"
            "  Get a free key at: https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def generate_summary(client: genai.Client, source_text: str, model: str) -> str:
    """
    Call Gemini to generate a handoff summary.
    Returns the summary string or empty string on failure.
    """
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT.format(source_text=source_text[:4000])}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                    stop_sequences=["\n\n\n"],
                ),
            )
            return response.text.strip() if response.text else ""

        except APIError as e:
            error_str = str(e)

            # Rate limit — wait and retry
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = RETRY_WAIT * attempt   # exponential back-off
                print(f"\n    [rate limit] attempt {attempt}/{MAX_RETRIES} "
                      f"— waiting {wait}s...")
                time.sleep(wait)

            # Quota exceeded — fatal
            elif "quota" in error_str.lower() or "billing" in error_str.lower():
                print(f"\n❌  Gemini quota exceeded.")
                print("  Check: https://aistudio.google.com/apikey")
                print("  Or upgrade at: https://ai.google.dev/pricing")
                raise

            # Safety block — skip this note
            elif "SAFETY" in error_str or "blocked" in error_str.lower():
                print(f"\n    [safety block] note skipped by Gemini safety filter")
                return ""

            # Other error — retry
            else:
                print(f"\n    [error] attempt {attempt}/{MAX_RETRIES}: "
                      f"{error_str[:100]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT)

        except Exception as e:
            print(f"\n    [unexpected] attempt {attempt}/{MAX_RETRIES}: {str(e)[:100]}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)

    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate handoff summaries from MIMIC-III notes using Google Gemini"
    )
    parser.add_argument("--in",    dest="input",  type=Path, default=INPUT_PATH)
    parser.add_argument("--out",   dest="output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max",   dest="max",    type=int,  default=None,
                        help="Max notes to process (default: all)")
    parser.add_argument("--model", dest="model",  type=str,  default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    print(f"\nHaloCheck — MIMIC Handoff Generator (Google Gemini)")
    print(f"  Model  : {args.model}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")

    # Build client — will raise if no API key
    client = build_client()
    print(f"\n✓ Gemini client ready\n")

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

    # Skip notes that already have summaries
    to_process = [n for n in corpus if not n.get("test_summary", "").strip()]
    already    = len(corpus) - len(to_process)

    cost_per = COST_PER_NOTE.get(args.model, 0.00015)
    est_cost = len(to_process) * cost_per

    print(f"Loaded {len(corpus)} notes")
    print(f"  Already have summaries : {already}")
    print(f"  Need to generate       : {len(to_process)}")
    print(f"  Estimated cost         : ~${est_cost:.3f} USD")
    print(f"  Rate limit             : auto-managed with back-off\n")

    if len(to_process) == 0:
        print("All notes already have summaries.")
        return

    failed  = 0
    t_start = time.time()

    for i, note in enumerate(to_process):
        elapsed = time.time() - t_start
        eta     = (elapsed / (i + 1)) * (len(to_process) - i - 1) if i > 0 else 0
        eta_min = eta / 60

        print(f"  [{i+1:>3}/{len(to_process)}] {note['note_id']}"
              f"  ETA: {eta_min:.1f}m", end=" ", flush=True)

        summary = generate_summary(client, note["source_text"], args.model)

        if not summary:
            # Fallback — use first 300 chars of source
            summary = note["source_text"][:300]
            failed += 1
            print("→ FAILED (fallback used)")
        else:
            print(f"→ {len(summary.split())} words")

        note["test_summary"] = summary
        note["generator"]    = args.model   # track which model generated this

        # Save checkpoint every 10 notes — resume-safe
        if (i + 1) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(corpus, f, indent=2)
            print(f"  [checkpoint] saved at note {i+1}")

    # Final save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    total_time = (time.time() - t_start) / 60
    actual_cost = (len(to_process) - failed) * cost_per

    print(f"\n{'='*50}")
    print(f"✓ Saved → {args.output}")
    print(f"  Summaries generated : {len(to_process) - failed}")
    print(f"  Fallbacks used      : {failed}")
    print(f"  Total time          : {total_time:.1f} min")
    print(f"  Avg per note        : {total_time * 60 / max(len(to_process), 1):.1f}s")
    print(f"  Approx cost         : ~${actual_cost:.4f} USD")
    print(f"\nNext steps:")
    print(f"  python run_pipeline.py --from 3")
    print(f"  python evaluation/evaluate_ehr.py")


if __name__ == "__main__":
    main()
