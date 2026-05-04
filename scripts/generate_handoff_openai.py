"""
generate_handoff.py — Generate handoff summaries from MIMIC-III notes

Takes real MIMIC-III discharge summaries and asks GPT-4o mini to generate
a shorter handoff summary. No error injection — GPT hallucinates naturally.

The natural hallucinations are then caught by HaloCheck and verified
against the EHR ground truth tables (labs, meds, diagnoses).

Run:
  python scripts/generate_handoff.py
  python scripts/generate_handoff.py --max 50   # quick test with 50 notes

Requires:
  - data/corpus_mimic.json  (from mimic_extractor.py)
  - OPENAI_API_KEY set in this file or as environment variable
"""

import json
import time
import os
import argparse
from pathlib import Path
from openai import OpenAI

# ── API Key ───────────────────────────────────────────────────────────────────

OPENAI_API_KEY = ""
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


client     = OpenAI(api_key=OPENAI_API_KEY)
_total_cost = 0.0
print("✓ OpenAI GPT-4o mini ready")

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_PATH  = Path("data/corpus_mimic.json")
OUTPUT_PATH = Path("data/corpus_mimic.json")   # overwrite with summaries added

REQUESTS_PER_MINUTE   = 50
SECONDS_BETWEEN_CALLS = 60.0 / REQUESTS_PER_MINUTE
MAX_RETRIES           = 5
RETRY_WAIT            = 15

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

# ── Rate-limited caller ───────────────────────────────────────────────────────

_last_call_time = 0.0

def generate_summary(source_text: str) -> str:
    global _last_call_time, _total_cost

    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = time.time() - _last_call_time
        if elapsed < SECONDS_BETWEEN_CALLS:
            time.sleep(SECONDS_BETWEEN_CALLS - elapsed)

        try:
            _last_call_time = time.time()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_PROMPT.format(
                        source_text=source_text[:3000]
                    )}
                ],
                max_tokens=500,
                temperature=0.3
            )

            # Track cost
            usage = response.usage
            cost  = (usage.prompt_tokens * 0.00000015) + (usage.completion_tokens * 0.0000006)
            _total_cost += cost

            return response.choices[0].message.content.strip()

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                print(f"    [rate limit] attempt {attempt}/{MAX_RETRIES} — waiting {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
            elif "insufficient_quota" in error_msg or "billing" in error_msg.lower():
                print(f"\n❌  OpenAI quota exceeded. Check: https://platform.openai.com/usage")
                raise
            else:
                print(f"    [error] attempt {attempt}/{MAX_RETRIES}: {error_msg[:80]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT)

    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",   dest="input",  type=Path, default=INPUT_PATH)
    parser.add_argument("--out",  dest="output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max",  dest="max",    type=int,  default=None,
                        help="Max notes to process (default: all)")
    args = parser.parse_args()

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

    estimated_cost = len(to_process) * 0.0002
    print(f"\nLoaded {len(corpus)} notes")
    print(f"  Already have summaries : {already}")
    print(f"  Need to generate       : {len(to_process)}")
    print(f"  Estimated cost         : ~${estimated_cost:.2f}")
    print(f"  Estimated time         : ~{len(to_process) * SECONDS_BETWEEN_CALLS / 60:.0f} minutes\n")

    failed = 0
    for i, note in enumerate(to_process):
        print(f"  [{i+1}/{len(to_process)}] {note['note_id']}  (spent: ${_total_cost:.4f})")

        summary = generate_summary(note["source_text"])

        if not summary:
            summary = note["source_text"][:300]
            failed += 1
            print(f"    [warn] fallback used")

        note["test_summary"] = summary

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✓ Saved → {args.output}")
    print(f"  Summaries generated : {len(to_process) - failed}")
    print(f"  Fallbacks used      : {failed}")
    print(f"  Total cost          : ${_total_cost:.4f}")
    print(f"\nNext: python run_pipeline.py --from 3")
    print(f"  Then: python evaluation/evaluate_ehr.py")


if __name__ == "__main__":
    main()
