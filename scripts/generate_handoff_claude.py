"""
generate_handoff_claude.py — Generate handoff summaries using Claude Sonnet 4.6

Takes real MIMIC-III discharge summaries and asks Claude Sonnet 4.6 to generate
a shorter handoff summary. No error injection — Claude hallucinates naturally.
The natural hallucinations are then caught by HaloCheck and verified against
the EHR ground truth tables (labs, meds, diagnoses).

Run:
  python scripts/generate_handoff_claude.py
  python scripts/generate_handoff_claude.py --max 50

Setup:
  pip install anthropic
  Set your API key below or as environment variable ANTHROPIC_API_KEY
  Get a key at: https://console.anthropic.com/

Model: claude-sonnet-4-6
  Smart, efficient, fast — ideal for batch summarization tasks
  Cost: ~$0.00045 per note (200 notes ≈ $0.09)
"""

import json
import time
import os
import argparse
from pathlib import Path

import anthropic
from anthropic import APIStatusError, APIConnectionError, RateLimitError

# ── API Key ───────────────────────────────────────────────────────────────────
# Option 1: hardcode here
ANTHROPIC_API_KEY = ""   # paste your key here

# Option 2: set environment variable ANTHROPIC_API_KEY (recommended)
# $env:ANTHROPIC_API_KEY="your-key"    (Windows PowerShell)
# export ANTHROPIC_API_KEY="your-key"  (Mac/Linux)

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_PATH    = Path("data/corpus_mimic.json")
OUTPUT_PATH   = Path("data/corpus_mimic.json")   # overwrite with summaries
MODEL         = "claude-sonnet-4-6"
MAX_TOKENS    = 500
TEMPERATURE   = 1.0    # Claude API requires temperature=1 for standard generation

MAX_RETRIES   = 5
RETRY_WAIT    = 15   # base wait in seconds — doubles on each retry

# Cost estimate per note (input ~800 tokens + output ~200 tokens)
# claude-sonnet-4-6: $0.003/1K input + $0.015/1K output
COST_PER_NOTE = 0.00045

# ── Prompts ───────────────────────────────────────────────────────────────────

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


# ── Client ────────────────────────────────────────────────────────────────────

def build_client() -> anthropic.Anthropic:
    """Build Anthropic client from API key."""
    api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "\n❌  No Anthropic API key found.\n"
            "  Option 1: Set ANTHROPIC_API_KEY in this file\n"
            "  Option 2: Set environment variable:\n"
            "    $env:ANTHROPIC_API_KEY='your-key'  (PowerShell)\n"
            "    export ANTHROPIC_API_KEY='your-key'  (Mac/Linux)\n"
            "  Get a key at: https://console.anthropic.com/"
        )
    return anthropic.Anthropic(api_key=api_key)


def generate_summary(client: anthropic.Anthropic, source_text: str) -> str:
    """
    Call Claude Sonnet 4.6 to generate a handoff summary.
    Returns the summary string or empty string on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(
                            source_text=source_text[:4000]
                        )
                    }
                ],
            )

            # Extract text from response
            for block in message.content:
                if block.type == "text":
                    return block.text.strip()
            return ""

        except RateLimitError:
            wait = RETRY_WAIT * (2 ** (attempt - 1))   # exponential back-off
            print(f"\n    [rate limit] attempt {attempt}/{MAX_RETRIES} "
                  f"— waiting {wait}s...")
            time.sleep(wait)

        except APIStatusError as e:
            # Overloaded — treat like rate limit
            if e.status_code == 529 or "overloaded" in str(e).lower():
                wait = RETRY_WAIT * attempt
                print(f"\n    [overloaded] attempt {attempt}/{MAX_RETRIES} "
                      f"— waiting {wait}s...")
                time.sleep(wait)

            # Credit/billing issue — fatal
            elif e.status_code in (401, 403):
                print(f"\n❌  Authentication error: {e.message}")
                print("  Check your API key at: https://console.anthropic.com/")
                raise

            # Other API error — retry
            else:
                print(f"\n    [API error {e.status_code}] attempt "
                      f"{attempt}/{MAX_RETRIES}: {str(e)[:100]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT)

        except APIConnectionError:
            print(f"\n    [connection error] attempt {attempt}/{MAX_RETRIES} "
                  f"— waiting {RETRY_WAIT}s...")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)

        except Exception as e:
            print(f"\n    [unexpected error] attempt {attempt}/{MAX_RETRIES}: "
                  f"{str(e)[:100]}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)

    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate handoff summaries from MIMIC-III notes using Claude Sonnet 4.6"
    )
    parser.add_argument("--in",  dest="input",  type=Path, default=INPUT_PATH)
    parser.add_argument("--out", dest="output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max", dest="max",    type=int,  default=None,
                        help="Max notes to process (default: all)")
    args = parser.parse_args()

    print(f"\nHaloCheck — MIMIC Handoff Generator (Claude Sonnet 4.6)")
    print(f"  Model  : {MODEL}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")

    # Build client — raises if no API key
    client = build_client()
    print(f"\n✓ Anthropic client ready\n")

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
    est_cost   = len(to_process) * COST_PER_NOTE

    print(f"Loaded {len(corpus)} notes")
    print(f"  Already have summaries : {already}")
    print(f"  Need to generate       : {len(to_process)}")
    print(f"  Estimated cost         : ~${est_cost:.3f} USD")
    print(f"  Rate limits            : auto-managed with exponential back-off\n")

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

        summary = generate_summary(client, note["source_text"])

        if not summary:
            summary = note["source_text"][:300]   # fallback
            failed += 1
            print("→ FAILED (fallback used)")
        else:
            print(f"→ {len(summary.split())} words")

        note["test_summary"] = summary
        note["generator"]    = MODEL   # track which model generated this

        # Checkpoint every 10 notes — safe to resume after interruption
        if (i + 1) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(corpus, f, indent=2)
            print(f"  [checkpoint] saved at note {i+1}")

    # Final save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    total_time  = (time.time() - t_start) / 60
    actual_cost = (len(to_process) - failed) * COST_PER_NOTE

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
