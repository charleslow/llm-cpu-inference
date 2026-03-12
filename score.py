"""Score completions and append results to summary.jsonl.

CLI:
    python score.py --file completions/trial-5-foo.jsonl   # score a specific file
    python score.py                                        # score latest (by mtime)
    python score.py --status crash                         # log a crash (0 accuracy)
    python score.py --status timeout                       # log a timeout (0 accuracy)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

COMPLETIONS_DIR = Path("completions")
RESULTS_DIR = Path("results")
SUMMARY_FILE = RESULTS_DIR / "summary.jsonl"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_choice(raw_output: str) -> str:
    """Extract a letter answer (A-E) from model output."""
    text = raw_output.strip()

    # 1. Entire output is a single letter
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()

    # 2. Pattern: "answer is A" / "correct: B"
    m = re.search(r"(?:answer|correct)\s*(?:is|:)\s*([A-E])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. First standalone letter A-E
    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1).upper()

    # 4. First character uppercased
    if text:
        return text[0].upper()

    return ""


def extract_numeric(raw_output: str) -> str:
    """Extract a numeric answer from model output."""
    text = raw_output.strip()

    # 1. #### pattern (GSM8K style)
    m = re.search(r"####\s*(.+?)(?:\s|$)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    # 2. "answer is 42" / "result = 42"
    m = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*([\d,.\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()

    # 3. Last number in text
    numbers = re.findall(r"(-?[\d,]+\.?\d*)", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""


def _compare_numeric(predicted: str, gold: str) -> bool:
    """Compare numeric answers: try float comparison, fallback to string."""
    try:
        return float(predicted) == float(gold)
    except (ValueError, TypeError):
        return predicted.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def score_entry(entry: dict) -> dict:
    """Score a single completion entry."""
    answer_type = entry["answer_type"]
    raw_output = entry.get("raw_output", entry.get("completion", ""))
    gold = entry["gold"]

    if answer_type == "choice":
        predicted = extract_choice(raw_output)
        correct = predicted.upper() == gold.upper()
    elif answer_type == "numeric":
        predicted = extract_numeric(raw_output)
        correct = _compare_numeric(predicted, gold)
    else:
        predicted = raw_output.strip()
        correct = predicted == gold

    return {
        "prompt": entry["prompt"][:200],  # truncate for readability
        "task": entry["task"],
        "gold": gold,
        "predicted": predicted,
        "correct": correct,
        "latency_ms": entry.get("latency_ms", 0),
    }


def _find_latest_completion() -> Path | None:
    """Find the most recently created completion file (by modification time)."""
    files = list(COMPLETIONS_DIR.glob("trial-*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _parse_run_id(path: Path) -> tuple[str, int]:
    """Extract run_id and trial number from a completion file path."""
    run_id = path.stem  # e.g. trial-3-qwen-q4-llama-cpp
    m = re.match(r"trial-(\d+)", run_id)
    trial_num = int(m.group(1)) if m else 0
    return run_id, trial_num


def _read_backend_metadata(backend_path: str = "backends/backend.py") -> dict:
    """Try to extract model/framework/quant info from a backend file's TRIAL_NAME."""
    try:
        with open(backend_path) as f:
            source = f.read()
        m = re.search(r'TRIAL_NAME\s*=\s*["\'](.+?)["\']', source)
        name = m.group(1) if m else "unknown"
        return {"description": name}
    except Exception:
        return {"description": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="ok", choices=["ok", "crash", "timeout"])
    parser.add_argument("--file", type=Path, default=None,
                        help="Explicit completion file to score (avoids guessing latest)")
    parser.add_argument("--backend", type=str, default="backends/backend.py",
                        help="Path to the backend file (for metadata extraction)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # For crash/timeout, write a summary line with 0 accuracy and exit
    if args.status in ("crash", "timeout"):
        meta = _read_backend_metadata(args.backend)
        latest = _find_latest_completion()
        if latest:
            run_id, _ = _parse_run_id(latest)
        else:
            run_id = "unknown"
        summary = {
            "run_id": run_id,
            "model": "",
            "framework": "",
            "quant": "",
            "accuracy": 0.0,
            "avg_latency_ms": 0.0,
            "p90_latency_ms": 0.0,
            "total_time_s": 0.0,
            "status": args.status,
            "description": meta.get("description", ""),
        }
        with open(SUMMARY_FILE, "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"Logged {args.status}: {run_id}")
        return

    # Normal scoring — use explicit file if given, else find latest by mtime
    if args.file:
        latest = args.file
        if not latest.exists():
            print(f"ERROR: Specified file {latest} does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        latest = _find_latest_completion()
        if not latest:
            print("ERROR: No completion files found in completions/", file=sys.stderr)
            sys.exit(1)

    run_id, _ = _parse_run_id(latest)
    print(f"Scoring {latest.name}...")

    with open(latest) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    results = [score_entry(e) for e in entries]

    # Write per-prompt results
    results_file = RESULTS_DIR / f"{run_id}.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute aggregate metrics
    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / len(results) if results else 0.0
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p90_latency = sorted(latencies)[int(len(latencies) * 0.9)] if latencies else 0.0
    total_time = sum(latencies) / 1000.0 if latencies else 0.0

    # Print per-task breakdown
    from collections import Counter
    task_correct: dict[str, int] = Counter()
    task_total: dict[str, int] = Counter()
    for r in results:
        task_total[r["task"]] += 1
        if r["correct"]:
            task_correct[r["task"]] += 1

    print(f"\n{'Task':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for task in sorted(task_total.keys()):
        tc = task_correct[task]
        tt = task_total[task]
        acc = tc / tt if tt else 0
        print(f"{task:<20} {tc:>8} {tt:>8} {acc:>10.1%}")
    print("-" * 50)
    print(f"{'OVERALL':<20} {n_correct:>8} {len(results):>8} {accuracy:>10.1%}")
    print(f"\nAvg latency: {avg_latency:.1f}ms | P90: {p90_latency:.1f}ms | Total: {total_time:.1f}s")

    # Append summary
    meta = _read_backend_metadata(args.backend)
    summary = {
        "run_id": run_id,
        "model": "",
        "framework": "",
        "quant": "",
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "p90_latency_ms": round(p90_latency, 1),
        "total_time_s": round(total_time, 1),
        "status": "ok",
        "description": meta.get("description", ""),
    }
    with open(SUMMARY_FILE, "a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"\nResults: {results_file}")
    print(f"Summary appended to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
