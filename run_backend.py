"""Run an inference backend against all prompts (or a dry-run subset).

CLI:
    python run_backend.py              # full run
    python run_backend.py --dry-run    # 3 prompts, validate, print latency
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import sys
from pathlib import Path

from constants import DRY_RUN_PROMPTS, TIMEOUT_SECONDS
from protocol import InferenceBackend

PROMPTS_FILE = Path("data/prompts.jsonl")
COMPLETIONS_DIR = Path("completions")
RESULTS_DIR = Path("results")
SUMMARY_FILE = RESULTS_DIR / "summary.jsonl"


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _load_backend() -> InferenceBackend:
    spec = importlib.util.spec_from_file_location("backend", "backends/backend.py")
    if spec is None or spec.loader is None:
        print("ERROR: Cannot load backends/backend.py", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "Backend"):
        print("ERROR: backends/backend.py must define a Backend class", file=sys.stderr)
        sys.exit(1)

    backend = mod.Backend()
    if not isinstance(backend, InferenceBackend):
        print("ERROR: Backend does not conform to InferenceBackend protocol", file=sys.stderr)
        sys.exit(1)

    return backend


def _get_trial_name() -> str:
    spec = importlib.util.spec_from_file_location("backend", "backends/backend.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "TRIAL_NAME", "unnamed")


def _next_trial_number() -> int:
    """Determine next trial number from summary.jsonl line count.

    Each completed trial (ok, crash, timeout) appends exactly one line.
    This is more reliable than counting files or parsing filenames.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE) as f:
            n = sum(1 for line in f if line.strip())
        return n + 1
    return 1


def _load_prompts() -> list[dict]:
    if not PROMPTS_FILE.exists():
        print(f"ERROR: {PROMPTS_FILE} not found. Run setup.py first.", file=sys.stderr)
        sys.exit(1)
    with open(PROMPTS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference backend")
    parser.add_argument("--dry-run", action="store_true", help="Run only a few prompts for validation")
    parser.add_argument("--trial-num", type=int, default=None,
                        help="Explicit trial number (overrides auto-detection)")
    args = parser.parse_args()

    prompts = _load_prompts()
    trial_name = _get_trial_name()
    trial_num = args.trial_num if args.trial_num is not None else _next_trial_number()
    run_id = f"trial-{trial_num}-{_slugify(trial_name)}"

    if args.dry_run:
        prompts = prompts[:DRY_RUN_PROMPTS]
        print(f"=== DRY RUN ({len(prompts)} prompts) ===")
    else:
        print(f"=== FULL RUN: {run_id} ({len(prompts)} prompts) ===")

    # Load and setup backend
    backend = _load_backend()
    print(f"Setting up backend: {trial_name}")
    backend.setup()

    COMPLETIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = COMPLETIONS_DIR / f"{run_id}.jsonl"

    try:
        with open(output_path, "w") as out:
            for i, entry in enumerate(prompts):
                result = backend.generate(entry["prompt"])
                record = {
                    "prompt": entry["prompt"],
                    "gold": entry["gold"],
                    "task": entry["task"],
                    "answer_type": entry["answer_type"],
                    "completion": result.completion,
                    "raw_output": result.raw_output,
                    "latency_ms": result.latency_ms,
                    "tokens_generated": result.tokens_generated,
                    "prompt_tokens": result.prompt_tokens,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()

                if args.dry_run:
                    print(f"  [{i+1}/{len(prompts)}] latency={result.latency_ms:.1f}ms "
                          f"tokens={result.tokens_generated} "
                          f"completion={result.completion[:80]!r}")
                else:
                    print(f"  [{i+1}/{len(prompts)}] {result.latency_ms:.1f}ms", end="\r")

        if not args.dry_run:
            print()
    finally:
        backend.teardown()

    if args.dry_run:
        print("Dry run passed.")
    else:
        print(f"Completions written to {output_path}")
        # Machine-readable line for harness to capture the output path
        print(f"COMPLETIONS_FILE={output_path}")


if __name__ == "__main__":
    main()
