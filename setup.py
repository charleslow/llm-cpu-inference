"""One-time setup: generate data/prompts.jsonl from HF datasets.

Distribution (configurable via NUM_PROMPTS in constants.py):
  MMLU          25%
  ARC-Challenge 20%
  HellaSwag     20%
  GSM8K         15%
  WinoGrande    10%
  TruthfulQA    10%
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

from datasets import load_dataset

from constants import NUM_PROMPTS, SEED

DATA_DIR = Path("data")
OUTPUT = DATA_DIR / "prompts.jsonl"

# Distribution ratios — must sum to 1.0
TASK_RATIOS = {
    "mmlu": 0.25,
    "arc_challenge": 0.20,
    "hellaswag": 0.20,
    "gsm8k": 0.15,
    "winogrande": 0.10,
    "truthfulqa": 0.10,
}

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _compute_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Distribute total prompts according to ratios, ensuring exact total."""
    raw = {k: v * total for k, v in ratios.items()}
    counts = {k: math.floor(v) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    # Give remainder to tasks with largest fractional parts
    frac = sorted(raw.keys(), key=lambda k: raw[k] - counts[k], reverse=True)
    for k in frac[:remainder]:
        counts[k] += 1
    return counts


def _format_mmlu(example: dict) -> dict:
    question = example["question"]
    choices = example["choices"]
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    prompt = f"{question}\n\n{options}\n\nAnswer:"
    gold = LETTERS[example["answer"]]
    return {"prompt": prompt, "gold": gold, "task": "mmlu", "answer_type": "choice"}


def _format_arc(example: dict) -> dict:
    question = example["question"]
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    prompt = f"{question}\n\n{options}\n\nAnswer:"
    gold = example["answerKey"]
    return {"prompt": prompt, "gold": gold, "task": "arc_challenge", "answer_type": "choice"}


def _format_hellaswag(example: dict) -> dict:
    ctx = example["ctx"]
    endings = example["endings"]
    options = "\n".join(f"{LETTERS[i]}. {e}" for i, e in enumerate(endings))
    prompt = f"{ctx}\n\n{options}\n\nAnswer:"
    gold = LETTERS[int(example["label"])]
    return {"prompt": prompt, "gold": gold, "task": "hellaswag", "answer_type": "choice"}


def _format_gsm8k(example: dict) -> dict:
    question = example["question"]
    prompt = f"{question}\n\nAnswer:"
    # Gold answer is after ####
    answer_text = example["answer"]
    gold = answer_text.split("####")[-1].strip()
    return {"prompt": prompt, "gold": gold, "task": "gsm8k", "answer_type": "numeric"}


def _format_winogrande(example: dict) -> dict:
    sentence = example["sentence"]
    opt1 = example["option1"]
    opt2 = example["option2"]
    prompt = f"{sentence}\n\nA. {opt1}\nB. {opt2}\n\nAnswer:"
    gold = "A" if example["answer"] == "1" else "B"
    return {"prompt": prompt, "gold": gold, "task": "winogrande", "answer_type": "choice"}


def _format_truthfulqa(example: dict) -> dict:
    question = example["question"]
    choices = example["mc1_targets"]["choices"]
    labels = example["mc1_targets"]["labels"]
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    prompt = f"{question}\n\n{options}\n\nAnswer:"
    gold_idx = labels.index(1)
    gold = LETTERS[gold_idx]
    return {"prompt": prompt, "gold": gold, "task": "truthfulqa", "answer_type": "choice"}


TASK_LOADERS = {
    "mmlu": {
        "load_args": ("cais/mmlu", "all"),
        "split": "test",
        "formatter": _format_mmlu,
    },
    "arc_challenge": {
        "load_args": ("allenai/ai2_arc", "ARC-Challenge"),
        "split": "test",
        "formatter": _format_arc,
    },
    "hellaswag": {
        "load_args": ("Rowan/hellaswag",),
        "split": "validation",
        "formatter": _format_hellaswag,
    },
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "split": "test",
        "formatter": _format_gsm8k,
    },
    "winogrande": {
        "load_args": ("allenai/winogrande", "winogrande_xl"),
        "split": "validation",
        "formatter": _format_winogrande,
    },
    "truthfulqa": {
        "load_args": ("truthfulqa/truthful_qa", "multiple_choice"),
        "split": "validation",
        "formatter": _format_truthfulqa,
    },
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts = _compute_counts(NUM_PROMPTS, TASK_RATIOS)
    rng = random.Random(SEED)

    all_prompts: list[dict] = []

    for task, n in counts.items():
        cfg = TASK_LOADERS[task]
        print(f"Loading {task} ({n} prompts)...")
        ds = load_dataset(*cfg["load_args"], split=cfg["split"])
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = indices[:n]
        for idx in selected:
            entry = cfg["formatter"](ds[idx])
            all_prompts.append(entry)

    # Shuffle all prompts together
    rng.shuffle(all_prompts)

    with open(OUTPUT, "w") as f:
        for entry in all_prompts:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(all_prompts)} prompts to {OUTPUT}")
    # Print distribution
    from collections import Counter
    dist = Counter(e["task"] for e in all_prompts)
    for task, count in sorted(dist.items()):
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
