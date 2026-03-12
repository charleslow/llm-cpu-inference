"""Seed backend: HuggingFace Transformers with Qwen2.5-0.5B, FP32."""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import MAX_TOKENS, SEED, TEMPERATURE
from protocol import GenerationResult

TRIAL_NAME = "hf-qwen2-fp32"


class Backend:
    def setup(self) -> None:
        model_name = "Qwen/Qwen2.5-0.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
        self.model.eval()

        # Warmup
        with torch.no_grad():
            ids = self.tokenizer.encode("Hello", return_tensors="pt")
            self.model.generate(ids, max_new_tokens=5, do_sample=False)

    def generate(self, prompt: str, max_tokens: int = MAX_TOKENS) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
            )
        t1 = time.perf_counter()

        new_ids = output_ids[0, prompt_len:]
        raw_output = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        completion = raw_output.strip()

        return GenerationResult(
            completion=completion,
            raw_output=raw_output,
            latency_ms=(t1 - t0) * 1000,
            tokens_generated=len(new_ids),
            prompt_tokens=prompt_len,
        )

    def teardown(self) -> None:
        del self.model
        del self.tokenizer
