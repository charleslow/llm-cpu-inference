from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class GenerationResult:
    completion: str          # extracted answer text
    raw_output: str          # full model output
    latency_ms: float        # wall-clock time
    tokens_generated: int
    prompt_tokens: int = 0


@runtime_checkable
class InferenceBackend(Protocol):
    def setup(self) -> None: ...
    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult: ...
    def teardown(self) -> None: ...
