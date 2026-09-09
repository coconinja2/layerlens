"""Stable, portable trace types and JSON serialization."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACE_SCHEMA_VERSION = "1.0"


@dataclass
class LayerEvent:
    sequence: int
    step: int
    phase: str
    token_index: int | None
    module: str
    module_type: str
    depth: int
    started_ns: int
    duration_ns: int
    input_shapes: list[list[int]] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)
    device: str = "unknown"
    memory_before_bytes: int | None = None
    memory_after_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["started_ms"] = self.started_ns / 1_000_000
        row["duration_ms"] = self.duration_ns / 1_000_000
        if self.memory_before_bytes is not None and self.memory_after_bytes is not None:
            row["memory_delta_bytes"] = self.memory_after_bytes - self.memory_before_bytes
        else:
            row["memory_delta_bytes"] = None
        return row


@dataclass
class StepEvent:
    step: int
    phase: str
    token_index: int | None
    started_ns: int
    duration_ns: int
    input_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["started_ms"] = self.started_ns / 1_000_000
        row["duration_ms"] = self.duration_ns / 1_000_000
        return row


@dataclass
class TokenEvent:
    token_index: int
    step: int
    token_id: int | None
    text: str | None
    emitted_ns: int | None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["emitted_ms"] = self.emitted_ns / 1_000_000 if self.emitted_ns is not None else None
        return row


@dataclass
class InferenceTrace:
    metadata: dict[str, Any]
    events: list[LayerEvent]
    steps: list[StepEvent] = field(default_factory=list)
    tokens: list[TokenEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
            "tokens": [token.to_dict() for token in self.tokens],
            "events": [event.to_dict() for event in self.events],
            "errors": self.errors,
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


def environment_metadata() -> dict[str, Any]:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        metadata["torch"] = None
    return metadata

