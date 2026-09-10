"""Public adapter API for plugging inference runtimes into LayerLens."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from .ollama import capture_ollama
from .vllm import capture_vllm


ENTRY_POINT_GROUP = "layerlens.adapters"
_ADAPTER_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SENSITIVE_OPTION_PARTS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
    "key",
)


@dataclass(frozen=True)
class CaptureRequest:
    """Runtime-neutral inputs passed to every LayerLens adapter."""

    model: str
    prompt: str
    max_tokens: int
    output_path: Path
    include_content: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        object.__setattr__(self, "output_path", Path(self.output_path))


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Contract implemented by built-in and third-party runtime adapters."""

    name: str

    def capture(self, request: CaptureRequest) -> Path:
        """Capture one request and return the written LayerLens trace path."""
        ...


class AdapterRegistry:
    """Explicit registry with lazy Python entry-point discovery."""

    def __init__(self) -> None:
        self._adapters: dict[str, RuntimeAdapter] = {}

    def register(
        self,
        adapter: RuntimeAdapter | type[RuntimeAdapter] | Any,
        *,
        name: str | None = None,
        replace: bool = False,
    ) -> RuntimeAdapter:
        resolved = self._instantiate(adapter)
        adapter_name = (name or getattr(resolved, "name", "")).strip().lower()
        if not _ADAPTER_NAME.fullmatch(adapter_name):
            raise ValueError(
                "adapter name must contain only lowercase letters, numbers, '.', '_' or '-'"
            )
        if not callable(getattr(resolved, "capture", None)):
            raise TypeError("adapter must provide capture(request)")
        if adapter_name in self._adapters and not replace:
            raise ValueError(f"adapter already registered: {adapter_name}")
        self._adapters[adapter_name] = resolved
        return resolved

    def get(self, name: str) -> RuntimeAdapter:
        normalized = name.strip().lower()
        if normalized in self._adapters:
            return self._adapters[normalized]
        for point in entry_points().select(group=ENTRY_POINT_GROUP):
            if point.name.lower() == normalized:
                return self.register(point.load(), name=normalized)
        available = ", ".join(self.names()) or "none"
        raise LookupError(f"unknown LayerLens runtime '{name}'; available: {available}")

    def names(self) -> tuple[str, ...]:
        discovered = {
            point.name.lower() for point in entry_points().select(group=ENTRY_POINT_GROUP)
        }
        return tuple(sorted(set(self._adapters) | discovered))

    def capture(self, name: str, request: CaptureRequest) -> Path:
        return self.get(name).capture(request)

    @staticmethod
    def _instantiate(candidate: Any) -> RuntimeAdapter:
        if isinstance(candidate, type):
            candidate = candidate()
        elif callable(candidate) and not callable(getattr(candidate, "capture", None)):
            candidate = candidate()
        return candidate


def _option(request: CaptureRequest, name: str, default: Any) -> Any:
    return request.options.get(name, default)


def _bool_option(request: CaptureRequest, name: str, default: bool) -> bool:
    value = _option(request, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


class OllamaAdapter:
    """Built-in adapter for local or cloud Ollama-compatible APIs."""

    name = "ollama"

    def capture(self, request: CaptureRequest) -> Path:
        api_key = os.environ.get("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        return capture_ollama(
            base_url=str(_option(request, "base_url", "http://localhost:11434")),
            model=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            output_path=request.output_path,
            include_content=request.include_content,
            keep_alive=_option(request, "keep_alive", "5m"),
            request_headers=headers,
            timeout=float(_option(request, "timeout", 600)),
        )


class VLLMAdapter:
    """Built-in adapter for standard or instrumented vLLM servers."""

    name = "vllm"

    def capture(self, request: CaptureRequest) -> Path:
        api_key = os.environ.get("VLLM_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        return capture_vllm(
            base_url=str(_option(request, "base_url", "http://localhost:8000")),
            model=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            control_path=Path(
                _option(request, "control_path", "traces/vllm_raw/layer-profiler.control")
            ),
            events_path=Path(
                _option(request, "events_path", "traces/vllm_raw/vllm-layer-events.jsonl")
            ),
            output_path=request.output_path,
            include_content=request.include_content,
            collect_metrics=_bool_option(request, "collect_metrics", True),
            metrics_url=_option(request, "metrics_url", None),
            metrics_sample_ms=int(_option(request, "metrics_sample_ms", 50)),
            layer_events=_bool_option(request, "layer_events", False),
            request_headers=headers,
        )


def parse_option(value: str) -> tuple[str, Any]:
    """Parse a CLI ``KEY=VALUE`` pair, decoding JSON scalars when possible."""
    key, separator, raw = value.partition("=")
    if not separator or not _ADAPTER_NAME.fullmatch(key):
        raise ValueError("options must use key=value with a lowercase key")
    if any(part in key for part in _SENSITIVE_OPTION_PARTS):
        raise ValueError("secret-bearing options are not allowed; use an environment variable")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = raw
    return key, decoded


adapter_registry = AdapterRegistry()
adapter_registry.register(OllamaAdapter())
adapter_registry.register(VLLMAdapter())


def capture(runtime: str, request: CaptureRequest) -> Path:
    """Capture through a built-in or installed third-party runtime adapter."""
    return adapter_registry.capture(runtime, request)
