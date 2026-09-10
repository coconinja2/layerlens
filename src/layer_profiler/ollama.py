"""Privacy-safe capture of Ollama inference and loaded-model metrics."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from .privacy import safe_model_identifier
from .trace import InferenceTrace, TokenEvent, environment_metadata


def _request_json(
    url: str,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 600,
) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json", **(headers or {})},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _ns_to_ms(value: Any) -> float:
    try:
        return round(int(value) / 1_000_000, 3)
    except (TypeError, ValueError):
        return 0.0


def _model_runtime(
    *, show_payload: dict[str, Any], ps_payload: dict[str, Any], model: str
) -> dict[str, Any]:
    details = show_payload.get("details", {})
    model_info = show_payload.get("model_info", {})
    architecture = model_info.get("general.architecture") or details.get("family")
    safe_architecture: dict[str, Any] = {}
    if architecture:
        prefix = f"{architecture}."
        suffixes = {
            "block_count",
            "context_length",
            "embedding_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "attention.sliding_window",
            "feed_forward_length",
        }
        for key, value in model_info.items():
            if key.startswith(prefix) and key[len(prefix) :] in suffixes:
                safe_architecture[key[len(prefix) :]] = value

    running = next(
        (
            item
            for item in ps_payload.get("models", [])
            if item.get("model") == model or item.get("name") == model
        ),
        {},
    )
    return {
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization": details.get("quantization_level"),
        "format": details.get("format"),
        "capabilities": sorted(show_payload.get("capabilities", [])),
        "model_size_mb": round(float(running.get("size", 0)) / (1024 * 1024), 3),
        "processor_memory_mb": round(float(running.get("size_vram", 0)) / (1024 * 1024), 3),
        "context_length": running.get("context_length") or safe_architecture.get("context_length"),
        "architecture": safe_architecture,
    }


def normalize_ollama_metrics(
    *,
    response_payload: dict[str, Any],
    show_payload: dict[str, Any],
    ps_payload: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Convert Ollama's final response into the common LayerLens metric shape."""
    prompt_tokens = int(response_payload.get("prompt_eval_count", 0) or 0)
    generation_tokens = int(response_payload.get("eval_count", 0) or 0)
    total_ms = _ns_to_ms(response_payload.get("total_duration"))
    load_ms = _ns_to_ms(response_payload.get("load_duration"))
    prompt_eval_ms = _ns_to_ms(response_payload.get("prompt_eval_duration"))
    eval_ms = _ns_to_ms(response_payload.get("eval_duration"))
    runtime = _model_runtime(show_payload=show_payload, ps_payload=ps_payload, model=model)
    context_length = int(runtime.get("context_length") or 0)
    active_tokens = prompt_tokens + generation_tokens

    return {
        "available": True,
        "source": "ollama_api",
        "counters": {
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
        },
        "latency_ms": {
            "total": total_ms,
            "load": load_ms,
            "prompt_eval": prompt_eval_ms,
            "generation": eval_ms,
            "mean_output_token": round(eval_ms / generation_tokens, 3) if generation_tokens else None,
            "runtime_overhead": round(max(0.0, total_ms - load_ms - prompt_eval_ms - eval_ms), 3),
        },
        "throughput": {
            "prompt_tokens_per_second": (
                round(prompt_tokens * 1000 / prompt_eval_ms, 3) if prompt_eval_ms else None
            ),
            "generation_tokens_per_second": (
                round(generation_tokens * 1000 / eval_ms, 3) if eval_ms else None
            ),
        },
        "cache": {
            "measurement": "estimated_context_utilization",
            "context_length": context_length or None,
            "active_tokens": active_tokens,
            "estimated_context_utilization": (
                round(active_tokens / context_length, 6) if context_length else None
            ),
            "kv_cache_usage_available": False,
        },
        "runtime": runtime,
    }


def capture_ollama(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    output_path: Path,
    include_content: bool = False,
    keep_alive: str | int = "5m",
    request_headers: dict[str, str] | None = None,
    timeout: float = 600,
) -> Path:
    """Run one deterministic Ollama generation and write a portable LayerLens trace."""
    root = base_url.rstrip("/")
    show_payload: dict[str, Any] = {}
    try:
        show_payload = _request_json(
            f"{root}/api/show",
            body={"model": model, "verbose": False},
            headers=request_headers,
            timeout=timeout,
        )
    except Exception:
        pass

    started = time.perf_counter()
    response_payload = _request_json(
        f"{root}/api/generate",
        body={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens},
        },
        headers=request_headers,
        timeout=timeout,
    )
    client_elapsed_s = time.perf_counter() - started

    ps_payload: dict[str, Any] = {}
    try:
        ps_payload = _request_json(f"{root}/api/ps", headers=request_headers, timeout=10)
    except Exception:
        pass

    metrics = normalize_ollama_metrics(
        response_payload=response_payload,
        show_payload=show_payload,
        ps_payload=ps_payload,
        model=model,
    )
    prompt_tokens = int(response_payload.get("prompt_eval_count", 0) or 0)
    generation_tokens = int(response_payload.get("eval_count", 0) or 0)
    metadata = environment_metadata()
    metadata.update(
        {
            "source": "ollama",
            "model": safe_model_identifier(model),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generation_tokens,
            "request_duration_s": client_elapsed_s,
            "selected_modules": 0,
            "step_timing_scope": "ollama_aggregate",
            "content_recorded": include_content,
            "server_metrics_collected": True,
        }
    )
    if include_content:
        metadata["prompt"] = prompt
        metadata["generated_text"] = response_payload.get("response", "")
        metadata["thinking_text"] = response_payload.get("thinking", "")

    tokens = [
        TokenEvent(token_index=index, step=index, token_id=None, text=None, emitted_ns=None)
        for index in range(generation_tokens)
    ]
    return InferenceTrace(
        metadata=metadata,
        events=[],
        steps=[],
        tokens=tokens,
        metrics=metrics,
    ).write(output_path)
