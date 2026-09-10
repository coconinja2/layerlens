"""Safe, dependency-free collection of selected vLLM Prometheus metrics."""

from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .trace import RuntimeEvent


_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)
_LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class PrometheusSample:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)


def parse_prometheus(text: str) -> list[PrometheusSample]:
    """Parse Prometheus text exposition without retaining comments or raw text."""
    samples: list[PrometheusSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels: dict[str, str] = {}
        for key, raw_value in _LABEL_RE.findall(match.group("labels") or ""):
            try:
                labels[key] = json.loads(f'"{raw_value}"')
            except json.JSONDecodeError:
                labels[key] = raw_value
        samples.append(PrometheusSample(match.group("name"), value, labels))
    return samples


def fetch_prometheus(
    url: str, timeout: float = 3.0, headers: dict[str, str] | None = None
) -> list[PrometheusSample]:
    request_headers = {"Accept": "text/plain", **(headers or {})}
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_prometheus(response.read().decode("utf-8", errors="replace"))


def _matches_model(sample: PrometheusSample, model: str | None) -> bool:
    sample_model = sample.labels.get("model_name")
    return model is None or sample_model is None or sample_model == model


def metric_total(
    samples: list[PrometheusSample], names: str | tuple[str, ...], model: str | None = None
) -> float:
    accepted = (names,) if isinstance(names, str) else names
    return sum(
        sample.value
        for sample in samples
        if sample.name in accepted and _matches_model(sample, model)
    )


def metric_max(
    samples: list[PrometheusSample], names: str | tuple[str, ...], model: str | None = None
) -> float:
    accepted = (names,) if isinstance(names, str) else names
    values = [
        sample.value
        for sample in samples
        if sample.name in accepted and _matches_model(sample, model)
    ]
    return max(values, default=0.0)


_KV_USAGE_NAMES = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
)


def _timeline_point(
    samples: list[PrometheusSample], model: str | None, elapsed_ms: float
) -> dict[str, float]:
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "kv_cache_usage": round(metric_max(samples, _KV_USAGE_NAMES, model), 6),
        "requests_running": metric_total(samples, "vllm:num_requests_running", model),
        "requests_waiting": metric_total(samples, "vllm:num_requests_waiting", model),
        "resident_memory_mb": round(
            metric_max(samples, "process_resident_memory_bytes") / (1024 * 1024), 3
        ),
    }


_COUNTERS = {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "cached_prompt_tokens": "vllm:prompt_tokens_cached_total",
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "external_prefix_cache_queries": "vllm:external_prefix_cache_queries_total",
    "external_prefix_cache_hits": "vllm:external_prefix_cache_hits_total",
    "preemptions": "vllm:num_preemptions_total",
    "successful_requests": "vllm:request_success_total",
    "multimodal_cache_queries": "vllm:mm_cache_queries_total",
    "multimodal_cache_hits": "vllm:mm_cache_hits_total",
    "speculative_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "speculative_accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "speculative_emitted_tokens": "vllm:spec_decode_num_emitted_tokens_total",
    "corrupted_requests": "vllm:corrupted_requests_total",
}

_LATENCIES = {
    "time_to_first_token": "vllm:time_to_first_token_seconds",
    "inter_token": "vllm:inter_token_latency_seconds",
    "end_to_end": "vllm:e2e_request_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "inference": "vllm:request_inference_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
}

_SAFE_CACHE_CONFIG_KEYS = {
    "block_size",
    "cache_dtype",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "kv_cache_max_concurrency",
    "kv_cache_size_tokens",
    "kv_offloading_backend",
    "kv_offloading_size",
    "num_cpu_blocks",
    "num_gpu_blocks",
    "sliding_window",
}


def _delta(
    before: list[PrometheusSample],
    after: list[PrometheusSample],
    name: str,
    model: str | None,
) -> float:
    return max(0.0, metric_total(after, name, model) - metric_total(before, name, model))


def _cache_config(samples: list[PrometheusSample], model: str | None) -> dict[str, Any]:
    for sample in samples:
        if sample.name not in {"vllm:cache_config_info", "vllm:cache_config"}:
            continue
        if not _matches_model(sample, model):
            continue
        return {key: sample.labels[key] for key in sorted(_SAFE_CACHE_CONFIG_KEYS) if key in sample.labels}
    return {}


def summarize_vllm_metrics(
    *,
    before: list[PrometheusSample],
    after: list[PrometheusSample],
    timeline: list[dict[str, float]],
    model: str | None,
    sample_interval_ms: int,
) -> dict[str, Any]:
    """Normalize a request-sized metric delta into a portable, sanitized record."""
    counters = {
        key: round(_delta(before, after, metric, model), 6)
        for key, metric in _COUNTERS.items()
    }
    latencies: dict[str, float | None] = {}
    for key, family in _LATENCIES.items():
        count = _delta(before, after, f"{family}_count", model)
        total_seconds = _delta(before, after, f"{family}_sum", model)
        latencies[f"{key}_mean_ms"] = round(total_seconds * 1000 / count, 3) if count else None

    queries = counters["prefix_cache_queries"]
    hits = counters["prefix_cache_hits"]
    prompt_tokens = counters["prompt_tokens"]
    cached_prompt_tokens = counters["cached_prompt_tokens"]
    multimodal_queries = counters["multimodal_cache_queries"]
    multimodal_hits = counters["multimodal_cache_hits"]
    draft_tokens = counters["speculative_draft_tokens"]
    accepted_tokens = counters["speculative_accepted_tokens"]
    peak_usage = max((point["kv_cache_usage"] for point in timeline), default=0.0)

    return {
        "available": True,
        "source": "vllm_prometheus",
        "capabilities": {
            "kv_cache_usage": "sampled",
            "scheduler_state": "sampled",
            "kernel_timing": "unavailable",
        },
        "sample_interval_ms": sample_interval_ms,
        "timeline": timeline,
        "cache": {
            "peak_usage": round(peak_usage, 6),
            "final_usage": round(metric_max(after, _KV_USAGE_NAMES, model), 6),
            "prefix_queries": queries,
            "prefix_hits": hits,
            "prefix_hit_rate": round(hits / queries, 6) if queries else None,
            "cached_prompt_tokens": cached_prompt_tokens,
            "prompt_cache_hit_rate": (
                round(cached_prompt_tokens / prompt_tokens, 6) if prompt_tokens else None
            ),
            "config": _cache_config(after, model),
        },
        "scheduler": {
            "peak_running": max((point["requests_running"] for point in timeline), default=0),
            "peak_waiting": max((point["requests_waiting"] for point in timeline), default=0),
            "preemptions": counters["preemptions"],
        },
        "process": {
            "peak_resident_memory_mb": max(
                (point["resident_memory_mb"] for point in timeline), default=0
            ),
            "cpu_seconds": round(
                _delta(before, after, "process_cpu_seconds_total", model=None), 6
            ),
        },
        "multimodal_cache": {
            "queries": multimodal_queries,
            "hits": multimodal_hits,
            "hit_rate": (
                round(multimodal_hits / multimodal_queries, 6) if multimodal_queries else None
            ),
        },
        "speculative_decode": {
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted_tokens,
            "emitted_tokens": counters["speculative_emitted_tokens"],
            "acceptance_rate": (
                round(accepted_tokens / draft_tokens, 6) if draft_tokens else None
            ),
        },
        "counters": counters,
        "latency_ms": latencies,
    }


def runtime_events_from_metrics(metrics: dict[str, Any]) -> list[RuntimeEvent]:
    """Convert a normalized runtime timeline into engine-neutral events."""
    events: list[RuntimeEvent] = []
    fields = (
        ("kv_cache", "usage", "kv_cache_usage", "ratio"),
        ("scheduler", "running", "requests_running", "requests"),
        ("scheduler", "waiting", "requests_waiting", "requests"),
    )
    for point in metrics.get("timeline", []):
        elapsed_ns = int(float(point.get("elapsed_ms", 0)) * 1_000_000)
        for category, name, key, unit in fields:
            if point.get(key) is None:
                continue
            events.append(
                RuntimeEvent(
                    sequence=len(events),
                    elapsed_ns=elapsed_ns,
                    category=category,
                    name=name,
                    value=float(point[key]),
                    unit=unit,
                    measurement="sampled",
                )
            )
    return events


class VLLMMetricsSampler:
    """Poll vLLM's Prometheus endpoint while one profiled request is active."""

    def __init__(
        self,
        url: str,
        model: str | None,
        interval_ms: int = 50,
        headers: dict[str, str] | None = None,
    ):
        self.url = url
        self.model = model
        self.interval_ms = max(10, interval_ms)
        self.headers = dict(headers or {})
        self.before: list[PrometheusSample] = []
        self.after: list[PrometheusSample] = []
        self.timeline: list[dict[str, float]] = []
        self.error: str | None = None
        self._origin = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self.before = fetch_prometheus(self.url, headers=self.headers)
        except Exception:
            self.error = "metrics_endpoint_unavailable"
            return
        self._origin = time.perf_counter()
        self.timeline.append(_timeline_point(self.before, self.model, 0.0))
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000):
            try:
                samples = fetch_prometheus(self.url, headers=self.headers)
            except Exception:
                continue
            elapsed_ms = (time.perf_counter() - self._origin) * 1000
            self.timeline.append(_timeline_point(samples, self.model, elapsed_ms))

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self.error:
            return {"available": False, "error": self.error}
        try:
            self.after = fetch_prometheus(self.url, headers=self.headers)
        except Exception:
            return {"available": False, "error": "metrics_endpoint_unavailable"}
        elapsed_ms = (time.perf_counter() - self._origin) * 1000
        self.timeline.append(_timeline_point(self.after, self.model, elapsed_ms))
        return summarize_vllm_metrics(
            before=self.before,
            after=self.after,
            timeline=self.timeline,
            model=self.model,
            sample_interval_ms=self.interval_ms,
        )
