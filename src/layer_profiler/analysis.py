"""Scalable aggregation and decision-oriented analysis for LayerLens traces."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


def _number(row: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return float(value)
    return default


def _layer(row: Mapping[str, Any]) -> int:
    if row.get("layer") is not None:
        return int(row["layer"])
    match = re.search(r"(\d+)$", str(row.get("module", "")))
    return int(match.group(1)) if match else -1


def choose_group_size(cardinality: int, target_groups: int) -> int:
    """Choose a power-of-two bin size that stays below a visual cell budget."""
    if cardinality <= 0 or target_groups <= 0:
        return 1
    size = 1
    while math.ceil(cardinality / size) > target_groups:
        size *= 2
    return size


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_event_grid(
    events: Iterable[Mapping[str, Any]],
    *,
    layer_group_size: int | None = None,
    token_group_size: int | None = None,
    max_layer_groups: int = 48,
    max_token_groups: int = 64,
) -> dict[str, Any]:
    """Build a bounded layer × token view without discarding the raw events."""
    rows = [row for row in events if _layer(row) >= 0]
    layers = sorted({_layer(row) for row in rows})
    decode_steps = sorted(
        {int(row.get("step", 0)) for row in rows if row.get("phase") != "prefill"}
    )
    layer_size = layer_group_size or choose_group_size(len(layers), max_layer_groups)
    token_size = token_group_size or choose_group_size(len(decode_steps), max_token_groups)
    if layer_size < 1 or token_size < 1:
        raise ValueError("group sizes must be at least 1")

    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        layer = _layer(row)
        step = int(row.get("step", 0))
        phase = str(row.get("phase", "decode"))
        layer_start = (layer // layer_size) * layer_size
        step_start = 0 if phase == "prefill" else 1 + ((max(step, 1) - 1) // token_size) * token_size
        grouped[(layer_start, step_start, phase)].append(
            _number(row, "duration_ms", "durationMs")
        )

    cells = []
    for (layer_start, step_start, phase), values in sorted(grouped.items()):
        layer_end = layer_start + layer_size - 1
        step_end = step_start if phase == "prefill" else step_start + token_size - 1
        layer_label = (
            f"Layer {layer_start}"
            if layer_size == 1
            else f"Layers {layer_start}–{layer_end}"
        )
        token_label = (
            "P0 · Prefill"
            if phase == "prefill"
            else f"D{step_start} · Decode"
            if token_size == 1
            else f"D{step_start}–D{step_end}"
        )
        cells.append(
            {
                "layer_start": layer_start,
                "layer_end": layer_end,
                "step_start": step_start,
                "step_end": step_end,
                "phase": phase,
                "layer_label": layer_label,
                "token_label": token_label,
                "count": len(values),
                "total_ms": round(sum(values), 3),
                "mean_ms": round(sum(values) / len(values), 3),
                "p95_ms": round(_percentile(values, 0.95), 3),
                "max_ms": round(max(values), 3),
            }
        )
    return {
        "layer_group_size": layer_size,
        "token_group_size": token_size,
        "aggregated": layer_size > 1 or token_size > 1,
        "raw_event_count": len(rows),
        "cell_count": len(cells),
        "cells": cells,
    }


def diagnose_trace(
    events: Iterable[Mapping[str, Any]],
    steps: Iterable[Mapping[str, Any]],
    metrics: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Translate observed timing/cache evidence into bounded next-step guidance."""
    rows = list(events)
    step_rows = list(steps)
    metrics = metrics or {}
    insights: list[dict[str, str]] = []

    if rows:
        durations = [_number(row, "duration_ms", "durationMs") for row in rows]
        hotspot = max(rows, key=lambda row: _number(row, "duration_ms", "durationMs"))
        hotspot_ms = _number(hotspot, "duration_ms", "durationMs")
        typical_ms = median(durations) or 0.0
        ratio = hotspot_ms / typical_ms if typical_ms else 0.0
        layer = _layer(hotspot)
        step = int(hotspot.get("step", 0))
        phase = str(hotspot.get("phase", "decode"))
        insights.append(
            {
                "kind": "hotspot",
                "title": f"Inspect layer {layer}, {phase} step {step}",
                "evidence": f"{hotspot_ms:.2f} ms — {ratio:.1f}× the median layer call.",
                "action": (
                    "Profile attention and MLP children for this layer; check tensor shapes, "
                    "device transfers, and kernel selection before changing the model."
                ),
            }
        )

        by_layer: dict[int, float] = defaultdict(float)
        for row in rows:
            by_layer[_layer(row)] += _number(row, "duration_ms", "durationMs")
        slow_layer, slow_total = max(by_layer.items(), key=lambda item: item[1])
        total = sum(by_layer.values()) or 1.0
        insights.append(
            {
                "kind": "cumulative",
                "title": f"Layer {slow_layer} has the largest cumulative cost",
                "evidence": f"{slow_total:.2f} ms, or {100 * slow_total / total:.1f}% of recorded layer time.",
                "action": (
                    "If this repeats across tokens, test quantization or an optimized attention/MLP "
                    "kernel and compare against a second trace."
                ),
            }
        )

    prefill = [
        _number(row, "duration_ms", "durationMs")
        for row in step_rows
        if row.get("phase") == "prefill"
    ]
    decode = [
        _number(row, "duration_ms", "durationMs")
        for row in step_rows
        if row.get("phase") == "decode"
    ]
    if prefill and decode:
        prefill_ms = sum(prefill)
        decode_mean = sum(decode) / len(decode)
        if prefill_ms > decode_mean * 2:
            insights.append(
                {
                    "kind": "phase",
                    "title": "Prompt processing dominates first-token latency",
                    "evidence": f"Prefill is {prefill_ms:.2f} ms versus {decode_mean:.2f} ms mean decode.",
                    "action": (
                        "Evaluate prompt length, prefix caching, and chunked prefill; optimize decode "
                        "only if later-token latency is also a user-facing problem."
                    ),
                }
            )

    cache = metrics.get("cache", {})
    scheduler = metrics.get("scheduler", {})
    peak_usage = cache.get("peak_usage")
    preemptions = float(scheduler.get("preemptions", 0) or 0)
    peak_waiting = float(scheduler.get("peak_waiting", 0) or 0)
    if peak_usage is not None and float(peak_usage) >= 0.8:
        insights.append(
            {
                "kind": "cache",
                "title": "KV-cache pressure is high",
                "evidence": f"Peak measured occupancy reached {100 * float(peak_usage):.1f}%.",
                "action": (
                    "Test a smaller maximum sequence or concurrency limit, more cache memory, or a "
                    "lower-precision cache; confirm the change reduces waiting or preemption."
                ),
            }
        )
    elif preemptions > 0 or peak_waiting > 0:
        insights.append(
            {
                "kind": "scheduler",
                "title": "Scheduler contention occurred",
                "evidence": f"Peak waiting {peak_waiting:g}; preemptions {preemptions:g}.",
                "action": (
                    "Correlate the affected window with batch size and KV occupancy, then test lower "
                    "concurrency or a larger cache allocation."
                ),
            }
        )

    return insights[:4]
