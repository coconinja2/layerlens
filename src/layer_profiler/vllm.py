"""Capture and convert server-side vLLM layer events."""

from __future__ import annotations

import json
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .metrics import VLLMMetricsSampler
from .privacy import safe_model_identifier
from .trace import InferenceTrace, LayerEvent, StepEvent, TokenEvent, environment_metadata


def capture_vllm(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    control_path: Path,
    events_path: Path,
    output_path: Path,
    include_content: bool = False,
    collect_metrics: bool = True,
    metrics_url: str | None = None,
    metrics_sample_ms: int = 50,
    layer_events: bool = True,
    request_headers: dict[str, str] | None = None,
) -> Path:
    run_id = uuid.uuid4().hex
    if layer_events:
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.write_text(run_id, encoding="utf-8")
    sampler = None
    if collect_metrics:
        sampler = VLLMMetricsSampler(
            metrics_url or f"{base_url.rstrip('/')}/metrics",
            model=model,
            interval_ms=metrics_sample_ms,
            headers=request_headers,
        )
        sampler.start()
    started = time.perf_counter()
    response_payload: dict[str, Any] = {}
    try:
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json", **(request_headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            response_payload = json.load(response)
    finally:
        if layer_events:
            control_path.write_text("off", encoding="utf-8")
        elapsed_s = time.perf_counter() - started
        server_metrics = sampler.stop() if sampler is not None else {}

    rows = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id") == run_id:
                rows.append(row)
    if layer_events and not rows:
        raise RuntimeError(
            "vLLM returned a completion but produced no matching layer events. "
            "Check the injection mount and LLM_LAYER_CLASS_PATTERN."
        )

    origin_ns = min((row["started_ns"] for row in rows), default=0)
    events = [
        LayerEvent(
            sequence=index,
            step=int(row["step"]),
            phase=row["phase"],
            token_index=row.get("token_index"),
            module=row["module"],
            module_type=row["module_type"],
            depth=2,
            started_ns=int(row["started_ns"]) - origin_ns,
            duration_ns=int(row["duration_ns"]),
            input_shapes=row.get("input_shapes", []),
            output_shapes=row.get("output_shapes", []),
            device=row.get("device", "unknown"),
        )
        for index, row in enumerate(sorted(rows, key=lambda item: item["started_ns"]))
    ]

    steps = []
    for step_number in sorted({event.step for event in events}):
        step_events = [event for event in events if event.step == step_number]
        first_ns = min(event.started_ns for event in step_events)
        last_ns = max(event.started_ns + event.duration_ns for event in step_events)
        output_shapes = step_events[0].output_shapes
        input_tokens = output_shapes[0][-2] if output_shapes and len(output_shapes[0]) >= 2 else None
        steps.append(
            StepEvent(
                step=step_number,
                phase=step_events[0].phase,
                token_index=step_number,
                started_ns=first_ns,
                duration_ns=last_ns - first_ns,
                input_tokens=input_tokens,
            )
        )

    usage = response_payload.get("usage", {})
    completion_count = int(usage.get("completion_tokens", len(steps)))
    tokens = [
        TokenEvent(
            token_index=index,
            step=index,
            token_id=None,
            text=None,
            emitted_ns=(steps[index].started_ns + steps[index].duration_ns if index < len(steps) else None),
        )
        for index in range(completion_count)
    ]
    metadata = environment_metadata()
    metadata.update(
        {
            "source": "vllm-injected" if layer_events else "vllm-prometheus",
            "run_id": run_id,
            "model": safe_model_identifier(model),
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": completion_count,
            "request_duration_s": elapsed_s,
            "selected_modules": len({event.module for event in events}),
            "step_timing_scope": "decoder_layer_span" if layer_events else "not_recorded",
            "content_recorded": include_content,
            "server_metrics_collected": bool(server_metrics.get("available", False)),
        }
    )
    if include_content:
        metadata["prompt"] = prompt
        metadata["generated_text"] = response_payload.get("choices", [{}])[0].get("text", "")
    return InferenceTrace(
        metadata=metadata,
        events=events,
        steps=steps,
        tokens=tokens,
        metrics=server_metrics,
    ).write(output_path)
