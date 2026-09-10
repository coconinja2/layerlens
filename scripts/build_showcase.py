#!/usr/bin/env python3
"""Build the small, publishable dataset used by the LayerLens showcase."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = [
    PROJECT_ROOT / "traces" / "qwen-vllm-4-token.json",
    PROJECT_ROOT / "traces" / "qwen-vllm-8-token.json",
    PROJECT_ROOT / "traces" / "gemma4-12b-ollama.json",
    PROJECT_ROOT / "traces" / "gpt2-huggingface.json",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "runs.json"


def layer_number(module: str) -> int:
    match = re.search(r"(\d+)$", module)
    return int(match.group(1)) if match else -1


def compact_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    events = payload.get("events", [])
    steps = payload.get("steps", [])
    tokens = payload.get("tokens", [])
    engine_metrics = payload.get("metrics", {})
    source = metadata.get("source", "unknown")
    backend = "Ollama" if source == "ollama" else "vLLM" if source.startswith("vllm") else "Hugging Face"

    normalized_events = [
        {
            "step": int(event["step"]),
            "phase": event["phase"],
            "layer": layer_number(event["module"]),
            "module": event["module"],
            "startMs": round(float(event["started_ms"]), 3),
            "durationMs": round(float(event["duration_ms"]), 3),
        }
        for event in events
    ]
    normalized_steps = [
        {
            "step": int(step["step"]),
            "phase": step["phase"],
            "inputTokens": step.get("input_tokens"),
            "startMs": round(float(step["started_ms"]), 3),
            "durationMs": round(float(step["duration_ms"]), 3),
        }
        for step in steps
    ]

    prefill = [step["durationMs"] for step in normalized_steps if step["phase"] == "prefill"]
    decode = [step["durationMs"] for step in normalized_steps if step["phase"] == "decode"]
    mean_decode = sum(decode) / len(decode) if decode else 0.0
    total_layer_ms = sum(event["durationMs"] for event in normalized_events)
    request_ms = float(metadata.get("request_duration_s", 0)) * 1000
    token_count = int(metadata.get("generated_tokens", len(tokens)))
    if source == "ollama":
        latency = engine_metrics.get("latency_ms", {})
        throughput = engine_metrics.get("throughput", {})
        prefill_ms = float(latency.get("prompt_eval") or 0)
        mean_decode = float(latency.get("mean_output_token") or 0)
        request_ms = float(latency.get("total") or request_ms)
        decode_rate = float(throughput.get("generation_tokens_per_second") or 0)
    else:
        prefill_ms = sum(prefill)
        decode_rate = 1000 / mean_decode if mean_decode else 0
    model_name = metadata.get("model", metadata.get("model_class", "Unknown model"))
    short_model = str(model_name).split("/")[-1]

    return {
        "id": path.stem,
        "label": f"{short_model} · {backend} · {token_count} tok",
        "model": model_name,
        "backend": backend,
        "device": events[0].get("device", "unknown") if events else backend.lower(),
        "promptTokens": metadata.get("prompt_tokens"),
        "tokenCount": token_count,
        "layerCount": len({event["layer"] for event in normalized_events}),
        "eventCount": len(normalized_events),
        "requestMs": round(request_ms, 3),
        "ttftMs": round(prefill_ms, 3),
        "meanDecodeMs": round(mean_decode, 3),
        "decodeTokensPerSecond": round(decode_rate, 2),
        "totalLayerMs": round(total_layer_ms, 3),
        "timingScope": metadata.get("step_timing_scope", "model_forward"),
        "engineMetrics": engine_metrics,
        "steps": normalized_steps,
        "events": normalized_events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    runs = [compact_trace(path.resolve()) for path in args.inputs]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": runs}, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(runs)} showcase runs to {output}")


if __name__ == "__main__":
    main()
