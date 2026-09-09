"""CLI for one profiled request against an instrumented vLLM server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .vllm import capture_vllm


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture layer timings from instrumented vLLM")
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--control", type=Path, default=Path("traces/vllm_raw/layer-profiler.control"))
    parser.add_argument("--events", type=Path, default=Path("traces/vllm_raw/vllm-layer-events.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("traces/vllm.json"))
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="Store prompt and generated text (off by default for privacy)",
    )
    parser.add_argument(
        "--no-metrics", action="store_true", help="Disable vLLM /metrics collection"
    )
    parser.add_argument(
        "--metrics-url",
        help="Prometheus endpoint; defaults to BASE_URL/metrics and is never stored",
    )
    parser.add_argument("--metrics-sample-ms", type=int, default=50)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Capture portable vLLM engine/KV metrics without the layer injection shim",
    )
    args = parser.parse_args()
    api_key = os.environ.get("VLLM_API_KEY")
    request_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    path = capture_vllm(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        control_path=args.control,
        events_path=args.events,
        output_path=args.output,
        include_content=args.include_content,
        collect_metrics=not args.no_metrics,
        metrics_url=args.metrics_url,
        metrics_sample_ms=args.metrics_sample_ms,
        layer_events=not args.metrics_only,
        request_headers=request_headers,
    )
    print(f"Wrote vLLM layer trace to {path}")


if __name__ == "__main__":
    main()
