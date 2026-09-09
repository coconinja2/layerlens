"""CLI for one profiled request against an instrumented vLLM server."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    path = capture_vllm(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        control_path=args.control,
        events_path=args.events,
        output_path=args.output,
    )
    print(f"Wrote vLLM layer trace to {path}")


if __name__ == "__main__":
    main()

