"""CLI for capturing one privacy-safe Ollama benchmark."""

from __future__ import annotations

import argparse
import os
import urllib.error
from pathlib import Path

from .ollama import capture_ollama


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Ollama inference metrics")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--timeout", type=float, default=600, help="Request timeout in seconds")
    parser.add_argument("--output", type=Path, default=Path("traces/ollama.json"))
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="Store prompt, response, and thinking text (off by default for privacy)",
    )
    args = parser.parse_args()
    api_key = os.environ.get("OLLAMA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        path = capture_ollama(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            output_path=args.output,
            include_content=args.include_content,
            keep_alive=args.keep_alive,
            request_headers=headers,
            timeout=args.timeout,
        )
    except (TimeoutError, urllib.error.URLError):
        parser.exit(2, "LayerLens: Ollama request timed out or the endpoint was unavailable.\n")
    print(f"Wrote Ollama benchmark trace to {path}")


if __name__ == "__main__":
    main()
