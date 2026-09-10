"""Runtime-neutral command-line entry point for LayerLens adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import CaptureRequest, adapter_registry, parse_option


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture an LLM benchmark through a LayerLens runtime adapter"
    )
    parser.add_argument("--list-runtimes", action="store_true")
    parser.add_argument("--runtime", help="Built-in or installed adapter name")
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("traces/latest.json"))
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Runtime-specific option; repeat as needed. JSON scalars are accepted.",
    )
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="Store prompt and response content; disabled by default",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_runtimes:
        print("\n".join(adapter_registry.names()))
        return
    if not args.runtime or not args.model or args.prompt is None:
        parser.error("--runtime, --model, and --prompt are required for capture")
    try:
        options = dict(parse_option(value) for value in args.option)
        path = adapter_registry.capture(
            args.runtime,
            CaptureRequest(
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                output_path=args.output,
                include_content=args.include_content,
                options=options,
            ),
        )
    except (LookupError, TypeError, ValueError) as error:
        parser.exit(2, f"LayerLens: {error}\n")
    print(f"Wrote {args.runtime} benchmark trace to {path}")


if __name__ == "__main__":
    main()
