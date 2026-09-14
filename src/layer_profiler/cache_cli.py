"""Command-line cache opportunity planner for tokenized prompt workloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cache_plan import PrefixRequest, compare_prefix_schedules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan cache-aware request order without storing prompt content"
    )
    parser.add_argument("input", type=Path, help="JSON workload containing token ID arrays")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--capacity-blocks", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument(
        "--mlp-projections",
        type=int,
        choices=(2, 3),
        default=3,
        help="Two for a dense MLP, three for a gated MLP",
    )
    parser.add_argument(
        "--cache-namespace",
        default="layerlens-default",
        help="Model/revision namespace used only inside block hashes",
    )
    parser.add_argument("--output", type=Path, help="Optional report path; stdout by default")
    return parser


def _requests(payload: Any) -> list[PrefixRequest]:
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        raise ValueError("workload must be an object with a requests array")
    requests = []
    for index, row in enumerate(payload["requests"]):
        if not isinstance(row, dict) or not isinstance(row.get("token_ids"), list):
            raise ValueError("each request must contain a token_ids array")
        requests.append(
            PrefixRequest(
                request_ref=row.get("request_ref", f"request-{index}"),
                token_ids=tuple(row["token_ids"]),
            )
        )
    return requests


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        report = compare_prefix_schedules(
            _requests(payload),
            block_size=args.block_size,
            capacity_blocks=args.capacity_blocks,
            num_layers=args.layers,
            cache_namespace=args.cache_namespace,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            mlp_projections=args.mlp_projections,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.exit(2, f"LayerLens: {error}\n")
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"Wrote cache plan to {args.output}")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
