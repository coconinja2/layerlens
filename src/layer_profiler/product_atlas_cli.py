"""Command-line entry point for the exact Product Atlas experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from .privacy import safe_model_identifier
from .product_atlas import analyze_weight_family, benchmark_product_atlas


DEFAULT_PROJECTIONS = ("mlp.gate_proj.weight", "mlp.up_proj.weight")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find exact, bit-identical weight products shared by projections that consume "
            "the same activation. No weights or activations are quantized."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    parser.add_argument("--layer", type=int, default=0, help="Decoder layer to inspect")
    parser.add_argument(
        "--projection",
        action="append",
        dest="projections",
        help=(
            "Parameter suffix within the selected layer; repeat for a projection family. "
            "Defaults to MLP gate_proj and up_proj."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Stored model dtype to analyze; this does not add quantization",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=0,
        help="Run the inspectable CPU reference executor (0 only analyzes operation counts)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face to download missing model files",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def _resolve_parameters(
    model: torch.nn.Module, layer: int, projections: list[str]
) -> tuple[list[str], list[torch.Tensor]]:
    parameters = dict(model.named_parameters())
    names: list[str] = []
    tensors: list[torch.Tensor] = []
    for projection in projections:
        suffix = f"layers.{layer}.{projection}"
        matches = [(name, value) for name, value in parameters.items() if name.endswith(suffix)]
        if len(matches) != 1:
            available = [
                name
                for name, value in parameters.items()
                if value.ndim == 2 and f"layers.{layer}." in name
            ]
            raise ValueError(
                f"expected one parameter ending in {suffix!r}, found {len(matches)}. "
                f"Two-dimensional parameters in layer {layer}: {available[:20]}"
            )
        name, value = matches[0]
        names.append(name)
        tensors.append(value)
    return names, tensors


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.benchmark_iterations < 0:
        raise SystemExit("--benchmark-iterations cannot be negative")
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    projections = args.projections or list(DEFAULT_PROJECTIONS)
    names, weights = _resolve_parameters(model, args.layer, projections)
    analysis = analyze_weight_family(weights)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "experiment": "exact_product_atlas",
        "model": safe_model_identifier(args.model),
        "layer": args.layer,
        "parameters": names,
        "analysis": analysis.to_dict(),
        "contract": {
            "kv_cache": False,
            "activation_quantization": False,
            "weight_quantization_added": False,
            "reuse_key": "input coordinate + bit-identical stored weight value",
            "production_kernel": False,
        },
    }
    if args.benchmark_iterations:
        _, benchmark = benchmark_product_atlas(
            weights, iterations=args.benchmark_iterations
        )
        report["reference_benchmark"] = benchmark.to_dict()

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
