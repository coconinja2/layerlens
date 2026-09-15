"""Benchmark precomputed BF16 mantissa multiplication on real model values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from .precomputed_multiply import (
    benchmark_precomputed_products,
    benchmark_precomputed_projection,
    bf16_bits,
)
from .privacy import safe_model_identifier
from .repeatability_cli import DEFAULT_PROMPT


DEFAULT_TARGET = "layers.0.linear_attn.in_proj_qkv"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native BF16 multiplication with fully precomputed mantissa tables."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Never written to the report")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-vectors", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--output-rows", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--parallel-threads", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/precomputed-bf16-multiply.json"),
    )
    return parser


def _resolve_linear(model: torch.nn.Module, suffix: str) -> tuple[str, torch.nn.Linear]:
    matches = [(name, module) for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1 or not isinstance(matches[0][1], torch.nn.Linear):
        raise ValueError(f"expected one torch.nn.Linear ending in {suffix!r}")
    return matches[0][0], matches[0][1]


def _unique_vectors(values: torch.Tensor) -> int:
    digests = {
        hashlib.blake2b(row.contiguous().view(torch.uint8).numpy().tobytes(), digest_size=16).digest()
        for row in values
    }
    return len(digests)


def _weight_classes(weight: torch.Tensor) -> dict[str, int | float]:
    bits = bf16_bits(weight)
    magnitude = bits & 0x7FFF
    exponent = (bits >> 7) & 0xFF
    fraction = bits & 0x7F
    total = bits.numel()
    zero = int((magnitude == 0).sum())
    one = int((magnitude == 0x3F80).sum())
    power_mask = (fraction == 0) & (exponent > 0) & (exponent < 0xFF)
    power_of_two = int(power_mask.sum())
    fast_path = int(((magnitude == 0) | (magnitude == 0x3F80) | power_mask).sum())
    unique = int(torch.unique(bits).numel())
    model_specific_table_bytes = unique * 65536 * 4
    return {
        "total_values": total,
        "unique_bit_patterns": unique,
        "exact_zero_values": zero,
        "exact_one_magnitude_values": one,
        "exact_power_of_two_magnitude_values": power_of_two,
        "algebraic_fast_path_rate": fast_path / total,
        "model_specific_all_activation_products_fp32_bytes": model_specific_table_bytes,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if min(
        args.max_new_tokens,
        args.max_vectors,
        args.max_pairs,
        args.output_rows,
        args.iterations,
    ) < 1:
        raise SystemExit("numeric benchmark arguments must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    ).cpu().eval()
    module_name, module = _resolve_linear(model, args.target)
    captured: list[torch.Tensor] = []
    captured_count = 0

    def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        nonlocal captured_count
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        remaining = args.max_vectors - captured_count
        if remaining <= 0:
            return
        tensor = inputs[0]
        rows = tensor.detach().reshape(-1, tensor.shape[-1])[:remaining].cpu().contiguous()
        if rows.numel():
            captured.append(rows)
            captured_count += rows.shape[0]

    handle = module.register_forward_pre_hook(hook)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    activations = torch.cat(captured, dim=0)
    vectors = activations.shape[0]
    output_width, input_width = module.weight.shape
    available_pairs = vectors * output_width * input_width
    pair_count = min(args.max_pairs, available_pairs)
    indices = torch.arange(pair_count, dtype=torch.int64)
    vector_indices = indices % vectors
    column_indices = (indices // vectors) % input_width
    row_indices = (indices // (vectors * input_width)) % output_width
    left = module.weight.detach().cpu()[row_indices, column_indices].contiguous()
    right = activations[vector_indices, column_indices].contiguous()
    parallel = args.parallel_threads or min(os.cpu_count() or 1, torch.get_num_threads())
    thread_counts = tuple(dict.fromkeys((1, max(1, parallel))))

    report = {
        "schema_version": "1.0",
        "experiment": "precomputed_bf16_multiplication",
        "model": safe_model_identifier(args.model),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "device": "cpu",
            "dtype": "bfloat16",
        },
        "workload": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(generated.shape[-1]) - prompt_tokens,
            "captured_vectors": vectors,
            "different_captured_vectors": _unique_vectors(activations),
            "module": module_name,
            "thread_counts": list(thread_counts),
        },
        "method": {
            "dynamic_result_cache": False,
            "hash_lookup": False,
            "mantissa_table_entries": 256 * 256,
            "nibble_table_entries": 16 * 16,
            "sign_operation": "xor",
            "exponent_operation": "integer addition",
            "table_addressing": "direct packed integer index",
            "operand_components_predecoded_before_timing": True,
            "special_values": "excluded from timed sample; native fallback required",
        },
        "weight_classes": _weight_classes(module.weight.detach().cpu()),
        "scalar_product_benchmark": benchmark_precomputed_products(
            left,
            right,
            iterations=args.iterations,
            thread_counts=thread_counts,
        ),
        "projection_benchmark": benchmark_precomputed_projection(
            module.weight.detach().cpu(),
            activations[0],
            output_rows=min(args.output_rows, output_width),
            iterations=args.iterations,
            thread_counts=thread_counts,
        ),
        "privacy": {
            "contains_prompt": False,
            "contains_token_ids": False,
            "contains_weights": False,
            "contains_activations": False,
            "contains_products": False,
            "contains_hostname": False,
            "contains_username": False,
            "aggregate_statistics_only": True,
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
