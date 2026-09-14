"""Raw with/without exact-cache experiments on real model projection inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as torch_functional
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from .exact_result_cache import benchmark_exact_result_caches
from .privacy import safe_model_identifier
from .repeatability_cli import DEFAULT_PROMPT


DEFAULT_TARGETS = (
    "layers.0.linear_attn.in_proj_qkv",
    "layers.3.mlp.gate_proj",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark dense matvec against exact in-memory and SQLite result caches "
            "using projection inputs captured from a real model run."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Never written to the report")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-vectors", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Linear-module suffix to capture; repeat for multiple levels",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disk-cache", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/exact-result-cache.json"),
    )
    return parser


def _resolve_module(model: torch.nn.Module, suffix: str) -> tuple[str, torch.nn.Module]:
    matches = [(name, module) for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one module ending in {suffix!r}, found {len(matches)}")
    name, module = matches[0]
    if not isinstance(module, torch.nn.Linear):
        raise ValueError(f"{name} is {type(module).__name__}, not torch.nn.Linear")
    return name, module


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if min(args.max_new_tokens, args.max_vectors, args.cycles) < 1:
        raise SystemExit("tokens, vectors, and cycles must be positive")
    targets = args.targets or list(DEFAULT_TARGETS)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device).eval()

    captured: dict[str, list[torch.Tensor]] = defaultdict(list)
    captured_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    output_counts: dict[str, int] = defaultdict(int)
    resolved: list[tuple[str, torch.nn.Linear]] = []
    handles: list[Any] = []
    for suffix in targets:
        name, module = _resolve_module(model, suffix)
        resolved.append((name, module))

        def hook(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            *,
            signal: str = name,
        ) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            remaining = args.max_vectors - counts[signal]
            if remaining <= 0:
                return
            tensor = inputs[0]
            rows = tensor.detach().reshape(-1, tensor.shape[-1])[:remaining].cpu().contiguous()
            if rows.numel():
                captured[signal].append(rows)
                counts[signal] += rows.shape[0]

        handles.append(module.register_forward_pre_hook(hook))

        def output_hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            signal: str = name,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                return
            remaining = args.max_vectors - output_counts[signal]
            if remaining <= 0:
                return
            rows = output.detach().reshape(-1, output.shape[-1])[:remaining].cpu().contiguous()
            if rows.numel():
                captured_outputs[signal].append(rows)
                output_counts[signal] += rows.shape[0]

        handles.append(module.register_forward_hook(output_hook))

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
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
        for handle in handles:
            handle.remove()

    disk_path = args.disk_cache or Path(
        f"/tmp/layerlens-exact-result-{os.getpid()}-{time.time_ns()}.sqlite"
    )
    namespace_seed = f"raw-{time.time_ns()}"
    experiments: list[dict[str, object]] = []
    for name, module in resolved:
        values = torch.cat(captured[name], dim=0)
        native_outputs = torch.cat(captured_outputs[name], dim=0)
        replayed_outputs = torch_functional.linear(
            values.to(module.weight.dtype),
            module.weight.detach().cpu(),
            module.bias.detach().cpu() if module.bias is not None else None,
        )
        result = benchmark_exact_result_caches(
            module.weight,
            values,
            disk_path=disk_path,
            label=name,
            cycles=args.cycles,
            bias=module.bias,
            namespace_seed=namespace_seed,
        )
        result["native_model_projection_bitwise_agreement"] = (
            native_outputs.dtype == replayed_outputs.dtype
            and native_outputs.shape == replayed_outputs.shape
            and native_outputs.contiguous().view(torch.uint8).numpy().tobytes()
            == replayed_outputs.contiguous().view(torch.uint8).numpy().tobytes()
        )
        experiments.append(result)

    report = {
        "schema_version": "1.0",
        "experiment": "exact_result_cache_raw_benchmark",
        "model": safe_model_identifier(args.model),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "device": args.device,
            "dtype": args.dtype,
        },
        "workload": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(generated.shape[-1]) - prompt_tokens,
            "max_vectors_per_target": args.max_vectors,
            "cycles": args.cycles,
        },
        "cache_contract": {
            "key": "blake2b(exact activation dtype + shape + bytes)",
            "namespace": "sha256(exact weight and bias bytes)",
            "value": "native-dtype projection output",
            "input_bytes_stored": False,
            "disk_backend": "sqlite_wal_synchronous_full",
            "disk_path_recorded": False,
            "approximation_added": False,
        },
        "experiments": experiments,
        "privacy": {
            "contains_prompt": False,
            "contains_token_ids": False,
            "contains_activations": False,
            "contains_cached_outputs": False,
            "contains_cache_path": False,
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
