"""Capture and aggregate bottom-up repeatability patterns in a Hugging Face model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from .privacy import safe_model_identifier
from .product_atlas import analyze_weight_family
from .repeatability import (
    FiniteDomainUnaryAtlas,
    profile_attention_heads,
    profile_tensor_repetition,
)


DEFAULT_PROMPT = "Explain why the moon changes shape in the sky using simple words."


class _TensorCollector:
    def __init__(self, max_vectors: int):
        self.max_vectors = max_vectors
        self.values: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _first_tensor(value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                result = _TensorCollector._first_tensor(item)
                if result is not None:
                    return result
        if isinstance(value, dict):
            for item in value.values():
                result = _TensorCollector._first_tensor(item)
                if result is not None:
                    return result
        return None

    def add(self, signal: str, value: Any) -> None:
        tensor = self._first_tensor(value)
        if tensor is None or tensor.ndim == 0:
            return
        remaining = self.max_vectors - self.counts[signal]
        if remaining <= 0:
            return
        rows = tensor.detach().reshape(-1, tensor.shape[-1])[:remaining].cpu().contiguous()
        if rows.numel():
            self.values[signal].append(rows)
            self.counts[signal] += rows.shape[0]

    def tensor(self, signal: str) -> torch.Tensor | None:
        chunks = self.values.get(signal)
        return torch.cat(chunks, dim=0) if chunks else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile bit-exact repetition from scalar/unary operations through matmul "
            "families, attention heads, transformer blocks, and the layer stack."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Never written to the report")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--layers",
        default="0,3",
        help="Comma-separated decoder layers; include an attention layer when possible",
    )
    parser.add_argument("--max-vectors", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--unary-iterations", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/repeatability-profile.json"),
    )
    return parser


def _module_by_suffix(model: torch.nn.Module, suffix: str) -> tuple[str, torch.nn.Module] | None:
    matches = [(name, module) for name, module in model.named_modules() if name.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def _parameters_by_suffixes(
    model: torch.nn.Module, suffixes: list[str]
) -> tuple[list[str], list[torch.Tensor]] | None:
    parameters = dict(model.named_parameters())
    names: list[str] = []
    values: list[torch.Tensor] = []
    for suffix in suffixes:
        matches = [(name, value) for name, value in parameters.items() if name.endswith(suffix)]
        if len(matches) != 1:
            return None
        name, value = matches[0]
        names.append(name)
        values.append(value)
    return names, values


def _pre_hook(collector: _TensorCollector, signal: str) -> Callable[..., None]:
    def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        collector.add(signal, inputs)

    return hook


def _post_hook(collector: _TensorCollector, signal: str) -> Callable[..., None]:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        collector.add(signal, output)

    return hook


def _layer_signal(layer: int, name: str) -> str:
    return f"layer_{layer}.{name}"


def _attach_hooks(
    model: torch.nn.Module,
    layers: list[int],
    collector: _TensorCollector,
    warnings: list[str],
) -> list[Any]:
    handles: list[Any] = []
    targets = {
        "layer_input": ("", "pre"),
        "layer_output": ("", "post"),
        "mlp_input": (".mlp", "pre"),
        "gate_preactivation": (".mlp.gate_proj", "post"),
        "silu_output": (".mlp.act_fn", "post"),
        "query_projection": (".self_attn.q_proj", "post"),
    }
    for layer in layers:
        base = f"layers.{layer}"
        for signal_name, (tail, phase) in targets.items():
            resolved = _module_by_suffix(model, base + tail)
            if resolved is None:
                if signal_name not in {"query_projection"}:
                    warnings.append(f"layer {layer}: module {base + tail!r} was not uniquely resolved")
                continue
            _, module = resolved
            signal = _layer_signal(layer, signal_name)
            hook = _pre_hook(collector, signal) if phase == "pre" else _post_hook(collector, signal)
            handles.append(
                module.register_forward_pre_hook(hook)
                if phase == "pre"
                else module.register_forward_hook(hook)
            )
    return handles


def _projection_profiles(model: torch.nn.Module, layer: int) -> list[dict[str, object]]:
    families = {
        "mlp_gate_up": [
            f"layers.{layer}.mlp.gate_proj.weight",
            f"layers.{layer}.mlp.up_proj.weight",
        ],
        "attention_qkv": [
            f"layers.{layer}.self_attn.q_proj.weight",
            f"layers.{layer}.self_attn.k_proj.weight",
            f"layers.{layer}.self_attn.v_proj.weight",
        ],
    }
    profiles: list[dict[str, object]] = []
    for family, suffixes in families.items():
        resolved = _parameters_by_suffixes(model, suffixes)
        if resolved is None:
            continue
        names, weights = resolved
        profiles.append(
            {
                "family": family,
                "parameters": names,
                "analysis": analyze_weight_family(weights).to_dict(),
            }
        )
    return profiles


def _build_findings(layers: list[dict[str, object]]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for layer in layers:
        layer_number = layer["layer"]
        for family in layer["projection_families"]:
            analysis = family["analysis"]
            findings.append(
                {
                    "level": "matrix_product",
                    "layer": layer_number,
                    "candidate": family["family"],
                    "exact_reuse_ratio": analysis["multiplication_savings_ratio"],
                    "decision": "native_kernel_required",
                }
            )
        for unary in layer.get("unary_atlases", []):
            findings.append(
                {
                    "level": "unary_function",
                    "layer": layer_number,
                    "candidate": f"complete_{unary['function']}_domain_table",
                    "exact_reuse_ratio": unary["observed_replay_hit_rate"],
                    "wall_time_ratio": unary["lookup_to_direct_ratio"],
                    "decision": (
                        "benchmark_native_kernel"
                        if unary["lookup_to_direct_ratio"] <= 1.25
                        else "reject_current_lookup_path"
                    ),
                }
            )
        for signal in layer["signals"]:
            findings.append(
                {
                    "level": "activation_tile",
                    "layer": layer_number,
                    "candidate": signal["signal"],
                    "exact_reuse_ratio": signal["tile_replay_hit_rate"],
                    "decision": (
                        "measure_partial_product_cache"
                        if signal["tile_replay_hit_rate"] >= 0.1
                        else "insufficient_exact_tile_reuse"
                    ),
                }
            )
    return findings


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    layers = sorted({int(value.strip()) for value in args.layers.split(",") if value.strip()})
    if not layers:
        raise SystemExit("--layers must select at least one layer")
    if args.max_vectors < 2 or args.tile_size < 1 or args.max_new_tokens < 1:
        raise SystemExit("max vectors/tokens must be at least 2/1 and tile size must be positive")

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
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {name: tensor.to(args.device) for name, tensor in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])

    warnings: list[str] = []
    collector = _TensorCollector(args.max_vectors)
    handles = _attach_hooks(model, layers, collector, warnings)
    started = time.perf_counter_ns()
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
    capture_ms = (time.perf_counter_ns() - started) / 1_000_000
    generated_tokens = int(generated.shape[-1]) - prompt_tokens

    text_config = getattr(model.config, "text_config", model.config)
    num_heads = int(getattr(text_config, "num_attention_heads", 0) or 0)
    head_dim = int(getattr(text_config, "head_dim", 0) or 0)
    unary_atlases = (
        {
            name: FiniteDomainUnaryAtlas(name, dtype)
            for name in ("sigmoid", "silu")
        }
        if dtype in (torch.bfloat16, torch.float16)
        else {}
    )

    layer_reports: list[dict[str, object]] = []
    stack_inputs: list[torch.Tensor] = []
    for layer in layers:
        signal_reports: list[dict[str, object]] = []
        for signal_name in (
            "layer_input",
            "mlp_input",
            "gate_preactivation",
            "silu_output",
            "query_projection",
            "layer_output",
        ):
            signal = _layer_signal(layer, signal_name)
            values = collector.tensor(signal)
            if values is None:
                continue
            signal_reports.append(
                profile_tensor_repetition(
                    values, signal=signal_name, tile_size=args.tile_size
                ).to_dict()
            )
            if signal_name == "layer_input":
                stack_inputs.append(values)

        gate_values = collector.tensor(_layer_signal(layer, "gate_preactivation"))
        unary_reports: list[dict[str, object]] = []
        if gate_values is not None:
            for atlas in unary_atlases.values():
                unary_reports.append(
                    {
                        "source_signal": "mlp_gate_preactivation",
                        **atlas.profile(
                            gate_values, iterations=args.unary_iterations
                        ).to_dict(),
                    }
                )

        head_report: dict[str, object] | None = None
        query_values = collector.tensor(_layer_signal(layer, "query_projection"))
        expected_query_width = num_heads * head_dim
        query_states: torch.Tensor | None = None
        attention_gate: torch.Tensor | None = None
        if query_values is not None and expected_query_width:
            if query_values.shape[1] == expected_query_width:
                query_states = query_values
            elif query_values.shape[1] == expected_query_width * 2:
                paired = query_values.reshape(-1, num_heads, head_dim * 2)
                query_states, attention_gate = torch.split(paired, head_dim, dim=2)
                query_states = query_states.reshape(-1, expected_query_width)
                attention_gate = attention_gate.reshape(-1, expected_query_width)
        if query_states is not None:
            signal_reports.append(
                profile_tensor_repetition(
                    query_states,
                    signal="query_state",
                    tile_size=args.tile_size,
                ).to_dict()
            )
            head_report = profile_attention_heads(
                query_states,
                signal="query_state",
                heads=num_heads,
                head_dim=head_dim,
            ).to_dict()
        if attention_gate is not None:
            signal_reports.append(
                profile_tensor_repetition(
                    attention_gate,
                    signal="attention_output_gate",
                    tile_size=args.tile_size,
                ).to_dict()
            )
            sigmoid_atlas = unary_atlases.get("sigmoid")
            if sigmoid_atlas is not None:
                unary_reports.append(
                    {
                        "source_signal": "attention_output_gate",
                        **sigmoid_atlas.profile(
                            attention_gate, iterations=args.unary_iterations
                        ).to_dict(),
                    }
                )

        layer_reports.append(
            {
                "layer": layer,
                "signals": signal_reports,
                "projection_families": _projection_profiles(model, layer),
                "unary_atlases": unary_reports,
                "attention_heads": head_report,
            }
        )

    stack_report: dict[str, object] | None = None
    if stack_inputs and len({tensor.shape[1] for tensor in stack_inputs}) == 1:
        stack_report = profile_tensor_repetition(
            torch.cat(stack_inputs),
            signal="selected_layer_inputs",
            tile_size=args.tile_size,
        ).to_dict()

    report: dict[str, object] = {
        "schema_version": "1.0",
        "experiment": "bottom_up_repeatability",
        "model": safe_model_identifier(args.model),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
        "capture": {
            "selected_layers": layers,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "max_vectors_per_signal": args.max_vectors,
            "tile_size": args.tile_size,
            "dtype": args.dtype,
            "device": args.device,
            "capture_ms": capture_ms,
        },
        "hierarchy": [
            "scalar_bit_pattern",
            "unary_sigmoid_silu",
            "activation_coordinate_and_tile",
            "sibling_matrix_products",
            "attention_head",
            "transformer_block",
            "selected_layer_stack",
        ],
        "layers": layer_reports,
        "stack": stack_report,
        "findings": _build_findings(layer_reports),
        "warnings": warnings,
        "privacy": {
            "contains_prompt": False,
            "contains_token_ids": False,
            "contains_activations": False,
            "contains_generated_text": False,
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
