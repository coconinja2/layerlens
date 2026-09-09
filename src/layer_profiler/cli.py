"""CLI for profiling a Hugging Face causal language model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .profiler import LayerProfiler, ProfileConfig
from .privacy import safe_model_identifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record offline, layer-by-layer prefill and decode timings"
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, or cuda:N")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--output", type=Path, default=Path("traces/latest.json"))
    parser.add_argument("--module-pattern", default=ProfileConfig.module_pattern)
    parser.add_argument("--leaf-modules", action="store_true")
    parser.add_argument("--no-sync", action="store_true", help="Disable accurate accelerator synchronization")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="Store prompt and token text (off by default for privacy)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device).eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {name: tensor.to(args.device) for name, tensor in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])

    config = ProfileConfig(
        module_pattern=args.module_pattern,
        leaf_modules=args.leaf_modules,
        synchronize_device=not args.no_sync,
    )
    with torch.inference_mode(), LayerProfiler(model, config) as profiler:
        generated = profiler.profile_generate(
            model.generate,
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_ids = generated[0, prompt_length:].tolist()
    profiler.attach_generated_tokens(
        generated_ids,
        decode=(lambda token_id: tokenizer.decode([token_id])) if args.include_content else None,
        include_ids=args.include_content,
    )
    trace_metadata = dict(
        model=safe_model_identifier(args.model),
        prompt_tokens=prompt_length,
        generated_tokens=len(generated_ids),
        device=args.device,
        dtype=args.dtype,
        content_recorded=args.include_content,
    )
    if args.include_content:
        trace_metadata["prompt"] = args.prompt
    trace = profiler.trace(**trace_metadata)
    path = trace.write(args.output)
    print(f"Wrote {len(trace.events)} layer events across {len(trace.steps)} steps to {path}")
    if trace.errors:
        print("Warnings:")
        for error in trace.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
