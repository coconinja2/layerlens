# LayerLens

**Every layer. Every token. Every millisecond.**

[![Tests](https://github.com/coconinja2/layerlens/actions/workflows/tests.yml/badge.svg)](https://github.com/coconinja2/layerlens/actions/workflows/tests.yml)
[![Live benchmark](https://img.shields.io/badge/live-benchmark-49d7ff)](https://coconinja2.github.io/layerlens/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-9b87ff)](https://www.python.org/)

[Explore the live LayerLens benchmark →](https://coconinja2.github.io/layerlens/)

![LayerLens dashboard showing a transformer layer-by-token heatmap and latency benchmark](docs/layerlens-dashboard.png)

LayerLens is an offline observability tool for answering a concrete inference question:
**where does each generated token spend its time inside the model?**

It records every transformer layer during:

- the prefill pass that produces the first token;
- each subsequent KV-cache decode pass;
- in-process Hugging Face inference or an instrumented vLLM server.

For standard vLLM and Ollama endpoints, LayerLens also captures the native
engine/runtime metrics those APIs expose, without requiring model-specific
code. Exact layer timing remains opt-in because it requires execution inside
the model process.

The output is a portable JSON trace. A Streamlit dashboard turns it into a
first-token/decode scorecard, layer-by-token heatmap, sequential layer chart,
token timeline, bottleneck ranking, and raw trace explorer. Collection does
not need to run continuously or in real time.

## What the visualization shows

- **Layer × token heatmap:** which decoder layers dominate prefill and decode.
- **Execution timeline:** the exact start, end, and duration of every layer call.
- **Token-step latency:** time-to-first-token separated from KV-cache decode cost.
- **vLLM engine telemetry:** KV-cache pressure, prefix-cache effectiveness,
  request concurrency, queueing, preemptions, TTFT, and prefill/decode timing.
- **Benchmark comparison:** comparable traces across models, runtimes, hardware, and generation windows.

![LayerLens vLLM KV-cache and scheduler telemetry](docs/layerlens-kv-metrics.png)

The included showcase contains two real Qwen3.5 0.8B runs captured from an
instrumented vLLM CPU server, a direct Hugging Face GPT-2 run, and a native
Ollama Gemma4 12B run. The browser dashboard is static and deploys on GitHub
Pages; the Streamlit explorer supports arbitrary local trace uploads.

## Validated model matrix

| Model | Runtime | Layer timeline | Engine/runtime metrics | Result |
|---|---|---:|---:|---:|
| Qwen3.5 0.8B | vLLM 0.27.1 | Yes, 24 layers | Yes, including KV cache | Pass |
| GPT-2 | Hugging Face Transformers | Yes, 12 layers | In-process timing | Pass |
| Gemma4 12B Q4_K_M | Ollama 0.32.14 | Not exposed by Ollama | Yes, native API | Pass |

![LayerLens comparing an Ollama model with vLLM and Hugging Face runs](docs/layerlens-multi-model.png)

Model selection is runtime-driven rather than hard-coded. Hugging Face layer
paths are discovered with a configurable regex, vLLM class matching is
configurable through `LLM_LAYER_CLASS_PATTERN`, and Ollama accepts any locally
installed or cloud-accessible model name.

## Measurement model

Accelerator synchronization is enabled at layer boundaries by default. That
makes a profiled request slower, but prevents asynchronous CUDA/MPS kernel
launches from being reported as completed computation. Profiled numbers should
only be compared with other profiled numbers collected under the same settings.

Step `0` is the prompt prefill and produces generated token `0`. Steps `1..N`
are single-token decode passes. Whole-forward durations and per-layer durations
are stored independently. Parent and child module time is never silently added
together.

For direct Hugging Face profiling, step duration covers the full model forward.
For injected vLLM traces, it covers the decoder stack from the start of layer 0
through the end of the final layer; the trace records this distinction as
`step_timing_scope` metadata.

## Install with Poetry

```bash
poetry install --with dashboard,dev
```

## Profile a Hugging Face model

```bash
poetry run llm-layer-profile \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain what a KV cache is." \
  --max-new-tokens 8 \
  --device cpu \
  --output traces/qwen-hf.json
```

The default selector recognizes common paths such as `model.layers.0`,
`transformer.h.0`, and `blocks.0`. Use `--module-pattern` for another model
layout. `--leaf-modules` adds attention/MLP/operator detail, with substantially
more instrumentation overhead.

## Profile a vLLM server layer by layer

vLLM's official PyTorch profiling endpoints provide rich engine and operator
traces, but HTTP clients cannot install hooks into server-side model layers.
This repository includes an opt-in `sitecustomize.py` shim for local development
servers. It patches PyTorch's module call boundary inside the container, stays
dormant unless a capture control file contains a run ID, and records only
decoder-layer calls during the requested window.

Example container launch (adjust image, cache, model, and memory flags):

```bash
mkdir -p traces/vllm_raw
docker run --name vllm-layer-profiler -p 8001:8000 --shm-size=1g \
  -e PYTHONPATH=/layer-profiler-inject \
  -e LLM_LAYER_PROFILER=1 \
  -e LLM_LAYER_PROFILE_CONTROL=/profiles/layer-profiler.control \
  -e LLM_LAYER_PROFILE_EVENTS=/profiles/vllm-layer-events.jsonl \
  -v "$PWD/vllm_inject:/layer-profiler-inject:ro" \
  -v "$PWD/traces/vllm_raw:/profiles" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  vllm-cpu:arm64 \
  --model Qwen/Qwen3.5-0.8B --enforce-eager --max-model-len 2048
```

Capture exactly one request after the server is ready:

```bash
poetry run llm-layer-profile-vllm \
  --base-url http://localhost:8001 \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "Explain a KV cache in one sentence." \
  --max-tokens 8 \
  --output traces/qwen-vllm.json
```

The capture polls vLLM's Prometheus-compatible `/metrics` endpoint during the
request and records request-scoped counter deltas plus a KV-cache/concurrency
timeline. Endpoint URLs, raw Prometheus labels, prompts, generated text,
hostnames, usernames, and filesystem paths are not stored. Use
`--include-content` only when the text is intentionally safe to retain.

### Metrics-only mode: no injection required

Every standard vLLM server can produce an engine/KV trace without mounting the
optional layer hook:

```bash
poetry run layerlens-vllm \
  --base-url http://localhost:8000 \
  --model your-org/your-model \
  --prompt "Benchmark prompt" \
  --max-tokens 32 \
  --metrics-only \
  --output traces/engine-metrics.json
```

This mode works across model architectures because it consumes vLLM's public
OpenAI-compatible completion and Prometheus APIs. Add the injection shim only
when exact decoder-layer timing is required. For servers launched with API-key
authentication, set `VLLM_API_KEY`; the value is used as a bearer token for the
request and metrics endpoint and is never serialized.

### Python integration

```python
from pathlib import Path
from layer_profiler.vllm import capture_vllm

capture_vllm(
    base_url="http://localhost:8000",
    model="your-org/your-model",
    prompt="Benchmark prompt",
    max_tokens=32,
    control_path=Path("traces/layer-profiler.control"),
    events_path=Path("traces/layer-events.jsonl"),
    output_path=Path("traces/run.json"),
    layer_events=False,  # metrics-only; works on an unmodified vLLM server
)
```

## Benchmark Ollama models

Ollama exposes aggregate load, prompt-evaluation, generation, and token metrics
through its native API. LayerLens also records safe model architecture,
quantization, context capacity, and loaded processor-memory data:

```bash
poetry run layerlens-ollama \
  --model gemma4:12b \
  --prompt "Benchmark prompt" \
  --max-tokens 8 \
  --output traces/gemma4.json
```

Ollama does not currently expose actual block-level KV-cache occupancy through
its API. LayerLens therefore reports estimated context utilization separately
and never labels it as measured KV-cache usage. Cloud authentication uses
`OLLAMA_API_KEY`, which is never written to the trace.

The class matcher covers common `DecoderLayer`, `TransformerLayer`, and
`Block` names. Override `LLM_LAYER_CLASS_PATTERN` for a custom architecture.
Only use the shim on a local profiling instance, never a production server.

For lower-level engine and kernel analysis, vLLM 0.13+ also supports its
official `--profiler-config` flag and `/start_profile` and `/stop_profile`
endpoints. Those traces complement this tool rather than replacing layer-level
attribution.

## Open the visual dashboard

```bash
./scripts/run_dashboard.sh
```

The dashboard discovers `traces/*.json` automatically, or accepts an uploaded
trace. Use the sidebar to isolate prefill, decode, or individual token steps.

To rebuild the publishable GitHub Pages dataset from captured traces:

```bash
poetry run python scripts/build_showcase.py
python3 -m http.server 4173 --directory docs
```

Then open `http://localhost:4173`.

## Custom generation loops

For exact control, wrap every forward call explicitly:

```python
with LayerProfiler(model) as profiler:
    first = profiler.profile_forward(model, **prompt_inputs, step=0)
    next_result = profiler.profile_forward(model, **decode_inputs, step=1)
    profiler.mark_token(token_id=42, text=" answer")

profiler.trace(model="my-model").write("traces/run.json")
```

See `examples/manual_generation.py` for a complete greedy KV-cache loop.

## Trace fields

Each trace includes privacy-safe model/environment metadata, whole forward
steps, generated-token indices, layer events, tensor shapes, device, supported
CUDA/MPS allocation deltas, and optional normalized vLLM engine metrics. Schema
`1.1` adds the top-level `metrics` object while preserving the existing event
format.

## Privacy defaults

- Prompt, generated text, and reconstructable generated token IDs are excluded
  unless `--include-content` is passed.
- Server and metrics endpoint URLs are used for the request but never serialized.
- Raw Prometheus labels are discarded; cache configuration uses a strict allowlist.
- Environment metadata records only OS family and machine architecture—not a
  hostname, username, home directory, or local path.
- Absolute local model paths are serialized as `local-model`; registry model IDs
  such as `Qwen/Qwen3.5-0.8B` are preserved.
- Raw captures under `traces/` are ignored by Git; only the compact, sanitized
  showcase dataset is published.

CI also runs `scripts/privacy_audit.py` to reject common API keys, private keys,
absolute home-directory paths, raw prompt content, endpoint URLs, and run IDs
from publishable artifacts.

## Caveats

- Hooks measure Python module boundaries. Use vLLM/PyTorch Profiler or Nsight
  for individual kernels and asynchronous overlap.
- Warm up compilation, graph capture, and disk caches before representative
  measurements.
- vLLM continuous batching can mix multiple requests in one engine iteration.
  The included vLLM capture command intentionally sends one request at a time.
- Profiling is invasive by nature. Never treat these timings as unprofiled
  production throughput.
