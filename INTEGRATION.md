# Integrating LayerLens

LayerLens can be used as a command-line application, embedded as a Python
library, or extended with an adapter distributed by another Python package.
The adapter boundary is intentionally small: a runtime receives one
`CaptureRequest` and writes one portable LayerLens trace.

## 1. Install

From a clone of this repository:

```bash
poetry install --with dashboard,dev
poetry run layerlens-capture --list-runtimes
poetry run layerlens-capture --describe-runtime vllm
```

Or install the standalone commands directly from GitHub:

```bash
python3 -m pip install "git+https://github.com/coconinja2/layerlens.git"
layerlens-capture --list-runtimes
```

The built-in adapter names are `ollama` and `vllm`. The specialized
`layerlens`, `layerlens-ollama`, and `layerlens-vllm` commands remain available
when runtime-specific flags are more convenient.

`--describe-runtime` reports whether an adapter can provide exact, sampled,
estimated, or unavailable layer, KV-cache, scheduler, and kernel data before a
capture is started.

Prompts and generated content are excluded from traces by default. Keep API
keys in `OLLAMA_API_KEY`, `VLLM_API_KEY`, or your adapter's environment
variable; do not pass secrets through `--option`.

## 2. Capture through the generic CLI

### Ollama

```bash
poetry run layerlens-capture \
  --runtime ollama \
  --model gemma4:12b \
  --prompt "Benchmark prompt" \
  --max-tokens 8 \
  --output traces/ollama.json \
  --option base_url=http://localhost:11434
```

Supported Ollama options are `base_url`, `keep_alive`, and `timeout`.

### vLLM

The generic vLLM adapter defaults to metrics-only mode, so it works with an
unmodified OpenAI-compatible vLLM server:

```bash
poetry run layerlens-capture \
  --runtime vllm \
  --model your-org/your-model \
  --prompt "Benchmark prompt" \
  --max-tokens 32 \
  --output traces/vllm.json \
  --option base_url=http://localhost:8000
```

Supported vLLM options are `base_url`, `collect_metrics`, `metrics_url`,
`metrics_sample_ms`, `layer_events`, `control_path`, and `events_path`. Enable
`layer_events=true` only when the optional injection shim is mounted into the
server; then provide control and event paths visible to the LayerLens process.

`--option` values accept JSON scalars, for example:

```bash
--option layer_events=true --option metrics_sample_ms=25
```

### Cache-aware workload planning

For a queue of already-tokenized prompts, LayerLens can compare FCFS execution
with an exact longest-prefix-first order under a bounded block budget:

```bash
layerlens-cache-plan workload.json \
  --block-size 16 \
  --capacity-blocks 4096 \
  --layers 32 \
  --hidden-size 4096 \
  --intermediate-size 11008 \
  --mlp-projections 3 \
  --cache-namespace model-revision-and-adapter
```

The workload format deliberately excludes prompt text:

```json
{
  "requests": [
    {"request_ref": "request-0", "token_ids": [1, 2, 3, 4]},
    {"request_ref": "request-1", "token_ids": [1, 2, 3, 9]}
  ]
}
```

Only complete blocks are counted as reusable. A block key is a chained SHA-256
digest over its model namespace, parent block, and token IDs. Reports expose
only aggregate hits and trace-local references. Execute the returned
`prefix_aware.order` against a runtime with compatible prefix caching enabled;
the planner itself does not hold model tensors or replace the engine cache.

When model dimensions are supplied, the report adds a deliberately bounded
dense-decoder estimate:

```text
fixed FLOPs per layer-token = 8*d^2 + 2*P*d*m
```

Here `P=2` for a two-projection MLP and `P=3` for a gated MLP. The estimate
counts Q/K/V/output and MLP matrix multiplications only; consult the report's
`cost_model.excludes` list before comparing hybrid, MoE, or custom architectures.

The same planner is available as a library:

```python
from layer_profiler import PrefixRequest, compare_prefix_schedules

report = compare_prefix_schedules(
    [
        PrefixRequest("request-0", (1, 2, 3, 4)),
        PrefixRequest("request-1", (1, 2, 3, 9)),
    ],
    block_size=2,
    capacity_blocks=64,
    num_layers=32,
    cache_namespace="model-revision-and-adapter",
)
```

## 3. Embed LayerLens in Python

```python
from pathlib import Path

from layer_profiler import CaptureRequest, capture

trace_path = capture(
    "ollama",
    CaptureRequest(
        model="gemma4:12b",
        prompt="Benchmark prompt",
        max_tokens=8,
        output_path=Path("traces/ollama.json"),
        options={"base_url": "http://localhost:11434", "timeout": 120},
    ),
)
```

For dependency injection inside a larger application, create a private
`AdapterRegistry`, register adapter instances explicitly, and call
`registry.capture(name, request)`. This avoids global state and is convenient
for tests.

### Embed the exact projection-result cache

First profile a candidate module with `layerlens-cache-benchmark`. Then wrap
only modules whose measured exact hits save more time than hashing and lookup:

```python
from pathlib import Path
from layer_profiler import MemoryExactResultCache, SQLiteExactResultCache

memory_cache = MemoryExactResultCache(linear.weight, linear.bias)
output = memory_cache(input_vector)

with SQLiteExactResultCache(
    linear.weight,
    Path("projection-results.sqlite"),
    bias=linear.bias,
    experiment_namespace="model-revision-and-adapter",
) as disk_cache:
    output = disk_cache(input_vector)
```

Inputs must be individual vectors of the module's input width. Keys include the
exact input dtype, shape, and bytes plus an exact weight-and-bias fingerprint;
outputs retain the projection's native dtype. A hit therefore replaces one
complete matrix-vector product without approximation. CPU measurements are not
portable to GPU engines, and the reference wrapper is not a fused GPU kernel.

The database is created with owner-only permissions, but it still contains
derived hidden-state outputs. Keep it local, assign a model-specific namespace,
and do not commit it. See [CACHE_BENCHMARK.md](CACHE_BENCHMARK.md) for the raw
with/without-cache methodology and admission result.

## 4. Add another runtime

An adapter needs a lowercase `name` and a `capture()` method:

```python
from pathlib import Path

from layer_profiler import CaptureRequest


class MyRuntimeAdapter:
    name = "my-runtime"
    capabilities = {
        "layer_timing": "unavailable",
        "kv_cache_usage": "sampled",
        "scheduler_state": "sampled",
        "kernel_timing": "unavailable",
    }

    def capture(self, request: CaptureRequest) -> Path:
        # Call the runtime, normalize its response, build an InferenceTrace,
        # write it to request.output_path, and return that Path.
        ...
```

See [`examples/custom_adapter.py`](examples/custom_adapter.py) for a complete
privacy-safe HTTP adapter template. It demonstrates request execution,
aggregate metric normalization, trace construction, and optional content
recording.

### Register inside one process

```python
from layer_profiler import adapter_registry
from my_package import MyRuntimeAdapter

adapter_registry.register(MyRuntimeAdapter())
adapter_registry.capture("my-runtime", request)
```

### Make an adapter discoverable after installation

Add a Python entry point to the adapter package's `pyproject.toml`:

```toml
[project.entry-points."layerlens.adapters"]
my-runtime = "my_package.adapter:MyRuntimeAdapter"
```

After installing that package in the same environment, no LayerLens source
changes are needed:

```bash
poetry run layerlens-capture --list-runtimes
poetry run layerlens-capture \
  --runtime my-runtime \
  --model provider-model-name \
  --prompt "Benchmark prompt" \
  --output traces/my-runtime.json
```

Entry points may reference an adapter instance, an adapter class with a
zero-argument constructor, or a zero-argument factory returning an adapter.

### Engine support model

LayerLens is engine-agnostic at the trace and visualization layers, not at the
collection boundary. Scheduler, KV-cache, and kernel implementations differ
between vLLM, SGLang, TensorRT-LLM, llama.cpp, and other runtimes, so each
adapter owns its version-specific collection code. It translates native data
into the same categories:

- `LayerEvent` for exact model-boundary execution;
- `RuntimeEvent(category="kv_cache", ...)` for cache state;
- `RuntimeEvent(category="scheduler", ...)` for running/waiting/batch state;
- `RuntimeEvent(category="kernel", ...)` when a profiler can correlate kernels;
- capability declarations when a measurement is exact, sampled, estimated, or
  unavailable.

`request_ref` and `batch_ref` are optional trace-local aliases for correlation.
Adapters must generate neutral values such as `request-0` and `batch-3`; never
copy provider request IDs, user identifiers, prompts, or scheduler labels into
these fields.

The dashboards consume only these normalized records. Adding deeper support
for one engine therefore changes its adapter rather than requiring a dashboard
or trace-backend rewrite.

## 5. Adapter requirements

- Return the `Path` written to `request.output_path`.
- Emit schema-compatible data with `InferenceTrace`, `LayerEvent`,
  `StepEvent`, `TokenEvent`, and `RuntimeEvent` rather than inventing a separate
  JSON format.
- Use `safe_model_identifier()` before storing a model name that may be a local
  path.
- Do not store endpoint URLs, headers, credentials, raw provider responses, or
  provider labels that have not been allowlisted.
- Store prompts, generated text, or token IDs only when
  `request.include_content` is true.
- Set `metrics.available`, `metrics.source`, and explicit measurement names so
  dashboards never confuse measured KV occupancy with an estimate.
- Declare collector capabilities as `exact`, `sampled`, `estimated`, or
  `unavailable`. Emit scheduler and KV data through engine-neutral
  `RuntimeEvent` records; keep version-specific collection inside the adapter.
- Leave `events` empty when a runtime exposes only aggregate metrics. Never
  fabricate layer timing.

Before publishing traces, run:

```bash
poetry run pytest -q
poetry run python scripts/privacy_audit.py
```

## 6. Visualize the result

Open the Streamlit dashboard and upload the generated JSON trace:

```bash
./scripts/run_dashboard.sh
```

To publish selected sanitized captures in the static showcase, add their paths
to `DEFAULT_INPUTS` in `scripts/build_showcase.py`, rebuild `docs/runs.json`,
run the privacy audit, and then deploy `docs/`.
