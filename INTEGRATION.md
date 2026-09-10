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
```

Or install the standalone commands directly from GitHub:

```bash
python3 -m pip install "git+https://github.com/coconinja2/layerlens.git"
layerlens-capture --list-runtimes
```

The built-in adapter names are `ollama` and `vllm`. The specialized
`layerlens`, `layerlens-ollama`, and `layerlens-vllm` commands remain available
when runtime-specific flags are more convenient.

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

## 4. Add another runtime

An adapter needs a lowercase `name` and a `capture()` method:

```python
from pathlib import Path

from layer_profiler import CaptureRequest


class MyRuntimeAdapter:
    name = "my-runtime"

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

## 5. Adapter requirements

- Return the `Path` written to `request.output_path`.
- Emit schema-compatible data with `InferenceTrace`, `LayerEvent`,
  `StepEvent`, and `TokenEvent` rather than inventing a separate JSON format.
- Use `safe_model_identifier()` before storing a model name that may be a local
  path.
- Do not store endpoint URLs, headers, credentials, raw provider responses, or
  provider labels that have not been allowlisted.
- Store prompts, generated text, or token IDs only when
  `request.include_content` is true.
- Set `metrics.available`, `metrics.source`, and explicit measurement names so
  dashboards never confuse measured KV occupancy with an estimate.
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
