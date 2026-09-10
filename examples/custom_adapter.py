"""Template for a privacy-safe third-party LayerLens runtime adapter."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from layer_profiler import CaptureRequest
from layer_profiler.privacy import safe_model_identifier
from layer_profiler.trace import InferenceTrace, TokenEvent, environment_metadata


class ExampleHTTPAdapter:
    """Adapt an example JSON generation endpoint to the LayerLens schema."""

    name = "example-http"

    def capture(self, request: CaptureRequest) -> Path:
        base_url = str(request.options.get("base_url", "http://localhost:9000"))
        timeout = float(request.options.get("timeout", 120))
        api_key = os.environ.get("EXAMPLE_RUNTIME_API_KEY")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = json.dumps(
            {
                "model": request.model,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
            }
        ).encode("utf-8")
        http_request = urllib.request.Request(
            f"{base_url.rstrip('/')}/generate",
            data=body,
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            payload: dict[str, Any] = json.load(response)
        client_ms = round((time.perf_counter() - started) * 1000, 3)

        prompt_tokens = int(payload.get("prompt_tokens", 0))
        generated_tokens = int(payload.get("generated_tokens", 0))
        total_ms = float(payload.get("total_ms", client_ms))
        metadata = environment_metadata()
        metadata.update(
            {
                "source": self.name,
                "model": safe_model_identifier(request.model),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "content_recorded": request.include_content,
                "step_timing_scope": "runtime_aggregate",
            }
        )
        if request.include_content:
            metadata["prompt"] = request.prompt
            metadata["generated_text"] = payload.get("text", "")

        metrics = {
            "available": True,
            "source": self.name,
            "counters": {
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generated_tokens,
            },
            "latency_ms": {"total": total_ms},
            "cache": {"kv_cache_usage_available": False},
        }
        tokens = [
            TokenEvent(index, index, None, None, None)
            for index in range(generated_tokens)
        ]
        return InferenceTrace(
            metadata=metadata,
            events=[],
            tokens=tokens,
            metrics=metrics,
        ).write(request.output_path)


adapter = ExampleHTTPAdapter()
