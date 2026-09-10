from pathlib import Path

import pytest

from layer_profiler.adapters import AdapterRegistry, CaptureRequest, parse_option


class DemoAdapter:
    name = "demo"

    def __init__(self):
        self.request = None

    def capture(self, request: CaptureRequest) -> Path:
        self.request = request
        return request.output_path


def test_registry_dispatches_a_runtime_neutral_capture_request(tmp_path):
    registry = AdapterRegistry()
    adapter = registry.register(DemoAdapter())
    request = CaptureRequest(
        model="org/model",
        prompt="safe test prompt",
        max_tokens=3,
        output_path=tmp_path / "trace.json",
        options={"region": "test"},
    )

    assert registry.names() == ("demo",)
    assert registry.capture("DEMO", request) == tmp_path / "trace.json"
    assert adapter.request is request


def test_registry_rejects_invalid_and_duplicate_adapters():
    registry = AdapterRegistry()
    registry.register(DemoAdapter())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DemoAdapter())
    with pytest.raises(ValueError, match="adapter name"):
        registry.register(DemoAdapter(), name="Not Valid")
    with pytest.raises(TypeError, match="capture"):
        registry.register(object(), name="broken")


def test_capture_request_and_option_validation(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        CaptureRequest("model", "prompt", 0, tmp_path / "trace.json")

    assert parse_option("timeout=30") == ("timeout", 30)
    assert parse_option("layer_events=true") == ("layer_events", True)
    assert parse_option("base_url=http://localhost:8000") == (
        "base_url",
        "http://localhost:8000",
    )
    with pytest.raises(ValueError, match="key=value"):
        parse_option("missing-value")
    with pytest.raises(ValueError, match="environment variable"):
        parse_option("api_key=do-not-put-secrets-here")
