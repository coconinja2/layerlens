import json

import pytest

from layer_profiler.trace import InferenceTrace, LayerEvent, RuntimeEvent, StepEvent, TokenEvent


def test_trace_serializes_derived_units(tmp_path):
    event = LayerEvent(
        sequence=0,
        step=0,
        phase="prefill",
        token_index=0,
        module="model.layers.0",
        module_type="DecoderLayer",
        depth=3,
        started_ns=1_000_000,
        duration_ns=2_500_000,
        memory_before_bytes=100,
        memory_after_bytes=140,
    )
    trace = InferenceTrace(
        metadata={"model": "tiny"},
        events=[event],
        steps=[StepEvent(0, "prefill", 0, 500_000, 3_000_000, 8)],
        tokens=[TokenEvent(0, 0, 42, "x", 3_500_000)],
        runtime_events=[RuntimeEvent(0, 2_000_000, "scheduler", "running", 1, "requests")],
    )
    output = trace.write(tmp_path / "trace.json")
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == "1.2"
    assert payload["events"][0]["duration_ms"] == 2.5
    assert payload["events"][0]["memory_delta_bytes"] == 40
    assert payload["steps"][0]["duration_ms"] == 3.0
    assert payload["tokens"][0]["emitted_ms"] == 3.5
    assert payload["metrics"] == {}
    assert payload["runtime_events"][0]["elapsed_ms"] == 2.0
    assert payload["runtime_events"][0]["category"] == "scheduler"


def test_runtime_event_rejects_provider_identifiers_and_freeform_strings():
    with pytest.raises(ValueError, match="trace-local"):
        RuntimeEvent(
            0,
            0,
            "scheduler",
            "running",
            1,
            "requests",
            request_ref="provider-request-id",
        )
    with pytest.raises(ValueError, match="numeric or boolean"):
        RuntimeEvent(
            0,
            0,
            "scheduler",
            "batch_state",
            1,
            "requests",
            attributes={"raw_label": "private-value"},
        )
