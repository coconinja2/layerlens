import json

from layer_profiler.trace import InferenceTrace, LayerEvent, StepEvent, TokenEvent


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
    )
    output = trace.write(tmp_path / "trace.json")
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == "1.1"
    assert payload["events"][0]["duration_ms"] == 2.5
    assert payload["events"][0]["memory_delta_bytes"] == 40
    assert payload["steps"][0]["duration_ms"] == 3.0
    assert payload["tokens"][0]["emitted_ms"] == 3.5
    assert payload["metrics"] == {}
