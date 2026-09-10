from layer_profiler.analysis import aggregate_event_grid, choose_group_size, diagnose_trace


def make_events(layers: int, steps: int):
    return [
        {
            "layer": layer,
            "module": f"model.layers.{layer}",
            "step": step,
            "phase": "prefill" if step == 0 else "decode",
            "durationMs": 50.0 if (layer, step) == (17, 42) else 2.0,
        }
        for layer in range(layers)
        for step in range(steps)
    ]


def test_large_grid_is_bounded_but_preserves_distribution_statistics():
    events = make_events(96, 257)
    overview = aggregate_event_grid(events, max_layer_groups=24, max_token_groups=32)

    assert overview["layer_group_size"] == 4
    assert overview["token_group_size"] == 8
    assert overview["aggregated"] is True
    assert overview["raw_event_count"] == 96 * 257
    assert overview["cell_count"] <= 24 * 33
    hotspot = max(overview["cells"], key=lambda cell: cell["max_ms"])
    assert hotspot["max_ms"] == 50.0
    assert hotspot["mean_ms"] > 2.0
    assert hotspot["p95_ms"] == 2.0


def test_small_grid_stays_exact_and_prefill_is_not_mixed_with_decode():
    overview = aggregate_event_grid(make_events(12, 4))

    assert overview["layer_group_size"] == 1
    assert overview["token_group_size"] == 1
    assert overview["aggregated"] is False
    assert {cell["token_label"] for cell in overview["cells"]} == {
        "P0 · Prefill",
        "D1 · Decode",
        "D2 · Decode",
        "D3 · Decode",
    }


def test_diagnostics_turn_hotspots_and_runtime_pressure_into_actions():
    insights = diagnose_trace(
        make_events(24, 43),
        [
            {"phase": "prefill", "durationMs": 100},
            {"phase": "decode", "durationMs": 20},
            {"phase": "decode", "durationMs": 25},
        ],
        {
            "cache": {"peak_usage": 0.91},
            "scheduler": {"peak_waiting": 2, "preemptions": 1},
        },
    )

    assert any("layer 17" in insight["title"].lower() for insight in insights)
    assert any(insight["kind"] == "phase" for insight in insights)
    assert any(insight["kind"] == "cache" for insight in insights)
    assert all(insight["action"] for insight in insights)


def test_group_size_uses_powers_of_two():
    assert choose_group_size(120, 48) == 4
