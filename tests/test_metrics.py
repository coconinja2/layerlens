from layer_profiler.metrics import (
    parse_prometheus,
    runtime_events_from_metrics,
    summarize_vllm_metrics,
)


BEFORE = """
# HELP vllm:kv_cache_usage_perc KV cache usage
vllm:kv_cache_usage_perc{engine="0",model_name="demo/model"} 0.0
vllm:prompt_tokens_total{engine="0",model_name="demo/model"} 100
vllm:generation_tokens_total{engine="0",model_name="demo/model"} 50
vllm:prefix_cache_queries_total{engine="0",model_name="demo/model"} 20
vllm:prefix_cache_hits_total{engine="0",model_name="demo/model"} 5
vllm:prompt_tokens_cached_total{engine="0",model_name="demo/model"} 5
vllm:time_to_first_token_seconds_count{engine="0",model_name="demo/model"} 2
vllm:time_to_first_token_seconds_sum{engine="0",model_name="demo/model"} 0.8
"""

AFTER = """
vllm:kv_cache_usage_perc{engine="0",model_name="demo/model"} 0.02
vllm:prompt_tokens_total{engine="0",model_name="demo/model"} 110
vllm:generation_tokens_total{engine="0",model_name="demo/model"} 54
vllm:prefix_cache_queries_total{engine="0",model_name="demo/model"} 30
vllm:prefix_cache_hits_total{engine="0",model_name="demo/model"} 13
vllm:prompt_tokens_cached_total{engine="0",model_name="demo/model"} 13
vllm:time_to_first_token_seconds_count{engine="0",model_name="demo/model"} 3
vllm:time_to_first_token_seconds_sum{engine="0",model_name="demo/model"} 1.1
vllm:cache_config_info{engine="0",block_size="16",cache_dtype="auto",enable_prefix_caching="True",kv_cache_size_tokens="4096",model_name="demo/model",served_model_name="private alias"} 1
"""


def test_prometheus_metrics_are_normalized_and_labels_are_allowlisted():
    before = parse_prometheus(BEFORE)
    after = parse_prometheus(AFTER)
    metrics = summarize_vllm_metrics(
        before=before,
        after=after,
        timeline=[
            {"elapsed_ms": 0.0, "kv_cache_usage": 0.0, "requests_running": 0, "requests_waiting": 0, "resident_memory_mb": 512},
            {"elapsed_ms": 20.0, "kv_cache_usage": 0.04, "requests_running": 1, "requests_waiting": 0, "resident_memory_mb": 640},
        ],
        model="demo/model",
        sample_interval_ms=20,
    )

    assert metrics["counters"]["prompt_tokens"] == 10
    assert metrics["counters"]["generation_tokens"] == 4
    assert metrics["cache"]["prefix_hit_rate"] == 0.8
    assert metrics["cache"]["prompt_cache_hit_rate"] == 0.8
    assert metrics["cache"]["peak_usage"] == 0.04
    assert metrics["latency_ms"]["time_to_first_token_mean_ms"] == 300
    assert metrics["scheduler"]["peak_running"] == 1
    assert metrics["process"]["peak_resident_memory_mb"] == 640
    assert metrics["cache"]["config"] == {
        "block_size": "16",
        "cache_dtype": "auto",
        "enable_prefix_caching": "True",
        "kv_cache_size_tokens": "4096",
    }
    assert "served_model_name" not in metrics["cache"]["config"]
    events = runtime_events_from_metrics(metrics)
    assert {event.category for event in events} == {"kv_cache", "scheduler"}
    assert all(event.measurement == "sampled" for event in events)


def test_parser_ignores_comments_invalid_values_and_unescapes_labels():
    samples = parse_prometheus(
        '# comment\nmetric_total{reason="line\\nvalue"} 2.5\nmetric_bad NaN\ninvalid\n'
    )
    assert len(samples) == 1
    assert samples[0].name == "metric_total"
    assert samples[0].labels["reason"] == "line\nvalue"
    assert samples[0].value == 2.5
