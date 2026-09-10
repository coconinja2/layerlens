from layer_profiler.ollama import normalize_ollama_metrics


def test_ollama_response_is_normalized_without_raw_content_or_identifiers():
    metrics = normalize_ollama_metrics(
        response_payload={
            "response": "private output",
            "thinking": "private chain",
            "total_duration": 2_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500_000_000,
            "eval_count": 5,
            "eval_duration": 1_250_000_000,
        },
        show_payload={
            "template": "private template",
            "license": "large license",
            "capabilities": ["thinking", "completion"],
            "details": {
                "family": "demo",
                "format": "gguf",
                "parameter_size": "1B",
                "quantization_level": "Q4_K_M",
            },
            "model_info": {
                "general.architecture": "demo",
                "demo.block_count": 16,
                "demo.context_length": 4096,
                "tokenizer.ggml.tokens": ["private", "tokens"],
            },
        },
        ps_payload={
            "models": [
                {
                    "model": "demo:latest",
                    "digest": "private digest",
                    "size": 1024 * 1024 * 100,
                    "size_vram": 1024 * 1024 * 80,
                    "context_length": 4096,
                }
            ]
        },
        model="demo:latest",
    )

    assert metrics["latency_ms"]["total"] == 2000
    assert metrics["latency_ms"]["mean_output_token"] == 250
    assert metrics["throughput"]["generation_tokens_per_second"] == 4
    assert metrics["cache"]["estimated_context_utilization"] == round(15 / 4096, 6)
    assert metrics["cache"]["kv_cache_usage_available"] is False
    assert metrics["runtime"]["processor_memory_mb"] == 80
    assert metrics["runtime"]["architecture"] == {
        "block_count": 16,
        "context_length": 4096,
    }
    serialized = repr(metrics)
    assert "private output" not in serialized
    assert "private chain" not in serialized
    assert "private template" not in serialized
    assert "private digest" not in serialized
