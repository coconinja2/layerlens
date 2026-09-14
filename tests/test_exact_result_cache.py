from pathlib import Path
import stat

import torch

from layer_profiler.exact_result_cache import (
    MemoryExactResultCache,
    SQLiteExactResultCache,
    benchmark_exact_result_caches,
    weight_fingerprint,
)


def fixtures():
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    activations = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.bfloat16
    )
    return weight, activations


def test_memory_cache_requires_exact_activation_bytes():
    weight, activations = fixtures()
    cache = MemoryExactResultCache(weight)

    outputs = [cache(row) for row in activations]

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert cache.stats.lookups == 3
    assert cache.stats.hits == 1
    assert cache.stats.misses == 2


def test_memory_cache_output_cannot_be_mutated_through_a_hit():
    weight, activations = fixtures()
    cache = MemoryExactResultCache(weight)
    expected = cache(activations[0]).clone()

    mutable_hit = cache(activations[0])
    mutable_hit.zero_()

    assert torch.equal(cache(activations[0]), expected)


def test_sqlite_cache_survives_reopen_without_storing_raw_keys(tmp_path: Path):
    weight, activations = fixtures()
    path = tmp_path / "exact.sqlite"
    first = SQLiteExactResultCache(weight, path, experiment_namespace="test")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    expected = [first(row) for row in activations]
    first.close()

    reopened = SQLiteExactResultCache(weight, path, experiment_namespace="test")
    actual = [reopened(row) for row in activations]
    stats = reopened.stats
    reopened.close()

    assert stats.hits == 3
    assert stats.misses == 0
    for result, reference in zip(actual, expected, strict=True):
        assert result.dtype == reference.dtype
        assert torch.equal(result, reference)


def test_raw_benchmark_compares_empty_and_warm_cache_modes(tmp_path: Path):
    weight, activations = fixtures()

    report = benchmark_exact_result_caches(
        weight,
        activations,
        disk_path=tmp_path / "benchmark.sqlite",
        label="tiny",
        cycles=2,
    )

    assert report["bitwise_output_agreement"] is True
    assert report["memory_stats_after_empty_pass"]["hit_rate"] == 1 / 3
    assert report["disk_stats_after_empty_pass"]["hit_rate"] == 1 / 3
    assert report["disk_stats_after_reopen"]["hit_rate"] == 1.0
    assert report["disk_file_bytes"] > 0
    assert report["mathematical_work"]["dense_matvecs_avoided_on_empty_pass"] == 1
    assert report["mathematical_work"]["scalar_multiplications_avoided_on_empty_pass"] == 4
    assert "recommendation" in report["admission_decision"]


def test_weight_fingerprint_changes_with_weight_bits():
    first = torch.tensor([[1.0]], dtype=torch.bfloat16)
    second = torch.tensor([[2.0]], dtype=torch.bfloat16)

    assert weight_fingerprint(first) != weight_fingerprint(second)
