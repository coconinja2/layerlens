import pytest

from layer_profiler.cache_plan import (
    PrefixCacheSimulator,
    PrefixRequest,
    compare_prefix_schedules,
)


def _workload():
    prefix_a = tuple(range(12))
    prefix_b = tuple(range(100, 112))
    return [
        PrefixRequest("request-0", prefix_a + (20, 21, 22, 23)),
        PrefixRequest("request-1", prefix_b + (120, 121, 122, 123)),
        PrefixRequest("request-2", prefix_a + (24, 25, 26, 27)),
        PrefixRequest("request-3", prefix_b + (124, 125, 126, 127)),
        PrefixRequest("request-4", prefix_a + (28, 29, 30, 31)),
        PrefixRequest("request-5", prefix_b + (128, 129, 130, 131)),
    ]


def test_prefix_aware_order_reduces_exact_layer_token_operations():
    report = compare_prefix_schedules(
        _workload(),
        block_size=4,
        capacity_blocks=4,
        num_layers=24,
        hidden_size=1024,
        intermediate_size=3584,
        mlp_projections=3,
    )

    assert report["fcfs"]["prefix_hit_tokens"] == 0
    assert report["prefix_aware"]["prefix_hit_tokens"] == 48
    assert report["prefix_aware"]["operation_reduction"] == 0.5
    assert report["additional_operations_avoided"] == 48 * 24
    assert report["prefix_aware"]["avoided_fixed_matmul_flops"] == 35_030_827_008
    assert report["cost_model"]["fixed_flops_per_layer_token"] == 30_408_704
    assert report["prefix_aware"]["order"] == [
        "request-0",
        "request-2",
        "request-4",
        "request-1",
        "request-3",
        "request-5",
    ]
    assert report["changes_model_output"] is False


def test_only_complete_contiguous_prefix_blocks_hit():
    simulator = PrefixCacheSimulator(block_size=4, capacity_blocks=8)
    first = PrefixRequest("request-0", (1, 2, 3, 4, 5, 6))
    second = PrefixRequest("request-1", (1, 2, 3, 4, 9, 10))

    assert simulator.process(first)["hit_tokens"] == 0
    assert simulator.process(second)["hit_tokens"] == 4


def test_cache_inputs_are_bounded_and_privacy_safe():
    with pytest.raises(ValueError, match="trace-local"):
        PrefixRequest("customer@example.com", (1, 2, 3))
    with pytest.raises(ValueError, match="non-negative"):
        PrefixRequest("request-0", (1, -1))
    with pytest.raises(ValueError, match="block_size"):
        PrefixCacheSimulator(block_size=0, capacity_blocks=1)
    with pytest.raises(ValueError, match="capacity_blocks"):
        PrefixCacheSimulator(block_size=4, capacity_blocks=0)
    with pytest.raises(ValueError, match="provided together"):
        compare_prefix_schedules(
            _workload(),
            block_size=4,
            capacity_blocks=4,
            num_layers=24,
            hidden_size=1024,
        )
