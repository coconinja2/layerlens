import pytest
import torch

from layer_profiler.product_atlas import (
    CoordinateProductCache,
    ProductAtlas,
    analyze_weight_family,
    benchmark_product_atlas,
    exact_coordinate_hit_rate,
)


def sibling_weights():
    return [
        torch.tensor(
            [[1.0, 10.0], [1.0, 20.0], [2.0, 10.0]], dtype=torch.bfloat16
        ),
        torch.tensor([[1.0, 30.0], [3.0, 10.0]], dtype=torch.bfloat16),
    ]


def test_analysis_counts_exact_products_shared_across_siblings():
    report = analyze_weight_family(sibling_weights())

    assert report.conventional_multiplications == 10
    assert report.atlas_multiplications == 6
    assert report.multiplications_saved == 4
    assert report.multiplication_savings_ratio == pytest.approx(0.4)
    assert report.quantization_added is False
    assert report.exact_weight_values is True


def test_reference_atlas_matches_dense_float32_projection():
    weights = sibling_weights()
    activation = torch.tensor([0.25, -2.0], dtype=torch.bfloat16)
    atlas = ProductAtlas(weights)

    actual = atlas(activation)
    expected = [torch.mv(weight.float(), activation.float()) for weight in weights]

    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_reference_benchmark_reports_numerical_fidelity():
    _, report = benchmark_product_atlas(sibling_weights(), iterations=2)

    assert report.max_absolute_error <= 1e-5
    assert report.relative_l2_error <= 1e-6
    assert report.cosine_similarity == pytest.approx(1.0, abs=1e-6)


def test_coordinate_cache_preserves_products_only_for_bit_exact_values():
    atlas = ProductAtlas(sibling_weights())
    cache = CoordinateProductCache(atlas, entries_per_coordinate=2)

    first = torch.tensor([0.25, -2.0], dtype=torch.bfloat16)
    second = torch.tensor([0.25, 4.0], dtype=torch.bfloat16)
    cache(first)
    actual = cache(second)
    expected = [torch.mv(weight.float(), second.float()) for weight in sibling_weights()]

    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert cache.stats.lookups == 4
    assert cache.stats.hits == 1
    assert cache.stats.multiplications_performed == 9
    assert cache.stats.multiplications_avoided == 3
    assert cache.stats.resident_bytes == 36


def test_coordinate_reuse_requires_bit_identical_values_at_same_coordinate():
    activations = torch.tensor(
        [[1.0, 2.0], [1.0, 3.0], [4.0, 3.0]], dtype=torch.bfloat16
    )

    report = exact_coordinate_hit_rate(activations)

    assert report == {
        "samples": 3,
        "input_features": 2,
        "lookups": 6,
        "hits": 2,
        "hit_rate": pytest.approx(1 / 3),
    }


def test_weight_family_rejects_different_input_widths():
    with pytest.raises(ValueError, match="same input width"):
        analyze_weight_family([torch.ones(2, 3), torch.ones(2, 4)])


def test_dictionary_keys_signed_zero_by_stored_bits():
    weight = torch.tensor([[-0.0], [0.0]], dtype=torch.bfloat16)

    report = analyze_weight_family([weight])

    assert report.atlas_multiplications == 2
