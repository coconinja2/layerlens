import torch

from layer_profiler.precomputed_multiply import (
    benchmark_precomputed_products,
    benchmark_precomputed_projection,
    decode_bf16,
    full_mantissa_table,
    mantissa_table_multiply,
    nibble_product_table,
    nibble_table_multiply,
)


def test_tables_have_expected_small_sizes():
    assert full_mantissa_table().numel() * full_mantissa_table().element_size() == 131072
    assert nibble_product_table().numel() * nibble_product_table().element_size() == 256


def test_table_products_match_native_for_finite_bf16_values():
    bits = torch.arange(0, 0x7F80, 37, dtype=torch.int32).to(torch.int16)
    left = bits.view(torch.bfloat16)
    right = bits.flip(0).contiguous().view(torch.bfloat16)
    expected = left.float() * right.float()
    left_parts = decode_bf16(left)
    right_parts = decode_bf16(right)

    mantissa = mantissa_table_multiply(left_parts, right_parts, full_mantissa_table())
    nibble = nibble_table_multiply(left_parts, right_parts, nibble_product_table())

    assert torch.equal(expected.view(torch.int32), mantissa.view(torch.int32))
    assert torch.equal(expected.view(torch.int32), nibble.view(torch.int32))


def test_benchmarks_report_exact_products_and_projection_costs():
    left = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.bfloat16)
    right = torch.tensor([4.0, 0.25, -8.0, 2.0], dtype=torch.bfloat16)
    products = benchmark_precomputed_products(left, right, iterations=1)
    assert products["runs"][0]["mantissa_products_bitwise_equal"] is True
    assert products["runs"][0]["nibble_products_bitwise_equal"] is True

    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.bfloat16)
    projection = benchmark_precomputed_projection(
        weight, left, output_rows=4, iterations=1
    )
    assert projection["scalar_products"] == 16
    assert projection["runs"][0]["maximum_absolute_difference"] >= 0
