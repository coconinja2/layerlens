"""Exact BF16 product reconstruction from small precomputed mantissa tables."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Callable

import torch


@dataclass(frozen=True)
class BF16Parts:
    sign: torch.Tensor
    exponent: torch.Tensor
    mantissa: torch.Tensor
    special: torch.Tensor


def bf16_bits(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.bfloat16:
        raise ValueError("values must use bfloat16")
    return values.detach().cpu().contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def decode_bf16(values: torch.Tensor) -> BF16Parts:
    bits = bf16_bits(values)
    stored_exponent = (bits >> 7) & 0xFF
    fraction = bits & 0x7F
    normal = stored_exponent != 0
    return BF16Parts(
        sign=(bits >> 15) & 1,
        exponent=torch.where(normal, stored_exponent, torch.ones_like(stored_exponent)),
        mantissa=torch.where(normal, fraction + 128, fraction),
        special=stored_exponent == 0xFF,
    )


def full_mantissa_table() -> torch.Tensor:
    """All 256 x 256 unsigned significand products in 128 KiB."""

    values = torch.arange(256, dtype=torch.int32)
    return (values[:, None] * values[None, :]).to(torch.uint16).contiguous()


def nibble_product_table() -> torch.Tensor:
    """All 16 x 16 four-bit products in 256 bytes."""

    values = torch.arange(16, dtype=torch.int16)
    return (values[:, None] * values[None, :]).to(torch.uint8).contiguous()


def _assemble_products(
    left: BF16Parts,
    right: BF16Parts,
    unsigned_products: torch.Tensor,
) -> torch.Tensor:
    power = left.exponent + right.exponent - 268
    result = torch.ldexp(unsigned_products.to(torch.float32), power)
    return torch.where((left.sign ^ right.sign).bool(), -result, result)


def mantissa_table_multiply(
    left: BF16Parts,
    right: BF16Parts,
    table: torch.Tensor,
) -> torch.Tensor:
    indices = left.mantissa.to(torch.int64) * 256 + right.mantissa.to(torch.int64)
    unsigned = table.reshape(-1)[indices].to(torch.int32)
    return _assemble_products(left, right, unsigned)


def nibble_table_multiply(
    left: BF16Parts,
    right: BF16Parts,
    table: torch.Tensor,
) -> torch.Tensor:
    left_low = left.mantissa & 0xF
    left_high = left.mantissa >> 4
    right_low = right.mantissa & 0xF
    right_high = right.mantissa >> 4
    flat = table.reshape(-1)

    def lookup(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return flat[(first * 16 + second).to(torch.int64)].to(torch.int32)

    unsigned = (
        lookup(left_low, right_low)
        + ((lookup(left_low, right_high) + lookup(left_high, right_low)) << 4)
        + (lookup(left_high, right_high) << 8)
    )
    return _assemble_products(left, right, unsigned)


def _median_ms(operation: Callable[[], torch.Tensor], iterations: int) -> tuple[float, torch.Tensor]:
    for _ in range(3):
        output = operation()
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        output = operation()
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(timings), output


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.contiguous().view(torch.uint8).numpy().tobytes()
        == right.contiguous().view(torch.uint8).numpy().tobytes()
    )


def benchmark_precomputed_products(
    left_values: torch.Tensor,
    right_values: torch.Tensor,
    *,
    iterations: int = 10,
    thread_counts: tuple[int, ...] = (1,),
) -> dict[str, object]:
    if left_values.shape != right_values.shape or left_values.dtype != torch.bfloat16:
        raise ValueError("left and right must be equally shaped BF16 tensors")
    if right_values.dtype != torch.bfloat16 or iterations < 1:
        raise ValueError("right must be BF16 and iterations must be positive")
    left = left_values.detach().cpu().contiguous()
    right = right_values.detach().cpu().contiguous()
    left_parts = decode_bf16(left)
    right_parts = decode_bf16(right)
    if bool((left_parts.special | right_parts.special).any()):
        raise ValueError("benchmark inputs must be finite; special values use a native slow path")
    mantissa = full_mantissa_table()
    nibble = nibble_product_table()
    direct_indices = (
        left_parts.mantissa.to(torch.int64) * 256
        + right_parts.mantissa.to(torch.int64)
    )
    flat_mantissa = mantissa.reshape(-1)
    original_threads = torch.get_num_threads()
    results: list[dict[str, object]] = []
    try:
        for threads in dict.fromkeys(thread_counts):
            torch.set_num_threads(max(1, threads))
            native_ms, native = _median_ms(lambda: left.float() * right.float(), iterations)
            direct_lookup_ms, _ = _median_ms(
                lambda: flat_mantissa[direct_indices], iterations
            )
            mantissa_ms, mantissa_output = _median_ms(
                lambda: mantissa_table_multiply(left_parts, right_parts, mantissa), iterations
            )
            nibble_ms, nibble_output = _median_ms(
                lambda: nibble_table_multiply(left_parts, right_parts, nibble), iterations
            )
            results.append(
                {
                    "threads": max(1, threads),
                    "native_median_ms": native_ms,
                    "mantissa_direct_lookup_only_median_ms": direct_lookup_ms,
                    "mantissa_direct_lookup_only_vs_native": direct_lookup_ms / native_ms,
                    "mantissa_table_median_ms": mantissa_ms,
                    "mantissa_table_vs_native": mantissa_ms / native_ms,
                    "nibble_table_median_ms": nibble_ms,
                    "nibble_table_vs_native": nibble_ms / native_ms,
                    "mantissa_products_bitwise_equal": _bitwise_equal(native, mantissa_output),
                    "nibble_products_bitwise_equal": _bitwise_equal(native, nibble_output),
                }
            )
    finally:
        torch.set_num_threads(original_threads)
    return {
        "pairs": left.numel(),
        "iterations": iterations,
        "mantissa_table_bytes": mantissa.numel() * mantissa.element_size(),
        "nibble_table_bytes": nibble.numel() * nibble.element_size(),
        "special_operand_pairs": int((left_parts.special | right_parts.special).sum()),
        "runs": results,
    }


def benchmark_precomputed_projection(
    weight: torch.Tensor,
    activation: torch.Tensor,
    *,
    output_rows: int = 256,
    iterations: int = 10,
    thread_counts: tuple[int, ...] = (1,),
) -> dict[str, object]:
    if weight.dtype != torch.bfloat16 or activation.dtype != torch.bfloat16:
        raise ValueError("weight and activation must use BF16")
    if weight.ndim != 2 or activation.ndim != 1 or activation.numel() != weight.shape[1]:
        raise ValueError("expected weight [out, in] and activation [in]")
    selected = weight.detach().cpu()[:output_rows].contiguous()
    vector = activation.detach().cpu().contiguous()
    weight_parts = decode_bf16(selected)
    activation_parts = decode_bf16(vector)
    if bool((weight_parts.special | activation_parts.special.unsqueeze(0)).any()):
        raise ValueError("projection contains special BF16 values")
    mantissa = full_mantissa_table()

    def table_projection() -> torch.Tensor:
        broadcast_activation = BF16Parts(
            sign=activation_parts.sign.unsqueeze(0),
            exponent=activation_parts.exponent.unsqueeze(0),
            mantissa=activation_parts.mantissa.unsqueeze(0),
            special=activation_parts.special.unsqueeze(0),
        )
        products = mantissa_table_multiply(weight_parts, broadcast_activation, mantissa)
        return products.sum(dim=1)

    original_threads = torch.get_num_threads()
    results: list[dict[str, object]] = []
    try:
        for threads in dict.fromkeys(thread_counts):
            torch.set_num_threads(max(1, threads))
            native_bf16_ms, native_bf16 = _median_ms(
                lambda: torch.mv(selected, vector), iterations
            )
            native_fp32_ms, native_fp32 = _median_ms(
                lambda: torch.mv(selected.float(), vector.float()), iterations
            )
            table_ms, table_output = _median_ms(table_projection, iterations)
            difference = (native_fp32 - table_output).abs()
            results.append(
                {
                    "threads": max(1, threads),
                    "native_bf16_projection_median_ms": native_bf16_ms,
                    "native_fp32_projection_median_ms": native_fp32_ms,
                    "mantissa_table_projection_median_ms": table_ms,
                    "mantissa_table_vs_native_bf16": table_ms / native_bf16_ms,
                    "fp32_projection_bitwise_equal": _bitwise_equal(
                        native_fp32, table_output
                    ),
                    "native_bf16_projection_bitwise_equal": _bitwise_equal(
                        native_bf16, table_output.to(torch.bfloat16)
                    ),
                    "maximum_absolute_difference": float(difference.max()),
                    "mean_absolute_difference": float(difference.mean()),
                }
            )
    finally:
        torch.set_num_threads(original_threads)
    return {
        "weight_shape": list(selected.shape),
        "scalar_products": selected.numel(),
        "iterations": iterations,
        "mantissa_table_bytes": mantissa.numel() * mantissa.element_size(),
        "runs": results,
    }
