"""Bottom-up, privacy-safe repeatability measurements for inference tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
import time
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TensorRepeatProfile:
    signal: str
    dtype: str
    vectors: int
    width: int
    elements: int
    distinct_scalar_bit_patterns: int
    scalar_replay_hits: int
    scalar_replay_hit_rate: float
    coordinate_replay_hits: int
    coordinate_replay_hit_rate: float
    tile_size: int
    tile_lookups: int
    tile_replay_hits: int
    tile_replay_hit_rate: float
    vector_replay_hits: int
    vector_replay_hit_rate: float
    consecutive_unchanged_elements: int
    consecutive_unchanged_ratio: float
    consecutive_cosine_similarity_mean: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UnaryAtlasProfile:
    function: str
    input_dtype: str
    domain_entries: int
    table_bytes: int
    observed_values: int
    observed_distinct_inputs: int
    observed_replay_hit_rate: float
    table_build_ms: float
    direct_median_ms: float
    lookup_median_ms: float
    lookup_to_direct_ratio: float
    bitwise_output_agreement: float
    max_absolute_error: float
    approximation_added: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionHeadProfile:
    signal: str
    samples: int
    heads: int
    head_dim: int
    head_vectors: int
    exact_head_replay_hits: int
    exact_head_replay_hit_rate: float
    same_head_position_hits: int
    same_head_position_hit_rate: float
    within_sample_duplicate_heads: int
    consecutive_same_head_unchanged_ratio: float
    consecutive_same_head_cosine_mean: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _bit_integer_dtype(dtype: torch.dtype) -> torch.dtype:
    mapping = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"repeatability profiling requires a floating dtype, got {dtype}")
    return mapping[dtype]


def _bit_matrix(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().contiguous()
    return values.view(_bit_integer_dtype(values.dtype))


def _byte_key(values: torch.Tensor) -> bytes:
    return bytes(values.contiguous().view(torch.uint8).reshape(-1).tolist())


def _exact_replay_hits(rows: torch.Tensor) -> int:
    seen: set[bytes] = set()
    hits = 0
    for row in rows:
        key = _byte_key(row)
        if key in seen:
            hits += 1
        else:
            seen.add(key)
    return hits


def profile_tensor_repetition(
    values: torch.Tensor,
    *,
    signal: str,
    tile_size: int = 4,
) -> TensorRepeatProfile:
    """Measure exact scalar, coordinate, tile, vector, and delta repetition."""

    if values.ndim != 2:
        raise ValueError("values must have shape [vectors, width]")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("values cannot be empty")
    source = values.detach().cpu().contiguous()
    bits = _bit_matrix(source)
    vector_count, width = source.shape
    elements = source.numel()

    distinct_scalars = int(torch.unique(bits).numel())
    scalar_hits = elements - distinct_scalars
    coordinate_distinct = sum(int(torch.unique(bits[:, column]).numel()) for column in range(width))
    coordinate_hits = elements - coordinate_distinct

    tile_hits = 0
    tile_lookups = 0
    for start in range(0, width, tile_size):
        tile_hits += _exact_replay_hits(bits[:, start : start + tile_size])
        tile_lookups += vector_count

    vector_hits = _exact_replay_hits(bits)
    comparisons = max(0, vector_count - 1) * width
    unchanged = int((bits[1:] == bits[:-1]).sum()) if vector_count > 1 else 0
    cosine: float | None = None
    if vector_count > 1:
        numeric = source.to(torch.float32)
        similarities = F.cosine_similarity(numeric[1:], numeric[:-1], dim=1)
        finite = similarities[torch.isfinite(similarities)]
        if finite.numel():
            cosine = float(finite.mean())

    return TensorRepeatProfile(
        signal=signal,
        dtype=str(source.dtype).removeprefix("torch."),
        vectors=vector_count,
        width=width,
        elements=elements,
        distinct_scalar_bit_patterns=distinct_scalars,
        scalar_replay_hits=scalar_hits,
        scalar_replay_hit_rate=scalar_hits / elements,
        coordinate_replay_hits=coordinate_hits,
        coordinate_replay_hit_rate=coordinate_hits / elements,
        tile_size=tile_size,
        tile_lookups=tile_lookups,
        tile_replay_hits=tile_hits,
        tile_replay_hit_rate=tile_hits / tile_lookups,
        vector_replay_hits=vector_hits,
        vector_replay_hit_rate=vector_hits / vector_count,
        consecutive_unchanged_elements=unchanged,
        consecutive_unchanged_ratio=unchanged / comparisons if comparisons else 0.0,
        consecutive_cosine_similarity_mean=cosine,
    )


class FiniteDomainUnaryAtlas:
    """Complete lookup table for a unary function over FP16 or BF16 inputs."""

    FUNCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "sigmoid": torch.sigmoid,
        "silu": F.silu,
    }

    def __init__(self, function: str, dtype: torch.dtype):
        if function not in self.FUNCTIONS:
            raise ValueError(f"unsupported unary function: {function}")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("the complete unary atlas supports float16 and bfloat16")
        self.function = function
        self.dtype = dtype
        unsigned = torch.arange(1 << 16, dtype=torch.int32)
        signed = torch.where(unsigned < (1 << 15), unsigned, unsigned - (1 << 16))
        domain = signed.to(torch.int16).contiguous().view(dtype)
        start = time.perf_counter_ns()
        self.table = self.FUNCTIONS[function](domain).to(dtype)
        self.build_ms = (time.perf_counter_ns() - start) / 1_000_000

    @staticmethod
    def _indices(values: torch.Tensor) -> torch.Tensor:
        bits = values.detach().cpu().contiguous().view(torch.int16).to(torch.int32)
        return torch.bitwise_and(bits, (1 << 16) - 1).to(torch.int64)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        if values.dtype != self.dtype:
            raise ValueError(f"expected {self.dtype}, got {values.dtype}")
        return self.table[self._indices(values)]

    def profile(self, values: torch.Tensor, *, iterations: int = 20) -> UnaryAtlasProfile:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        source = values.detach().cpu().contiguous().reshape(-1)
        if source.dtype != self.dtype:
            raise ValueError(f"expected {self.dtype}, got {source.dtype}")
        operation = self.FUNCTIONS[self.function]
        operation(source)
        self(source)
        direct_times: list[float] = []
        lookup_times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            expected = operation(source)
            direct_times.append((time.perf_counter_ns() - start) / 1_000_000)
            start = time.perf_counter_ns()
            actual = self(source)
            lookup_times.append((time.perf_counter_ns() - start) / 1_000_000)

        expected_bits = _bit_matrix(expected)
        actual_bits = _bit_matrix(actual)
        bitwise_agreement = float((expected_bits == actual_bits).to(torch.float32).mean())
        finite = torch.isfinite(expected.float()) & torch.isfinite(actual.float())
        max_error = (
            float((expected.float()[finite] - actual.float()[finite]).abs().max())
            if finite.any()
            else 0.0
        )
        input_bits = _bit_matrix(source)
        distinct = int(torch.unique(input_bits).numel())
        direct_median = statistics.median(direct_times)
        lookup_median = statistics.median(lookup_times)
        return UnaryAtlasProfile(
            function=self.function,
            input_dtype=str(self.dtype).removeprefix("torch."),
            domain_entries=self.table.numel(),
            table_bytes=self.table.numel() * self.table.element_size(),
            observed_values=source.numel(),
            observed_distinct_inputs=distinct,
            observed_replay_hit_rate=1.0 - (distinct / source.numel()),
            table_build_ms=self.build_ms,
            direct_median_ms=direct_median,
            lookup_median_ms=lookup_median,
            lookup_to_direct_ratio=lookup_median / direct_median,
            bitwise_output_agreement=bitwise_agreement,
            max_absolute_error=max_error,
        )


def profile_attention_heads(
    values: torch.Tensor,
    *,
    signal: str,
    heads: int,
    head_dim: int,
) -> AttentionHeadProfile:
    """Measure exact and directional reuse at an attention-head boundary."""

    if values.ndim != 2 or values.shape[1] != heads * head_dim:
        raise ValueError(
            f"expected [samples, {heads * head_dim}] for {heads} × {head_dim} heads"
        )
    source = values.detach().cpu().contiguous().reshape(values.shape[0], heads, head_dim)
    bits = _bit_matrix(source)
    flattened_heads = bits.reshape(-1, head_dim)
    global_hits = _exact_replay_hits(flattened_heads)

    same_position_hits = 0
    for head in range(heads):
        same_position_hits += _exact_replay_hits(bits[:, head, :])

    within_sample_hits = 0
    for sample in bits:
        within_sample_hits += _exact_replay_hits(sample)

    comparisons = max(0, source.shape[0] - 1) * heads * head_dim
    unchanged = int((bits[1:] == bits[:-1]).sum()) if source.shape[0] > 1 else 0
    cosine: float | None = None
    if source.shape[0] > 1:
        numeric = source.to(torch.float32)
        similarities = F.cosine_similarity(numeric[1:], numeric[:-1], dim=2)
        finite = similarities[torch.isfinite(similarities)]
        if finite.numel():
            cosine = float(finite.mean())

    head_vectors = source.shape[0] * heads
    same_position_lookups = head_vectors
    return AttentionHeadProfile(
        signal=signal,
        samples=source.shape[0],
        heads=heads,
        head_dim=head_dim,
        head_vectors=head_vectors,
        exact_head_replay_hits=global_hits,
        exact_head_replay_hit_rate=global_hits / head_vectors,
        same_head_position_hits=same_position_hits,
        same_head_position_hit_rate=same_position_hits / same_position_lookups,
        within_sample_duplicate_heads=within_sample_hits,
        consecutive_same_head_unchanged_ratio=(
            unchanged / comparisons if comparisons else 0.0
        ),
        consecutive_same_head_cosine_mean=cosine,
    )
