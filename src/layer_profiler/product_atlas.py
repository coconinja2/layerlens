"""Exact shared-product analysis for families of dense projections.

The Product Atlas is an experimental matmul representation.  When several
weight matrices consume the same activation vector, it finds bit-identical
stored weight values at each input coordinate and computes ``x[j] * value``
once for every distinct value.  Index maps route those products back into the
ordinary output dot products.

No activation or weight is quantized by this module.  The reference executor
uses float32 products and accumulation so it is useful for validating the
algebra and measuring indexing overhead; it is not a production kernel.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
import statistics
import time
from typing import Sequence

import torch


@dataclass(frozen=True)
class ProductAtlasAnalysis:
    """Hardware-neutral operation and storage counts for a weight family."""

    dtype: str
    matrix_shapes: list[list[int]]
    input_features: int
    output_features: int
    dictionary_values: int
    dictionary_values_per_column_median: float
    dictionary_values_per_column_max: int
    conventional_multiplications: int
    atlas_multiplications: int
    multiplications_saved: int
    multiplication_savings_ratio: float
    original_weight_bytes: int
    theoretical_packed_atlas_bytes: int
    theoretical_storage_ratio: float
    exact_weight_values: bool = True
    quantization_added: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProductAtlasBenchmark:
    """Reference-executor fidelity and latency measurements."""

    iterations: int
    baseline_median_ms: float
    atlas_median_ms: float
    atlas_to_baseline_ratio: float
    max_absolute_error: float
    mean_absolute_error: float
    relative_l2_error: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CoordinateProductCacheStats:
    """Counters for bit-exact product preservation across activation vectors."""

    lookups: int
    hits: int
    misses: int
    multiplications_performed: int
    multiplications_avoided: int
    resident_bytes: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "hit_rate": self.hit_rate}


def _validate_weights(weights: Sequence[torch.Tensor]) -> tuple[int, torch.dtype]:
    if not weights:
        raise ValueError("at least one weight matrix is required")
    if any(weight.ndim != 2 for weight in weights):
        raise ValueError("every weight must be a two-dimensional [out, in] matrix")
    input_features = weights[0].shape[1]
    dtype = weights[0].dtype
    if any(weight.shape[1] != input_features for weight in weights):
        raise ValueError("all matrices must consume the same input width")
    if any(weight.dtype != dtype for weight in weights):
        raise ValueError("all matrices must use the same stored dtype")
    return input_features, dtype


def _integer_dtype_for_bits(dtype: torch.dtype) -> torch.dtype:
    mapping = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported floating weight dtype: {dtype}")
    return mapping[dtype]


def _unique_bit_patterns(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return floating values and inverse indices keyed by their exact bits."""

    integer_dtype = _integer_dtype_for_bits(values.dtype)
    bit_patterns = values.contiguous().view(integer_dtype)
    unique_bits, inverse = torch.unique(bit_patterns, sorted=True, return_inverse=True)
    return unique_bits.contiguous().view(values.dtype), inverse


def analyze_weight_family(weights: Sequence[torch.Tensor]) -> ProductAtlasAnalysis:
    """Count exact product sharing without constructing runtime index maps."""

    input_features, dtype = _validate_weights(weights)
    cpu_weights = [weight.detach().cpu() for weight in weights]
    output_features = sum(weight.shape[0] for weight in cpu_weights)
    element_bytes = cpu_weights[0].element_size()
    dictionary_sizes: list[int] = []
    packed_index_bits = 0

    for column in range(input_features):
        values = torch.cat([weight[:, column] for weight in cpu_weights])
        dictionary_size = int(_unique_bit_patterns(values)[0].numel())
        dictionary_sizes.append(dictionary_size)
        bits = max(1, math.ceil(math.log2(dictionary_size)))
        packed_index_bits += bits * output_features

    conventional = sum(weight.numel() for weight in cpu_weights)
    atlas = sum(dictionary_sizes)
    original_bytes = conventional * element_bytes
    packed_bytes = atlas * element_bytes + math.ceil(packed_index_bits / 8)
    return ProductAtlasAnalysis(
        dtype=str(dtype).removeprefix("torch."),
        matrix_shapes=[list(weight.shape) for weight in cpu_weights],
        input_features=input_features,
        output_features=output_features,
        dictionary_values=atlas,
        dictionary_values_per_column_median=float(statistics.median(dictionary_sizes)),
        dictionary_values_per_column_max=max(dictionary_sizes),
        conventional_multiplications=conventional,
        atlas_multiplications=atlas,
        multiplications_saved=conventional - atlas,
        multiplication_savings_ratio=1.0 - (atlas / conventional),
        original_weight_bytes=original_bytes,
        theoretical_packed_atlas_bytes=packed_bytes,
        theoretical_storage_ratio=packed_bytes / original_bytes,
    )


class ProductAtlas:
    """Reference implementation of exact product sharing across projections.

    The representation is deliberately simple and inspectable.  Its integer
    maps are not bit-packed, and advanced indexing is not fused with reduction,
    so its wall time should be treated as a go/no-go signal for a future native
    kernel rather than as an optimized implementation.
    """

    def __init__(self, weights: Sequence[torch.Tensor]):
        input_features, dtype = _validate_weights(weights)
        cpu_weights = [weight.detach().cpu().contiguous() for weight in weights]
        self.analysis = analyze_weight_family(cpu_weights)
        self.input_features = input_features
        self.source_dtype = dtype
        self.output_sizes = [weight.shape[0] for weight in cpu_weights]

        dictionaries: list[torch.Tensor] = []
        dictionary_columns: list[torch.Tensor] = []
        local_maps: list[list[torch.Tensor]] = [[] for _ in cpu_weights]
        offset = 0

        for column in range(input_features):
            column_values = [weight[:, column] for weight in cpu_weights]
            combined = torch.cat(column_values)
            unique, inverse = _unique_bit_patterns(combined)
            dictionaries.append(unique.to(torch.float32))
            dictionary_columns.append(
                torch.full((unique.numel(),), column, dtype=torch.int64)
            )
            cursor = 0
            for matrix_index, values in enumerate(column_values):
                count = values.numel()
                local_maps[matrix_index].append(inverse[cursor : cursor + count] + offset)
                cursor += count
            offset += unique.numel()

        self.dictionary = torch.cat(dictionaries)
        self.dictionary_columns = torch.cat(dictionary_columns)
        self.index_maps = [torch.stack(columns, dim=1) for columns in local_maps]
        self.dictionary_lengths = torch.bincount(
            self.dictionary_columns, minlength=self.input_features
        ).tolist()
        self.dictionary_offsets: list[int] = []
        offset = 0
        for length in self.dictionary_lengths:
            self.dictionary_offsets.append(offset)
            offset += length

    def __call__(self, activation: torch.Tensor) -> list[torch.Tensor]:
        """Evaluate every projection for one activation vector."""

        if activation.ndim != 1 or activation.numel() != self.input_features:
            raise ValueError(
                f"activation must have shape [{self.input_features}], got {list(activation.shape)}"
            )
        activation = activation.detach().cpu().to(torch.float32)
        products = self.dictionary * activation[self.dictionary_columns]
        return [products[index_map].sum(dim=1) for index_map in self.index_maps]


class CoordinateProductCache:
    """Preserve exact Product Atlas rows for repeated activation coordinates.

    Keys use the activation value's original bytes and its coordinate.  The
    cache is intentionally in-memory: hidden activations can encode user input
    and must not be written to a public trace or shared disk cache.
    """

    def __init__(self, atlas: ProductAtlas, *, entries_per_coordinate: int = 2):
        if entries_per_coordinate < 1:
            raise ValueError("entries_per_coordinate must be positive")
        self.atlas = atlas
        self.entries_per_coordinate = entries_per_coordinate
        self._cache: list[OrderedDict[bytes, torch.Tensor]] = [
            OrderedDict() for _ in range(atlas.input_features)
        ]
        self._lookups = 0
        self._hits = 0
        self._multiplications_performed = 0
        self._multiplications_avoided = 0

    def __call__(self, activation: torch.Tensor) -> list[torch.Tensor]:
        if activation.ndim != 1 or activation.numel() != self.atlas.input_features:
            raise ValueError(
                f"activation must have shape [{self.atlas.input_features}], "
                f"got {list(activation.shape)}"
            )
        source = activation.detach().cpu().contiguous()
        numeric = source.to(torch.float32)
        product_chunks: list[torch.Tensor] = []
        for column in range(self.atlas.input_features):
            key = bytes(source[column].reshape(1).view(torch.uint8).tolist())
            column_cache = self._cache[column]
            length = self.atlas.dictionary_lengths[column]
            self._lookups += 1
            if key in column_cache:
                products = column_cache.pop(key)
                column_cache[key] = products
                self._hits += 1
                self._multiplications_avoided += length
            else:
                offset = self.atlas.dictionary_offsets[column]
                values = self.atlas.dictionary[offset : offset + length]
                products = values * numeric[column]
                column_cache[key] = products
                self._multiplications_performed += length
                if len(column_cache) > self.entries_per_coordinate:
                    column_cache.popitem(last=False)
            product_chunks.append(products)
        product_table = torch.cat(product_chunks)
        return [product_table[index_map].sum(dim=1) for index_map in self.atlas.index_maps]

    @property
    def stats(self) -> CoordinateProductCacheStats:
        resident_values = sum(
            products.numel()
            for column_cache in self._cache
            for products in column_cache.values()
        )
        return CoordinateProductCacheStats(
            lookups=self._lookups,
            hits=self._hits,
            misses=self._lookups - self._hits,
            multiplications_performed=self._multiplications_performed,
            multiplications_avoided=self._multiplications_avoided,
            resident_bytes=resident_values * torch.empty((), dtype=torch.float32).element_size(),
        )


def benchmark_product_atlas(
    weights: Sequence[torch.Tensor],
    *,
    iterations: int = 10,
    seed: int = 0,
) -> tuple[ProductAtlas, ProductAtlasBenchmark]:
    """Compare the reference atlas with float32 dense matrix-vector products."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    atlas = ProductAtlas(weights)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    activation = torch.randn(atlas.input_features, generator=generator, dtype=torch.float32)
    references = [weight.detach().cpu().to(torch.float32) for weight in weights]

    def baseline() -> list[torch.Tensor]:
        return [torch.mv(weight, activation) for weight in references]

    baseline()
    atlas(activation)
    baseline_times: list[float] = []
    atlas_times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        expected = baseline()
        baseline_times.append((time.perf_counter_ns() - start) / 1_000_000)
        start = time.perf_counter_ns()
        actual = atlas(activation)
        atlas_times.append((time.perf_counter_ns() - start) / 1_000_000)

    expected_flat = torch.cat(expected)
    actual_flat = torch.cat(actual)
    difference = (expected_flat - actual_flat).abs()
    denominator = max(float(torch.linalg.vector_norm(expected_flat)), torch.finfo(torch.float32).tiny)
    baseline_median = statistics.median(baseline_times)
    atlas_median = statistics.median(atlas_times)
    benchmark = ProductAtlasBenchmark(
        iterations=iterations,
        baseline_median_ms=baseline_median,
        atlas_median_ms=atlas_median,
        atlas_to_baseline_ratio=atlas_median / baseline_median,
        max_absolute_error=float(difference.max()),
        mean_absolute_error=float(difference.mean()),
        relative_l2_error=float(torch.linalg.vector_norm(expected_flat - actual_flat)) / denominator,
        cosine_similarity=float(torch.nn.functional.cosine_similarity(expected_flat, actual_flat, dim=0)),
    )
    return atlas, benchmark


def exact_coordinate_hit_rate(activations: torch.Tensor) -> dict[str, float | int]:
    """Measure bit-identical activation-coordinate reuse across vectors."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape [samples, input_features]")
    values = activations.detach().cpu().contiguous()
    seen: list[set[bytes]] = [set() for _ in range(values.shape[1])]
    hits = 0
    for row in values:
        for column, value in enumerate(row):
            key = bytes(value.reshape(1).view(torch.uint8).tolist())
            if key in seen[column]:
                hits += 1
            else:
                seen[column].add(key)
    lookups = values.numel()
    return {
        "samples": values.shape[0],
        "input_features": values.shape[1],
        "lookups": lookups,
        "hits": hits,
        "hit_rate": hits / lookups if lookups else 0.0,
    }
