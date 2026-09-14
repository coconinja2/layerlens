"""Bit-exact in-memory and SQLite caches for dense matrix-vector results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Callable, Sequence

import torch
import torch.nn.functional as torch_functional


@dataclass(frozen=True)
class ExactCacheStats:
    lookups: int
    hits: int
    misses: int
    output_bytes_stored: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "hit_rate": self.hit_rate}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _activation_key(activation: torch.Tensor) -> bytes:
    digest = hashlib.blake2b(digest_size=32, person=b"layerlens-input")
    digest.update(str(activation.dtype).encode("ascii"))
    digest.update(str(tuple(activation.shape)).encode("ascii"))
    digest.update(_tensor_bytes(activation))
    return digest.digest()


def weight_fingerprint(weight: torch.Tensor, bias: torch.Tensor | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(str(weight.dtype).encode("ascii"))
    digest.update(str(tuple(weight.shape)).encode("ascii"))
    digest.update(_tensor_bytes(weight))
    if bias is not None:
        digest.update(str(bias.dtype).encode("ascii"))
        digest.update(str(tuple(bias.shape)).encode("ascii"))
        digest.update(_tensor_bytes(bias))
    return digest.hexdigest()


class _ExactResultComputer:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        if weight.ndim != 2:
            raise ValueError("weight must have shape [out, in]")
        if bias is not None and (bias.ndim != 1 or bias.numel() != weight.shape[0]):
            raise ValueError("bias must match the output width")
        self.weight = weight.detach().cpu().contiguous()
        self.bias = bias.detach().cpu().contiguous() if bias is not None else None
        self.input_features = weight.shape[1]
        self.output_features = weight.shape[0]
        self.namespace = weight_fingerprint(weight, bias)

    def compute(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 1 or activation.numel() != self.input_features:
            raise ValueError(
                f"activation must have shape [{self.input_features}], got {list(activation.shape)}"
            )
        native_input = activation.detach().cpu().to(self.weight.dtype)
        return torch_functional.linear(native_input, self.weight, self.bias)


class MemoryExactResultCache(_ExactResultComputer):
    """Process-local exact full-result cache."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__(weight, bias)
        self._values: dict[bytes, torch.Tensor] = {}
        self._lookups = 0
        self._hits = 0

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        key = _activation_key(activation)
        self._lookups += 1
        cached = self._values.get(key)
        if cached is not None:
            self._hits += 1
            return cached.clone()
        result = self.compute(activation)
        self._values[key] = result.clone()
        return result

    @property
    def stats(self) -> ExactCacheStats:
        stored = sum(value.numel() * value.element_size() for value in self._values.values())
        return ExactCacheStats(
            lookups=self._lookups,
            hits=self._hits,
            misses=self._lookups - self._hits,
            output_bytes_stored=stored,
        )


class SQLiteExactResultCache(_ExactResultComputer):
    """Persistent cache that stores hashes and native-dtype outputs, never inputs."""

    def __init__(
        self,
        weight: torch.Tensor,
        path: Path,
        *,
        bias: torch.Tensor | None = None,
        experiment_namespace: str = "default",
    ):
        super().__init__(weight, bias)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cache_already_existed = self.path.exists()
        self.cache_namespace = f"{self.namespace}:{experiment_namespace}"
        self.connection = sqlite3.connect(self.path)
        if not cache_already_existed:
            self.path.chmod(0o600)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS exact_results (
                namespace TEXT NOT NULL,
                input_hash BLOB NOT NULL,
                output BLOB NOT NULL,
                PRIMARY KEY (namespace, input_hash)
            ) WITHOUT ROWID
            """
        )
        self.connection.commit()
        self._lookups = 0
        self._hits = 0
        self._bytes_written = 0

    def __enter__(self) -> "SQLiteExactResultCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        key = _activation_key(activation)
        self._lookups += 1
        row = self.connection.execute(
            "SELECT output FROM exact_results WHERE namespace = ? AND input_hash = ?",
            (self.cache_namespace, key),
        ).fetchone()
        if row is not None:
            self._hits += 1
            return torch.frombuffer(bytearray(row[0]), dtype=self.weight.dtype).clone()

        result = self.compute(activation).contiguous()
        payload = _tensor_bytes(result)
        self.connection.execute(
            "INSERT INTO exact_results(namespace, input_hash, output) VALUES (?, ?, ?)",
            (self.cache_namespace, key, payload),
        )
        self.connection.commit()
        self._bytes_written += len(payload)
        return result

    @property
    def stats(self) -> ExactCacheStats:
        return ExactCacheStats(
            lookups=self._lookups,
            hits=self._hits,
            misses=self._lookups - self._hits,
            output_bytes_stored=self._bytes_written,
        )

    def close(self) -> None:
        self.connection.execute("PRAGMA wal_checkpoint(FULL)")
        self.connection.commit()
        self.connection.close()


def _elapsed_ms(operation: Callable[[], object]) -> tuple[float, object]:
    start = time.perf_counter_ns()
    result = operation()
    return (time.perf_counter_ns() - start) / 1_000_000, result


def _all_bitwise_equal(
    expected: Sequence[torch.Tensor], actual: Sequence[torch.Tensor]
) -> bool:
    return all(
        reference.dtype == result.dtype
        and reference.shape == result.shape
        and _tensor_bytes(reference) == _tensor_bytes(result)
        for reference, result in zip(expected, actual, strict=True)
    )


def benchmark_exact_result_caches(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    disk_path: Path,
    label: str,
    cycles: int = 5,
    bias: torch.Tensor | None = None,
    namespace_seed: str = "run",
) -> dict[str, object]:
    """Measure uncached, empty-cache, and restarted warm-cache workloads."""

    if activations.ndim != 2 or activations.shape[1] != weight.shape[1]:
        raise ValueError("activations must have shape [vectors, weight input width]")
    if cycles < 1:
        raise ValueError("cycles must be positive")
    rows = [row.contiguous() for row in activations.detach().cpu()]
    computer = _ExactResultComputer(weight, bias)

    dense_times: list[float] = []
    memory_initialization_times: list[float] = []
    memory_cold_times: list[float] = []
    memory_warm_times: list[float] = []
    disk_initialization_times: list[float] = []
    disk_cold_times: list[float] = []
    disk_reopen_initialization_times: list[float] = []
    disk_reopen_times: list[float] = []
    memory_cold_stats: ExactCacheStats | None = None
    memory_warm_stats: ExactCacheStats | None = None
    disk_stats: ExactCacheStats | None = None
    disk_reopen_stats: ExactCacheStats | None = None
    correctness = True

    for cycle in range(cycles):
        dense_ms, references = _elapsed_ms(
            lambda: [computer.compute(row) for row in rows]
        )
        dense_times.append(dense_ms)

        memory_init_ms, memory_result = _elapsed_ms(
            lambda: MemoryExactResultCache(weight, bias)
        )
        memory = memory_result
        assert isinstance(memory, MemoryExactResultCache)
        memory_initialization_times.append(memory_init_ms)
        memory_ms, memory_outputs = _elapsed_ms(lambda: [memory(row) for row in rows])
        memory_cold_times.append(memory_ms)
        memory_cold_stats = memory.stats
        warm_ms, warm_outputs = _elapsed_ms(lambda: [memory(row) for row in rows])
        memory_warm_times.append(warm_ms)
        memory_warm_stats = memory.stats

        namespace = f"{namespace_seed}:{label}:cycle:{cycle}"
        disk_init_ms, disk_result = _elapsed_ms(
            lambda: SQLiteExactResultCache(
                weight,
                disk_path,
                bias=bias,
                experiment_namespace=namespace,
            )
        )
        disk = disk_result
        assert isinstance(disk, SQLiteExactResultCache)
        disk_initialization_times.append(disk_init_ms)
        disk_ms, disk_outputs = _elapsed_ms(lambda: [disk(row) for row in rows])
        disk_cold_times.append(disk_ms)
        disk_stats = disk.stats
        disk.close()

        reopen_init_ms, reopen_result = _elapsed_ms(
            lambda: SQLiteExactResultCache(
                weight,
                disk_path,
                bias=bias,
                experiment_namespace=namespace,
            )
        )
        reopened = reopen_result
        assert isinstance(reopened, SQLiteExactResultCache)
        disk_reopen_initialization_times.append(reopen_init_ms)
        reopen_ms, reopened_outputs = _elapsed_ms(lambda: [reopened(row) for row in rows])
        disk_reopen_times.append(reopen_ms)
        disk_reopen_stats = reopened.stats
        reopened.close()

        correctness = correctness and all(
            _all_bitwise_equal(references, outputs)
            for outputs in (memory_outputs, warm_outputs, disk_outputs, reopened_outputs)
        )

    dense_median = statistics.median(dense_times)
    memory_initialization_median = statistics.median(memory_initialization_times)
    memory_cold_median = statistics.median(memory_cold_times)
    memory_warm_median = statistics.median(memory_warm_times)
    disk_initialization_median = statistics.median(disk_initialization_times)
    disk_cold_median = statistics.median(disk_cold_times)
    disk_reopen_initialization_median = statistics.median(
        disk_reopen_initialization_times
    )
    disk_reopen_median = statistics.median(disk_reopen_times)
    memory_query_saving = dense_median - memory_cold_median
    memory_first_pass = memory_initialization_median + memory_cold_median
    empty_hits = memory_cold_stats.hits if memory_cold_stats else 0
    empty_misses = memory_cold_stats.misses if memory_cold_stats else 0
    scalar_products_per_matvec = int(weight.shape[0] * weight.shape[1])
    return {
        "label": label,
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype).removeprefix("torch."),
        "activation_vectors": activations.shape[0],
        "activation_width": activations.shape[1],
        "cycles": cycles,
        "bitwise_output_agreement": correctness,
        "without_cache_median_ms": dense_median,
        "memory_initialization_median_ms": memory_initialization_median,
        "memory_empty_cache_median_ms": memory_cold_median,
        "memory_empty_cache_vs_without": memory_cold_median / dense_median,
        "memory_first_pass_including_initialization_median_ms": memory_first_pass,
        "memory_first_pass_including_initialization_vs_without": (
            memory_first_pass / dense_median
        ),
        "memory_warm_cache_median_ms": memory_warm_median,
        "memory_warm_cache_vs_without": memory_warm_median / dense_median,
        "disk_initialization_median_ms": disk_initialization_median,
        "disk_empty_cache_median_ms": disk_cold_median,
        "disk_empty_cache_vs_without": disk_cold_median / dense_median,
        "disk_reopen_initialization_median_ms": disk_reopen_initialization_median,
        "disk_reopen_page_cache_warm_median_ms": disk_reopen_median,
        "disk_reopen_page_cache_warm_vs_without": disk_reopen_median / dense_median,
        "memory_stats_after_empty_pass": (
            memory_cold_stats.to_dict() if memory_cold_stats else {}
        ),
        "memory_stats_after_warm_pass": (
            memory_warm_stats.to_dict() if memory_warm_stats else {}
        ),
        "disk_stats_after_empty_pass": disk_stats.to_dict() if disk_stats else {},
        "disk_stats_after_reopen": (
            disk_reopen_stats.to_dict() if disk_reopen_stats else {}
        ),
        "disk_file_bytes": disk_path.stat().st_size if disk_path.exists() else 0,
        "mathematical_work": {
            "dense_matvecs_without_cache": len(rows),
            "dense_matvecs_with_empty_memory_cache": empty_misses,
            "dense_matvecs_avoided_on_empty_pass": empty_hits,
            "scalar_multiplications_per_matvec": scalar_products_per_matvec,
            "scalar_multiplications_avoided_on_empty_pass": (
                empty_hits * scalar_products_per_matvec
            ),
            "dense_matvecs_avoided_on_fully_warm_pass": len(rows),
            "scalar_multiplications_avoided_on_fully_warm_pass": (
                len(rows) * scalar_products_per_matvec
            ),
        },
        "admission_decision": {
            "empty_pass_query_path_faster": memory_query_saving > 0,
            "empty_pass_query_path_saved_percent": (
                100 * memory_query_saving / dense_median
            ),
            "first_pass_including_initialization_faster": memory_first_pass < dense_median,
            "setup_cost_break_even_equivalent_workloads": (
                memory_initialization_median / memory_query_saving
                if memory_query_saving > 0
                else None
            ),
            "recommendation": (
                "admit_after_reuse_is_observed"
                if memory_query_saving > 0
                else "reject_exact_full_result_cache"
            ),
        },
    }
