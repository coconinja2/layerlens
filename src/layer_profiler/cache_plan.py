"""Privacy-safe prefix-cache simulation and cache-aware request scheduling."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


_REQUEST_REF = re.compile(r"^request-\d+$")


@dataclass(frozen=True)
class PrefixRequest:
    """A prompt represented only by trace-local identity and token IDs."""

    request_ref: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _REQUEST_REF.fullmatch(self.request_ref):
            raise ValueError("request_ref must be trace-local, such as request-0")
        if not self.token_ids:
            raise ValueError("token_ids must not be empty")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in self.token_ids):
            raise ValueError("token_ids must contain non-negative integers")


class PrefixCacheSimulator:
    """Simulate full-block prefix reuse with a bounded LRU cache."""

    def __init__(
        self,
        *,
        block_size: int,
        capacity_blocks: int,
        cache_namespace: str = "layerlens-default",
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if capacity_blocks < 1:
            raise ValueError("capacity_blocks must be at least 1")
        if not cache_namespace:
            raise ValueError("cache_namespace must not be empty")
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        self._root = hashlib.sha256(cache_namespace.encode("utf-8")).digest()
        self._blocks: OrderedDict[bytes, None] = OrderedDict()

    def block_digests(self, token_ids: Sequence[int]) -> tuple[bytes, ...]:
        """Return chained digests for complete blocks without retaining token content."""
        parent = self._root
        digests = []
        full_length = len(token_ids) - (len(token_ids) % self.block_size)
        for start in range(0, full_length, self.block_size):
            encoded = json.dumps(
                token_ids[start : start + self.block_size], separators=(",", ":")
            ).encode("ascii")
            parent = hashlib.sha256(parent + b":" + encoded).digest()
            digests.append(parent)
        return tuple(digests)

    def peek_hit_blocks(self, token_ids: Sequence[int]) -> int:
        """Count the longest contiguous cached prefix without mutating LRU state."""
        hits = 0
        for digest in self.block_digests(token_ids):
            if digest not in self._blocks:
                break
            hits += 1
        return hits

    def process(self, request: PrefixRequest) -> dict[str, int | str]:
        """Process one request, returning cache hits and updating the simulated cache."""
        digests = self.block_digests(request.token_ids)
        hit_blocks = self.peek_hit_blocks(request.token_ids)
        for digest in digests[:hit_blocks]:
            self._blocks.move_to_end(digest)
        for digest in digests[hit_blocks:]:
            self._blocks[digest] = None
            self._blocks.move_to_end(digest)
            while len(self._blocks) > self.capacity_blocks:
                self._blocks.popitem(last=False)
        return {
            "request_ref": request.request_ref,
            "prompt_tokens": len(request.token_ids),
            "cacheable_blocks": len(digests),
            "hit_blocks": hit_blocks,
            "hit_tokens": hit_blocks * self.block_size,
        }


def _simulate(
    requests: Iterable[PrefixRequest],
    *,
    block_size: int,
    capacity_blocks: int,
    num_layers: int,
    cache_namespace: str,
    fixed_flops_per_layer_token: int | None,
) -> dict[str, Any]:
    simulator = PrefixCacheSimulator(
        block_size=block_size,
        capacity_blocks=capacity_blocks,
        cache_namespace=cache_namespace,
    )
    rows = [simulator.process(request) for request in requests]
    total_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    hit_tokens = sum(int(row["hit_tokens"]) for row in rows)
    baseline_ops = total_tokens * num_layers
    avoided_ops = hit_tokens * num_layers
    result = {
        "order": [str(row["request_ref"]) for row in rows],
        "requests": rows,
        "prompt_tokens": total_tokens,
        "prefix_hit_tokens": hit_tokens,
        "prefix_hit_rate": round(hit_tokens / total_tokens, 6) if total_tokens else 0.0,
        "baseline_layer_token_operations": baseline_ops,
        "avoided_layer_token_operations": avoided_ops,
        "remaining_layer_token_operations": baseline_ops - avoided_ops,
        "operation_reduction": round(avoided_ops / baseline_ops, 6) if baseline_ops else 0.0,
    }
    if fixed_flops_per_layer_token is not None:
        result.update(
            {
                "baseline_fixed_matmul_flops": baseline_ops * fixed_flops_per_layer_token,
                "avoided_fixed_matmul_flops": avoided_ops * fixed_flops_per_layer_token,
                "remaining_fixed_matmul_flops": (
                    baseline_ops - avoided_ops
                )
                * fixed_flops_per_layer_token,
            }
        )
    return result


def prefix_aware_order(
    requests: Sequence[PrefixRequest],
    *,
    block_size: int,
    capacity_blocks: int,
    cache_namespace: str = "layerlens-default",
) -> tuple[PrefixRequest, ...]:
    """Greedily schedule the request with the longest currently cached prefix."""
    simulator = PrefixCacheSimulator(
        block_size=block_size,
        capacity_blocks=capacity_blocks,
        cache_namespace=cache_namespace,
    )
    remaining = list(enumerate(requests))
    ordered: list[PrefixRequest] = []
    while remaining:
        position = max(
            range(len(remaining)),
            key=lambda index: (
                simulator.peek_hit_blocks(remaining[index][1].token_ids),
                -remaining[index][0],
            ),
        )
        _, request = remaining.pop(position)
        ordered.append(request)
        simulator.process(request)
    return tuple(ordered)


def compare_prefix_schedules(
    requests: Sequence[PrefixRequest],
    *,
    block_size: int,
    capacity_blocks: int,
    num_layers: int,
    cache_namespace: str = "layerlens-default",
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    mlp_projections: int = 3,
) -> dict[str, Any]:
    """Compare FCFS with exact, prefix-aware scheduling under one cache budget."""
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if (hidden_size is None) != (intermediate_size is None):
        raise ValueError("hidden_size and intermediate_size must be provided together")
    if hidden_size is not None and (hidden_size < 1 or intermediate_size < 1):
        raise ValueError("model dimensions must be at least 1")
    if mlp_projections not in {2, 3}:
        raise ValueError("mlp_projections must be 2 or 3")
    fixed_flops = None
    cost_model = None
    if hidden_size is not None and intermediate_size is not None:
        fixed_flops = 8 * hidden_size**2 + 2 * mlp_projections * hidden_size * intermediate_size
        cost_model = {
            "name": "dense_decoder_fixed_matmuls",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "mlp_projections": mlp_projections,
            "fixed_flops_per_layer_token": fixed_flops,
            "formula": "8*d^2 + 2*mlp_projections*d*intermediate_size",
            "excludes": [
                "attention score and value products",
                "normalization, activation, and elementwise operations",
                "mixture-of-experts and nonstandard attention projections",
            ],
        }
    fcfs = _simulate(
        requests,
        block_size=block_size,
        capacity_blocks=capacity_blocks,
        num_layers=num_layers,
        cache_namespace=cache_namespace,
        fixed_flops_per_layer_token=fixed_flops,
    )
    planned_requests = prefix_aware_order(
        requests,
        block_size=block_size,
        capacity_blocks=capacity_blocks,
        cache_namespace=cache_namespace,
    )
    planned = _simulate(
        planned_requests,
        block_size=block_size,
        capacity_blocks=capacity_blocks,
        num_layers=num_layers,
        cache_namespace=cache_namespace,
        fixed_flops_per_layer_token=fixed_flops,
    )
    incremental = int(planned["avoided_layer_token_operations"]) - int(
        fcfs["avoided_layer_token_operations"]
    )
    return {
        "method": "exact_prefix_lru",
        "block_size": block_size,
        "capacity_blocks": capacity_blocks,
        "num_layers": num_layers,
        "request_count": len(requests),
        "fcfs": fcfs,
        "prefix_aware": planned,
        "additional_operations_avoided": incremental,
        "changes_model_output": False,
        "cost_model": cost_model,
        "limitations": [
            "Only complete prefix blocks are reusable.",
            "The serving engine must expose compatible prefix caching.",
            "Layer-token operations are a hardware-neutral proxy, not measured FLOPs.",
        ],
    }
