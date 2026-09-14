# Cache and computation-reduction strategies

LayerLens separates exact optimizations from approximations that can change
model quality. Timing similarity alone is never considered proof that an
activation is reusable.

## Implemented: exact prefix-aware scheduling

`layerlens-cache-plan` models chained, full-block prefix keys in a bounded LRU
cache and greedily selects the queued request with the longest currently cached
prefix. The prompt and its token order are unchanged. An inference engine with
compatible prefix caching can therefore skip the corresponding prefill blocks.

The report measures reusable tokens and avoided layer-token evaluations:

```text
avoided_layer_token_operations = prefix_hit_tokens * num_layers
```

This hardware-neutral count deliberately avoids claiming FLOP or latency
savings that were not measured on the target engine.

For conventional dense decoders, optional dimensions add a bounded fixed-matmul
estimate:

```text
fixed_flops_per_layer_token = 8*d^2 + 2*P*d*m
```

`P` is the number of MLP projections. Attention-score products and nonstandard
architectures remain excluded instead of being guessed.

Related systems include SGLang's RadixAttention cache-aware scheduling and
ChunkAttention's prefix tree for shared KV tensors:

- [SGLang paper](https://openreview.net/forum?id=VqkAKQibpq)
- [ChunkAttention](https://arxiv.org/abs/2402.15220)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)

## Exact candidates for later adapters

### Modular prompt state

Prompt Cache names reusable prompt modules and retains their attention state.
This is useful for stable system instructions, tool schemas, and documents, but
the cache key must include model revision, adapter, tokenizer, positions, and
attention configuration.

- [Prompt Cache](https://arxiv.org/abs/2311.04934)

### Non-prefix document chunks

CacheBlend reuses KV state for chunks that are not at the beginning of the new
prompt, then selectively recomputes affected tokens. This is promising for RAG
but requires engine-level attention instrumentation and quality validation.

- [CacheBlend](https://arxiv.org/abs/2405.16444)

### Multi-tier KV storage

Hot blocks can remain on the accelerator, warm blocks in host memory, and cold
blocks on local storage. LayerLens should compare compute time with hashing,
read, decompression, and device-transfer time before recommending this:

```text
net_saved_ms = recompute_ms - hash_ms - read_ms - transfer_ms
```

Disk is useful for large prefixes reused across sessions; it is usually too
slow for small per-token activations.

## Approximate candidates: opt-in only

### Attention search-space reduction

H2O keeps recent tokens and attention heavy hitters. SnapKV uses an observation
window to select important per-head KV positions. Both reduce the number of
keys searched during later attention, but they can change outputs.

- [H2O](https://arxiv.org/abs/2306.14048)
- [SnapKV](https://arxiv.org/abs/2404.14469)

LayerLens would require attention-mass coverage, retained-token ratio, logit
divergence, exact-token agreement, and task-quality gates before enabling such
a policy.

### Layer reduction

ShortGPT measures block influence and removes low-influence layers. LayerSkip
trains intermediate layers for early exit and verifies draft tokens with the
remaining model. Ordinary checkpoints must not be treated as safely skippable
just because neighboring layer timings or tensor shapes look similar.

- [ShortGPT](https://arxiv.org/abs/2403.03853)
- [LayerSkip](https://arxiv.org/abs/2404.16710)
- [Draft & Verify](https://arxiv.org/abs/2309.08168)

## Required validation

Every optimization should be compared with the unmodified runtime using:

- time to first token and inter-token latency;
- actual prefix-hit tokens and KV bytes retained;
- layer-token evaluations and, when available, measured kernel FLOPs;
- greedy token agreement or distribution divergence;
- task-level quality on a representative evaluation set;
- peak cache occupancy, eviction, scheduler waiting, and preemption.

Only exact prefix reuse is enabled by the current implementation. Approximate
methods remain analysis targets until an adapter can measure their quality and
runtime trade-offs directly.
