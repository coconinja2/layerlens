# Cache and computation-reduction strategies

LayerLens separates exact optimizations from approximations that can change
model quality. Timing similarity alone is never considered proof that an
activation is reusable.

## Implemented experiment: exact Product Atlas

Product Atlas factors a family of projections that consume the same activation
through dictionaries of bit-identical stored weight values. It shares each
`activation coordinate × weight value` product across matching outputs and
sibling projections. The optional in-memory coordinate cache preserves a
product row only when the later activation value has exactly the same bits.

This adds no quantization and is separate from attention/KV caching. The Qwen3.5
0.8B MLP experiment removes 76.97% of scalar multiplications on paper, but the
current reference executor is 9.54× slower because gathers and reductions are
not fused. It is deliberately not wired into model inference until a native
kernel wins on wall time, memory, numerical agreement, and task quality.

See [PRODUCT_ATLAS.md](PRODUCT_ATLAS.md) for the equations and reproducible CLI.

## Implemented measurement: bottom-up repeatability

`layerlens-repeatability` profiles exact recurrence at scalar, unary-function,
coordinate, tile, vector, sibling-projection, attention-head, block, and layer
boundaries. The report distinguishes “the same BF16 number occurred again” from
“the same dependency-complete calculation can be reused.”

The first Qwen3.5 0.8B run found 97.81–98.33% repeated unary inputs, but a
complete bit-exact sigmoid/SiLU table was 2.35–2.45× slower than the native CPU
operation. Contextual four-value tiles, full vectors, and query-head vectors had
0% exact reuse. These rejected paths remain in the saved aggregate report so a
different model or backend can be compared rather than assumed equivalent.

See [REPEATABILITY.md](REPEATABILITY.md) for the hierarchy and interpretation.

## Implemented experiment: exact full-projection results

`layerlens-cache-benchmark` captures real inputs to chosen linear modules and
compares native dense execution with empty and warm memory caches plus a
reopened SQLite cache. A key includes the exact activation dtype, shape, and
bytes under an exact weight-and-bias fingerprint. A hit therefore reuses the
whole native-dtype projection output with no approximation.

On the saved Qwen3.5 0.8B CPU run, the layer-0 linear-attention QKV input had
18.92% exact full-vector reuse. It avoided 7 of 37 matvecs, or 44,040,192 scalar
multiplications, and made the query path 12.88% faster. Initialization made the
first short pass slower, so the reported admission decision waits until reuse is
observed. The layer-3 MLP gate input had 0% reuse and 10.31% overhead and was
rejected. This is the key boundary: repeated scalar values do not imply a
reusable dependency-complete matrix result.

See [CACHE_BENCHMARK.md](CACHE_BENCHMARK.md) for formulas, raw commands,
measured timings, integration code, and the warning not to publish the SQLite
file containing derived hidden-state outputs.

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

Exact prefix planning is available as an engine-facing feature. Exact Product
Atlas analysis is available as a kernel experiment, but its reference executor
is not installed into model inference. Approximate methods remain analysis
targets until an adapter can measure their quality and runtime trade-offs
directly.
