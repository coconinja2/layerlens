# Bottom-up repeatability profiling

`layerlens-repeatability` starts at stored floating-point bit patterns and moves
up through unary functions, matrix products, activation tiles, attention heads,
transformer blocks, and selected layers. It separates a repeated number from a
reusable computation: the same scalar appearing at a different matrix column
does not automatically save a dot product.

The selected Qwen checkpoint declares BF16 storage, SiLU, 8 query heads, and an
attention output gate. The profiler follows the corresponding Hugging Face
implementation, where MLP gate/up projections share an input and full attention
multiplies its output by `sigmoid(gate)`:

- [Qwen3.5 0.8B configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)
- [Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

## Profile hierarchy

### 1. Scalar and unary functions

For an observed tensor with \(N\) elements and \(U\) distinct bit patterns, a
perfectly keyed unary memo table could avoid \(N-U\) repeated evaluations:

\[
r_{unary}=1-\frac{U}{N}
\]

BF16 and FP16 each have exactly \(2^{16}=65,536\) input bit patterns. LayerLens
can therefore construct a complete 128 KiB BF16 output table for sigmoid or
SiLU. Every possible input is covered without rounding it into a new bucket.
`FiniteDomainUnaryAtlas` indexes the table using the original input bits.

On the Qwen3.5 0.8B sample, 97.81–98.33% of observed unary inputs had appeared
before and the table produced 100% bitwise agreement with PyTorch. The lookup
still took 2.35–2.45× as long as the native vector operation on CPU. Repetition
exists, but this implementation should not replace the native kernel.

### 2. Matrix products

Product Atlas groups bit-identical weights at each input coordinate across
projections that consume the same vector:

\[
P_j[k]=x_jD_j[k],\qquad y_i^{(r)}=\sum_jP_j[I_{ij}^{(r)}]
\]

Measured exact scalar-multiplication reductions were:

| Projection family | Layer | Multiplications reusable |
|---|---:|---:|
| MLP gate + up | 0 | 76.97% |
| MLP gate + up | 3 | 75.74% |
| Attention Q + K + V | 3 | 68.86% |

These are multiplication counts, not latency claims. Additions remain, and the
current gather-based reference is slower than dense matvec. A fused native
kernel is required.

### 3. Coordinates, tiles, and exact deltas

LayerLens tests progressively stronger keys:

- scalar reuse anywhere, suitable only for a unary function;
- the same scalar at the same coordinate;
- the same contiguous tile at the same tile position;
- the entire vector;
- unchanged coordinates between consecutive vectors.

For exact incremental matmul,

\[
Wx_t = Wx_{t-1}+W(x_t-x_{t-1})
\]

only exact zeros in \(x_t-x_{t-1}\) remove columns of multiplication. In the
measured contextual signals, only 0.08–1.71% of consecutive elements were
unchanged. Four-value tile and whole-vector reuse was 0% after the first
contextual transformation.

Layer 0 was the exception: its input had 18.92% exact vector/tile reuse because
identical token embeddings can recur. Only context-free projections of that
input are candidates for reuse; an entire attention block cannot be cached
because position and context still change its result.

### 4. Attention heads and blocks

The full-attention layer exposed 296 query-head vectors (37 samples × 8 heads).
There were no bit-identical head vectors, no same-head-position repeats, and no
duplicate heads within a sample. Consecutive query heads were directionally
similar (mean cosine 0.728), but similarity is not exact reuse and is therefore
reported only as a diagnostic.

After contextual mixing, every profiled MLP input, gate vector, SiLU output,
query state, attention gate, and layer output had 0% exact four-value tile and
whole-vector reuse. Whole-state caching is not supported by this measurement.

## Run and save a profile

```bash
poetry install --with dev
poetry run layerlens-repeatability \
  --model Qwen/Qwen3.5-0.8B \
  --layers 0,3 \
  --max-new-tokens 24 \
  --max-vectors 64 \
  --tile-size 4 \
  --output benchmarks/repeatability-profile.json
```

Model files must already be local unless `--allow-download` is passed. Include
a full-attention layer to obtain head measurements. Module suffixes are resolved
from the model rather than hard-coded to a top-level prefix.

The saved report contains the model revision, library versions, aggregate
counts, ratios, timings, shapes, and parameter names. Prompts, token IDs,
generated text, raw activations, hostnames, usernames, and environment variables
are excluded. The checked-in measurement is
[benchmarks/qwen35-08b-bottom-up-repeatability.json](benchmarks/qwen35-08b-bottom-up-repeatability.json).

## What survives the measurement

1. **Keep investigating sibling-product fusion.** It has the largest exact
   arithmetic reuse, but needs a native Metal/CUDA kernel and must beat GEMV.
2. **Keep the full-domain unary table as a backend experiment.** It is bounded,
   deterministic, and bit-exact, but the CPU lookup loses today.
3. **Consider layer-0 projection caching for repeated token embeddings.** Scope
   the cache before position- or context-dependent operations.
4. **Reject contextual tile, whole-vector, and attention-head caches for this
   sample.** Their exact hit rate is zero.
5. **Do not convert high cosine similarity into reuse.** That would be an
   approximation and violates the current accuracy contract.
