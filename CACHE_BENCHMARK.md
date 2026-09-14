# Exact projection-result cache benchmark

LayerLens now tests the simplest possible non-KV cache from first principles:
if the complete input vector to a fixed linear projection repeats bit for bit,
reuse the complete output vector instead of executing the matrix-vector product
again.

This is an experiment and an integration primitive, not a claim that every
projection should be cached. The included profiler measures the admission
decision separately for each module.

## Exact contract

For a linear projection

```text
y = W x + b
```

the key is

```text
K = (SHA256(W bytes, b bytes), BLAKE2b(dtype(x), shape(x), x bytes))
```

and the stored value is the native-dtype output `y`. A hit skips one complete
matrix-vector product. There is no tolerance, interpolation, quantization, or
similarity search. Different weights, bias, input dtype, input shape, or input
bits cannot match.

For `W` with shape `[m, n]`, one hit avoids exactly

```text
m × n scalar multiplications
```

plus the corresponding accumulation and bias work. LayerLens reports the
multiplication count and does not guess hardware FLOPs.

## Reproduce the raw experiment

```bash
poetry install --with dev

poetry run layerlens-cache-benchmark \
  --model Qwen/Qwen3.5-0.8B \
  --max-new-tokens 24 \
  --max-vectors 64 \
  --cycles 7 \
  --disk-cache /tmp/layerlens-exact-result-cache.sqlite \
  --output benchmarks/qwen35-08b-exact-result-cache.json
```

The default targets are the layer-0 linear-attention QKV input and the layer-3
MLP gate input. Repeat `--target MODULE_SUFFIX` to benchmark other
`torch.nn.Linear` modules. The CLI captures real inputs from a normal greedy
model run, then replays the same native BF16 projection through four paths:

1. dense projection with no cache;
2. empty in-memory cache, including natural hits inside that pass;
3. fully warm in-memory cache;
4. empty SQLite cache and a later lookup after closing and reopening SQLite.

Initialization time is reported separately. The reopened SQLite number is
explicitly a page-cache-warm measurement, not a claim about cold physical-disk
latency.

## Qwen3.5 0.8B result

This saved run used 14 prompt tokens, 24 generated tokens, BF16 CPU inference,
37 captured vectors per target, and the median of seven cycles.

| Projection | Exact hits on empty pass | Dense | Memory empty | First pass with setup | Memory warm | Reopened SQLite lookup | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| layer 0 linear-attention QKV | 7 / 37 (18.92%) | 5.444 ms | 4.743 ms | 11.026 ms | 0.224 ms | 0.880 ms | Admit only after reuse is observed |
| layer 3 MLP gate | 0 / 37 (0%) | 2.296 ms | 2.533 ms | 5.915 ms | 0.191 ms | 0.742 ms | Reject |

Both the replayed native projection and every cache mode agreed bit for bit
with the outputs captured from the model.

The early projection avoided 7 dense matvecs and 44,040,192 scalar
multiplications in the first empty-cache pass. Its query path was 12.88%
faster, but initialization made that first pass 2.03× the uncached time. If the
same reuse rate and timings persist, setup amortizes after about 8.96 equivalent
workloads. The contextual layer had no hits, avoided no multiplication, and
added 10.31% query-path overhead, so the profiler rejects it.

The fully warm results show an upper bound for repeated requests, not the
expected behavior of arbitrary new generations. Disk lookup is slower than
memory lookup but remains faster than recomputation in this page-cache-warm
test; opening the database and hashing the weights still costs 3–6 ms per
projection and is reported separately.

## Integration

The reusable classes are public:

```python
from pathlib import Path
from layer_profiler import MemoryExactResultCache, SQLiteExactResultCache

memory = MemoryExactResultCache(linear.weight, linear.bias)
y = memory(x)

with SQLiteExactResultCache(
    linear.weight,
    Path("projection-results.sqlite"),
    bias=linear.bias,
    experiment_namespace="model-revision-and-adapter",
) as disk:
    y = disk(x)
```

An engine adapter should begin in observation mode and admit only modules whose
measured saved compute exceeds hashing, lookup, initialization, memory, and
transfer costs on that engine. CPU timings should not be projected onto a GPU.

## Privacy and storage

The public JSON contains only aggregate counts and timings. It contains no
prompt, token IDs, activations, cached outputs, cache path, username, hostname,
or raw Prometheus labels.

The SQLite database itself contains derived hidden-state outputs. Treat it as
potentially sensitive local model data: do not commit or share it, restrict its
permissions, assign a model/revision/adapter-specific namespace, and apply an
explicit retention policy in a production integration. LayerLens stores only
the activation hash as the key, not the activation bytes, but a hash does not
make the stored output public-safe.
