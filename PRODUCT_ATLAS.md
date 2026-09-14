# Exact Product Atlas

Product Atlas is a LayerLens experiment for preserving scalar products inside
dense inference matrix multiplications. It is not a KV cache, prompt cache, or
new quantization format.

## The idea

Several projections in a transformer consume the same activation vector. In a
Qwen MLP, for example, both `gate_proj(x)` and `up_proj(x)` evaluate a dense
matrix-vector product. For a family of matrices \(W^{(r)}\), collect the exact
stored weight bit patterns used at input coordinate \(j\):

\[
D_j = \operatorname{unique\_bits}\{W^{(r)}_{ij}\}_{r,i}
\]

Compute a product atlas once for that coordinate:

\[
P_j[k] = x_j D_j[k]
\]

Each ordinary output is reconstructed through a precomputed integer map
\(I^{(r)}\):

\[
y^{(r)}_i = \sum_j P_j[I^{(r)}_{ij}]
\]

The conventional family performs \(\sum_r d\,o_r\) scalar multiplications. The
atlas performs \(\sum_j |D_j|\). Additions are not eliminated. BF16 and FP16
checkpoints naturally contain repeated bit patterns because their weights come
from finite sets; Product Atlas does not reduce their precision further.

The optional `CoordinateProductCache` also keeps \(P_j\) in memory when the
same activation coordinate recurs with exactly the same bits. It never treats
nearby floating-point values as equivalent. Hidden activations can encode user
input, so this cache is process-local and is not exported to traces or disk.

## Measured result, not a speed claim

The checked-in experiment used the first MLP gate/up pair from the unmodified
BF16 `Qwen/Qwen3.5-0.8B` checkpoint:

| Measurement | Dense pair | Product Atlas |
|---|---:|---:|
| Matrix shapes | 2 × (3584 × 1024) | same weights |
| Scalar multiplications | 7,340,032 | 1,690,196 |
| Multiplications removed | — | 5,649,836 (76.97%) |
| Added quantization | none | none |
| Packed representation estimate | 14,680,064 bytes | 13,472,936 bytes |
| Median CPU reference time | 0.520 ms | 4.961 ms |
| Relative L2 difference vs `torch.mv` | — | 1.46e-7 |

The operation count is promising; the Python/PyTorch prototype is **9.54×
slower**. It materializes large integer gathers, while the baseline uses a
highly optimized dense kernel. That makes the current code an analyzer and
correctness oracle, not a replacement inference kernel.

The small numerical difference is floating-point reduction order, not an
approximation of weights or activations. A production kernel still needs greedy
token agreement and task-quality tests; “mathematically equivalent” is not
enough to claim unchanged model behavior.

Across 61 real activation vectors captured in one synthetic Qwen3.5 0.8B run,
only 2.76% of same-coordinate BF16 values repeated exactly (4.90% over the last
16 vectors). Persisting products across tokens is therefore a weak path for
this sample and grows memory quickly. The stronger measured reuse is spatial:
bit-identical weight values shared inside and across sibling projections for the
same current activation.

## Run it on any Hugging Face model

The default family is the MLP gate/up pair in decoder layer 0:

```bash
poetry install --with dev
poetry run layerlens-product-atlas \
  --model Qwen/Qwen3.5-0.8B \
  --layer 0 \
  --benchmark-iterations 20
```

Model files must already be local unless `--allow-download` is passed. Supply
different projection suffixes for another architecture:

```bash
poetry run layerlens-product-atlas \
  --model provider/model \
  --layer 0 \
  --projection self_attn.q_proj.weight \
  --projection self_attn.k_proj.weight \
  --projection self_attn.v_proj.weight
```

The JSON report contains model and parameter identifiers, dimensions, operation
counts, storage estimates, and optional timings. It contains no prompts,
activations, generated text, hostnames, usernames, or environment variables.

## Qwen3.5 27B reality check

The official `Qwen/Qwen3.5-27B` checkpoint is BF16, has 27B parameters, 64
language layers, hidden size 5120, and intermediate size 17408. Its repository
is approximately 55.6 GB:

- [Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B)
- [Qwen3.5-27B configuration](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)

An exact method cannot promise to fit that checkpoint in 16 GB while retaining
the performance of an in-memory dense kernel. Product Atlas estimates only an
8.2% packed storage reduction on the measured 0.8B MLP pair, far short of the
roughly 3.5× reduction required. Disk streaming can make weights addressable,
but it sacrifices latency to storage bandwidth.

The practical next milestone is a fused Metal/CUDA kernel that avoids
materializing the gather tensor. It must beat the vendor GEMV/GEMM kernel in
wall time and memory on several layers before LayerLens should attempt a 27B
integration. If it fails that gate, the experiment should remain a profiler
finding rather than become an inference feature.

