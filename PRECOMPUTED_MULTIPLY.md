# Precomputed BF16 multiplication experiment

This experiment tests a fixed arithmetic table, not a cache of prompts,
activations, or previous results. Every possible BF16 significand product is
calculated before inference and addressed directly with integer bits. There is
no hash, similarity search, hit, or miss.

## Methods

For a finite BF16 number, LayerLens separates the sign, exponent, and unsigned
significand. It compares two tables:

- a 256-entry, 256-byte table for four-bit pieces, requiring four reads per
  significand product;
- a 65,536-entry, 128 KiB table containing every 8-bit significand product,
  requiring one read.

Sign is reconstructed with XOR, exponents are added, and the exact unsigned
product is scaled by a power of two. Operand components are decoded before the
timed section, which favors the table implementation. NaNs and infinities are
excluded from the timed real-model sample and would require a native slow path.

## Reproduce

```bash
poetry run layerlens-precomputed-multiply \
  --model Qwen/Qwen3.5-0.8B \
  --max-new-tokens 12 \
  --max-vectors 16 \
  --max-pairs 1000000 \
  --output-rows 6144 \
  --iterations 7 \
  --output benchmarks/qwen35-08b-precomputed-bf16-multiply.json
```

The saved run uses one million real weight/activation pairs captured from the
layer-0 linear-attention QKV projection. It also reconstructs the complete
6,144 × 1,024 projection. Timings are medians of seven iterations on CPU.

## Results

### One million independent products

| Threads | Native FP32 products | Direct table read only | Complete 128 KiB table path | Complete 256-byte four-bit path |
|---:|---:|---:|---:|---:|
| 1 | 1.120 ms | 0.811 ms (0.72×) | 9.193 ms (8.21×) | 20.288 ms (18.11×) |
| 4 | 0.628 ms | 0.236 ms (0.38×) | 4.204 ms (6.69×) | 9.398 ms (14.96×) |

Every reconstructed individual FP32 product was bit-for-bit identical to
native multiplication. Parallel direct lookup alone was 62.4% faster than
native multiplication. That is a useful positive result: table access itself
is not the bottleneck. Reconstructing signs, exponents, normalized FP32 values,
and materializing intermediate tensors erased the advantage.

### Complete QKV projection

| Threads | Native BF16 projection | Native FP32 projection | Table reconstruction | Table / native BF16 |
|---:|---:|---:|---:|---:|
| 1 | 0.324 ms | 2.898 ms | 57.441 ms | 177.40× |
| 4 | 0.162 ms | 1.751 ms | 26.327 ms | 162.76× |

The table projection rounded to the exact same BF16 output as the native BF16
projection. Its FP32 accumulation differed by at most 9.54e-7 because the
reduction order was different. Therefore the measured model-format output gate
passed, but the current unfused implementation is not a performance win.

## Algebraic observations

The 6,291,456 QKV weights contained 4,811 distinct BF16 bit patterns. Only
0.81% were exact powers of two suitable for a simple exponent-shift fast path;
none of the measured weights were exact zero or magnitude one.

A model-specific table containing the FP32 result of every one of those 4,811
weight values multiplied by every BF16 activation value would require about
1.26 GB for this projection's value vocabulary. Such a table would eliminate
sign/exponent reconstruction from the hot path and is the strongest next table
experiment, but it still must perform a direct memory read for every product
and preserve the projection's accumulation behavior.

## Conclusion

The experiment rejects the current software arithmetic-table implementation:
it is exact at the BF16 projection boundary but 162.76× slower than the native
full projection. It also isolates the opportunity: parallel direct lookup by
itself is faster than native elementwise multiplication. Any viable successor
must fuse lookup, FP32 reconstruction, and accumulation into one native kernel,
or precompute complete model-specific FP32 products so reconstruction disappears.

The public JSON contains aggregate timings and counts only. It contains no
prompt, token IDs, weights, activations, products, machine name, username, or
filesystem path.
