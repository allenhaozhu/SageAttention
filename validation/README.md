# SageAttention Optimization Validation Suite

This directory contains benchmarks to validate 5 high-impact optimizations for SageAttention.

## Quick Start

Run all validations:

```bash
cd validation
python run_all_validations.py
```

This will test all 5 optimizations and generate a comprehensive report.

## Individual Benchmarks

### 1. L2 Cache Control (Expected: 10-15% speedup)

Tests L2 cache persistence for KV cache on Ada/Hopper GPUs.

```bash
python bench_l2_cache.py --seq-len 2048 --decode-steps 20
```

**Implementation Status:** ✓ Proof-of-concept available in `sageattention/l2_cache_optimizer.py`

**Key Benefit:** Keeps KV cache resident in L2, reducing memory latency for decoding workloads.

---

### 2. CUDA Graphs (Expected: 25-35% latency reduction)

Tests CUDA graph capture to eliminate kernel launch overhead.

```bash
python bench_cuda_graphs.py --batch-size 1 --seq-len 2048
```

**Implementation Status:** ⚠ Requires SageAttention kernel modifications to support graph capture

**Key Benefit:** Eliminates ~10-20μs launch overhead per call, critical for low-latency inference.

---

### 3. Adaptive Quantization (Expected: 8-12% speedup with better accuracy)

Tests automatic selection of quantization granularity based on activation distribution.

```bash
python bench_adaptive_quant.py --seq-len 2048
```

**Implementation Status:** ✓ Proof-of-concept available in `sageattention/adaptive_quant.py`

**Key Benefit:** Automatically balances speed vs accuracy based on data characteristics.

---

### 4. KV Cache Layout (Expected: 15-25% speedup for decode)

Tests pre-quantized KV cache to eliminate preprocessing overhead.

```bash
python bench_kv_cache_layout.py --seq-len 2048 --decode-steps 20
```

**Implementation Status:** ⚠ Requires API changes to accept pre-quantized inputs

**Key Benefit:** Quantize once, reuse for all decode steps in multi-turn conversations.

---

### 5. Kernel Fusion (Expected: 20-30% bandwidth reduction)

Estimates benefit of fusing quantization + smoothing + attention into single kernel.

```bash
python bench_kernel_fusion.py --seq-len 2048
```

**Implementation Status:** ⚠ Requires new fused CUDA kernel implementation

**Key Benefit:** Reduces memory traffic by keeping intermediate results in registers/shared memory.

---

## Combined Impact

The validation suite estimates combined speedup using two models:

1. **Conservative**: Assumes 70% effectiveness when combining optimizations
2. **Optimistic**: Assumes optimizations are orthogonal (multiplicative)

Example output:

```
Conservative Estimate: 1.82x additional speedup
Optimistic Estimate:   2.34x additional speedup

Current SageAttention: 2.7x vs FlashAttention2 (RTX 5090)
With optimizations:    4.91x - 6.32x
```

## Configuration

All benchmarks support the following arguments:

- `--batch-size`: Batch size (default: 1)
- `--num-heads`: Number of attention heads (default: 32)
- `--seq-len`: Sequence length (default: 2048)
- `--head-dim`: Head dimension (default: 128)
- `--device`: Device to use (default: cuda)

Example:

```bash
python run_all_validations.py \
  --batch-size 1 \
  --num-heads 32 \
  --seq-len 4096 \
  --head-dim 128 \
  --output-dir my_results
```

## Output

Results are saved to `validation_results/` directory:

- `validation_results_TIMESTAMP.json`: Machine-readable results
- `validation_report_TIMESTAMP.txt`: Human-readable report

## Requirements

- PyTorch >= 2.0
- CUDA >= 11.4 (for L2 cache optimization)
- SageAttention installed
- FlashAttention (for reference comparisons)

## Implementation Priority

Based on impact vs difficulty:

| Optimization | Difficulty | Impact | Priority |
|-------------|-----------|--------|----------|
| L2 Cache Control | Low | Medium | ⭐⭐⭐⭐⭐ |
| CUDA Graphs | Low | High | ⭐⭐⭐⭐⭐ |
| Adaptive Quant | Low | Medium | ⭐⭐⭐⭐⭐ |
| KV Cache Layout | Low | High (decode) | ⭐⭐⭐⭐⭐ |
| Kernel Fusion | Medium | High | ⭐⭐⭐⭐⭐ |

All 5 optimizations are high priority!

## Validation Framework

The validation framework (`validation_framework.py`) provides utilities for:

- **Performance measurement**: Latency, throughput, memory usage
- **Accuracy measurement**: Max error, mean error, cosine similarity
- **L2 cache flushing**: Fair benchmarking
- **TFLOPS calculation**: Compute efficiency metrics

Example usage:

```python
from validation.validation_framework import ValidationFramework

framework = ValidationFramework(device="cuda", num_runs=100)

# Benchmark a function
perf = framework.measure_performance(
    fn=my_attention_function,
    args=(q, k, v),
    kwargs={},
    flops_per_call=flops,
)

print(f"Time: {perf.mean_time_ms:.3f} ms")
print(f"Throughput: {perf.throughput_tflops:.2f} TFLOPS")
```

## Notes

### L2 Cache Optimization

- Only effective on Ada (RTX 4090, L40) and Hopper (H100) GPUs
- Requires CUDA 11.4+
- May not work on all systems due to driver/CUDA limitations

### CUDA Graphs

- Requires static input shapes
- Not all operations are graph-capturable
- May need kernel modifications to remove dynamic operations

### Kernel Fusion

- Current benchmarks measure individual kernel timings
- Actual fused kernel needs to be implemented in CUDA
- Estimated speedup is conservative

## Contributing

To add a new optimization benchmark:

1. Create `bench_<optimization_name>.py`
2. Implement benchmark function
3. Add to `run_all_validations.py`
4. Update this README

## References

- [OPTIMIZATION_PROPOSALS.md](../OPTIMIZATION_PROPOSALS.md): Detailed technical analysis
- [OPTIMIZATION_SUMMARY.md](../OPTIMIZATION_SUMMARY.md): Executive summary
- [SageAttention Paper](https://arxiv.org/abs/2410.02367)
