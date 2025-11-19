"""
Benchmark for L2 Cache Optimization

Tests the impact of L2 cache persistence on attention performance.
This optimization is particularly effective for decoding workloads where
KV cache is reused across multiple steps.

Copyright (c) 2024 by SageAttention team.
Licensed under the Apache License, Version 2.0
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sageattention import sageattn
from validation.validation_framework import ValidationFramework, compute_attention_flops


def benchmark_l2_cache_optimization(
    batch_size: int = 1,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 128,
    device: str = "cuda",
    num_decode_steps: int = 20,
):
    """
    Benchmark L2 cache optimization for decoding workload.

    In decoding, we:
    1. Have a fixed KV cache
    2. Generate one token at a time
    3. Reuse KV cache for each new query

    L2 cache persistence should help keep KV cache in L2.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length (KV cache size)
        head_dim: Head dimension
        device: Device to use
        num_decode_steps: Number of decoding steps to simulate
    """
    print(f"\n{'='*80}")
    print(f"L2 Cache Optimization Benchmark")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Num Heads:      {num_heads}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Head Dim:       {head_dim}")
    print(f"  Decode Steps:   {num_decode_steps}")
    print(f"{'='*80}\n")

    # Initialize framework
    framework = ValidationFramework(
        device=device,
        num_warmup=5,
        num_runs=50,
    )

    # Allocate KV cache (persistent across decode steps)
    k_cache = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float16, device=device
    )
    v_cache = torch.randn_like(k_cache)

    # Simulate decoding: single query at a time
    q = torch.randn(
        batch_size, num_heads, 1, head_dim,
        dtype=torch.float16, device=device
    )

    # FLOPs calculation
    flops = compute_attention_flops(batch_size, num_heads, seq_len, head_dim, is_causal=False)

    # Baseline: without L2 optimization
    def baseline_attention():
        outputs = []
        for _ in range(num_decode_steps):
            out = sageattn(q, k_cache, v_cache, tensor_layout="HND", is_causal=False)
            outputs.append(out)
        return outputs

    # Optimized: with L2 cache persistence
    def optimized_attention_l2():
        # Import L2 optimizer (may not work on all systems)
        try:
            from sageattention.l2_cache_optimizer import enable_kv_cache_l2_persistence

            # Enable L2 persistence for KV cache
            success = enable_kv_cache_l2_persistence(k_cache, v_cache)

            if not success:
                print("⚠ Warning: L2 cache persistence not available on this system")
                print("  This requires CUDA 11.4+ and specific GPU architectures")
                print("  Falling back to baseline (no optimization)")
                return baseline_attention()

            print("✓ L2 cache persistence enabled for KV cache")

        except (ImportError, NotImplementedError) as e:
            print(f"⚠ Warning: L2 optimization not available: {e}")
            print("  Falling back to baseline")
            return baseline_attention()

        outputs = []
        for _ in range(num_decode_steps):
            out = sageattn(q, k_cache, v_cache, tensor_layout="HND", is_causal=False)
            outputs.append(out)
        return outputs

    # Note: Since we can't easily enable/disable L2 persistence per benchmark,
    # we'll measure the optimized version and report the result
    # In practice, you would need to restart the process or use different streams

    print("Measuring baseline performance...")
    baseline_perf = framework.measure_performance(
        baseline_attention,
        args=(),
        kwargs={},
        name="baseline",
        flops_per_call=flops * num_decode_steps,
    )

    print("\nMeasuring optimized performance (with L2 cache persistence)...")
    opt_perf = framework.measure_performance(
        optimized_attention_l2,
        args=(),
        kwargs={},
        name="l2_optimized",
        flops_per_call=flops * num_decode_steps,
    )

    # Calculate speedup
    speedup = baseline_perf.mean_time_ms / opt_perf.mean_time_ms

    print(f"\n{'='*80}")
    print(f"L2 Cache Optimization Results")
    print(f"{'='*80}")
    print(f"Baseline:   {baseline_perf.mean_time_ms:.3f} ms (±{baseline_perf.std_time_ms:.3f})")
    print(f"Optimized:  {opt_perf.mean_time_ms:.3f} ms (±{opt_perf.std_time_ms:.3f})")
    print(f"Speedup:    {speedup:.2f}x")
    print(f"Expected:   1.10-1.15x (10-15% improvement)")

    if speedup >= 1.05:
        print(f"Status:     ✓ IMPROVEMENT DETECTED ({(speedup-1)*100:.1f}%)")
    else:
        print(f"Status:     ⚠ MINIMAL/NO IMPROVEMENT")
        print(f"            (May not be supported on this GPU or CUDA version)")

    print(f"{'='*80}\n")

    return speedup


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark L2 Cache Optimization")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    speedup = benchmark_l2_cache_optimization(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        device=args.device,
        num_decode_steps=args.decode_steps,
    )

    # Print summary
    print(f"\nFinal Speedup: {speedup:.2f}x")
    print(f"Expected Range: 1.10-1.15x (10-15%)")

    if speedup >= 1.10:
        print("✓ L2 Cache Optimization: SUCCESSFUL")
        sys.exit(0)
    elif speedup >= 1.05:
        print("⚠ L2 Cache Optimization: MARGINAL BENEFIT")
        sys.exit(0)
    else:
        print("✗ L2 Cache Optimization: NOT EFFECTIVE")
        print("  (This optimization requires Ada/Hopper GPUs and CUDA 11.4+)")
        sys.exit(1)
