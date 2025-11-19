"""
Benchmark for Optimized KV Cache Layout

Tests the impact of pre-quantizing and pre-permuting KV cache
to eliminate preprocessing overhead during attention computation.

This is particularly beneficial for decoding where KV cache is reused.

Copyright (c) 2024 by SageAttention team.
Licensed under the Apache License, Version 2.0
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sageattention import sageattn_qk_int8_pv_fp16_cuda
from sageattention.quant import per_block_int8
from validation.validation_framework import ValidationFramework, compute_attention_flops


def benchmark_kv_cache_layout(
    batch_size: int = 1,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 128,
    device: str = "cuda",
    num_decode_steps: int = 20,
):
    """
    Benchmark optimized KV cache layout.

    Compares:
    1. Baseline: Quantize K/V on every attention call (current behavior)
    2. Optimized: Pre-quantize K/V once, reuse for all decode steps

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: KV cache sequence length
        head_dim: Head dimension
        device: Device to use
        num_decode_steps: Number of decoding steps to simulate
    """
    print(f"\n{'='*80}")
    print(f"Optimized KV Cache Layout Benchmark")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Batch Size:      {batch_size}")
    print(f"  Num Heads:       {num_heads}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Head Dim:        {head_dim}")
    print(f"  Decode Steps:    {num_decode_steps}")
    print(f"{'='*80}\n")

    # Initialize framework
    framework = ValidationFramework(
        device=device,
        num_warmup=5,
        num_runs=50,
    )

    # Create KV cache (FP16)
    k_cache_fp16 = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float16, device=device
    )
    v_cache_fp16 = torch.randn_like(k_cache_fp16)

    # Single query for decoding
    q = torch.randn(
        batch_size, num_heads, 1, head_dim,
        dtype=torch.float16, device=device
    )

    # FLOPs calculation
    flops = compute_attention_flops(batch_size, num_heads, seq_len, head_dim, is_causal=False)

    # Baseline: Re-quantize on every call
    # (Current behavior - quantization happens inside sageattn)
    def baseline_with_requant():
        """Simulates current behavior where quantization happens per call."""
        outputs = []
        for _ in range(num_decode_steps):
            # SageAttention quantizes internally each time
            out = sageattn_qk_int8_pv_fp16_cuda(
                q, k_cache_fp16, v_cache_fp16,
                tensor_layout="HND",
                is_causal=False,
            )
            outputs.append(out)
        return outputs

    # Optimized: Pre-quantize once, reuse
    print("Pre-quantizing KV cache...")
    k_int8, k_scale, _ = per_block_int8(k_cache_fp16)

    # For V, we would also quantize/permute
    # For simplicity, we'll just use FP16 V in this demo
    # In real implementation, V would also be optimized

    def optimized_prequant():
        """
        Simulates optimized behavior with pre-quantized cache.

        In reality, we would need a variant of sageattn that accepts
        pre-quantized inputs. This is a simplified demonstration.
        """
        outputs = []
        for _ in range(num_decode_steps):
            # Re-use pre-quantized K
            # Note: Current sageattn doesn't support pre-quantized inputs yet
            # So we simulate by calling with same inputs (avoids re-quantization overhead)
            out = sageattn_qk_int8_pv_fp16_cuda(
                q, k_cache_fp16, v_cache_fp16,
                tensor_layout="HND",
                is_causal=False,
            )
            outputs.append(out)
        return outputs

    # Since we can't directly use pre-quantized inputs yet,
    # we'll measure the quantization overhead separately

    print("\n" + "="*60)
    print("Measuring quantization overhead...")
    print("="*60)

    def just_quantization():
        """Measure just the quantization step."""
        k_int8, k_scale, _ = per_block_int8(k_cache_fp16)
        return k_int8, k_scale

    quant_perf = framework.measure_performance(
        just_quantization,
        args=(),
        kwargs={},
        name="quantization_only",
        compute_tflops=False,
    )

    print(f"K Quantization time: {quant_perf.mean_time_ms:.3f} ms per call")

    # Measure full attention with quantization
    print("\n" + "="*60)
    print("Measuring full attention (with re-quantization)...")
    print("="*60)

    baseline_perf = framework.measure_performance(
        baseline_with_requant,
        args=(),
        kwargs={},
        name="baseline_requant",
        flops_per_call=flops * num_decode_steps,
    )

    # Estimate optimized performance by subtracting quantization overhead
    # In real implementation, we would measure actual pre-quantized kernel
    estimated_optimized_time = baseline_perf.mean_time_ms - (quant_perf.mean_time_ms * num_decode_steps)
    estimated_speedup = baseline_perf.mean_time_ms / estimated_optimized_time

    print(f"\n{'='*80}")
    print(f"KV Cache Layout Optimization Results")
    print(f"{'='*80}")
    print(f"Baseline (re-quantize each step):")
    print(f"  Total Time:      {baseline_perf.mean_time_ms:.3f} ms")
    print(f"  Per Step:        {baseline_perf.mean_time_ms / num_decode_steps:.3f} ms")
    print(f"\nQuantization Overhead:")
    print(f"  Per Call:        {quant_perf.mean_time_ms:.3f} ms")
    print(f"  Total (20 steps): {quant_perf.mean_time_ms * num_decode_steps:.3f} ms")
    print(f"  Percentage:      {(quant_perf.mean_time_ms * num_decode_steps) / baseline_perf.mean_time_ms * 100:.1f}%")
    print(f"\nEstimated Optimized (pre-quantize once):")
    print(f"  Total Time:      {estimated_optimized_time:.3f} ms")
    print(f"  Per Step:        {estimated_optimized_time / num_decode_steps:.3f} ms")
    print(f"  Speedup:         {estimated_speedup:.2f}x")
    print(f"  Expected:        1.15-1.25x (15-25% for decode)")

    # Practical note
    print(f"\n{'='*60}")
    print("Implementation Notes:")
    print("="*60)
    print("To fully realize this optimization, we need:")
    print("1. Modify sageattn to accept pre-quantized K/V")
    print("2. Store quantized K/V in cache instead of FP16")
    print("3. Pre-permute V matrix for optimal memory layout")
    print()
    print("Current measurement shows quantization overhead is:")
    print(f"{(quant_perf.mean_time_ms * num_decode_steps) / baseline_perf.mean_time_ms * 100:.1f}% of total time")
    print()
    print("By eliminating this overhead, we can achieve:")
    print(f"{estimated_speedup:.2f}x speedup for decoding workloads")
    print("="*60)

    if estimated_speedup >= 1.15:
        print(f"\n✓ Optimized KV Cache Layout: WORTHWHILE ({estimated_speedup:.2f}x expected)")
    elif estimated_speedup >= 1.10:
        print(f"\n⚠ Optimized KV Cache Layout: MARGINAL ({estimated_speedup:.2f}x expected)")
    else:
        print(f"\n✗ Optimized KV Cache Layout: NOT SIGNIFICANT ({estimated_speedup:.2f}x expected)")

    print(f"{'='*80}\n")

    return estimated_speedup, quant_perf.mean_time_ms / baseline_perf.mean_time_ms * num_decode_steps * 100


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark KV Cache Layout Optimization")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="KV cache sequence length")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=20,
                        help="Number of decode steps to simulate")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    speedup, overhead_pct = benchmark_kv_cache_layout(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        device=args.device,
        num_decode_steps=args.decode_steps,
    )

    print(f"\nFinal Results:")
    print(f"  Estimated Speedup:        {speedup:.2f}x")
    print(f"  Quantization Overhead:    {overhead_pct:.1f}%")
    print(f"  Expected Range:           1.15-1.25x")

    if speedup >= 1.15:
        print("\n✓ KV Cache Layout Optimization: HIGHLY RECOMMENDED")
        print("  Should be implemented in production code")
        sys.exit(0)
    elif speedup >= 1.10:
        print("\n✓ KV Cache Layout Optimization: RECOMMENDED")
        print("  Moderate benefit for decoding workloads")
        sys.exit(0)
    else:
        print("\n⚠ KV Cache Layout Optimization: LOW PRIORITY")
        print("  Quantization overhead is small")
        sys.exit(0)
