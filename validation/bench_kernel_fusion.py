"""
Benchmark for Kernel Fusion

Measures the overhead of separate quantization kernels vs.
a hypothetical fused kernel that combines quantization + smoothing + attention.

This is a simplified benchmark that estimates the benefit of fusion
by measuring individual kernel overheads.

Copyright (c) 2024 by SageAttention team.
Licensed under the Apache License, Version 2.0
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sageattention import sageattn_qk_int8_pv_fp16_cuda
from sageattention.quant import per_block_int8, sub_mean
from validation.validation_framework import ValidationFramework, compute_attention_flops


def benchmark_kernel_fusion(
    batch_size: int = 1,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 128,
    is_causal: bool = False,
    device: str = "cuda",
):
    """
    Benchmark kernel fusion opportunity.

    Current pipeline:
    1. Kernel 1: sub_mean(k) - smoothing
    2. Kernel 2: quantize(q) - INT8 quantization
    3. Kernel 3: quantize(k) - INT8 quantization
    4. Kernel 4: attention(q_int8, k_int8, v)

    Fused kernel would do all in one:
    - Load Q, K, V
    - Smooth K (subtract mean)
    - Quantize Q and K
    - Compute attention
    - Write output

    Benefits:
    - Reduced memory traffic (no intermediate results)
    - Reduced kernel launch overhead
    - Better instruction-level parallelism

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        is_causal: Whether to use causal masking
        device: Device to use
    """
    print(f"\n{'='*80}")
    print(f"Kernel Fusion Benchmark")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Batch Size:      {batch_size}")
    print(f"  Num Heads:       {num_heads}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Head Dim:        {head_dim}")
    print(f"  Causal:          {is_causal}")
    print(f"{'='*80}\n")

    # Initialize framework
    framework = ValidationFramework(
        device=device,
        num_warmup=5,
        num_runs=100,
    )

    # Create inputs
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float16, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # FLOPs calculation
    flops = compute_attention_flops(batch_size, num_heads, seq_len, head_dim, is_causal)

    # Measure individual kernel timings
    print("Measuring individual kernel timings...")
    print("="*60)

    # 1. K smoothing
    def kernel_smooth_k():
        km = k.mean(dim=-2, keepdim=True)
        k_smooth = k - km
        return k_smooth

    smooth_perf = framework.measure_performance(
        kernel_smooth_k,
        args=(),
        kwargs={},
        name="smooth_k",
        compute_tflops=False,
    )
    print(f"1. K Smoothing:      {smooth_perf.mean_time_ms:.3f} ms")

    # 2. Q quantization
    def kernel_quant_q():
        q_int8, q_scale, _ = per_block_int8(q)
        return q_int8, q_scale

    quant_q_perf = framework.measure_performance(
        kernel_quant_q,
        args=(),
        kwargs={},
        name="quant_q",
        compute_tflops=False,
    )
    print(f"2. Q Quantization:   {quant_q_perf.mean_time_ms:.3f} ms")

    # 3. K quantization
    def kernel_quant_k():
        k_smooth = kernel_smooth_k()
        k_int8, k_scale, _ = per_block_int8(k_smooth)
        return k_int8, k_scale

    quant_k_perf = framework.measure_performance(
        kernel_quant_k,
        args=(),
        kwargs={},
        name="quant_k",
        compute_tflops=False,
    )
    print(f"3. K Quantization:   {quant_k_perf.mean_time_ms:.3f} ms")

    # 4. Attention kernel (assume it uses pre-quantized inputs)
    # We'll measure the full sageattn to get the attention kernel time
    def full_attention():
        return sageattn_qk_int8_pv_fp16_cuda(
            q, k, v,
            tensor_layout="HND",
            is_causal=is_causal,
            smooth_k=True,
        )

    full_perf = framework.measure_performance(
        full_attention,
        args=(),
        kwargs={},
        name="full_attention",
        flops_per_call=flops,
    )
    print(f"4. Full Attention:   {full_perf.mean_time_ms:.3f} ms")

    # Calculate preprocessing overhead
    preprocessing_time = smooth_perf.mean_time_ms + quant_q_perf.mean_time_ms + quant_k_perf.mean_time_ms
    preprocessing_pct = (preprocessing_time / full_perf.mean_time_ms) * 100

    # Estimate kernel launch overhead (typical: ~2-10μs per kernel)
    # We have 3 preprocessing kernels + 1 attention kernel = 4 launches
    # With fusion, we'd have only 1 launch
    kernel_launch_overhead_us = 5.0  # conservative estimate
    num_unfused_kernels = 4
    num_fused_kernels = 1
    launch_overhead_ms = (num_unfused_kernels - num_fused_kernels) * kernel_launch_overhead_us / 1000.0

    # Estimate memory traffic saved
    # Unfused: Write Q_int8, K_int8 to memory, then read back
    # Fused: Keep in registers/shared memory
    tensor_size_mb = (batch_size * num_heads * seq_len * head_dim * 2) / (1024 ** 2)  # FP16
    memory_traffic_saved_mb = tensor_size_mb * 2  # Q_int8 and K_int8 write+read

    # Estimate bandwidth saved (typical GPU: 1-2 TB/s)
    # Assume 1.5 TB/s = 1536 GB/s
    bandwidth_gbs = 1536
    bandwidth_saved_ms = (memory_traffic_saved_mb * 1024) / bandwidth_gbs

    total_overhead_saved = launch_overhead_ms + bandwidth_saved_ms

    # Estimated fused performance
    estimated_fused_time = full_perf.mean_time_ms - total_overhead_saved
    estimated_speedup = full_perf.mean_time_ms / estimated_fused_time

    print(f"\n{'='*80}")
    print(f"Kernel Fusion Analysis")
    print(f"{'='*80}")
    print(f"Current (Unfused) Pipeline:")
    print(f"  1. K Smoothing:      {smooth_perf.mean_time_ms:.3f} ms")
    print(f"  2. Q Quantization:   {quant_q_perf.mean_time_ms:.3f} ms")
    print(f"  3. K Quantization:   {quant_k_perf.mean_time_ms:.3f} ms")
    print(f"  4. Attention:        {full_perf.mean_time_ms - preprocessing_time:.3f} ms (estimated)")
    print(f"  Total:               {full_perf.mean_time_ms:.3f} ms")
    print(f"\nPreprocessing Overhead:")
    print(f"  Time:                {preprocessing_time:.3f} ms")
    print(f"  Percentage:          {preprocessing_pct:.1f}%")
    print(f"\nFusion Savings:")
    print(f"  Kernel Launches:     {launch_overhead_ms:.3f} ms")
    print(f"  Memory Traffic:      {bandwidth_saved_ms:.3f} ms ({memory_traffic_saved_mb:.1f} MB saved)")
    print(f"  Total Saved:         {total_overhead_saved:.3f} ms")
    print(f"\nEstimated Fused Performance:")
    print(f"  Time:                {estimated_fused_time:.3f} ms")
    print(f"  Speedup:             {estimated_speedup:.2f}x")
    print(f"  Expected:            1.20-1.30x (20-30% bandwidth reduction)")

    print(f"\n{'='*60}")
    print("Implementation Roadmap:")
    print("="*60)
    print("To implement fused kernel:")
    print("1. Combine all preprocessing + attention into single CUDA kernel")
    print("2. Stream data through shared memory")
    print("3. Keep intermediate results in registers")
    print("4. Benefits:")
    print(f"   - Eliminate {memory_traffic_saved_mb:.1f} MB of memory traffic")
    print(f"   - Save {num_unfused_kernels - num_fused_kernels} kernel launches")
    print(f"   - Achieve ~{estimated_speedup:.2f}x speedup")
    print("="*60)

    if estimated_speedup >= 1.20:
        print(f"\n✓ Kernel Fusion: HIGHLY WORTHWHILE ({estimated_speedup:.2f}x expected)")
    elif estimated_speedup >= 1.10:
        print(f"\n✓ Kernel Fusion: WORTHWHILE ({estimated_speedup:.2f}x expected)")
    else:
        print(f"\n⚠ Kernel Fusion: MARGINAL ({estimated_speedup:.2f}x expected)")

    print(f"{'='*80}\n")

    return estimated_speedup, preprocessing_pct, total_overhead_saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Kernel Fusion")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    speedup, preproc_pct, overhead_saved = benchmark_kernel_fusion(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        is_causal=args.causal,
        device=args.device,
    )

    print(f"\nFinal Results:")
    print(f"  Estimated Speedup:        {speedup:.2f}x")
    print(f"  Preprocessing Overhead:   {preproc_pct:.1f}%")
    print(f"  Total Overhead Saved:     {overhead_saved:.3f} ms")
    print(f"  Expected Range:           1.20-1.30x")

    if speedup >= 1.20:
        print("\n✓ Kernel Fusion: HIGHLY RECOMMENDED")
        print("  Should be high priority for implementation")
        sys.exit(0)
    elif speedup >= 1.15:
        print("\n✓ Kernel Fusion: RECOMMENDED")
        print("  Moderate benefit expected")
        sys.exit(0)
    else:
        print("\n⚠ Kernel Fusion: CONSIDER")
        print("  Benefits may vary by workload")
        sys.exit(0)
