"""
Benchmark for CUDA Graph Integration

Tests the impact of CUDA graphs on attention latency.
CUDA graphs eliminate kernel launch overhead, which can be significant
for small batch sizes and low-latency inference.

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


def benchmark_cuda_graphs(
    batch_size: int = 1,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 128,
    is_causal: bool = False,
    device: str = "cuda",
):
    """
    Benchmark CUDA graph optimization.

    CUDA graphs capture a sequence of operations and replay them
    with minimal CPU overhead. This is particularly beneficial for:
    - Low latency inference (batch_size=1)
    - Repeated operations with same shapes
    - Reducing kernel launch overhead

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        is_causal: Whether to use causal masking
        device: Device to use
    """
    print(f"\n{'='*80}")
    print(f"CUDA Graphs Optimization Benchmark")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Num Heads:      {num_heads}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Head Dim:       {head_dim}")
    print(f"  Causal:         {is_causal}")
    print(f"{'='*80}\n")

    # Initialize framework
    framework = ValidationFramework(
        device=device,
        num_warmup=10,
        num_runs=100,
    )

    # Create static inputs (CUDA graphs require static shapes)
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float16, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # FLOPs calculation
    flops = compute_attention_flops(batch_size, num_heads, seq_len, head_dim, is_causal)

    # Baseline: regular execution
    def baseline_attention():
        return sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)

    # Optimized: CUDA graph execution
    # First, check if CUDA graphs are supported
    cuda_version = torch.version.cuda
    print(f"CUDA Version: {cuda_version}")

    if cuda_version is None:
        print("⚠ Warning: CUDA not available, skipping CUDA graphs test")
        return 1.0

    # Capture CUDA graph
    print("Capturing CUDA graph...")

    # Create placeholder for output
    static_output = None

    # Warmup before capture
    for _ in range(3):
        _ = sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)
    torch.cuda.synchronize()

    # Capture graph
    try:
        graph = torch.cuda.CUDAGraph()

        # Important: Use the same stream for graph capture and replay
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                static_output = sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)

        torch.cuda.synchronize()
        print("✓ CUDA graph captured successfully")

        def optimized_attention_graph():
            graph.replay()
            return static_output

    except Exception as e:
        print(f"⚠ Warning: CUDA graph capture failed: {e}")
        print("  This may be due to dynamic operations in the kernel")
        print("  Falling back to baseline for comparison")

        def optimized_attention_graph():
            return baseline_attention()

    # Benchmark baseline
    print("\nMeasuring baseline performance...")
    baseline_perf = framework.measure_performance(
        baseline_attention,
        args=(),
        kwargs={},
        name="baseline",
        flops_per_call=flops,
    )

    # Benchmark CUDA graphs
    print("\nMeasuring CUDA graph performance...")
    opt_perf = framework.measure_performance(
        optimized_attention_graph,
        args=(),
        kwargs={},
        name="cuda_graph",
        flops_per_call=flops,
    )

    # Measure accuracy
    output_baseline = baseline_attention()
    output_graph = optimized_attention_graph()

    # Handle tuple outputs
    if isinstance(output_baseline, tuple):
        output_baseline = output_baseline[0]
    if isinstance(output_graph, tuple):
        output_graph = output_graph[0]

    accuracy = framework.measure_accuracy(output_graph, output_baseline, atol=1e-5, rtol=1e-5)

    # Calculate speedup
    speedup = baseline_perf.mean_time_ms / opt_perf.mean_time_ms

    print(f"\n{'='*80}")
    print(f"CUDA Graphs Optimization Results")
    print(f"{'='*80}")
    print(f"Baseline:   {baseline_perf.mean_time_ms:.3f} ms (±{baseline_perf.std_time_ms:.3f})")
    print(f"CUDAGraph:  {opt_perf.mean_time_ms:.3f} ms (±{opt_perf.std_time_ms:.3f})")
    print(f"Speedup:    {speedup:.2f}x ({(speedup-1)*100:.1f}% improvement)")
    print(f"Expected:   1.25-1.35x (25-35% latency reduction)")
    print(f"\nAccuracy:")
    print(f"  Max Error:  {accuracy.max_abs_diff:.8f}")
    print(f"  Cos Sim:    {accuracy.cosine_similarity:.8f}")
    print(f"  Exact:      {accuracy.matches_exactly}")

    if speedup >= 1.20:
        print(f"\nStatus:     ✓ SIGNIFICANT IMPROVEMENT ({(speedup-1)*100:.1f}%)")
    elif speedup >= 1.10:
        print(f"\nStatus:     ✓ MODERATE IMPROVEMENT ({(speedup-1)*100:.1f}%)")
    elif speedup >= 1.05:
        print(f"\nStatus:     ⚠ MINOR IMPROVEMENT ({(speedup-1)*100:.1f}%)")
    else:
        print(f"\nStatus:     ⚠ MINIMAL/NO IMPROVEMENT")
        print(f"            (Graph capture may have failed or kernel has high overhead)")

    print(f"{'='*80}\n")

    # Note about when CUDA graphs help most
    print("Note: CUDA graphs provide the most benefit for:")
    print("  - Small batch sizes (batch_size=1)")
    print("  - Repeated inference with same shapes")
    print("  - Latency-critical applications")
    print("  - When kernel execution time is comparable to launch overhead")
    print()

    return speedup


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark CUDA Graphs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (smaller benefits more from graphs)")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    speedup = benchmark_cuda_graphs(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        is_causal=args.causal,
        device=args.device,
    )

    # Print summary
    print(f"\nFinal Speedup: {speedup:.2f}x")
    print(f"Expected Range: 1.25-1.35x (25-35%)")

    if speedup >= 1.20:
        print("✓ CUDA Graphs: HIGHLY EFFECTIVE")
        sys.exit(0)
    elif speedup >= 1.10:
        print("✓ CUDA Graphs: MODERATELY EFFECTIVE")
        sys.exit(0)
    elif speedup >= 1.05:
        print("⚠ CUDA Graphs: MARGINALLY EFFECTIVE")
        sys.exit(0)
    else:
        print("✗ CUDA Graphs: NOT EFFECTIVE")
        print("  (May not be supported or kernel overhead is low)")
        sys.exit(1)
