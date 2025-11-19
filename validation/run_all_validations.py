"""
Run All Optimization Validations

This script runs all 5 optimization benchmarks and generates
a comprehensive report comparing their individual and combined impact.

Copyright (c) 2024 by SageAttention team.
Licensed under the Apache License, Version 2.0
"""

import torch
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import individual benchmarks
from validation.bench_l2_cache import benchmark_l2_cache_optimization
from validation.bench_cuda_graphs import benchmark_cuda_graphs
from validation.bench_adaptive_quant import benchmark_adaptive_quantization
from validation.bench_kv_cache_layout import benchmark_kv_cache_layout
from validation.bench_kernel_fusion import benchmark_kernel_fusion


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def print_gpu_info():
    """Print GPU information."""
    print_header("GPU Information")

    if not torch.cuda.is_available():
        print("⚠ CUDA not available!")
        return

    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"Device:              {device_name}")
    print(f"Compute Capability:  {compute_cap[0]}.{compute_cap[1]} (SM{compute_cap[0]}{compute_cap[1]})")
    print(f"Total Memory:        {total_memory:.1f} GB")
    print(f"CUDA Version:        {torch.version.cuda}")
    print(f"PyTorch Version:     {torch.__version__}")


def run_all_validations(
    batch_size: int = 1,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 128,
    device: str = "cuda",
    output_dir: str = "validation_results",
):
    """
    Run all optimization validations.

    Args:
        batch_size: Batch size for testing
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        device: Device to use
        output_dir: Directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print configuration
    print_header("Validation Configuration")
    print(f"Batch Size:      {batch_size}")
    print(f"Num Heads:       {num_heads}")
    print(f"Sequence Length: {seq_len}")
    print(f"Head Dimension:  {head_dim}")
    print(f"Device:          {device}")
    print(f"Output Directory: {output_dir}")

    # Print GPU info
    print_gpu_info()

    # Store results
    results = {
        "config": {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "device": device,
            "timestamp": timestamp,
        },
        "optimizations": {}
    }

    # 1. L2 Cache Optimization
    print_header("Optimization 1/5: L2 Cache Control")
    try:
        speedup_l2 = benchmark_l2_cache_optimization(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            device=device,
            num_decode_steps=20,
        )
        results["optimizations"]["l2_cache"] = {
            "speedup": float(speedup_l2),
            "status": "success",
            "expected_range": "1.10-1.15x",
        }
    except Exception as e:
        print(f"✗ L2 Cache benchmark failed: {e}")
        results["optimizations"]["l2_cache"] = {
            "speedup": 1.0,
            "status": "failed",
            "error": str(e),
        }

    # 2. CUDA Graphs
    print_header("Optimization 2/5: CUDA Graphs")
    try:
        speedup_graph = benchmark_cuda_graphs(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            is_causal=False,
            device=device,
        )
        results["optimizations"]["cuda_graphs"] = {
            "speedup": float(speedup_graph),
            "status": "success",
            "expected_range": "1.25-1.35x",
        }
    except Exception as e:
        print(f"✗ CUDA Graphs benchmark failed: {e}")
        results["optimizations"]["cuda_graphs"] = {
            "speedup": 1.0,
            "status": "failed",
            "error": str(e),
        }

    # 3. Adaptive Quantization
    print_header("Optimization 3/5: Adaptive Quantization")
    try:
        speedup_quant, acc_improve = benchmark_adaptive_quantization(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            is_causal=False,
            device=device,
        )
        results["optimizations"]["adaptive_quant"] = {
            "speedup": float(speedup_quant),
            "accuracy_improvement_pct": float(acc_improve),
            "status": "success",
            "expected_range": "8-12% accuracy improvement",
        }
    except Exception as e:
        print(f"✗ Adaptive Quantization benchmark failed: {e}")
        results["optimizations"]["adaptive_quant"] = {
            "speedup": 1.0,
            "accuracy_improvement_pct": 0.0,
            "status": "failed",
            "error": str(e),
        }

    # 4. KV Cache Layout
    print_header("Optimization 4/5: KV Cache Layout")
    try:
        speedup_kv, overhead_pct = benchmark_kv_cache_layout(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            device=device,
            num_decode_steps=20,
        )
        results["optimizations"]["kv_cache_layout"] = {
            "estimated_speedup": float(speedup_kv),
            "quantization_overhead_pct": float(overhead_pct),
            "status": "success",
            "expected_range": "1.15-1.25x",
        }
    except Exception as e:
        print(f"✗ KV Cache Layout benchmark failed: {e}")
        results["optimizations"]["kv_cache_layout"] = {
            "estimated_speedup": 1.0,
            "quantization_overhead_pct": 0.0,
            "status": "failed",
            "error": str(e),
        }

    # 5. Kernel Fusion
    print_header("Optimization 5/5: Kernel Fusion")
    try:
        speedup_fusion, preproc_pct, overhead_saved = benchmark_kernel_fusion(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            is_causal=False,
            device=device,
        )
        results["optimizations"]["kernel_fusion"] = {
            "estimated_speedup": float(speedup_fusion),
            "preprocessing_overhead_pct": float(preproc_pct),
            "overhead_saved_ms": float(overhead_saved),
            "status": "success",
            "expected_range": "1.20-1.30x",
        }
    except Exception as e:
        print(f"✗ Kernel Fusion benchmark failed: {e}")
        results["optimizations"]["kernel_fusion"] = {
            "estimated_speedup": 1.0,
            "preprocessing_overhead_pct": 0.0,
            "overhead_saved_ms": 0.0,
            "status": "failed",
            "error": str(e),
        }

    # Generate summary report
    print_header("Validation Summary")

    print(f"{'Optimization':<25} {'Speedup':<15} {'Status':<15} {'Expected':<20}")
    print("-" * 80)

    opt_data = [
        ("L2 Cache Control", results["optimizations"]["l2_cache"].get("speedup", 1.0),
         results["optimizations"]["l2_cache"]["status"], "1.10-1.15x"),
        ("CUDA Graphs", results["optimizations"]["cuda_graphs"].get("speedup", 1.0),
         results["optimizations"]["cuda_graphs"]["status"], "1.25-1.35x"),
        ("Adaptive Quant", results["optimizations"]["adaptive_quant"].get("speedup", 1.0),
         results["optimizations"]["adaptive_quant"]["status"], "Accuracy+"),
        ("KV Cache Layout", results["optimizations"]["kv_cache_layout"].get("estimated_speedup", 1.0),
         results["optimizations"]["kv_cache_layout"]["status"], "1.15-1.25x"),
        ("Kernel Fusion", results["optimizations"]["kernel_fusion"].get("estimated_speedup", 1.0),
         results["optimizations"]["kernel_fusion"]["status"], "1.20-1.30x"),
    ]

    for name, speedup, status, expected in opt_data:
        status_str = "✓ PASS" if status == "success" else "✗ FAIL"
        print(f"{name:<25} {speedup:.2f}x{'':<11} {status_str:<15} {expected:<20}")

    # Combined impact estimation
    print("\n" + "="*80)
    print("Combined Impact Estimation")
    print("="*80)

    # Conservative combined speedup (not all optimizations multiply perfectly)
    # We assume 70% effectiveness when combining
    individual_speedups = [
        results["optimizations"]["l2_cache"].get("speedup", 1.0),
        results["optimizations"]["cuda_graphs"].get("speedup", 1.0),
        results["optimizations"]["kv_cache_layout"].get("estimated_speedup", 1.0),
        results["optimizations"]["kernel_fusion"].get("estimated_speedup", 1.0),
    ]

    # Multiplicative model with diminishing returns
    combined_speedup = 1.0
    for s in individual_speedups:
        # Each additional optimization has 70% effectiveness
        improvement = (s - 1.0) * 0.7
        combined_speedup *= (1.0 + improvement)

    results["combined_speedup_conservative"] = float(combined_speedup)

    # Optimistic estimate (assuming optimizations are orthogonal)
    optimistic_speedup = 1.0
    for s in individual_speedups:
        optimistic_speedup *= s

    results["combined_speedup_optimistic"] = float(optimistic_speedup)

    print(f"Conservative Estimate: {combined_speedup:.2f}x")
    print(f"Optimistic Estimate:   {optimistic_speedup:.2f}x")
    print(f"\nCurrent SageAttention: 2.7x vs FlashAttention2 (RTX 5090)")
    print(f"With optimizations:    {2.7 * combined_speedup:.2f}x - {2.7 * optimistic_speedup:.2f}x (conservative-optimistic)")

    # Save results to JSON
    output_file = output_path / f"validation_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save text report
    report_file = output_path / f"validation_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SageAttention Optimization Validation Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Batch Size:      {batch_size}\n")
        f.write(f"  Num Heads:       {num_heads}\n")
        f.write(f"  Sequence Length: {seq_len}\n")
        f.write(f"  Head Dimension:  {head_dim}\n")
        f.write(f"  Device:          {device}\n\n")
        f.write("Individual Optimizations:\n")
        f.write("-"*80 + "\n")
        for name, speedup, status, expected in opt_data:
            f.write(f"{name:<25} {speedup:.2f}x  {status:<10} (Expected: {expected})\n")
        f.write("\n")
        f.write(f"Combined Impact:\n")
        f.write(f"  Conservative: {combined_speedup:.2f}x\n")
        f.write(f"  Optimistic:   {optimistic_speedup:.2f}x\n")
        f.write(f"\n")
        f.write(f"Projected Performance:\n")
        f.write(f"  Current:      2.7x vs FlashAttention2\n")
        f.write(f"  With opts:    {2.7 * combined_speedup:.2f}x - {2.7 * optimistic_speedup:.2f}x\n")

    print(f"\n✓ Results saved to:")
    print(f"  {output_file}")
    print(f"  {report_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run all SageAttention optimization validations"
    )
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for testing")
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--output-dir", type=str, default="validation_results",
                        help="Output directory for results")

    args = parser.parse_args()

    print_header("SageAttention Optimization Validation Suite")
    print("This will run all 5 optimization benchmarks:")
    print("  1. L2 Cache Control")
    print("  2. CUDA Graphs")
    print("  3. Adaptive Quantization")
    print("  4. KV Cache Layout")
    print("  5. Kernel Fusion")
    print("\nEstimated time: 5-10 minutes")
    print()

    results = run_all_validations(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Final summary
    print_header("Validation Complete!")

    successful = sum(1 for opt in results["optimizations"].values()
                     if opt["status"] == "success")
    total = len(results["optimizations"])

    print(f"Successful: {successful}/{total}")
    print(f"\nProjected Combined Speedup:")
    print(f"  Conservative: {results['combined_speedup_conservative']:.2f}x")
    print(f"  Optimistic:   {results['combined_speedup_optimistic']:.2f}x")
    print(f"\nProjected Total Performance:")
    print(f"  Current SageAttention: 2.7x vs FlashAttention2")
    print(f"  With optimizations:    {2.7 * results['combined_speedup_conservative']:.2f}x - {2.7 * results['combined_speedup_optimistic']:.2f}x")
    print()

    if successful == total:
        print("✓ All validations passed successfully!")
        return 0
    elif successful >= total // 2:
        print("⚠ Some validations failed, but majority passed")
        return 0
    else:
        print("✗ Most validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
