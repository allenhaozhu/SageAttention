"""
Validation Framework for SageAttention Optimizations

This module provides utilities to measure both performance and accuracy
of different optimizations against baseline implementations.

Copyright (c) 2024 by SageAttention team.
Licensed under the Apache License, Version 2.0
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single benchmark run."""
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_tflops: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    num_runs: int


@dataclass
class AccuracyMetrics:
    """Accuracy metrics comparing against reference."""
    max_abs_diff: float
    mean_abs_diff: float
    relative_error: float
    cosine_similarity: float
    matches_exactly: bool
    atol: float = 1e-2
    rtol: float = 1e-2


@dataclass
class BenchmarkResult:
    """Combined result for a single optimization."""
    optimization_name: str
    performance: PerformanceMetrics
    accuracy: AccuracyMetrics
    config: Dict[str, Any] = field(default_factory=dict)
    speedup_vs_baseline: float = 1.0
    passed: bool = True
    error_message: Optional[str] = None


class ValidationFramework:
    """
    Framework for validating SageAttention optimizations.

    Measures:
    1. Performance (latency, throughput, memory)
    2. Accuracy (vs FP16 reference)
    3. Numerical stability
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        num_warmup: int = 10,
        num_runs: int = 100,
        cache_flush_size_mb: int = 256,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_warmup = num_warmup
        self.num_runs = num_runs

        # L2 cache flushing buffer
        self.cache_flush_buffer = torch.empty(
            cache_flush_size_mb * 1024 * 1024 // 4,
            dtype=torch.int32,
            device=self.device
        )

        self.results: List[BenchmarkResult] = []

    def flush_cache(self):
        """Flush L2 cache to ensure fair benchmarking."""
        self.cache_flush_buffer.zero_()
        torch.cuda.synchronize()

    def measure_performance(
        self,
        fn: Callable,
        args: Tuple,
        kwargs: Dict,
        name: str = "function",
        compute_tflops: bool = True,
        flops_per_call: Optional[int] = None,
    ) -> PerformanceMetrics:
        """
        Measure performance of a function.

        Args:
            fn: Function to benchmark
            args: Positional arguments
            kwargs: Keyword arguments
            name: Name for logging
            compute_tflops: Whether to compute TFLOPS
            flops_per_call: Number of FLOPs per call (for TFLOPS calculation)

        Returns:
            PerformanceMetrics object
        """
        # Warmup
        for _ in range(self.num_warmup):
            _ = fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        torch.cuda.reset_peak_memory_stats()

        for _ in range(self.num_runs):
            self.flush_cache()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = fn(*args, **kwargs)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # in milliseconds

        times = np.array(times)
        mean_time_ms = float(np.mean(times))
        std_time_ms = float(np.std(times))
        min_time_ms = float(np.min(times))
        max_time_ms = float(np.max(times))

        # Memory stats
        memory_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)

        # TFLOPS calculation
        throughput_tflops = 0.0
        if compute_tflops and flops_per_call is not None:
            throughput_tflops = (flops_per_call / (mean_time_ms / 1000)) / 1e12

        return PerformanceMetrics(
            mean_time_ms=mean_time_ms,
            std_time_ms=std_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            throughput_tflops=throughput_tflops,
            memory_allocated_mb=memory_allocated,
            memory_reserved_mb=memory_reserved,
            num_runs=self.num_runs,
        )

    def measure_accuracy(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ) -> AccuracyMetrics:
        """
        Measure accuracy against reference implementation.

        Args:
            output: Output from optimized kernel
            reference: Output from reference implementation
            atol: Absolute tolerance
            rtol: Relative tolerance

        Returns:
            AccuracyMetrics object
        """
        # Convert to float32 for accurate comparison
        output_f32 = output.float()
        reference_f32 = reference.float()

        # Absolute difference
        abs_diff = torch.abs(output_f32 - reference_f32)
        max_abs_diff = float(abs_diff.max())
        mean_abs_diff = float(abs_diff.mean())

        # Relative error
        relative_error = float((abs_diff / (torch.abs(reference_f32) + 1e-8)).mean())

        # Cosine similarity
        output_flat = output_f32.flatten()
        reference_flat = reference_f32.flatten()
        cosine_sim = float(
            torch.nn.functional.cosine_similarity(
                output_flat.unsqueeze(0),
                reference_flat.unsqueeze(0)
            )
        )

        # Exact match check
        matches_exactly = bool(torch.allclose(output, reference, atol=atol, rtol=rtol))

        return AccuracyMetrics(
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            relative_error=relative_error,
            cosine_similarity=cosine_sim,
            matches_exactly=matches_exactly,
            atol=atol,
            rtol=rtol,
        )

    def validate_optimization(
        self,
        optimization_fn: Callable,
        baseline_fn: Callable,
        args: Tuple,
        kwargs: Dict,
        name: str,
        config: Optional[Dict] = None,
        flops_per_call: Optional[int] = None,
        accuracy_atol: float = 1e-2,
        accuracy_rtol: float = 1e-2,
    ) -> BenchmarkResult:
        """
        Validate a single optimization against baseline.

        Args:
            optimization_fn: Optimized implementation
            baseline_fn: Baseline implementation
            args: Input arguments
            kwargs: Keyword arguments
            name: Optimization name
            config: Configuration dict
            flops_per_call: FLOPs for throughput calculation
            accuracy_atol: Absolute tolerance for accuracy check
            accuracy_rtol: Relative tolerance for accuracy check

        Returns:
            BenchmarkResult object
        """
        config = config or {}

        try:
            # Measure baseline performance
            print(f"Benchmarking baseline for {name}...")
            baseline_perf = self.measure_performance(
                baseline_fn, args, kwargs,
                name=f"{name}_baseline",
                flops_per_call=flops_per_call,
            )

            # Measure optimized performance
            print(f"Benchmarking optimized for {name}...")
            opt_perf = self.measure_performance(
                optimization_fn, args, kwargs,
                name=f"{name}_optimized",
                flops_per_call=flops_per_call,
            )

            # Measure accuracy
            print(f"Measuring accuracy for {name}...")
            with torch.no_grad():
                output_baseline = baseline_fn(*args, **kwargs)
                output_optimized = optimization_fn(*args, **kwargs)

            # Handle tuple outputs (e.g., attention + lse)
            if isinstance(output_baseline, tuple):
                output_baseline = output_baseline[0]
            if isinstance(output_optimized, tuple):
                output_optimized = output_optimized[0]

            accuracy = self.measure_accuracy(
                output_optimized,
                output_baseline,
                atol=accuracy_atol,
                rtol=accuracy_rtol,
            )

            # Calculate speedup
            speedup = baseline_perf.mean_time_ms / opt_perf.mean_time_ms

            # Check if passed
            passed = accuracy.matches_exactly or (
                accuracy.max_abs_diff < accuracy_atol and
                accuracy.cosine_similarity > 0.99
            )

            result = BenchmarkResult(
                optimization_name=name,
                performance=opt_perf,
                accuracy=accuracy,
                config=config,
                speedup_vs_baseline=speedup,
                passed=passed,
            )

            self.results.append(result)

            # Print summary
            print(f"\n{'='*60}")
            print(f"Results for {name}")
            print(f"{'='*60}")
            print(f"Baseline:   {baseline_perf.mean_time_ms:.3f} ms (±{baseline_perf.std_time_ms:.3f})")
            print(f"Optimized:  {opt_perf.mean_time_ms:.3f} ms (±{opt_perf.std_time_ms:.3f})")
            print(f"Speedup:    {speedup:.2f}x")
            print(f"Max Error:  {accuracy.max_abs_diff:.6f}")
            print(f"Mean Error: {accuracy.mean_abs_diff:.6f}")
            print(f"Cos Sim:    {accuracy.cosine_similarity:.6f}")
            print(f"Status:     {'✓ PASSED' if passed else '✗ FAILED'}")
            print(f"{'='*60}\n")

            return result

        except Exception as e:
            print(f"\n✗ Error validating {name}: {e}")
            result = BenchmarkResult(
                optimization_name=name,
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0),
                accuracy=AccuracyMetrics(0, 0, 0, 0, False),
                config=config,
                speedup_vs_baseline=0.0,
                passed=False,
                error_message=str(e),
            )
            self.results.append(result)
            return result

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a summary report of all validations.

        Args:
            output_file: Optional path to save JSON report

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("\n" + "="*80)
        report_lines.append("SageAttention Optimization Validation Report")
        report_lines.append("="*80 + "\n")

        # Summary table
        report_lines.append(f"{'Optimization':<30} {'Speedup':<12} {'Max Error':<12} {'Status':<10}")
        report_lines.append("-" * 80)

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report_lines.append(
                f"{result.optimization_name:<30} "
                f"{result.speedup_vs_baseline:.2f}x{'':<8} "
                f"{result.accuracy.max_abs_diff:.6f}{'':<4} "
                f"{status:<10}"
            )

        report_lines.append("\n" + "="*80)

        # Detailed results
        report_lines.append("\nDetailed Results:\n")

        for result in self.results:
            report_lines.append(f"\n{result.optimization_name}")
            report_lines.append("-" * 40)
            report_lines.append(f"  Performance:")
            report_lines.append(f"    Mean Time:     {result.performance.mean_time_ms:.3f} ms")
            report_lines.append(f"    Std Dev:       {result.performance.std_time_ms:.3f} ms")
            report_lines.append(f"    Throughput:    {result.performance.throughput_tflops:.2f} TFLOPS")
            report_lines.append(f"    Memory (MB):   {result.performance.memory_allocated_mb:.1f}")
            report_lines.append(f"  Accuracy:")
            report_lines.append(f"    Max Abs Diff:  {result.accuracy.max_abs_diff:.6f}")
            report_lines.append(f"    Mean Abs Diff: {result.accuracy.mean_abs_diff:.6f}")
            report_lines.append(f"    Rel Error:     {result.accuracy.relative_error:.6f}")
            report_lines.append(f"    Cosine Sim:    {result.accuracy.cosine_similarity:.6f}")
            report_lines.append(f"  Speedup:         {result.speedup_vs_baseline:.2f}x")
            report_lines.append(f"  Status:          {'✓ PASSED' if result.passed else '✗ FAILED'}")
            if result.error_message:
                report_lines.append(f"  Error:           {result.error_message}")

        report_lines.append("\n" + "="*80 + "\n")

        report = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            # Save text report
            with open(output_file, 'w') as f:
                f.write(report)

            # Save JSON report
            json_file = output_file.replace('.txt', '.json')
            json_data = {
                'results': [
                    {
                        'optimization': r.optimization_name,
                        'speedup': r.speedup_vs_baseline,
                        'passed': r.passed,
                        'performance': {
                            'mean_time_ms': r.performance.mean_time_ms,
                            'std_time_ms': r.performance.std_time_ms,
                            'throughput_tflops': r.performance.throughput_tflops,
                            'memory_mb': r.performance.memory_allocated_mb,
                        },
                        'accuracy': {
                            'max_abs_diff': r.accuracy.max_abs_diff,
                            'mean_abs_diff': r.accuracy.mean_abs_diff,
                            'cosine_similarity': r.accuracy.cosine_similarity,
                        },
                        'config': r.config,
                        'error': r.error_message,
                    }
                    for r in self.results
                ]
            }

            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)

            print(f"\nReports saved to {output_file} and {json_file}")

        return report

    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()


def compute_attention_flops(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool = False,
) -> int:
    """
    Compute FLOPs for attention operation.

    Attention: O = softmax(QK^T / sqrt(d)) @ V

    FLOPs breakdown:
    - QK^T: batch * heads * seq * seq * head_dim * 2 (matmul)
    - Softmax: batch * heads * seq * seq * 5 (exp, sum, div, etc.)
    - PV: batch * heads * seq * seq * head_dim * 2 (matmul)

    For causal attention, approximately half the FLOPs (triangular matrix)
    """
    # QK^T matmul
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim

    # Softmax (approximate)
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len

    # PV matmul
    pv_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim

    total_flops = qk_flops + softmax_flops + pv_flops

    # Causal attention is roughly half (triangular)
    if is_causal:
        total_flops = total_flops // 2

    return int(total_flops)
