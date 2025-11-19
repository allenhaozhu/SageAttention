"""
SageAttention Optimization Validation Suite

This package provides benchmarks and validation tools for testing
various optimizations to SageAttention.

Available benchmarks:
- bench_l2_cache: L2 cache persistence optimization
- bench_cuda_graphs: CUDA graph integration
- bench_adaptive_quant: Adaptive quantization granularity
- bench_kv_cache_layout: Optimized KV cache layout
- bench_kernel_fusion: Kernel fusion analysis

Main entry point:
- run_all_validations: Run all benchmarks and generate comprehensive report

Copyright (c) 2024 by SageAttention team.
Licensed under the Apache License, Version 2.0
"""

from .validation_framework import (
    ValidationFramework,
    PerformanceMetrics,
    AccuracyMetrics,
    BenchmarkResult,
    compute_attention_flops,
)

__all__ = [
    "ValidationFramework",
    "PerformanceMetrics",
    "AccuracyMetrics",
    "BenchmarkResult",
    "compute_attention_flops",
]
