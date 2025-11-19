"""
Adaptive Quantization Granularity for SageAttention

Dynamically selects quantization granularity based on activation distribution
to optimize the accuracy/performance trade-off.

Copyright (c) 2024 by SageAttention team.
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Literal, Optional
from .quant import per_block_int8, per_warp_int8


def compute_activation_statistics(
    tensor: torch.Tensor,
    dim: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute statistics for activation distribution.

    Args:
        tensor: Input tensor
        dim: Dimension along which to compute statistics

    Returns:
        (mean, std, coefficient_of_variation) tuple
    """
    mean = tensor.abs().mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    cv = std / (mean + 1e-8)  # Coefficient of variation
    return mean, std, cv


def detect_outliers(
    tensor: torch.Tensor,
    threshold: float = 3.0,
    dim: int = -1
) -> torch.Tensor:
    """
    Detect outliers using z-score method.

    Args:
        tensor: Input tensor
        threshold: Z-score threshold (default: 3.0 sigma)
        dim: Dimension for statistics

    Returns:
        Boolean mask indicating outliers
    """
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    z_score = (tensor - mean) / (std + 1e-8)
    outliers = z_score.abs() > threshold
    return outliers


def adaptive_quantization_granularity(
    tensor: torch.Tensor,
    cv_threshold_fine: float = 0.5,
    cv_threshold_coarse: float = 0.2,
    outlier_ratio_threshold: float = 0.05,
) -> Literal["per_thread", "per_warp", "per_block"]:
    """
    Automatically select quantization granularity based on tensor statistics.

    Strategy:
    - High variance (CV > 0.5) or many outliers -> per_thread (finest)
    - Medium variance (0.2 < CV < 0.5) -> per_warp (balanced)
    - Low variance (CV < 0.2) -> per_block (fastest)

    Args:
        tensor: Input tensor to quantize
        cv_threshold_fine: CV threshold for per_thread quantization
        cv_threshold_coarse: CV threshold for per_block quantization
        outlier_ratio_threshold: Ratio of outliers to trigger fine quantization

    Returns:
        Recommended granularity: "per_thread", "per_warp", or "per_block"
    """
    # Compute coefficient of variation along head dimension
    _, _, cv = compute_activation_statistics(tensor, dim=-1)
    mean_cv = cv.mean().item()

    # Detect outliers
    outliers = detect_outliers(tensor, threshold=3.0, dim=-1)
    outlier_ratio = outliers.float().mean().item()

    # Decision logic
    if mean_cv > cv_threshold_fine or outlier_ratio > outlier_ratio_threshold:
        # High variance or many outliers -> finest granularity
        return "per_thread"
    elif mean_cv > cv_threshold_coarse:
        # Medium variance -> balanced granularity
        return "per_warp"
    else:
        # Low variance -> coarsest granularity (fastest)
        return "per_block"


@torch.jit.script
def adaptive_per_block_int8(
    x: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-block INT8 quantization with adaptive scaling.

    Args:
        x: Input tensor (BF16/FP16)
        block_size: Block size for quantization

    Returns:
        (quantized_tensor, scale_factors)
    """
    orig_shape = x.shape
    seq_len = orig_shape[-2]
    head_dim = orig_shape[-1]

    # Reshape to blocks
    num_blocks = (seq_len + block_size - 1) // block_size
    padding = num_blocks * block_size - seq_len

    if padding > 0:
        x = F.pad(x, (0, 0, 0, padding))

    # Reshape: [..., num_blocks, block_size, head_dim]
    x_blocked = x.reshape(*orig_shape[:-2], num_blocks, block_size, head_dim)

    # Compute scale per block (max absolute value)
    scale = x_blocked.abs().max(dim=-2, keepdim=True).values.max(dim=-1, keepdim=True).values
    scale = scale / 127.0 + 1e-8

    # Quantize
    x_int8 = (x_blocked / scale).round().clamp(-127, 127).to(torch.int8)

    # Reshape back
    x_int8 = x_int8.reshape(*orig_shape[:-2], num_blocks * block_size, head_dim)
    scale = scale.reshape(*orig_shape[:-2], num_blocks, 1)

    # Remove padding
    if padding > 0:
        x_int8 = x_int8[..., :seq_len, :]

    return x_int8, scale


def adaptive_int8_quantization(
    q: torch.Tensor,
    k: torch.Tensor,
    quantization_backend: str = "cuda",
    auto_granularity: bool = True,
    q_granularity: Optional[str] = None,
    k_granularity: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
    """
    Adaptive INT8 quantization for Q and K tensors.

    Args:
        q: Query tensor
        k: Key tensor
        quantization_backend: "cuda" or "triton"
        auto_granularity: Automatically select granularity based on statistics
        q_granularity: Force specific granularity for Q (overrides auto)
        k_granularity: Force specific granularity for K (overrides auto)

    Returns:
        (q_int8, k_int8, q_scale, k_scale, q_granularity_used, k_granularity_used)

    Example:
        >>> q = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device='cuda')
        >>> k = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device='cuda')
        >>> q_i8, k_i8, q_s, k_s, q_gran, k_gran = adaptive_int8_quantization(q, k)
        >>> print(f"Selected granularities: Q={q_gran}, K={k_gran}")
    """
    # Determine granularity for Q
    if q_granularity is None and auto_granularity:
        q_granularity = adaptive_quantization_granularity(q)
    elif q_granularity is None:
        q_granularity = "per_block"  # Default

    # Determine granularity for K
    if k_granularity is None and auto_granularity:
        k_granularity = adaptive_quantization_granularity(k)
    elif k_granularity is None:
        k_granularity = "per_block"  # Default

    # Quantize Q
    if quantization_backend == "cuda":
        if q_granularity == "per_warp":
            from .quant import per_warp_int8 as per_warp_int8_cuda
            q_int8, q_scale, _ = per_warp_int8_cuda(q)
        elif q_granularity == "per_thread":
            # Per-thread quantization (finest granularity)
            # Currently not implemented in CUDA, fall back to per_warp
            from .quant import per_warp_int8 as per_warp_int8_cuda
            q_int8, q_scale, _ = per_warp_int8_cuda(q)
            q_granularity = "per_warp"  # Update actual granularity used
        else:  # per_block
            q_int8, q_scale, _ = per_block_int8(q)
    else:  # triton
        from .triton.quant_per_block import per_block_int8 as per_block_int8_triton
        q_int8, q_scale = per_block_int8_triton(q)
        q_granularity = "per_block"  # Triton backend only supports per_block

    # Quantize K
    if quantization_backend == "cuda":
        if k_granularity == "per_warp":
            from .quant import per_warp_int8 as per_warp_int8_cuda
            k_int8, k_scale, _ = per_warp_int8_cuda(k)
        elif k_granularity == "per_thread":
            # Per-thread quantization
            from .quant import per_warp_int8 as per_warp_int8_cuda
            k_int8, k_scale, _ = per_warp_int8_cuda(k)
            k_granularity = "per_warp"  # Update actual granularity used
        else:  # per_block
            k_int8, k_scale, _ = per_block_int8(k)
    else:  # triton
        from .triton.quant_per_block import per_block_int8 as per_block_int8_triton
        k_int8, k_scale = per_block_int8_triton(k)
        k_granularity = "per_block"

    return q_int8, k_int8, q_scale, k_scale, q_granularity, k_granularity


class AdaptiveQuantConfig:
    """
    Configuration for adaptive quantization.

    This allows fine-tuning the adaptation strategy for different models.
    """

    def __init__(
        self,
        enable_adaptive: bool = True,
        cv_threshold_fine: float = 0.5,
        cv_threshold_coarse: float = 0.2,
        outlier_ratio_threshold: float = 0.05,
        prefer_accuracy: bool = False,
        prefer_speed: bool = False,
    ):
        """
        Args:
            enable_adaptive: Enable adaptive granularity selection
            cv_threshold_fine: CV threshold for finest granularity
            cv_threshold_coarse: CV threshold for coarsest granularity
            outlier_ratio_threshold: Outlier ratio threshold
            prefer_accuracy: Bias towards finer granularity
            prefer_speed: Bias towards coarser granularity
        """
        self.enable_adaptive = enable_adaptive
        self.cv_threshold_fine = cv_threshold_fine
        self.cv_threshold_coarse = cv_threshold_coarse
        self.outlier_ratio_threshold = outlier_ratio_threshold

        # Adjust thresholds based on preferences
        if prefer_accuracy and not prefer_speed:
            self.cv_threshold_fine *= 0.7
            self.cv_threshold_coarse *= 0.7
        elif prefer_speed and not prefer_accuracy:
            self.cv_threshold_fine *= 1.3
            self.cv_threshold_coarse *= 1.3


# Global configuration
_global_adaptive_config = AdaptiveQuantConfig()


def set_adaptive_quant_config(config: AdaptiveQuantConfig) -> None:
    """Set global adaptive quantization configuration."""
    global _global_adaptive_config
    _global_adaptive_config = config


def get_adaptive_quant_config() -> AdaptiveQuantConfig:
    """Get global adaptive quantization configuration."""
    return _global_adaptive_config


# Profiling utilities
class QuantizationProfiler:
    """
    Profile quantization granularity decisions and their impact.

    Useful for understanding when adaptive quantization is beneficial.
    """

    def __init__(self):
        self.decisions = {
            "per_thread": 0,
            "per_warp": 0,
            "per_block": 0,
        }
        self.total_calls = 0

    def record_decision(self, granularity: str) -> None:
        """Record a granularity decision."""
        self.decisions[granularity] += 1
        self.total_calls += 1

    def get_statistics(self) -> dict:
        """Get profiling statistics."""
        if self.total_calls == 0:
            return {k: 0.0 for k in self.decisions}

        return {
            k: v / self.total_calls
            for k, v in self.decisions.items()
        }

    def reset(self) -> None:
        """Reset profiler."""
        self.decisions = {k: 0 for k in self.decisions}
        self.total_calls = 0

    def __str__(self) -> str:
        stats = self.get_statistics()
        return (
            f"Quantization Granularity Profile:\n"
            f"  per_thread: {stats['per_thread']:.1%}\n"
            f"  per_warp:   {stats['per_warp']:.1%}\n"
            f"  per_block:  {stats['per_block']:.1%}\n"
            f"  Total calls: {self.total_calls}"
        )


# Global profiler instance
_global_profiler = QuantizationProfiler()


def get_quant_profiler() -> QuantizationProfiler:
    """Get global quantization profiler."""
    return _global_profiler
