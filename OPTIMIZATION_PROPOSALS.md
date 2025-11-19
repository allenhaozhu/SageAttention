# SageAttention Optimization Proposals

## Executive Summary

After analyzing the SageAttention codebase, I've identified **8 high-impact optimization opportunities** that could further accelerate attention computation beyond the current 2.7x speedup over FlashAttention2. These optimizations target different architectural levels: algorithmic improvements, kernel fusion, memory optimization, and hardware-specific features.

**Current State**: SageAttention achieves impressive speedups through:
- INT8 quantization for QK^T computation
- FP8 quantization for PV computation
- Multi-granularity quantization (per-block/warp/thread)
- Two-level accumulation strategy
- Tensor core utilization
- Outlier smoothing techniques

**Performance Baseline**:
- RTX 5090: 560 TOPS, 2.7x faster than FlashAttention2
- RTX 4090: Up to 2x speedup
- H100/H20: Matches FlashAttention3-FP8 speed with better accuracy

---

## Optimization Direction 1: Enhanced Hopper Architecture Utilization ⭐⭐⭐⭐⭐

### Current State
- SM90 kernel (`qk_int_sv_f8_cuda_sm90.cu`) uses TMA (Tensor Memory Accelerator) and WGMMA
- Basic TMA usage for async memory transfers
- WGMMA for matrix multiplication

### Proposed Optimizations

#### 1.1 TMA Multicast for Shared Q/K Across Thread Blocks
**Benefit**: Reduce redundant memory transfers when multiple CTAs need the same Q/K data

```cuda
// Current: Each CTA loads its own copy
load_async_4D(sQ, &tensorMapQ, &barrier_Q, 0, bx * CTA_Q, head_id, batch_id);

// Proposed: Use TMA multicast for K tiles shared across Q blocks
// In non-causal attention, all Q blocks use the same K tiles
cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
```

**Expected Impact**: 15-20% memory bandwidth reduction for non-causal attention

#### 1.2 Warp-Specialized WGMMA with Persistent Threads
**Benefit**: Better instruction-level parallelism and reduced warp scheduling overhead

```cuda
// Current: All warps do the same work pattern
wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);

// Proposed: Specialize warps for different stages
if (warp_idx < 2) {
    // Producer warps: Handle TMA and data movement
    producer_warp_TMA_load(...);
} else {
    // Consumer warps: Focus on WGMMA computation
    consumer_warp_compute(...);
}
```

**Expected Impact**: 10-15% throughput improvement on H100/H200

#### 1.3 Leverage Hopper's FP32 Accumulation for FP8
**Current**: Two-level accumulation with manual FP16->FP32 conversion
**Proposed**: Use native FP32 accumulators in Hopper WGMMA

```cuda
// Hopper supports FP8 x FP8 -> FP32 directly with full precision
// No need for two-level accumulation on SM90+
wgmma::wgmma_f8f8f32<...>(RO, RS_f8, RV);  // Already implemented
// But can be optimized further with better register allocation
```

**Expected Impact**: 5-10% improvement in accuracy-sensitive workloads

---

## Optimization Direction 2: Advanced Quantization Strategies ⭐⭐⭐⭐⭐

### 2.1 Adaptive Quantization Granularity
**Current**: Fixed granularity (per-block, per-warp, or per-thread) chosen at compile time

**Proposed**: Runtime-adaptive granularity based on activation distribution

```python
def adaptive_quantization(x, outlier_threshold=3.0):
    """Choose quantization granularity based on coefficient of variation"""
    # Measure activation variance
    std = x.std(dim=-1, keepdim=True)
    mean = x.abs().mean(dim=-1, keepdim=True)
    cv = std / (mean + 1e-6)

    # High variance -> finer granularity needed
    # Low variance -> coarser granularity is sufficient
    if cv.mean() > outlier_threshold:
        return per_thread_int8(x)  # Finest, more overhead
    elif cv.mean() > outlier_threshold / 2:
        return per_warp_int8(x)    # Medium
    else:
        return per_block_int8(x)   # Coarsest, least overhead
```

**Expected Impact**: 8-12% speedup on mixed workloads with better accuracy

### 2.2 INT4 Quantization for QK^T
**Current**: INT8 for QK^T (already very good)
**Proposed**: INT4 with higher precision scaling

```cuda
// From attn_utils.cuh, INT4 support is already in the enum:
// enum class DataType { kHalf, kInt8, kInt4, kE4M3, kE5M2 };

// Implement INT4 kernel variant
template<...>
__device__ void compute_int4_qk(...) {
    // INT4 x INT4 -> INT32 MMA instruction
    mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInit>(RS, RQ, RK);
    // 2x more elements per instruction vs INT8
}
```

**Benefit**:
- 2x reduction in QK memory footprint
- Potential 30-40% faster QK^T computation
- May need per-warp or per-thread quantization for accuracy

**Trade-off**: Accuracy degradation needs careful evaluation

### 2.3 Hybrid Precision for Different Sequence Lengths
**Current**: Same precision for all sequence lengths
**Proposed**: Use FP16 for short sequences, INT8 for medium, INT4 for very long

```python
def sageattn_adaptive(q, k, v, ...):
    seq_len = q.shape[2]  # Assuming HND layout

    if seq_len < 512:
        # Short sequences: FP16 FlashAttention is competitive
        return flash_attn(q, k, v, ...)
    elif seq_len < 4096:
        # Medium sequences: INT8 optimal
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, ...)
    else:
        # Long sequences: INT4 worth the accuracy trade-off
        return sageattn_qk_int4_pv_fp8_cuda(q, k, v, ...)
```

**Expected Impact**: 15-25% improvement for long-context scenarios (>8K tokens)

---

## Optimization Direction 3: Kernel Fusion and Memory Optimization ⭐⭐⭐⭐

### 3.1 Fuse Quantization + Smoothing + Attention
**Current**: Separate kernels for quantization and attention

```python
# Current pipeline (simplified):
km = k.mean(dim=-1, keepdim=True)  # Kernel 1: Mean reduction
k = k - km                          # Kernel 2: Subtraction
k_int8, k_scale = quantize(k)      # Kernel 3: Quantization
output = attention(q_int8, k_int8, v_fp8)  # Kernel 4: Attention
```

**Proposed**: Single fused kernel

```cuda
__global__ void fused_smooth_quant_attention_kernel(
    half* Q, half* K, half* V, half* O,
    int8_t* workspace_Q, int8_t* workspace_K, int8_t* workspace_V,
    ...) {

    // Stage 1: Stream Q into shared memory, compute statistics, quantize
    __shared__ half sQ_fp16[CTA_Q * head_dim];
    __shared__ int8_t sQ_int8[CTA_Q * head_dim];

    // Load + compute mean + subtract + quantize in single pass
    cooperative_load_smooth_quantize(Q, sQ_fp16, sQ_int8, ...);

    // Stage 2: Attention computation reuses quantized data
    // No need to write back to global memory and reload
    compute_attention(sQ_int8, sK_int8, sV_fp8, O, ...);
}
```

**Expected Impact**:
- 20-30% reduction in memory bandwidth
- 10-15% speedup from reduced kernel launch overhead
- Particularly beneficial for decoding (batch_size=1)

### 3.2 Persistent Kernel Design for Iterative Decoding
**Current**: Launch new kernel for each decoding step
**Proposed**: Single persistent kernel that processes multiple steps

```cuda
__global__ void persistent_decode_attention(
    TokenBuffer* token_buffer,  // Ring buffer for tokens
    KVCache* kv_cache,
    ...) {

    __shared__ volatile bool* continue_flag;

    while (*continue_flag) {
        // Wait for new token
        wait_for_token_available();

        // Process attention for new token
        uint32_t current_pos = token_buffer->pos;
        process_single_token_attention(current_pos, ...);

        // Signal completion
        signal_token_complete();
    }
}
```

**Expected Impact**:
- 40-60% latency reduction for decode steps (batch_size=1)
- Eliminates kernel launch overhead (~10-20μs per launch)
- Critical for real-time inference scenarios

---

## Optimization Direction 4: Optimized Memory Access Patterns ⭐⭐⭐⭐

### 4.1 Improved Shared Memory Bank Conflict Avoidance
**Current**: Uses 128B swizzling
**Analysis**: Bank conflicts still occur in certain configurations

```cuda
// Current swizzling in permuted_smem.cuh
template<SwizzleMode mode, uint32_t stride>
class smem_t {
    // 128B swizzle pattern
};

// Proposed: Adaptive swizzling based on tile size
template<uint32_t CTA_SIZE, uint32_t head_dim>
constexpr SwizzleMode get_optimal_swizzle() {
    // For head_dim=64, 64B swizzle is better
    if (head_dim == 64) return SwizzleMode::k64B;
    // For head_dim=128, 128B swizzle is better
    if (head_dim == 128) return SwizzleMode::k128B;
    // For head_dim=256, consider 256B (if supported)
    if (head_dim == 256) return SwizzleMode::k128B;  // Fall back
}
```

**Expected Impact**: 5-8% reduction in shared memory access latency

### 4.2 Vectorized Global Memory Loads
**Current**: 128-bit loads via `load_128b_async`
**Proposed**: Larger vector loads where possible

```cuda
// For aligned, contiguous data
template<typename T>
__device__ void load_256b_async(uint32_t smem_offset, T* gmem_ptr) {
    // Use 256-bit loads (available on Ampere+)
    // Reduces number of memory transactions by 2x
    uint4 data0 = *reinterpret_cast<uint4*>(gmem_ptr);
    uint4 data1 = *reinterpret_cast<uint4*>(gmem_ptr + 16);
    // Store to shared memory
}
```

**Expected Impact**: 5-10% memory bandwidth improvement

### 4.3 Optimized V Matrix Layout
**Current**: `transpose_pad_permute_cuda(v)` preprocesses V
**Proposed**: Store V in pre-permuted format in KV cache

```python
class OptimizedKVCache:
    def __init__(self, ...):
        # Store K and V in formats optimized for SageAttention
        self.k_int8 = None  # Already quantized
        self.k_scale = None
        self.v_fp8_permuted = None  # Pre-permuted for tensor cores
        self.v_scale = None

    def append(self, k, v):
        # Quantize and permute once during insertion
        self.k_int8, self.k_scale = quantize_k(k)
        self.v_fp8_permuted, self.v_scale = quantize_permute_v(v)
        # No need to reprocess during attention
```

**Expected Impact**:
- Eliminates preprocessing overhead for decoding
- 15-25% speedup for multi-turn conversations

---

## Optimization Direction 5: Attention Sparsity Exploitation ⭐⭐⭐⭐

### 5.1 Block-Sparse Attention with Adaptive Patterns
**Current**: Dense attention computation
**Observation**: Many attention patterns are naturally sparse

**Proposed**: Integrate with SpargeAttn for automatic sparsity detection

```python
def sageattn_with_sparsity(q, k, v, sparsity_threshold=0.1, ...):
    """
    Combine SageAttention with dynamic sparsity detection
    """
    # Quick pass: Estimate attention sparsity with INT8 QK^T
    # (very fast due to INT8)
    q_int8, q_scale = per_block_int8(q)
    k_int8, k_scale = per_block_int8(k)
    scores_int8 = q_int8 @ k_int8.transpose(-2, -1)
    scores_approx = scores_int8.float() * q_scale * k_scale

    # Identify important blocks (top-k or threshold-based)
    block_mask = identify_important_blocks(scores_approx, threshold=sparsity_threshold)

    # Compute attention only on important blocks
    output = sparse_sageattn(q, k, v, block_mask, ...)
    return output
```

**Expected Impact**:
- 2-3x speedup for naturally sparse attention (e.g., local attention)
- Minimal accuracy loss (<1% for most models)
- Scales better to very long contexts (>16K tokens)

### 5.2 Causal Attention Optimization
**Current**: Computes full triangular attention
**Proposed**: Skip computation for masked-out regions earlier

```cuda
// Current: Computes QK^T for all blocks, then masks
const uint32_t num_iterations = div_ceil(
    mask_mode == MaskMode::kCausal ? min(kv_len, (bx + 1) * CTA_Q) : kv_len,
    CTA_K);

// Proposed: More aggressive early termination
// Don't even load K/V blocks that are fully masked
const uint32_t num_iterations = mask_mode == MaskMode::kCausal
    ? max(1, div_ceil(bx * CTA_Q, CTA_K))  // Much tighter bound
    : div_ceil(kv_len, CTA_K);

// Skip softmax computation for zero blocks
if (all_masked(bx, iter, CTA_Q, CTA_K)) {
    continue;  // Skip this iteration entirely
}
```

**Expected Impact**: 15-25% speedup for causal attention at long sequences

---

## Optimization Direction 6: Algorithmic Improvements ⭐⭐⭐

### 6.1 Improved Online Softmax Numerics
**Current**: Standard online softmax with log-sum-exp trick
**Proposed**: Reduce numerical operations with incremental scaling

```cuda
// Current approach (simplified):
m_new = max(m_old, m_current);
d_new = d_old * exp(m_old - m_new) + sum(exp(S - m_new));
O_new = O_old * exp(m_old - m_new) + ...;

// Proposed: Avoid repeated exp() calls
// Precompute and reuse scale factors
float scale_factor = exp(m_old - m_new);
d_new = d_old * scale_factor + sum(exp(S - m_new));
O_new = O_old * scale_factor + ...;  // Reuse scale_factor
```

**Expected Impact**: 3-5% reduction in compute time

### 6.2 Mixed Block Size Strategy
**Current**: Fixed CTA_Q and CTA_K (e.g., 64x64, 128x128)
**Proposed**: Adaptive block sizing based on sequence length

```cuda
template<uint32_t qo_len, uint32_t kv_len>
constexpr auto get_optimal_tile_size() {
    // Short sequences: Larger tiles (more reuse, less overhead)
    if (qo_len < 1024) return TileSize{128, 128};

    // Medium sequences: Balanced tiles
    if (qo_len < 4096) return TileSize{128, 64};

    // Long sequences: Smaller Q tiles (better load balancing)
    return TileSize{64, 128};
}
```

**Expected Impact**: 8-12% improvement across varying sequence lengths

### 6.3 Logarithmic Attention Scaling
**Current**: Linear scaling in attention computation
**Proposed**: Use logarithmic representation for extreme sequence lengths

```cuda
// For very long sequences (>32K), maintain attention in log-space
// Avoids underflow/overflow issues
struct LogSpaceAttention {
    float log_score[...];
    float log_denominator;

    __device__ void update(float new_score) {
        // LogSumExp trick in log space
        log_denominator = log_sum_exp(log_denominator, log(new_score));
    }
};
```

**Expected Impact**: Enables stable attention for >100K token contexts

---

## Optimization Direction 7: Architecture-Specific Tuning ⭐⭐⭐⭐

### 7.1 RTX 5090 / Blackwell Optimization
**Recent Addition**: B200 compile support added (commit `9eefd87`)

**Proposed Optimizations**:

```cuda
// Blackwell (SM100/120) has enhanced FP8 capabilities
// - Native FP4 support via SageAttention3
// - Improved FP8 tensor core throughput
// - Larger shared memory (228 KB vs 164 KB on Ada)

#if __CUDA_ARCH__ >= 1000
// Use larger tile sizes
constexpr uint32_t CTA_Q = 256;  // vs 128 on older GPUs
constexpr uint32_t CTA_K = 256;

// Leverage increased shared memory for deeper pipelining
constexpr uint32_t PIPELINE_STAGES = 4;  // vs 2 on Ampere/Ada
#endif
```

**Expected Impact**:
- 20-30% additional speedup on RTX 5090/B100/B200
- Better utilization of 560 TOPS peak performance

### 7.2 Ada (RTX 4090) L2 Cache Optimization
**Current**: Basic cache usage
**Proposed**: Explicit L2 residency control

```cuda
// Ada has 72MB L2 cache - keep hot data resident
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 72 * 1024 * 1024);

// Mark frequently accessed data for L2 persistence
cudaStreamAttrValue stream_attribute;
stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(kv_cache);
stream_attribute.accessPolicyWindow.num_bytes = kv_cache_size;
stream_attribute.accessPolicyWindow.hitRatio = 1.0;
stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

**Expected Impact**: 10-15% speedup for KV cache-heavy workloads

---

## Optimization Direction 8: System-Level Optimizations ⭐⭐⭐

### 8.1 CUDA Graph Integration
**Current**: Dynamic kernel launches
**Proposed**: Capture attention computation in CUDA graphs

```python
# Capture once, replay many times
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = sageattn(q, k, v, ...)

# Replay is much faster (no CPU-GPU synchronization)
for _ in range(num_iterations):
    graph.replay()
```

**Expected Impact**:
- 25-35% latency reduction for fixed-size workloads
- Critical for real-time inference

### 8.2 Multi-Stream Pipelining for Batch Processing
**Current**: Sequential batch processing
**Proposed**: Pipeline different batch elements across streams

```python
class PipelinedAttention:
    def __init__(self, num_streams=4):
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]

    def forward(self, q, k, v):
        batch_size = q.shape[0]
        chunk_size = batch_size // len(self.streams)

        # Launch attention for different batch elements concurrently
        for i, stream in enumerate(self.streams):
            with torch.cuda.stream(stream):
                start = i * chunk_size
                end = start + chunk_size
                output[start:end] = sageattn(
                    q[start:end], k[start:end], v[start:end], ...)
```

**Expected Impact**: 15-20% throughput improvement for large batches

### 8.3 Async Quantization for KV Cache
**Current**: Synchronous quantization during attention
**Proposed**: Background quantization for future steps

```python
class AsyncKVCache:
    def __init__(self):
        self.quant_stream = torch.cuda.Stream()
        self.cache = {}

    def insert(self, layer_id, k, v):
        # Store FP16 immediately
        self.cache[(layer_id, 'fp16')] = (k, v)

        # Quantize asynchronously in background
        with torch.cuda.stream(self.quant_stream):
            k_int8, k_scale = per_block_int8(k)
            v_fp8, v_scale = per_channel_fp8(v)
            self.cache[(layer_id, 'quant')] = (k_int8, v_fp8, k_scale, v_scale)

    def get(self, layer_id):
        # Use quantized if ready, otherwise fall back to FP16
        if (layer_id, 'quant') in self.cache:
            return self.cache[(layer_id, 'quant')]
        return self.cache[(layer_id, 'fp16')]
```

**Expected Impact**: Hides quantization latency, 10-20% throughput gain

---

## Implementation Priority Matrix

| Optimization | Difficulty | Expected Speedup | Accuracy Risk | Priority |
|-------------|-----------|------------------|---------------|----------|
| **Direction 1**: Hopper TMA Multicast | Medium | 15-20% | None | ⭐⭐⭐⭐⭐ |
| **Direction 2.1**: Adaptive Quantization | Low | 8-12% | Low | ⭐⭐⭐⭐⭐ |
| **Direction 2.2**: INT4 for QK | High | 30-40% | Medium | ⭐⭐⭐⭐ |
| **Direction 3.1**: Kernel Fusion | Medium | 20-30% | None | ⭐⭐⭐⭐⭐ |
| **Direction 3.2**: Persistent Kernels | High | 40-60% (decode) | None | ⭐⭐⭐⭐ |
| **Direction 4.1**: Bank Conflict Reduction | Low | 5-8% | None | ⭐⭐⭐ |
| **Direction 4.3**: Optimized KV Cache | Low | 15-25% (decode) | None | ⭐⭐⭐⭐⭐ |
| **Direction 5.1**: Sparsity Exploitation | Medium | 2-3x (sparse) | Low | ⭐⭐⭐⭐ |
| **Direction 5.2**: Causal Optimization | Low | 15-25% | None | ⭐⭐⭐⭐ |
| **Direction 6.2**: Mixed Block Sizes | Medium | 8-12% | None | ⭐⭐⭐ |
| **Direction 7.1**: Blackwell Optimization | Medium | 20-30% (B100+) | None | ⭐⭐⭐⭐ |
| **Direction 7.2**: L2 Cache Control | Low | 10-15% | None | ⭐⭐⭐⭐ |
| **Direction 8.1**: CUDA Graphs | Low | 25-35% (latency) | None | ⭐⭐⭐⭐⭐ |
| **Direction 8.3**: Async Quantization | Low | 10-20% | None | ⭐⭐⭐⭐ |

---

## Recommended Implementation Roadmap

### Phase 1: Low-Hanging Fruit (2-3 weeks)
1. **L2 Cache Optimization** (Direction 7.2)
   - Easy to implement, good gains
   - File: `sageattention/core.py`

2. **CUDA Graph Support** (Direction 8.1)
   - Significant latency reduction
   - Add `use_cuda_graph` parameter

3. **Optimized KV Cache Layout** (Direction 4.3)
   - Crucial for decoding performance
   - New file: `sageattention/kv_cache.py`

### Phase 2: Kernel Improvements (4-6 weeks)
1. **Kernel Fusion** (Direction 3.1)
   - Major bandwidth reduction
   - File: `csrc/fused/fused_attention.cu`

2. **Improved Causal Masking** (Direction 5.2)
   - Easy modification to existing kernels
   - File: `csrc/qattn/attn_utils.cuh`

3. **Adaptive Quantization** (Direction 2.1)
   - Better accuracy/speed trade-off
   - File: `sageattention/quant.py`

### Phase 3: Advanced Features (6-8 weeks)
1. **Hopper TMA Optimizations** (Direction 1)
   - H100/H200 performance boost
   - File: `csrc/qattn/qk_int_sv_f8_cuda_sm90.cu`

2. **INT4 Quantization** (Direction 2.2)
   - High speedup potential
   - New file: `csrc/qattn/qk_int4_sv_f8_cuda.cu`

3. **Persistent Kernels** (Direction 3.2)
   - Decoding latency optimization
   - New file: `csrc/persistent_decode.cu`

### Phase 4: Cutting Edge (8-12 weeks)
1. **Blackwell Optimizations** (Direction 7.1)
   - Future-proofing for RTX 5090/B100/B200
   - File: `csrc/qattn/qk_int_sv_f8_cuda_sm120.cu`

2. **Sparsity Integration** (Direction 5.1)
   - Long-context performance
   - Integration with SpargeAttn

3. **Multi-Stream Pipeline** (Direction 8.2)
   - Throughput optimization
   - File: `sageattention/pipeline.py`

---

## Expected Cumulative Impact

Combining these optimizations conservatively:

| Workload Type | Current Speedup | Projected Speedup | Improvement |
|--------------|----------------|-------------------|-------------|
| **Prefill (RTX 5090)** | 2.7x | 4.0-4.5x | +48-67% |
| **Prefill (H100)** | 1.0x (matches FA3) | 1.4-1.6x | +40-60% |
| **Decode (RTX 4090)** | 2.0x | 3.5-4.0x | +75-100% |
| **Long Context (>16K)** | 2.0x | 5.0-8.0x | +150-300% |
| **Sparse Patterns** | N/A | 5.0-10.0x | N/A (new capability) |

---

## Validation Strategy

For each optimization:

1. **Performance Validation**
   ```bash
   cd bench
   python benchmark_comprehensive.py --optimization <name>
   ```

2. **Accuracy Validation**
   ```python
   # Compare against FP16 baseline
   from sageattention.validation import validate_accuracy

   max_diff, mean_diff = validate_accuracy(
       sageattn_optimized,
       reference=torch.nn.functional.scaled_dot_product_attention,
       tolerance=1e-2
   )
   ```

3. **End-to-End Testing**
   - Image generation: FLUX, SD3
   - Video generation: CogVideoX
   - LLM: Llama-3, Mistral
   - Measure: SSIM for images, perplexity for LLM

---

## Conclusion

SageAttention has already achieved impressive speedups through quantization and careful kernel engineering. The proposed optimizations build on this foundation to push performance even further:

**Quick Wins** (Phases 1-2):
- 50-80% additional speedup with low implementation risk
- Focus on memory optimization and system-level improvements

**Long-Term Gains** (Phases 3-4):
- 2-4x additional speedup for specific workloads
- Unlock new capabilities (very long context, real-time decoding)

**Key Insight**: The biggest opportunities lie in:
1. Better hardware utilization (TMA, WGMMA, L2 cache)
2. Reducing memory traffic (kernel fusion, persistent kernels)
3. Algorithmic improvements (sparsity, adaptive quantization)

These directions are complementary and can be pursued in parallel by different team members.
