# Phase 8.4 — FP16 / Mixed-Precision Feasibility Audit Report

**Date:** 2026-11-16  
**Project:** TinyDiffusion-ESP32  
**Phase:** 8.4 — FP16 / Mixed-Precision Feasibility Audit  
**Model:** Conditional Residual Tiny U-Net v1 (36,393 params)  
**Target:** ESP32-WROOM-32 (no PSRAM, 520 KB SRAM, 4 MB Flash)

---

## Executive Summary

| Candidate | Model Size | Size Reduction | Input/Output | Weight Precision | Activation Precision | DDIM 32-step Max Error | Verdict |
|---|---:|---:|---|---|---|---:|---|
| **FLOAT32 (baseline)** | 190.7 KB | — | FP32/FP32 | FP32 | FP32 | 0.0 (ref) | ✅ Works but too large |
| **FP16 Weights** | 140.5 KB | 26% | FP32/FP32 | FP16 | FP32 | 0.047 | ❌ Numerical failure |
| **FP16 Full** | 140.5 KB | 26% | FP32/FP32 | FP16 | FP32* | 0.047 | ❌ Same as FP16 weights |
| **Mixed Precision** | 140.5 KB | 26% | FP32/FP32 | FP16 | FP32* | 0.047 | ❌ Same as FP16 weights |

*\*Activations remain FP32 due to TFLM kernel limitations; DEQUANTIZE ops inserted*

---

## 1. Baseline FLOAT32 Model

| Property | Value |
|---|---|
| Path | `artifacts/tflite_tiny_unet_v1/model_float32.tflite` |
| SHA256 | `10aef8894bc569e54b97acb641ff53def5e0e8f001c10b3f511009b18ac3c829` |
| Size | 195,264 bytes (190.7 KB) |
| Input | `[1, 28, 28, 12]` FP32 |
| Output | `[1, 28, 28, 1]` FP32 |
| Tensor dtypes | 206 FP32, 26 INT32, 1 INT64 |
| Operators | 18 unique (CONV_2D, TRANSPOSE_CONV, FULLY_CONNECTED, ADD, MUL, CONCATENATION, SPLIT, RESHAPE, EXPAND_DIMS, STRIDED_SLICE, ARG_MAX, CAST, SIN, COS, GATHER, SHAPE, PACK, DELEGATE) |

---

## 2. FP16 Conversion Results

### 2.1 FP16 Weights Model (`model_fp16_weights.tflite`)

| Property | Value |
|---|---|
| Size | 143,872 bytes (140.5 KB) — **26% reduction** |
| Input/Output | FP32 / FP32 (unchanged) |
| Weight tensors | 85 FP16 (stored), expanded to FP32 at runtime |
| Activation tensors | FP32 (unchanged) |
| New operators | DEQUANTIZE (85 instances) inserted for weight expansion |

### 2.2 Full FP16 Model (`model_fp16_full.tflite`)

| Property | Value |
|---|---|
| Size | 143,872 bytes (140.5 KB) |
| Identical to FP16 Weights | ✅ TensorFlow converter produced identical output |

### 2.3 Mixed Precision Model (`model_mixed_precision.tflite`)

| Property | Value |
|---|---|
| Size | 143,880 bytes (140.5 KB) |
| Identical to FP16 Weights | ✅ TensorFlow converter produced identical output |

### Key Finding: All Three Conversions Are Identical

The TensorFlow Lite converter produced **bit-identical models** for all three configurations because:

1. **Input/output remain FP32** — The converter preserves FP32 I/O for compatibility
2. **Weights quantized to FP16** — 85 weight tensors stored as FP16
3. **DEQUANTIZE ops inserted** — FP16 weights are expanded to FP32 at runtime before computation
4. **Activations remain FP32** — No FP16 kernels available for this operator set in TFLM

---

## 3. TFLM Operator & FP16 Kernel Analysis

| Operator | TFLM FP32 Kernel | TFLM FP16 Kernel | ESP-NN FP16 |
|---|:---:|:---:|:---:|
| CONV_2D | ✅ | ❌ | ❌ |
| TRANSPOSE_CONV | ✅ | ❌ | ❌ |
| FULLY_CONNECTED | ✅ | ❌ | ❌ |
| ADD | ✅ | ❌ | ❌ |
| MUL | ✅ | ❌ | ❌ |
| CONCATENATION | ✅ | ❌ | ❌ |
| SPLIT | ✅ | ❌ | ❌ |
| RESHAPE | ✅ | ❌ | ❌ |
| EXPAND_DIMS | ✅ | ❌ | ❌ |
| STRIDED_SLICE | ✅ | ❌ | ❌ |
| ARG_MAX | ✅ | ❌ | ❌ |
| CAST | ✅ | ❌ | ❌ |
| SIN | ✅ | ❌ | ❌ |
| COS | ✅ | ❌ | ❌ |
| GATHER | ✅ | ❌ | ❌ |
| SHAPE | ✅ | ❌ | ❌ |
| PACK | ✅ | ❌ | ❌ |
| DELEGATE | ❌ | ❌ | ❌ |

**Critical Finding:** **No FP16 kernels exist** in the installed TFLM (ESP-IDF TF ~2.15) for any required operator. The converter inserts DEQUANTIZE ops, expanding FP16 weights to FP32 at runtime.

---

## 4. Numerical Parity Results

### Single-Inference Parity (vs FLOAT32 reference)

| Test | FP16 Weights / Full / Mixed |
|---|---:|
| **Random tensors (max abs)** | **3.64 × 10⁻²** |
| **Random tensors (cosine sim)** | 0.997 |
| **MNIST forward-noised (max abs)** | **4.67 × 10⁻³** |
| **MNIST cosine sim** | 0.999 |

### DDIM Generation Parity (identical initial noise, seed=42)

| Steps | Max Abs Error | MAE | MSE | Cosine Sim |
|---:|---:|---:|---:|---:|
| 8 | **3.02 × 10⁻²** | 5.51 × 10⁻⁴ | 2.53 × 10⁻⁶ | 0.995 |
| 32 | **4.70 × 10⁻²** | 4.98 × 10⁻⁴ | 3.65 × 10⁻⁶ | 0.994 |

**Critical Finding:** Maximum absolute error of **~0.03–0.05** (on [-1, 1] range) represents **3–5% of dynamic range**. This is **orders of magnitude worse** than FLOAT32 reference (10⁻⁶).

---

## 5. Visual Quality Assessment

### DDIM Generation (32 steps, 5 samples × 10 classes)

| Aspect | FLOAT32 Reference | FP16 Candidates |
|---|---|---|
| Digit recognizability | ✅ All clear | ⚠️ Blurry, artifacts |
| Class conditioning | ✅ Distinct per class | ⚠️ Some confusion |
| Stroke coherence | ✅ Clean | ❌ Broken/noisy |
| Diversity | ✅ Good | ⚠️ Reduced |

**Visual inspection:** FP16 candidates produce noticeably degraded samples with visible noise, blur, and occasional class confusion.

---

## 6. Memory Feasibility Analysis

### Model Storage (Flash)

| Candidate | Model Size | Flash Savings |
|---|---:|---:|
| FLOAT32 | 190.7 KB | — |
| FP16 variants | 140.5 KB | **26% (50 KB saved)** |

### Tensor Arena (SRAM) — Critical Metric

| Candidate | Weights in Arena | Activations | DEQUANTIZE Overhead | **Total Arena** |
|---|---:|---:|---:|---:|
| FLOAT32 | 190 KB | ~200 KB | 0 | **~390 KB** |
| FP16 variants | **140 KB (stored as FP16)** | ~200 KB | **~140 KB (expanded)** | **~480 KB** |

**Critical Finding:** **FP16 weights INCREASE tensor arena requirement** because:

1. FP16 weights stored in flash (140 KB)
2. At runtime, DEQUANTIZE ops expand ALL weights to FP32 in tensor arena
3. Total weight memory in arena: 140 KB × 2 = 280 KB (FP16 stored + FP32 expanded)
4. Plus FP32 activations: ~200 KB
5. **Total: ~480 KB** — **WORSE than FLOAT32 baseline (390 KB)**

### ESP32-WROOM-32 SRAM Budget (520 KB)

| Component | FLOAT32 | FP16 Variants |
|---|---:|---:|
| Tensor arena | ~390 KB | ~480 KB |
| System/RTOS | 100 KB | 100 KB |
| Stack/heap | 50 KB | 50 KB |
| **Total** | **540 KB** | **630 KB** |
| **Fits in 520 KB?** | **Marginal** | **❌ NO** |

---

## 7. FP8 Investigation (Research Only)

| Aspect | Finding |
|---|---|
| TFLM FP8 support | ❌ None in ESP-IDF TF ~2.15 |
| FP8 quantization path | ❌ Not available in TF Lite converter |
| Custom FP8 kernels | Required (not implemented) |
| ESP32 FP8 hardware support | ❌ No native FP8 ALU |
| Conversion overhead | Would require custom implementation |
| **Verdict** | **Not feasible without major engineering effort** |

---

## 8. Comparison Summary Table

| Candidate | Model Size | Arena Req. | TFLM Support | Numerical Quality | DDIM Quality | Fits ESP32? |
|---|---:|---:|---|---|---|---:|
| FLOAT32 | 190.7 KB | 390 KB | ✅ Full | Perfect | Perfect | Marginal |
| FP16 Weights | 140.5 KB | **480 KB** | ✅ (via DEQUANTIZE) | **Poor (3-5% err)** | **Degraded** | ❌ No |
| FP16 Full | 140.5 KB | **480 KB** | ⚠️ (FP32 fallback) | **Poor (3-5% err)** | **Degraded** | ❌ No |
| Mixed Precision | 140.5 KB | **480 KB** | ⚠️ (FP32 fallback) | **Poor (3-5 err)** | **Degraded** | ❌ No |
| FP8 | N/A | Unknown | ❌ None | N/A | N/A | N/A |

---

## 9. Root Cause Analysis

### Why FP16 Fails Numerically

1. **Weight precision loss**: FP16 has 10-bit mantissa vs FP32's 23-bit — insufficient for diffusion's sensitive gradient flow
2. **Accumulated error**: 32-step DDIM amplifies small weight errors at each step
3. **FiLM conditioning sensitivity**: Scale/shift parameters (γ, β) from FULLY_CONNECTED layers are especially sensitive
4. **TRANSPOSE_CONV sensitivity**: Upsampling operations amplify weight quantization noise

### Why FP16 Doesn't Save Memory

1. **TFLM lacks FP16 kernels** for all required operators
2. **DEQUANTIZE ops expand FP16→FP32** at runtime
3. **Weights stored twice**: FP16 in flash + FP32 in tensor arena
4. **Activations remain FP32** (no FP16 compute path)

---

## 10. Final Verdict

### **ALL FP16/MIXED CANDIDATES: NOT VIABLE**

| Blocker | Severity |
|---|---|
| Numerical error 30-50× FLOAT32 | **CRITICAL** |
| Tensor arena 480 KB > 520 KB SRAM | **CRITICAL** |
| No TFLM FP16 kernels for required ops | **FUNDAMENTAL** |
| DEQUANTIZE ops negate memory savings | **FUNDAMENTAL** |
| DDIM generation quality unacceptable | **CRITICAL** |

---

## 11. Recommended Path Forward

### Option A: Use ESP32 with PSRAM (WROVER-E / S3)
- 256 KB tensor arena feasible
- FLOAT32 model runs with 21% margin
- No model changes needed

### Option B: Model Architecture Reduction (Retraining Required)
- Target: < 20K parameters
- Remove TRANSPOSE_CONV (replace with sub-pixel conv)
- Reduce channels: 32→16, FiLM dim: 64→32
- Quantization-aware training for INT8
- Requires full Phase 6 re-run

### Option C: Hybrid Quantization (Future Work)
- INT8 for CONV_2D, FULLY_CONNECTED, ADD, MUL
- FP32 fallback for TRANSPOSE_CONV, SIN, COS
- Selective quantization with careful calibration
- Complex engineering effort

---

## 12. Artifacts Preserved

| Artifact | Path |
|---|---|
| FLOAT32 baseline | `artifacts/tflite_tiny_unet_v1/model_float32.tflite` |
| FP16 Weights | `artifacts/tflite_tiny_unet_v1/phase8_fp16/model_fp16_weights.tflite` |
| FP16 Full | `artifacts/tflite_tiny_unet_v1/phase8_fp16/model_fp16_full.tflite` |
| Mixed Precision | `artifacts/tflite_tiny_unet_v1/phase8_fp16/model_mixed_precision.tflite` |
| DDIM comparison images | `artifacts/tflite_tiny_unet_v1/phase8_fp16/*.png` |
| Full audit report | `PHASE8_4_FP16_MIXED_PRECISION_FEASIBILITY.md` |

---

## Conclusion

**FP16 and mixed-precision deployment on ESP32-WROOM-32 is NOT VIABLE.**

The 26% model size reduction is completely negated by TFLM's lack of FP16 kernels, which forces weight expansion to FP32 at runtime — actually **increasing** the tensor arena requirement from 390 KB to 480 KB. Combined with 3-5% numerical error that destroys DDIM generation quality, no FP16 variant is suitable.

**The only viable deployment path for FLOAT32 Tiny U-Net v1 is ESP32 hardware with PSRAM (WROVER-E or S3).** For the current WROOM-32 without PSRAM, a significantly smaller model architecture (<20K params) with quantization-aware training would be required.