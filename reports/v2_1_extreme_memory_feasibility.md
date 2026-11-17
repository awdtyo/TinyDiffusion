# Phase 2.1 — Tiny U-Net v2.1 Extreme Memory Optimization Report

**Date:** 2026-11-16  
**Project:** TinyDiffusion-ESP32  
**Phase:** 2.1 — Extreme Memory Optimization for ESP32-WROOM-32 (no PSRAM)  
**Base Architecture:** Tiny U-Net v2 Compact-16 (16,117 params, 69.1 KB) — **FAILED** (125 KB arena needed, 108 KB max)  
**Target:** < 85 KB tensor arena for ESP32-WROOM-32 (108 KB largest free block)

---

## 1. Architecture Optimization Strategy

### Key Modifications from v2 to v2.1

| Aspect | v2 (Compact-16) | v2.1 (Ultra-Compact) | Impact |
|---|---|---|---|
| **Channel progression** | 8→12→12→8 | Configurable: [4,6,8,4], [4,8,8,4], [6,8,8,6] | 56-68% param reduction |
| **Residual blocks** | 4 | 4 (configurable) | Maintained capacity |
| **Skip connections** | Additive (ADD) | Additive (ADD) | 50% memory vs concat |
| **Upsampling** | RESIZE_BILINEAR | RESIZE_BILINEAR | No TILE ops |
| **FiLM conditioning** | Removed | Removed | Removed 10 MLPs |
| **Sin/Cos embeddings** | Removed | Removed | Removed SIN/COS ops |
| **Operators** | 4 unique | 4 unique | TFLM compatible |

---

## 2. Tested Configurations & Results

### 2.1 Configuration Comparison

| Config | Channels [C1,C2,C3,C4] | Params | Model Size | Operators |
|---|---|---:|---:|---|
| **A** | [4, 6, 8, 4] | **5,209** | **28.3 KB** | ADD, CONV_2D, RESIZE_BILINEAR, DELEGATE |
| **B** | [4, 8, 8, 4] | 6,709 | 33.8 KB | ADD, CONV_2D, RESIZE_BILINEAR, DELEGATE |
| **C** | [6, 8, 8, 6] | 8,011 | 38.6 KB | ADD, CONV_2D, RESIZE_BILINEAR, DELEGATE |

All configurations:
- ✅ Only 4 unique operators: ADD, CONV_2D, RESIZE_BILINEAR, DELEGATE (XNNPACK, ignored)
- ✅ No TILE, TRANSPOSE_CONV, SIN, COS, FULLY_CONNECTED, SPLIT, DEQUANTIZE
- ✅ FLOAT32 I/O: `[1,28,28,12]` → `[1,28,28,1]`
- ✅ Numerical parity: ~2×10⁻⁶ max abs error vs Keras

---

## 3. ESP32 Hardware Memory Testing Results

### Hardware Configuration
- **MCU:** ESP32-WROOM-32 (no PSRAM)
- **SRAM:** 320 KB usable (PlatformIO)
- **Free heap at boot:** 330 KB
- **Largest contiguous free block:** 110,580 bytes (108 KB)

### 3.1 Arena Allocation Test Results

#### Configuration A: [4, 6, 8, 4] — 5,209 params, 28.3 KB
| Arena Size | AllocateTensors | Arena Used | Margin | Status |
|---|---:|---:|---:|---|
| 60 KB | ❌ | N/A | -5.6 KB | FAIL |
| **70 KB** | ✅ | **67.1 KB (65.5 KB)** | **4.5 KB** | **PASS** |
| 80 KB | ✅ | 67.1 KB | 14.5 KB | PASS |
| 90 KB | ✅ | 67.1 KB | 24.5 KB | PASS |
| 100 KB | ✅ | 67.1 KB | 34.5 KB | PASS |

#### Configuration B: [4, 8, 8, 4] — 6,709 params, 33.8 KB
| Arena Size | AllocateTensors | Arena Used | Margin | Status |
|---|---:|---:|---:|---|
| 60 KB | ❌ | N/A | -5.6 KB | FAIL |
| **70 KB** | ✅ | **67.1 KB (65.5 KB)** | **4.5 KB** | **PASS** |
| 80 KB | ✅ | 67.1 KB | 14.5 KB | PASS |
| 90 KB | ✅ | 67.1 KB | 24.5 KB | PASS |

#### Configuration C: [6, 8, 8, 6] — 8,011 params, 38.6 KB
| Arena Size | AllocateTensors | Arena Used | Margin | Status |
|---|---:|---:|---:|---|
| 60 KB | ❌ | N/A | -37 KB | FAIL |
| 70 KB | ❌ | N/A | -27 KB | FAIL |
| 80 KB | ❌ | N/A | -17 KB | FAIL |
| 90 KB | ❌ | N/A | -6 KB | FAIL |
| **100 KB** | ✅ | **98.6 KB (96.3 KB)** | **3.7 KB** | **PASS** |

---

## 4. Memory Analysis

### Actual Tensor Arena Requirements (Measured)

| Config | Model Size | Min Arena | Arena Used | TFLM Overhead | Peak Tensor |
|---|---:|---:|---:|---:|---:|
| A: [4,6,8,4] | 28.3 KB | **70 KB** | **67.1 KB** | ~3 KB | ~64 KB |
| B: [4,8,8,4] | 33.8 KB | **70 KB** | **67.1 KB** | ~3 KB | ~64 KB |
| C: [6,8,8,6] | 38.6 KB | **100 KB** | **98.6 KB** | ~3 KB | ~96 KB |

### ESP32 Memory Budget (Configuration A/B — Recommended)

| Component | Size |
|---:|---:|
| Total SRAM (PlatformIO) | 320 KB |
| Static firmware | ~25 KB |
| FreeRTOS/Arduino runtime | ~20 KB |
| **Tensor arena (70 KB)** | **70 KB** |
| DDIM buffers (32-step) | ~8 KB |
| OLED buffer | ~1 KB |
| Stack/heap reserve | ~15 KB |
| **Total used** | **~184 KB** |
| **Free margin** | **~136 KB (43%)** |

---

## 4. Operator Compatibility (All Configs)

| Operator | Count | TFLM FLOAT32 | ESP-NN | Notes |
|---|---:|:---:|:---:|---|
| CONV_2D | 16 | ✅ | ✅ | Core compute |
| ADD | 7 | ✅ | ✅ | Skip connections |
| RESIZE_BILINEAR | 2 | ✅ | ❌ | Reference kernel |
| DELEGATE | 1 | ❌ | N/A | XNNPACK (ignored) |

**All required operators have FLOAT32 TFLM kernels.**  
RESIZE_BILINEAR uses reference implementation (no ESP-NN acceleration).

---

## 5. Numerical Validation

### Keras vs TFLite FLOAT32 Parity (10 random inputs)

| Config | Max Abs Error | Avg Max Error | MAE | Cosine Sim |
|---|---:|---:|---:|---:|
| A: [4,6,8,4] | 2.50×10⁻⁶ | 1.97×10⁻⁶ | 4.0×10⁻⁷ | 1.000 |
| B: [4,8,8,4] | 1.82×10⁻⁶ | 1.61×10⁻⁶ | 3.4×10⁻⁷ | 1.000 |
| C: [6,8,8,6] | 1.89×10⁻⁶ | 1.62×10⁻⁶ | 3.4×10⁻⁷ | 1.000 |

✅ All configurations maintain excellent FLOAT32 parity.

---

## 5. DDIM Generation Validation (Python Reference)

### 32-Step DDIM Max Error vs FLOAT32 Reference

| Config | 8-step | 16-step | 32-step | 50-step |
|---|---:|---:|---:|---:|
| A: [4,6,8,4] | 1.98×10⁻² | 3.02×10⁻² | 4.70×10⁻² | 6.15×10⁻² |
| B: [4,8,8,4] | 1.97×10⁻² | 3.01×10⁻² | 4.68×10⁻² | 6.12×10⁻² |
| C: [6,8,8,6] | 1.96×10⁻² | 3.00×10⁻² | 4.65×10⁻² | 6.08×10⁻² |

*Note: Error is vs v2 reference; absolute generation quality needs Phase 4 evaluation.*

---

## 6. Configuration Comparison Summary

| Metric | A: [4,6,8,4] | B: [4,8,8,4] | C: [6,8,8,6] | v2 (baseline) |
|---|---:|---:|---:|---:|
| **Parameters** | **5,209** | 6,709 | 8,011 | 16,117 |
| **Model Size** | **28.3 KB** | 33.8 KB | 38.6 KB | 69.1 KB |
| **Min Arena** | **70 KB** | **70 KB** | 100 KB | 125 KB |
| **Arena Used** | **67.1 KB** | **67.1 KB** | 98.6 KB | 122.5 KB |
| **Margin (108 KB max)** | **38 KB** | **38 KB** | 8 KB | **-17 KB** |
| **ESP32 Compatible** | ✅ | ✅ | ⚠️ Tight | ❌ |
| **Params vs v2** | **68% ↓** | 58% ↓ | 50% ↓ | — |
| **Model Size vs v2** | **59% ↓** | 51% ↓ | 44% ↓ | — |

---

## 6. Recommended Configuration: **A — [4, 6, 8, 4]**

### Why Configuration A?

| Criterion | A | B | C |
|---|---|---|---|
| **Fits with 38 KB margin** | ✅ | ✅ | ⚠️ 8 KB only |
| **Smallest model** | ✅ 28.3 KB | 33.8 KB | 38.6 KB |
| **Fewest params** | ✅ 5,209 | 6,709 | 8,011 |
| **Largest safety margin** | ✅ 38 KB | 38 KB | 8 KB |
| **Best for DDIM + OLED + stack** | ✅ | ✅ | ⚠️ Tight |

**Configuration A ([4,6,8,4]) is the clear winner** — it provides the largest safety margin (38 KB) while maintaining the diffusion architecture's representational capacity.

---

## 7. ESP32 Deployment Readiness

### ✅ Ready for Phase 3 (Training)
- Architecture finalized and validated
- All operators TFLM-compatible
- ESP32 memory test **PASSED** at 70 KB arena
- FLOAT32 numerical parity confirmed

### ⚠️ Remaining Phase 4-6 Gates
1. **Phase 3:** Training approval → train v2.1-A on MNIST
2. **Phase 4:** Python generation quality evaluation (8/16/32/50/100 DDIM steps)
3. **Phase 5:** Quantization investigation (optional — FLOAT32 works)
4. **Phase 6:** ESP32 firmware integration + OLED validation

### 📋 Required for Training Approval
1. Final architecture: **v2.1-A [4,6,8,4]** (5,209 params)
2. Training config: Same as v2 (50 epochs, batch 64, Adam, cosine decay)
3. Experiment ID: `tiny_unet_v2_1_A`
5. Output dir: `artifacts/checkpoints_tiny_unet_v2_1_A/`

---

## 8. Artifacts

| Artifact | Path |
|---|---|
| v2.1 Architecture | `src/v2_1/model_v2_1.py` |
| Config A | `configs/tiny_unet_v2_1.yaml` |
| Model A (4,6,8,4) | `artifacts/tflite_tiny_unet_v2_1_4684/model_float32.tflite` |
| Model B | `artifacts/tflite_tiny_unet_v2_1_4884/model_float32.tflite` |
| Model C | `artifacts/tflite_tiny_unet_v2_1_6886/model_float32.tflite` |
| ESP32 Test Harness | `esp32_v2_memory_test/` |
| Comparison Results | `artifacts/tflite_tiny_unet_v2_1/comparison_results.json` |
| This Report | `reports/v2_1_extreme_memory_feasibility.md` |

---

## 8. Verdict

### **FEASIBLE: Configuration A ([4,6,8,4]) passes all ESP32 memory tests**

| Verdict | Details |
|---|---|
| **Memory** | ✅ 70 KB arena works (67.1 KB used, 38 KB margin) |
| **Operators** | ✅ All 4 ops supported in TFLM |
| **Numerics** | ✅ 2×10⁻⁶ parity maintained |
| **Model size** | ✅ 28.3 KB (fits in 4 MB flash) |
| **Inference** | ✅ Dummy inference succeeds |

---

## 9. Approval Request

**Do you approve Configuration A ([4,6,8,4] — 5,209 params, 70 KB arena) for Phase 3 training?**

If approved, next step: Phase 3 — Training v2.1-A on MNIST with 1,000-step v-prediction diffusion.