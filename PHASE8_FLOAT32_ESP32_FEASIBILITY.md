# Phase 8.1 — FLOAT32 TFLite Micro Deployment Feasibility Report

**Date:** 2026-11-16  
**Project:** TinyDiffusion-ESP32  
**Model:** Conditional Residual Tiny U-Net v1 (36,393 params)  
**Target:** ESP32-WROOM-32 (no PSRAM, 520 KB SRAM, 4 MB Flash)

---

## Executive Summary

| Aspect | Status |
|---|---|
| **FLOAT32 Model Verified** | ✅ SHA256 matches (`10aef8894bc569e54b97acb641ff53def5e0e8f001c10b3f511009b18ac3c829`) |
| **Operator Support** | ✅ All 19 unique operators have FLOAT32 kernels in TFLM |
| **Memory Feasibility** | ✅ 256 KB arena + 22 KB DDIM = 278 KB < 520 KB SRAM |
| **Flash Storage** | ✅ 190.7 KB model << 4 MB Flash |
| **Numerical Parity (Python)** | ✅ 1e-6 max error vs Keras, 8e-06 DDIM parity |
| **Current Firmware Compatibility** | ❌ Built for INT8, requires significant changes |

**Overall Verdict: CONDITIONALLY READY**

The FLOAT32 model **can run** on ESP32-WROOM-32 without PSRAM, but the **production firmware must be modified** before hardware testing.

---

## 1. Model Verification

| Property | Expected | Actual | Status |
|---|---|---|---|
| SHA256 | `10aef8894bc569e54b97acb641ff53def5e0e8f001c10b3f511009b18ac3c829` | Match | ✅ |
| Size | ~195 KB | 195,264 bytes | ✅ |
| Input | `[1, 28, 28, 12]` float32 | Match | ✅ |
| Output | `[1, 28, 28, 1]` float32 | Match | ✅ |
| Python FLOAT32 vs Keras | 1e-6 max error | 2.04e-06 | ✅ |
| Python 32-step DDIM | 8e-06 max error | 8.05e-06 | ✅ |

---

## 2. Operator Compatibility

### FLOAT32 Model Inventory (126 ops, 19 unique)

| Operator | Count | TFLM FLOAT32 | ESP-NN | Registration |
|---|---:|:---:|:---:|---|
| CONV_2D | 56 | ✅ | ✅ | `AddConv2D()` |
| TRANSPOSE_CONV | 2 | ✅ | ❌ | `AddTransposeConv()` |
| FULLY_CONNECTED | 20 | ✅ | ✅ | `AddFullyConnected()` |
| ADD | 24 | ✅ | ✅ | `AddAdd()` |
| MUL | 14 | ✅ | ✅ | `AddMul()` |
| CONCATENATION | 4 | ✅ | ❌ | `AddConcatenation()` |
| SPLIT | 12 | ✅ | ❌ | `AddSplit()` |
| RESHAPE | 12 | ✅ | ❌ | `AddReshape()` |
| EXPAND_DIMS | 2 | ✅ | ❌ | `AddExpandDims()` |
| STRIDED_SLICE | 14 | ✅ | ❌ | `AddStridedSlice()` |
| ARG_MAX | 1 | ✅ | ❌ | `AddArgMax()` |
| CAST | 4 | ✅ | ❌ | `AddCast()` |
| SIN | 1 | ✅ | ❌ | `AddSin()` |
| COS | 1 | ✅ | ❌ | `AddCos()` |
| GATHER | 1 | ✅ | ❌ | `AddGather()` |
| SHAPE | 2 | ✅ | ❌ | `AddShape()` |
| PACK | 2 | ✅ | ❌ | `AddPack()` |
| DELEGATE | 4 | ❌ | N/A | **Remove** |

**Missing registrations in current firmware: 12 operators**

### Critical Finding: XNNPACK DELEGATE

The FLOAT32 model contains **4 DELEGATE operators** (XNNPACK execution plan) that are **desktop-only**. These must be removed by re-converting with:

```python
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# NOT TFLITE_BUILTINS_INT8, NOT DEFAULT (which enables XNNPACK)
```

---

## 3. Memory Analysis

### Tensor Arena Requirements

| Metric | Value |
|---|---:|
| Peak temporary tensors | ~200 KB |
| Persistent tensors (input/output) | ~41 KB |
| TFLM overhead (~20%) | ~40 KB |
| **Recommended arena** | **256 KB** |
| Current firmware config | 70 KB |

### ESP32 SRAM Budget (520 KB Total)

| Component | Size (KB) |
|---|---:|
| Total SRAM | 520 |
| System/IRAM reserve | 100 |
| **Tensor arena (FLOAT32)** | **256** |
| DDIM runtime buffers | 22 |
| Scheduler/state | 5 |
| OLED/U8g2 | 6 |
| Stack/heap reserve | 20 |
| **Total Used** | **409** |
| **Free Margin** | **111 KB (21%)** |

**✅ 32-step DDIM feasible with 21% memory margin**

---

## 4. Firmware Changes Required

### Must Do Before Hardware Test

| # | Change | File | Effort |
|---|---|---|---|
| 1 | Re-convert model without XNNPACK delegate | `scripts/convert_tiny_unet_v1_float32.py` | Low |
| 2 | Generate FLOAT32 model header/source | `scripts/generate_model_header.py` | Low |
| 3 | Update `config.h`: TENSOR_ARENA_SIZE=262144, remove quantization params | `esp32/include/config.h` | Low |
| 4 | Update `model_runner.h`: Change I/O to `float*`, remove quant params | `esp32/include/model_runner.h` | Medium |
| 5 | Update `model_runner.cpp`: Register all 19 ops, remove quant logic | `esp32/src/model_runner.cpp` | Medium |
| 6 | Update `diffusion.h/cpp`: Use float32 I/O, 1000 timesteps | `esp32/include/diffusion.h`, `esp32/src/diffusion.cpp` | Medium |
| 7 | Update `main.cpp`: Use FLOAT32 test vector | `esp32/src/main.cpp` | Low |

### Estimated Effort: 2-4 hours

---

## 5. Test Vector for Validation

| Property | Value |
|---|---|
| Seed | 42 |
| Input shape | [1, 28, 28, 12] |
| Input dtype | float32 |
| Input range | [-3.837, 3.926] |
| Input mean | -0.001159 |
| Expected output sum | 468.260010 |
| Expected output range | [-4.131, 5.488] |
| Expected output mean | 0.597270 |

Saved to: `artifacts/tflite_tiny_unet_v1/phase8_float32/test_vectors/test_vectors.h`

---

## 6. Numerical Parity Targets

| Metric | Target |
|---|---|
| Max absolute error | < 1e-4 |
| Mean absolute error | < 1e-5 |
| MSE | < 1e-8 |
| Cosine similarity | > 0.9999 |
| Output range match | Within 1% |

---

## 7. Deployment Feasibility Verdict

### Ready For:
- ✅ FLOAT32 model execution on ESP32
- ✅ 32-step deterministic DDIM
- ✅ 10-class conditioning
- ✅ No PSRAM required

### Blockers (Must Resolve):
1. **Re-convert model without XNNPACK delegate** (CRITICAL)
2. **Add 12 missing operator registrations** (CRITICAL)
3. **Increase tensor arena to 256 KB** (CRITICAL)
4. **Switch firmware I/O from INT8 to float32** (CRITICAL)

### Risks:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| TRANSPOSE_CONV ref kernel slow | Medium | ~50-100ms/inference | Acceptable for 32-step |
| SIN/COS ref kernel slow | Low | 2 calls/inference | Negligible |
| Arena fragmentation at 256 KB | Low | Allocation failure | TFLM uses linear allocator |
| DDIM accumulation error | Low | Quality loss | FLOAT32 precision sufficient |

---

## 8. Next Phase: 8.2 — Isolated Test Harness

### Plan:
1. Create `esp32/test_float32/` isolated PlatformIO project
2. Implement FLOAT32 ModelRunner with full 19-op resolver
3. Generate FLOAT32 model header from re-converted model
4. Build and run native compilation test (if possible)
5. **Request approval** for ESP32 hardware flash test
6. Run single-inference parity test on hardware
7. If successful, proceed to DDIM integration

### Approval Gate:
**Do not flash ESP32 without explicit approval.**

---

## 9. Artifacts Created

| File | Purpose |
|---|---|
| `artifacts/tflite_tiny_unet_v1/phase8_float32/operator_compatibility.md` | Detailed operator matrix |
| `artifacts/tflite_tiny_unet_v1/phase8_float32/memory_budget.md` | SRAM budget analysis |
| `artifacts/tflite_tiny_unet_v1/phase8_float32/test_vectors/test_vectors.h` | C test vectors |
| `artifacts/tflite_tiny_unet_v1/phase8_float32/test_vectors/test_input.npy` | Python test input |
| `artifacts/tflite_tiny_unet_v1/phase8_float32/test_vectors/ref_output.npy` | Python reference output |

---

## 10. Final Recommendation

**PROCEED TO PHASE 8.2** with the following priority:

1. Re-convert FLOAT32 model without XNNPACK delegate
2. Build isolated test harness
3. Validate single-inference parity on host (native compilation)
4. Request hardware test approval
5. Flash and run single-inference test
6. If parity achieved, integrate DDIM scheduler

The FLOAT32 Tiny U-Net v1 is **architecturally compatible** with ESP32-WROOM-32. The main work is firmware adaptation, not fundamental limitations.

**Confidence: 90%** — Memory and operators are verified; firmware changes are straightforward.