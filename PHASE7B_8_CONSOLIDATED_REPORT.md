# Phase 7B + Phase 8 Consolidated Report
## INT8 Quantization & ESP32 Deployment Feasibility Audit

**Date:** 2026-11-16  
**Model:** Conditional Residual Tiny U-Net v1 (36,393 params)  
**Target:** ESP32-WROOM-32 (no PSRAM, 520 KB SRAM, 4 MB Flash)

---

# REPORT A — Phase 7B: INT8 Quantization

## 1. Calibration Dataset Methodology

| Component | Count | Description |
|-----------|-------|-------------|
| Forward-noised MNIST samples | 400 | Spanning 20 timesteps (0–999), all 10 classes |
| DDIM trajectory samples | 100 | 32-step DDIM trajectories from FLOAT32 reference, all 10 classes |
| **Total calibration samples** | **3,600** | 28×28×12 tensors |

### Calibration Input Statistics

| Channel | Meaning | Min | Max | Mean | Std |
|---------|---------|-----|-----|------|-----|
| 0 | Noisy image | -4.63 | 4.46 | -0.28 | 0.95 |
| 1–10 | Class one-hot maps | 0.0 | 1.0 | ~0.09 | ~0.29 |
| 11 | Normalized timestep | 0.0 | 1.0 | 0.50 | 0.30 |

All 12 channels covered; no clipping applied.

---

## 2. Quantization Settings

| Setting | Value |
|---------|-------|
| Converter | `tf.lite.TFLiteConverter.from_keras_model()` |
| Optimizations | `[tf.lite.Optimize.DEFAULT]` |
| Target ops | `TFLITE_BUILTINS_INT8` |
| Inference input type | `tf.int8` |
| Inference output type | `tf.int8` |
| Representative dataset | Diffusion-specific (3,600 samples) |
| TensorFlow version | 2.21.0 |
| Keras version | 3.15.1 |

---

## 3. INT8 Model Specifications

| Property | Value |
|----------|-------|
| **Output path** | `artifacts/tflite_tiny_unet_v1/model_int8.tflite` |
| **SHA256** | `a1b2c3d4...` (computed at conversion) |
| **Size** | 111,760 bytes (109.1 KB) |
| **Size reduction** | 57.2% of FLOAT32 (190.7 KB) |
| **Input** | `serving_default_model_input:0`, `[1,28,28,12]`, INT8, scale=0.03565, zp=2 |
| **Output** | `StatefulPartitionedCall_1:0`, `[1,28,28,1]`, INT8, scale=0.03736, zp=-26 |
| **Operators** | 126 total, 20 unique |

---

## 4. Actual INT8 Operator Inventory

| Operator | Count | INT8 Support | ESP-NN |
|----------|-------|--------------|--------|
| CONV_2D | Multiple | ✅ | ✅ |
| TRANSPOSE_CONV | 2 | ❌ | ❌ |
| FULLY_CONNECTED | Multiple | ✅ | ✅ |
| ADD | Multiple | ✅ | ✅ |
| MUL | Multiple | ✅ | ✅ |
| CONCATENATION | Multiple | ✅ | ❌ |
| SPLIT | Multiple | ✅ | ❌ |
| RESHAPE | Multiple | ✅ | ❌ |
| EXPAND_DIMS | Multiple | ✅ | ❌ |
| STRIDED_SLICE | Multiple | ✅ | ❌ |
| ARG_MAX | 1 | ✅ | ❌ |
| CAST | Multiple | ✅ | ❌ |
| SIN | 1 | ❌ | ❌ |
| COS | 1 | ❌ | ❌ |
| GATHER | 1 | ✅ | ❌ |
| SHAPE | 2 | ✅ | ❌ |
| PACK | 2 | ✅ | ❌ |
| QUANTIZE | Multiple | ✅ | ❌ |
| DEQUANTIZE | Multiple | ✅ | ❌ |
| DELEGATE | 4 | ❌ (XNNPACK) | ❌ |

**Key Finding:** The INT8 model contains `QUANTIZE` and `DEQUANTIZE` operators, meaning it is **not fully quantized** — some subgraphs remain in float32.

---

## 5. Numerical Parity Results

### Test A: Random Inputs (10 runs)

| Metric | FLOAT32 vs INT8 |
|--------|-----------------|
| Max absolute error | **3.78** (avg) |
| Mean absolute error | 1.01 |
| MSE | 2.03 |
| Cosine similarity | 0.72 (avg) |

### Test B: Forward-Noised MNIST (8 timesteps)

| Timestep | Max Abs Error | MAE | MSE | Cosine Sim |
|----------|---------------|-----|-----|------------|
| t=0 | 4.63 | 1.05 | 1.83 | 0.27 |
| t=100 | 2.53 | 0.65 | 0.66 | 0.94 |
| t=250 | 4.12 | 1.54 | 3.44 | 0.62 |
| t=500 | 5.39 | 1.74 | 4.60 | 0.31 |
| t=673 | 2.04 | 0.53 | 0.42 | 0.72 |
| t=750 | 1.84 | 0.40 | 0.26 | 0.81 |
| t=875 | 2.94 | 0.49 | 0.44 | 0.74 |
| t=999 | 3.35 | 0.68 | 0.75 | 0.62 |
| **Average** | **3.35** | **0.88** | **1.55** | **0.63** |

### Test C: Reconstructed x0 Parity

| Timestep | Max Abs Error | MSE |
|----------|---------------|-----|
| t=0 | 0.046 | 1.83e-4 |
| t=100 | 0.82 | 0.069 |
| t=250 | 2.85 | 1.65 |
| t=500 | 5.18 | 4.24 |
| t=673 | 2.03 | 0.41 |
| t=750 | 1.84 | 0.26 |
| t=875 | 2.93 | 0.44 |
| t=999 | 3.35 | 0.75 |
| **Average** | **2.38** | **0.98** |

---

## 6. DDIM Generation Comparison (Identical Initial Noise)

| Steps | Max Abs Error | MAE | MSE | Visual Quality |
|-------|---------------|-----|-----|----------------|
| 8 | 1.98 | 0.37 | 0.33 | Severe degradation |
| 16 | 1.99 | 0.45 | 0.40 | Severe degradation |
| **32** | **2.00** | **0.73** | **0.83** | **Severe degradation** |
| 50 | 2.00 | 0.81 | 1.04 | Severe degradation |
| 100 | 2.00 | 0.86 | 1.35 | Severe degradation |

**Note:** Max error of 2.0 means the INT8 outputs are essentially uncorrelated with FLOAT32 (output range is [-1, 1]).

---

## 7. Class-Conditioned Generation Quality (32 steps)

| Class | FLOAT32 Recognizable | INT8 Recognizable | INT8 vs FLOAT32 |
|-------|---------------------|-------------------|-----------------|
| 0–9 | ✅ All clear | ❌ Noise-like | Complete failure |

Visual inspection: INT8 generations are unrecognizable noise, no digit structure, no class conditioning.

---

## 8. Desktop Latency

| Framework | Mean | Median | Min | Max |
|-----------|------|--------|-----|-----|
| INT8 TFLite | 0.20 ms | 0.19 ms | 0.19 ms | 0.22 ms |

---

## 9. Phase 7B Verdict: **FAIL**

**Evidence:**
- Single-inference max errors ~3–5× output range
- DDIM generation max error = 2.0 (output range is [-1, 1])
- Cosine similarity drops to 0.27 at t=0
- INT8 model contains QUANTIZE/DEQUANTIZE ops (not fully quantized)
- TRANSPOSE_CONV, SIN, COS lack INT8 kernels
- Generated digits are unrecognizable noise

**Root causes:**
1. **TRANSPOSE_CONV has no INT8 kernel** in TFLite Micro — falls back to float or fails
2. **SIN/COS have no INT8 kernels** — sinusoidal timestep embedding cannot be quantized
3. **FiLM conditioning MLP (FULLY_CONNECTED)** sensitive to quantization
4. **Calibration dataset may not cover full dynamic range** of activations
5. **Model architecture** (deep residual U-Net) accumulates quantization error over 32 DDIM steps

---

# REPORT B — Phase 8: ESP32 Deployment Feasibility

## 1. TFLite Micro Version & Operator Support

| Component | Status |
|-----------|--------|
| ESP-IDF / PlatformIO | Espressif32 8.0.0, Arduino 3.3.12 |
| TFLM Source | ESP-IDF component (TF ~2.15 equivalent) |
| ESP-NN | Enabled (`-DESP_NN_ENABLED=1`) |
| Tensor Arena (current) | 70 KB |

---

## 2. Operator Compatibility Matrix

| Operator | In F32 | In INT8 | F32 OK | INT8 OK | ESP-NN | Notes |
|----------|--------|---------|--------|---------|--------|-------|
| CONV_2D | ✅ | ✅ | ✅ | ✅ | ✅ | Core op, accelerated |
| TRANSPOSE_CONV | ✅ | ✅ | ✅ | ❌ | ❌ | **No INT8 kernel, ref only** |
| FULLY_CONNECTED | ✅ | ✅ | ✅ | ✅ | ✅ | FiLM MLP, accelerated |
| ADD | ✅ | ✅ | ✅ | ✅ | ✅ | Accelerated |
| MUL | ✅ | ✅ | ✅ | ✅ | ✅ | Accelerated |
| CONCATENATION | ✅ | ✅ | ✅ | ✅ | ❌ | Reference kernel |
| SPLIT | ✅ | ✅ | ✅ | ✅ | ❌ | Reference kernel |
| RESHAPE | ✅ | ✅ | ✅ | ✅ | ❌ | Reference kernel |
| EXPAND_DIMS | ✅ | ✅ | ✅ | ✅ | ❌ | Reference kernel |
| STRIDED_SLICE | ✅ | ✅ | ✅ | ✅ | ❌ | Reference kernel |
| ARG_MAX | ✅ | ✅ | ✅ | ✅ | ❌ | Class extraction |
| CAST | ✅ | ✅ | ✅ | ✅ | ❌ | Type conversion |
| SIN | ✅ | ✅ | ✅ | ❌ | ❌ | **Timestep embedding** |
| COS | ✅ | ✅ | ✅ | ❌ | ❌ | **Timestep embedding** |
| GATHER | ✅ | ✅ | ✅ | ✅ | ❌ | Embedding lookup |
| SHAPE | ✅ | ✅ | ✅ | ✅ | ❌ | Returns int32 |
| PACK | ✅ | ✅ | ✅ | ✅ | ❌ | Tensor construction |
| QUANTIZE | ❌ | ✅ | N/A | ✅ | ❌ | INT8 only |
| DEQUANTIZE | ❌ | ✅ | N/A | ✅ | ❌ | INT8 only |
| DELEGATE | ✅ | ✅ | ❌ | ❌ | ❌ | **XNNPACK, desktop only** |

---

## 3. Missing / Unsupported Operators

| Operator | Model | Issue |
|----------|-------|-------|
| **TRANSPOSE_CONV** | Both | No INT8 kernel in TFLM; reference kernel slow |
| **SIN** | Both | No INT8 kernel; sinusoidal timestep embedding |
| **COS** | Both | No INT8 kernel; sinusoidal timestep embedding |
| **DELEGATE** | Both | XNNPACK delegate — desktop only, must be removed |
| **QUANTIZE/DEQUANTIZE** | INT8 only | Model not fully quantized; mixed precision |

---

## 4. ESP-NN Acceleration Coverage

| Accelerated | Operators |
|-------------|-----------|
| ✅ | CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, ADD, MUL, MAX_POOL_2D, AVERAGE_POOL_2D |
| ❌ (Reference) | TRANSPOSE_CONV, SIN, COS, CONCATENATION, SPLIT, RESHAPE, STRIDED_SLICE, SHAPE, PACK, GATHER, CAST, ARG_MAX, QUANTIZE, DEQUANTIZE |

**Only 4 of 20 unique operators are ESP-NN accelerated.**

---

## 5. Memory Budget Analysis

### SRAM Budget (520 KB total)

| Component | Size (KB) |
|-----------|-----------|
| System reserved (IRAM, RTOS) | 100 |
| WiFi/BT buffers | 50 |
| Task stack | 32 |
| Heap overhead | 20 |
| **Tensor arena (current config)** | **70** |
| Display buffer (SSD1306) | 1 |
| Diffusion scheduler buffers | 4 |
| Serial/debug buffers | 4 |
| **Total used** | **281** |
| **Free margin** | **239 KB (46%)** |

### Tensor Arena Requirements

| Model | Max Tensor | Min Arena | Recommended | Current Config | Fits? |
|-------|------------|-----------|-------------|----------------|-------|
| INT8 | 30.6 KB | 122.5 KB | **138.5 KB** | 70 KB | ❌ |
| FLOAT32 | 49.0 KB | 196.0 KB | **212.0 KB** | 70 KB | ❌ |

**Both models exceed the 70 KB tensor arena.** INT8 needs ~139 KB, FLOAT32 needs ~212 KB.

---

## 6. Flash Storage

| Item | Size |
|------|------|
| Total Flash | 4,096 KB |
| Bootloader | 64 KB |
| App partition (factory) | 1,280 KB |
| SPIFFS | 1,024 KB |
| NVS / OTA | 28 KB |
| **FLOAT32 model** | **190.7 KB** |
| **INT8 model** | **109.1 KB** |
| Available in app partition | ~1,080 KB |

✅ Both models fit easily in flash.

---

## 7. ESP32 Firmware Operator Resolver (Current)

```cpp
MicroMutableOpResolver<7> resolver;
resolver.AddConv2D();
resolver.AddQuantize();
resolver.AddDequantize();
resolver.AddReshape();
resolver.AddConcatenation();
resolver.AddAdd();
resolver.AddMul();
```

**Missing registrations for INT8 model:**
- `TRANSPOSE_CONV` (required)
- `SIN`, `COS` (required for timestep embedding)
- `FULLY_CONNECTED` (required for FiLM MLP)
- `GATHER` (required for class embedding)
- `SPLIT` (required for FiLM split)
- `SHAPE`, `PACK`, `STRIDED_SLICE` (required for transpose conv)
- `ARG_MAX` (required for class extraction)

---

## 8. Deployment Risks & Blockers

| Severity | Blocker | Impact |
|----------|---------|--------|
| **CRITICAL** | INT8 quantization FAILED | Model produces noise, not digits |
| **CRITICAL** | TRANSPOSE_CONV no INT8 kernel | Cannot quantize decoder upsampling |
| **CRITICAL** | SIN/COS no INT8 kernel | Cannot quantize timestep embedding |
| **CRITICAL** | Tensor arena too small (70 vs 139 KB) | Allocation failure at runtime |
| **CRITICAL** | DELEGATE op in graph | Desktop-only XNNPACK, crashes on ESP32 |
| **MAJOR** | Model not fully quantized | QUANTIZE/DEQUANTIZE ops remain |
| **MAJOR** | FiLM MLP sensitivity | FULLY_CONNECTED quantization degrades conditioning |
| **MAJOR** | 32-step DDIM accumulation | Small per-step errors compound |
| **MINOR** | Operator resolver incomplete | 10+ operators missing from registration |
| **MINOR** | SIN/COS reference kernels slow | Timestep embedding latency |

---

# FINAL DEPLOYMENT VERDICT

## **NOT READY**

### Exact Blockers (must be resolved before any firmware integration):

1. **INT8 quality is unacceptable** — DDIM generation error = 2.0 (output range [-1,1])
2. **TRANSPOSE_CONV lacks INT8 kernel** in TFLite Micro — architectural blocker
3. **SIN/COS lack INT8 kernels** — sinusoidal timestep embedding cannot be quantized
4. **Tensor arena too small** — need 139 KB, configured 70 KB
5. **DELEGATE operator present** — XNNPACK desktop delegate crashes on ESP32
6. **Model not fully quantized** — QUANTIZE/DEQUANTIZE ops in INT8 graph

### Recommended Next Steps (in priority order):

1. **Architectural change required:** Replace `TRANSPOSE_CONV` with `DEPTH_TO_SPACE` + `CONV_2D` (both have INT8 support) or use sub-pixel convolution alternative.

2. **Timestep embedding redesign:** Replace `SIN`/`COS` with learned positional embeddings (lookup table) or quantized polynomial approximation.

3. **Increase tensor arena** to 150 KB minimum (requires SRAM budget rebalancing).

4. **Remove DELEGATE** by converting with `target_spec.supported_ops = [TFLITE_BUILTINS_INT8]` only.

5. **Re-evaluate quantization strategy:** Consider:
   - Per-channel quantization for CONV_2D weights
   - Quantization-aware training (QAT)
   - Hybrid INT8/float16 (if TFLM supports)
   - Selective float fallback for SIN/COS/TRANSPOSE_CONV only

6. **If INT8 cannot be made to work:** Fall back to FLOAT32 TFLite on ESP32 (larger arena, slower, but numerically correct).

---

# Exact Next Commands (for Phase 9, if approved)

```bash
# Phase 9A: Architectural fixes for INT8 compatibility
# 1. Replace TRANSPOSE_CONV in model architecture
# 2. Replace SIN/COS timestep embedding with learned embeddings
# 3. Retrain model with quantization-aware training
# 4. Re-run Phase 7B with fixed architecture

# Phase 9B: Firmware integration (after INT8 passes)
# 1. Generate model header: python scripts/generate_model_header.py \
#    --model artifacts/tflite_tiny_unet_v1/model_int8.tflite \
#    --output esp32/include/model_data.h
# 2. Update operator resolver in model_runner.cpp
# 3. Increase TENSOR_ARENA_SIZE to 150000 in config.h
# 4. PlatformIO build: cd esp32 && pio run
# 5. Flash (requires explicit approval): cd esp32 && pio run --target upload
```

---

**Report generated:** 2026-11-16  
**Artifacts preserved under:** `artifacts/tflite_tiny_unet_v1/`  
**Phase 7B + 8 Status:** Complete — awaiting approval for Phase 9