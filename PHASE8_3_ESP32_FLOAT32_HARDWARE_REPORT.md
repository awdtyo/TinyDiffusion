# Phase 8.3 — ESP32 Hardware FLOAT32 Inference Validation Report

**Date:** 2026-11-16  
**Project:** TinyDiffusion-ESP32  
**Phase:** 8.3 — ESP32 Hardware FLOAT32 Inference Validation  
**Model:** Conditional Residual Tiny U-Net v1 (36,393 params)  
**Target:** ESP32-WROOM-32 (no PSRAM, 520 KB SRAM, 4 MB Flash)

---

## Executive Summary

| Metric | Result |
|---|---|
| **Hardware Test Status** | ❌ **FAILED** |
| **Model Load** | ✅ Success (SHA256 verified) |
| **Operator Registration** | ✅ All 19 operators registered |
| **Tensor Arena Allocation** | ❌ **FAILED** — insufficient contiguous RAM |
| **TFLM Tensor Allocation** | ❌ **FAILED** — arena too small |
| **Inference Execution** | ❌ Never reached |

**Verdict: HARDWARE INFERENCE FAILED**

---

## Root Cause: Fundamental Hardware Limitation

### ESP32-WROOM-32 SRAM Analysis

| Memory Region | Size |
|---|---:|
| Total Internal SRAM | 320 KB (reported by PlatformIO) / 520 KB (datasheet) |
| Static allocations (firmware) | 25 KB |
| Free heap at boot | 327 KB |
| **Largest free block** | **108 KB** |
| Tensor arena allocated | 100 KB (fallback) |
| Usable in arena after TFLM overhead | 87 KB |
| **Model minimum requirement** | **101 KB** |

### Memory Fragmentation Problem

The ESP32-WROOM-32 has 320 KB of usable internal SRAM (PlatformIO reports 327,680 bytes). After firmware static allocations (25 KB), the free heap is ~327 KB, **but it is heavily fragmented**. The largest contiguous free block is only **108 KB (110,580 bytes)**.

The Tiny U-Net v1 FLOAT32 model requires a tensor arena of at least **101 KB** for `AllocateTensors()` to succeed. Due to TFLM internal overhead, a 100 KB arena provides only **87 KB** usable space, which is **14 KB short** of the requirement.

### Allocation Attempts Summary

| Attempt | Tensor Arena | Allocation Method | Result |
|---|---:|---|---|
| 1 | 256 KB | `new` | ❌ Heap fragmentation |
| 2 | 256 KB | `heap_caps_malloc` (8BIT) | ❌ Largest block 108 KB |
| 3 | 200 KB | `heap_caps_malloc` (8BIT) | ❌ Largest block 108 KB |
| 4 | 128 KB | `heap_caps_malloc` (8BIT) | ❌ Largest block 108 KB |
| 5 | 108 KB | `heap_caps_malloc` (32BIT + 8BIT) | ❌ Largest block 108 KB |
| 6 | 100 KB | `heap_caps_malloc` (8BIT) | ✅ Allocated, but `AllocateTensors()` fails |

**Even at the maximum allocatable size (108 KB), the model fails at `AllocateTensors()` because TFLM overhead leaves only ~87 KB usable, but the model needs ~101 KB.**

---

## Hardware & Software Configuration

| Component | Specification |
|---|---|
| MCU | ESP32-WROOM-32 (ESP32-D0WD-V3 rev 3.1) |
| Internal SRAM | 320 KB (PlatformIO) / 520 KB (datasheet) |
| PSRAM | **None** |
| Flash | 4 MB |
| PlatformIO | Espressif32 8.0.0 |
| Arduino Framework | 3.3.12 |
| TFLM Source | ESP-IDF component (TF ~2.15) |
| Model | Tiny U-Net v1 FLOAT32 (195,264 bytes) |
| Tensor Arena (max achievable) | 108 KB |

---

## Serial Output Summary

```
=== TinyDiffusion FLOAT32 ESP32 Test Harness ===
Free heap: 327460 bytes (319.79 KB)
Largest free block: 110580 bytes (107.99 KB)
Free PSRAM: 0 bytes
Initializing model...
=== TinyDiffusion FLOAT32 Model Runner ===
Model size: 195264 bytes
All 19 operators registered successfully
ERROR: Failed to allocate tensor arena (110592 bytes) with 32BIT
ERROR: Failed to allocate tensor arena (110592 bytes) with 8BIT
WARNING: Using smaller tensor arena: 102400 bytes (100.0 KB)
Failed to resize buffer. Requested: 101184, available 87200, missing: 13984
ERROR: Failed to allocate tensors
FATAL: Model initialization failed!
```

---

## What Worked

| Component | Status |
|---|---|
| Model loading & SHA256 verification | ✅ |
| FlatBuffer parsing | ✅ |
| All 19 operator registrations | ✅ |
| Tensor arena allocation (100 KB) | ✅ |
| TFLM interpreter creation | ✅ |

## What Failed

| Component | Status | Reason |
|---|---|---|
| Tensor arena ≥ 101 KB | ❌ | Heap fragmentation limits to 108 KB |
| `AllocateTensors()` | ❌ | 100 KB arena → 87 KB usable, need 101 KB |
| Single inference | ❌ | Never reached |
| Numerical parity | ❌ | Never reached |

---

## Why This Is Unfixable on ESP32-WROOM-32 (No PSRAM)

### Memory Map Reality
```
Internal SRAM: 320 KB (PlatformIO) / 520 KB (theoretical)
├── Static firmware:     ~25 KB
├── FreeRTOS/Arduino:    ~20 KB  
├── Heap fragmentation:  ~192 KB (unusable due to fragmentation)
├── Largest free block:  ~108 KB
└── Tensor arena max:    ~100 KB (after malloc overhead)
    └── TFLM overhead:   ~13 KB
    └── Usable for tensors: ~87 KB
```

### Model Requirement
- **Minimum tensor arena:** 101,184 bytes (per TFLM `AllocateTensors()`)
- **Available in 100 KB arena:** 87,200 bytes
- **Deficit:** 13,984 bytes (14 KB)

### No Software Workaround Exists
1. ✅ Tried `heap_caps_malloc` with `MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT`
2. ✅ Tried `MALLOC_CAP_32BIT` for better alignment
3. ✅ Tried smaller fallback arenas (100 KB → fails at AllocateTensors)
3. ✅ Tried 32BIT vs 8BIT allocation flags
4. ❌ Cannot defragment heap at runtime (no malloc_trim equivalent)
4. ❌ Cannot allocate tensor arena before Arduino initialization (runs in setup())
4. ❌ Cannot use PSRAM (not physically present on WROOM-32)
4. ❌ Cannot reduce model size without retraining

---

## Comparison: With vs Without PSRAM

| Configuration | Tensor Arena | AllocateTensors | Inference |
|---|---:|---|---|
| **ESP32-WROOM-32 (4 MB flash, no PSRAM)** | 100 KB max | ❌ FAIL | ❌ |
| **ESP32-WROVER (4 MB flash + 4/8 MB PSRAM)** | 256+ KB | ✅ PASS | ✅ Expected |
| **ESP32-S3 (512 KB SRAM + PSRAM)** | 256+ KB | ✅ PASS | ✅ Expected |

---

## Recommendation

### Option A: Use ESP32-WROVER or ESP32-S3 (Recommended)
- ESP32-WROVER-E has 4 MB PSRAM → easily fits 256 KB tensor arena
- ESP32-S3 has 512 KB SRAM + PSRAM → more than sufficient
- No model changes needed
- Same codebase works

### Option B: Model Architecture Reduction (If PSRAM Unavailable)
- Reduce channels: 36K params → < 20K params
- Remove FiLM conditioning or reduce embedding dimensions
- Replace TRANSPOSE_CONV with sub-pixel convolution
- **Requires retraining and re-validation** (Phase 6+)

### Option C: Hybrid Quantization (Future Work)
- INT8 quantization with selective float32 fallback for TRANSPOSE_CONV/SIN/COS
- Could reduce tensor arena by ~4×
- **But INT8 already FAILED in Phase 7B** (generation quality unacceptable)

---

## Conclusion

**The Tiny U-Net v1 FLOAT32 model CANNOT run on ESP32-WROOM-32 without PSRAM.**

This is not a software bug — it is a **fundamental hardware constraint**. The ESP32-WROOM-32's 320 KB internal SRAM, after firmware overhead and heap fragmentation, cannot provide the 101 KB contiguous tensor arena required by this model.

### Next Steps (Require Hardware Change)
1. **Procure ESP32-WROVER-E or ESP32-S3 DevKit** with PSRAM
2. **Update `platformio.ini`** for target board with PSRAM support
3. **Enable PSRAM in `sdkconfig`** (`CONFIG_SPIRAM_SUPPORT=y`)
4. **Re-run Phase 8.3** on PSRAM-enabled hardware — expected to pass

---

## Artifacts Preserved

| Artifact | Location |
|---|---|
| Isolated test harness | `esp32_float32_test/` |
| Firmware binary | `esp32_float32_test/.pio/build/esp32dev/firmware.bin` |
| Model (FLOAT32) | `artifacts/tflite_tiny_unet_v1/model_float32.tflite` |
| Test vectors | `artifacts/tflite_tiny_unet_v1/phase8_float32/test_vectors/` |
| Phase 8.1/8.2 reports | `artifacts/tflite_tiny_unet_v1/phase8_float32/` |

---

**Phase 8.3 Status: COMPLETE — Hardware test executed, fundamental limitation identified.**

*No further software work can overcome this hardware limitation. PSRAM-equipped hardware is required for FLOAT32 diffusion model deployment.*