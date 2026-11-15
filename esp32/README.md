# TinyDiffusion-ESP32 Firmware

ESP32 firmware for running the tiny conditional diffusion model on-device.

## Hardware Requirements

- **Board**: ESP32 DevKit V1 (ESP32-WROOM-32)
- **Display**: SSD1306 128×64 OLED (I2C)
- **Connections**:
  - OLED VCC → 3.3V
  - OLED GND → GND
  - OLED SDA → GPIO 21
  - OLED SCL → GPIO 22

## Software Requirements

- **PlatformIO** (VS Code extension or CLI)
- **ESP-IDF** (managed by PlatformIO)
- **Arduino Framework** (via PlatformIO)

## Building

### Using PlatformIO CLI

```bash
cd esp32
pio run
```

### Using VS Code

1. Open the `esp32` folder in VS Code
2. Install PlatformIO extension
3. Click "Build" (✓) in PlatformIO toolbar

## Flashing

```bash
# Flash to device
pio run --target upload

# Or with specific port
pio run --target upload --upload-port /dev/ttyUSB0
```

## Monitoring

```bash
# Open serial monitor (115200 baud)
pio device monitor

# With timestamps
pio device monitor --filter time
```

## Serial Commands

| Command | Description |
|---------|-------------|
| `G <0-9>` | Generate specific digit (e.g., `G 7`) |
| `R` | Generate random digit |
| `STEPS <8\|16\|32>` | Set DDIM sampling steps |
| `ETA <0.0-1.0>` | Set DDIM eta (0=deterministic) |
| `SEED <n>` | Deterministic noise seed for Python replay (`0` = hardware RNG) |
| `INFO` | Show model and memory info |
| `HELP` | Show command list |

### Example Session

```
> HELP
=== TinyDiffusion-ESP32 Commands ===
G <0-9>    - Generate specific digit
R          - Generate random digit
STEPS <8|16|32> - Set DDIM sampling steps
ETA <0-1>  - Set DDIM eta (0=deterministic)
INFO       - Show model and memory info
HELP       - Show this help
=====================================

> G 7
OK: Generated digit 7 in 234 ms (8 steps)

> STEPS 16
OK: Sampling steps set to 16

> G 3
OK: Generated digit 3 in 467 ms (16 steps)

> INFO
=== TinyDiffusion-ESP32 Info ===
Model size: 8536 bytes
Tensor arena: 70000 bytes
Arena used: 23356 bytes
Free heap: 231572 bytes
Largest free block: 110580 bytes
Sampling steps: 8
DDIM eta: 0.00
Input quant: scale=0.028883, zp=3
Output quant: scale=0.043014, zp=-21
Noise RNG: hardware seed=1
================================

> SEED 12345
OK: deterministic noise seed=12345

> G 0
... per-step diagnostics ...
OK: Generated digit 0 in 6374 ms (8 steps)
```

`SEED` exists so a generation can be reproduced exactly in Python:

```
PYTHONPATH=. python scripts/replay_reference.py --seed 12345 --class 0 \
    --log /tmp/opencode/serial_seed.log
```

Set `PRINT_STEP_STATS` and `PRINT_FINAL_NUMERIC_GRID` to `0` in
`esp32/src/main.cpp` to remove the temporary diagnostics.

## Memory Usage

| Component | Size |
|-----------|------|
| Model (INT8) | ~12 KB |
| Tensor Arena | 70 KB |
| Input/Output Buffers | ~10 KB |
| Display Buffer | ~1 KB |
| **Total** | **~93 KB** |

Well within the ESP32's ~520 KB SRAM (no PSRAM required).

## Performance (Typical)

| Steps | Generation Time | Per Step |
|-------|----------------|----------|
| 8 | ~200-300 ms | ~25-35 ms |
| 16 | ~400-600 ms | ~25-35 ms |
| 32 | ~800-1200 ms | ~25-35 ms |

Actual times may vary based on CPU frequency and thermal throttling.

## Model Integration

The model bytes are embedded as a 16-byte-aligned C array in
`src/model_data.cpp`; `include/model_data.h` contains only the declarations
and quantization metadata. To update:

1. Train and quantize model (see main project README)
2. Run conversion: `python ../scripts/generate_model_header.py`
3. Rebuild firmware: `pio run`

The generated source derives `g_model_len` from `sizeof(g_model)` and includes
a compile-time size check, so a truncated model array fails the build instead
of crashing at runtime.

## Quantization Parameters

The firmware expects specific quantization parameters that must match the exported model:

- **Input**: INT8, scale and zero-point from model
- **Output**: INT8, scale and zero-point from model

These are automatically extracted from the TFLite model at runtime.

## Display Output

The OLED shows:
- Generated digit (28×28 scaled 2× to 56×56)
- Requested digit class
- Generation time
- DDIM step count
- Free heap memory

## Troubleshooting

### Display not working
- Check I2C connections (SDA=21, SCL=22)
- Verify OLED address (0x3C or 0x3D)
- Try `i2cdetect` on another device

### Model load fails
- Verify `model_data.h` was generated correctly
- Check model size matches `g_tiny_diffusion_model_len`
- Ensure tensor arena is large enough (increase `TENSOR_ARENA_SIZE`)

### Generation crashes
- Increase tensor arena size
- Check for stack overflow (enable stack monitoring)
- Verify quantization parameters match

### Slow performance
- Ensure CPU is at 240 MHz (default)
- Check for thermal throttling
- Verify ESP-NN is being used (check build flags)

## Customization

### Change Display Pins
Edit `include/config.h`:
```cpp
#define DISPLAY_SDA_PIN  21
#define DISPLAY_SCL_PIN  22
```

### Change Sampling Steps
Edit `include/config.h`:
```cpp
#define DEFAULT_SAMPLING_STEPS 8
```

### Increase Tensor Arena
Edit `include/config.h`:
```cpp
#define TENSOR_ARENA_SIZE 80000  // Increase if needed
```

## License

Part of TinyDiffusion-ESP32 project. See main repository license.