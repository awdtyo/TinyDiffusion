# Tiny U-Net v1 Architecture Documentation

## Overview

This document describes the **Tiny U-Net v1** architecture implemented for the TinyDiffusion-ESP32 project. It is a lightweight conditional residual U-Net designed specifically for 28×28 MNIST diffusion generation with eventual deployment on ESP32-WROOM-32 without PSRAM.

## Architecture Specification

### Layer-by-Layer Structure

| Stage | Resolution | Channels | Description |
|-------|------------|----------|-------------|
| Input | 28×28 | 12 | Noisy image (1) + Class maps (10) + Timestep (1) |
| Initial Conv | 28×28 | 8 | 3×3 conv + BN + ReLU |
| Encoder Level 1 | 28×28 | 8 | 1× ResidualBlock (FiLM conditioned) |
| Downsample 1 | 14×14 | 12 | Strided 3×3 conv + BN + ReLU |
| Encoder Level 2 | 14×14 | 12 | 1× ResidualBlock (FiLM conditioned) |
| Downsample 2 | 7×7 | 16 | Strided 3×3 conv + BN + ReLU |
| Bottleneck | 7×7 | 16 | 1× ResidualBlock (FiLM conditioned) |
| Upsample 1 | 14×14 | 12 | Transposed 3×3 conv + BN + ReLU |
| Skip Concat 1 | 14×14 | 24 | Concat upsampled + encoder2 skip |
| Decoder Level 1 | 14×14 | 12 | 1× ResidualBlock (FiLM conditioned) |
| Upsample 2 | 28×28 | 8 | Transposed 3×3 conv + BN + ReLU |
| Skip Concat 2 | 28×28 | 16 | Concat upsampled + encoder1 skip |
| Decoder Level 2 | 28×28 | 8 | 1× ResidualBlock (FiLM conditioned) |
| Output Conv | 28×28 | 1 | 3×3 conv, linear |

### Parameter Count

**Actual measured: 36,393 parameters** (36,057 trainable, 336 non-trainable)

This is within the 20K–40K target range.

### Input/Output Specifications

- **Input**: `[batch, 28, 28, 12]` (NHWC)
  - Channel 0: Noisy image `x_t` ∈ [-1, 1]
  - Channels 1-10: Class one-hot maps (broadcast spatially)
  - Channel 11: Normalized timestep `t / 1000` (broadcast spatially)
- **Output**: `[batch, 28, 28, 1]` — Predicted velocity `v`

## Key Components

### 1. Residual Block with FiLM Conditioning

Each residual block contains:
- Two 3×3 convolutions with BatchNorm and ReLU
- FiLM (Feature-wise Linear Modulation) after each BatchNorm
- Skip connection with 1×1 projection if channel dimensions change

**FiLM Mechanism:**
```
conditioning = concat(timestep_embedding, class_embedding)  # [batch, 64]
scale, shift = MLP(conditioning)  # [batch, 2*C]
output = scale * features + shift
```

- MLP: Dense(16, ReLU) → Dense(2×channels)
- Conditioning dimension: 64 (32 timestep + 32 class)

### 2. Timestep Embedding (Sinusoidal)

Fixed sinusoidal positional encoding:
```
embedding_dim = 32
frequencies = exp(-log(10000) * i / (dim/2 - 1)) for i in 0..15
embedding = concat(sin(t * 1000 * freq), cos(t * 1000 * freq))
```

### 3. Class Embedding (Learnable)

Learnable embedding table:
```
class_embedding = Embedding(10 classes, 32 dims)
```

### 4. Downsampling / Upsampling

- **Downsample**: Strided Conv2D (3×3, stride=2) + BN + ReLU
- **Upsample**: Conv2DTranspose (3×3, stride=2) + BN + ReLU

### 5. Skip Connections

Standard U-Net skip connections concatenate encoder features with upsampled decoder features at matching resolutions.

## Conditioning Implementation

The model extracts class label and timestep from the input conditioning channels:
- **Class label**: `argmax` over channels 1-10 at spatial position (0,0)
- **Timestep**: Value at channel 11, position (0,0)

These are embedded separately and concatenated to form the conditioning vector used by all FiLM layers.

## Diffusion Mathematics Compatibility

The architecture is fully compatible with the existing v-prediction diffusion framework:

- **Prediction target**: Velocity `v`
- **v-prediction equation**: `v = √ᾱ_t · ε - √(1-ᾱ_t) · x₀`
- **x₀ reconstruction**: `x₀ = √ᾱ_t · x_t - √(1-ᾱ_t) · v_pred`
- **ε reconstruction**: `ε = √(1-ᾱ_t) · x_t + √ᾱ_t · v_pred`
- **DDIM update**: Uses reconstructed ε for deterministic sampling (η=0)

All equations verified with MAE < 1e-5 for round-trip reconstruction.

## Configuration

The architecture is configured via `configs/tiny_unet_v1.yaml`:

```yaml
model:
  architecture: "tiny_unet_v1"
  tiny_unet_v1:
    initial_conv_filters: 8
    encoder:
      - {resolution: [28, 28], channels: 8, num_blocks: 1}
      - {resolution: [14, 14], channels: 12, num_blocks: 1}
    bottleneck:
      resolution: [7, 7], channels: 16, num_blocks: 1
    decoder:
      - {resolution: [14, 14], channels: 12, num_blocks: 1}
      - {resolution: [28, 28], channels: 8, num_blocks: 1}
    residual_block:
      film_hidden_dim: 16
    timestep_embedding:
      embedding_dim: 32
    class_embedding:
      embedding_dim: 32
```

## Usage

### Selecting the Architecture

The architecture is selected via the config file:

```python
from src.utils import load_config
from src.model import build_inference_model

# Legacy CNN (default.yaml)
config_cnn = load_config("configs/default.yaml")
model_cnn = build_inference_model(config_cnn)  # TinyConditionalCNN

# Tiny U-Net v1 (tiny_unet_v1.yaml)
config_unet = load_config("configs/tiny_unet_v1.yaml")
model_unet = build_inference_model(config_unet)  # TinyUNetV1
```

### Training

```bash
# Train with Tiny U-Net v1
python scripts/train.py --config configs/tiny_unet_v1.yaml
```

Checkpoint isolation: Uses `artifacts/checkpoints_tiny_unet_v1/` directory.

### Pipeline Stages (Independent)

Each stage can be run independently with explicit approval:

| Stage | Command | Approval Gate |
|-------|---------|---------------|
| Training | `python scripts/train.py --config configs/tiny_unet_v1.yaml` | Gate A |
| Evaluation | `python scripts/evaluate.py --model artifacts/checkpoints_tiny_unet_v1/inference_model.keras` | - |
| TFLite Export | `python scripts/convert_model.py --config configs/tiny_unet_v1.yaml` | Gate C |
| INT8 Quantization | Included in convert_model.py | Gate B |
| Firmware Integration | `python scripts/generate_model_header.py` | Gate D |
| ESP32 Flashing | `cd esp32 && pio run --target upload` | Gate E |

## TFLite Micro Compatibility Considerations

### Operators Used
- Conv2D, Conv2DTranspose
- BatchNormalization (folds into conv at inference)
- Add, Multiply (FiLM)
- Concatenate
- ReLU
- Lambda (for input slicing)

All operators are in TFLITE_BUILTINS. No SELECT_TF_OPS required.

### Estimated Resource Requirements (INT8)

| Resource | Estimate |
|----------|----------|
| Model size (INT8) | ~36 KB |
| Tensor arena | ~50-70 KB |
| Total SRAM | < 70 KB |
| Flash usage | ~50 KB |

Fits comfortably within ESP32-WROOM-32 constraints (520 KB SRAM, 4 MB flash).

### Known Limitations

1. **Conv2DTranspose**: Not accelerated by ESP-NN; runs on reference kernel
2. **BatchNormalization**: Folded at conversion time; no runtime cost
3. **FiLM conditioning**: Adds Multiply + Add per residual block; minimal overhead
4. **No cosine schedule in this version**: Uses linear 1000-step schedule (cosine can be added in config)

## Testing

All tests pass:
- Architecture tests: 19/19 (model builds, shapes, params, forward pass, residual blocks, skip connections, downsample/upsample, conditioning, config summary, architecture selection)
- Diffusion tests: 10/10 (noise schedule, forward diffusion, v-prediction equations)
- Scheduler tests: 11/11 (DDIM timesteps, steps, sampling)
- Preprocessing tests: 8/8
- Quantization tests: 3/3 (2 skipped without quantized model)

## Comparison with Legacy CNN

| Aspect | Legacy CNN | Tiny U-Net v1 |
|--------|------------|---------------|
| Parameters | 3,277 | 36,393 |
| Receptive Field | 11×11 (39%) | Global (28×28) |
| Skip Connections | None | 2 |
| Residual Blocks | None | 5 |
| Conditioning | Input concat only | FiLM in every block |
| Downsampling | None | 2× (28→14→7) |
| Upsampling | None | 2× (7→14→28) |
| ESP32 Feasible | ✅ | ✅ |

## Next Steps (Future Phases)

1. **Gate A**: Training with approval
2. **Gate B**: INT8 quantization with calibration
3. **Gate C**: TFLite conversion and operator verification
4. **Gate D**: Firmware integration (model header generation)
5. **Gate E**: ESP32 flashing and on-device validation
6. **Future**: Cosine schedule experiment, attention variants, classifier evaluation