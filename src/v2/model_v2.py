"""
Tiny U-Net v2 — Compact-16 Architecture for ESP32 Deployment
Conditional diffusion model for MNIST with reduced parameters and memory footprint.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from typing import Dict, Any, Tuple
from src.utils import count_parameters, print_model_summary


def build_tiny_unet_v2(config: Dict[str, Any]) -> Model:
    """
    Build the Compact-16 Tiny U-Net v2 for diffusion.
    
    Architecture:
    Input: 28x28x12 (noisy image 1ch + 10 class maps + 1 timestep map)
    
    Stage                      Resolution    Channels
    -------------------------------------------------
    Input                      28x28         12
    Initial Conv               28x28         8
    Encoder Level 1            28x28         8   (1 residual block)
    Downsample                 14x14         16
    Encoder Level 2            14x14         16  (1 residual block)
    Bottleneck                 7x7           16  (1 residual block)
    Nearest-neighbor Upsample  14x14         16
    Decoder Level 1            14x14         16  (1 residual block + ADD skip)
    Nearest-neighbor Upsample  28x28         8
    Decoder Level 2            28x28         8   (1 residual block + ADD skip)
    Output Conv                28x28         1
    
    Output: 28x28x1 (predicted v)
    
    Key design choices:
    - Channel progression: 8 -> 16 -> 16 -> 8 (max 16 channels)
    - Only ONE downsampling stage (28->14->7, then 7->14->28)
    - Three residual blocks total (encoder1, encoder2/bottleneck shared, decoder1, decoder2)
    - Additive skip connections (not concatenate) - saves memory
    - Nearest-neighbor upsampling + Conv2D (no Conv2DTranspose)
    - FiLM conditioning removed - use direct spatial conditioning from input
    - No separate sinusoidal/class embeddings - spatial conditioning in input
    
    Target: ~14-15K parameters, < 100KB FLOAT32 TFLite
    """
    model_config = config.get('model', {})
    v2_config = model_config.get('tiny_unet_v2', {})
    
    input_height = model_config.get('input_height', 28)
    input_width = model_config.get('input_width', 28)
    total_input_channels = model_config.get('total_input_channels', 12)
    num_classes = model_config.get('num_classes', 10)
    
    # Architecture parameters
    initial_conv_filters = v2_config.get('initial_conv_filters', 8)
    initial_conv_kernel = v2_config.get('initial_conv_kernel', 3)
    
    encoder_config = v2_config.get('encoder', [
        {'resolution': [28, 28], 'channels': 8, 'num_blocks': 1},
        {'resolution': [14, 14], 'channels': 16, 'num_blocks': 1}
    ])
    
    bottleneck_config = v2_config.get('bottleneck', {
        'resolution': [7, 7], 'channels': 16, 'num_blocks': 1
    })
    
    decoder_config = v2_config.get('decoder', [
        {'resolution': [14, 14], 'channels': 16, 'num_blocks': 1},
        {'resolution': [28, 28], 'channels': 8, 'num_blocks': 1}
    ])
    
    output_channels = v2_config.get('output_channels', 1)
    
    residual_config = v2_config.get('residual_block', {
        'kernel_size': 3,
        'activation': 'relu',
        'use_bias': True,
        'padding': 'same'
    })
    
    downsample_config = v2_config.get('downsample', {
        'type': 'strided_conv', 'kernel_size': 3, 'stride': 2
    })
    
    upsample_config = v2_config.get('upsample', {
        'type': 'nearest_conv', 'kernel_size': 3  # nearest-neighbor + Conv2D
    })
    
    kernel_size = residual_config['kernel_size']
    activation = residual_config['activation']
    use_bias = residual_config['use_bias']
    padding = residual_config['padding']
    
    # Input
    inputs = Input(shape=(input_height, input_width, total_input_channels), name='model_input')
    
    # Initial convolution on full 12-channel input
    x = layers.Conv2D(
        filters=initial_conv_filters,
        kernel_size=initial_conv_kernel,
        padding=padding,
        use_bias=use_bias,
        name='initial_conv'
    )(inputs)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.Activation(activation, name='initial_act')(x)
    # x: 28x28x8
    
    # ==================== ENCODER ====================
    
    # Encoder Level 1: 28x28, 8 channels
    for i in range(encoder_config[0]['num_blocks']):
        x = _residual_block_v2(
            x, 
            channels=encoder_config[0]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            name=f'encoder1_block{i+1}'
        )
    skip1 = x  # 28x28x8 - save for decoder skip connection
    
    # Downsample: 28x28 -> 14x14, 8 -> 16
    x = _downsample_block_v2(
        x,
        out_channels=encoder_config[1]['channels'],
        kernel_size=downsample_config.get('kernel_size', 3),
        stride=downsample_config.get('stride', 2),
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='downsample1'
    )
    # x: 14x14x16
    
    # Encoder Level 2: 14x14, 16 channels
    for i in range(encoder_config[1]['num_blocks']):
        x = _residual_block_v2(
            x,
            channels=encoder_config[1]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            name=f'encoder2_block{i+1}'
        )
    skip2 = x  # 14x14x16 - save for decoder skip connection
    
    # Downsample: 14x14 -> 7x7, 16 -> 16
    x = _downsample_block_v2(
        x,
        out_channels=bottleneck_config['channels'],
        kernel_size=downsample_config.get('kernel_size', 3),
        stride=downsample_config.get('stride', 2),
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='downsample2'
    )
    # x: 7x7x16
    
    # Bottleneck: 7x7, 16 channels
    for i in range(bottleneck_config['num_blocks']):
        x = _residual_block_v2(
            x,
            channels=bottleneck_config['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            name=f'bottleneck_block{i+1}'
        )
    # x: 7x7x16
    
    # ==================== DECODER ====================
    
    # Upsample 1: 7x7 -> 14x14 using nearest-neighbor + Conv2D
    x = _upsample_block_v2(
        x,
        out_channels=decoder_config[0]['channels'],
        kernel_size=upsample_config.get('kernel_size', 3),
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='upsample1'
    )
    # x: 14x14x16
    
    # Additive skip connection from encoder level 2 (14x14x16)
    # ADD instead of CONCATENATE - saves memory
    x = layers.Add(name='skip_add1')([x, skip2])
    
    # Decoder Level 1: 14x14, 16 channels
    for i in range(decoder_config[0]['num_blocks']):
        x = _residual_block_v2(
            x,
            channels=decoder_config[0]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            name=f'decoder1_block{i+1}'
        )
    
    # Upsample 2: 14x14 -> 28x28 using nearest-neighbor + Conv2D
    x = _upsample_block_v2(
        x,
        out_channels=decoder_config[1]['channels'],
        kernel_size=upsample_config.get('kernel_size', 3),
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='upsample2'
    )
    # x: 28x28x8
    
    # Additive skip connection from encoder level 1 (28x28x8)
    x = layers.Add(name='skip_add2')([x, skip1])
    
    # Decoder Level 2: 28x28, 8 channels
    for i in range(decoder_config[1]['num_blocks']):
        x = _residual_block_v2(
            x,
            channels=decoder_config[1]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            name=f'decoder2_block{i+1}'
        )
    
    # Output convolution
    outputs = layers.Conv2D(
        filters=output_channels,
        kernel_size=3,
        padding=padding,
        use_bias=use_bias,
        activation='linear',
        name='output_conv'
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TinyUNetV2_Compact16')
    
    return model


def _residual_block_v2(x: tf.Tensor, channels: int, kernel_size: int = 3,
                        activation: str = 'relu', use_bias: bool = True,
                        padding: str = 'same', name: str = 'res_block') -> tf.Tensor:
    """
    Simplified residual block WITHOUT FiLM conditioning.
    Uses direct spatial conditioning from input channels.
    """
    residual = x
    
    # First conv
    x = layers.Conv2D(
        filters=channels,
        kernel_size=kernel_size,
        padding=padding,
        use_bias=use_bias,
        name=f'{name}_conv1'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation(activation, name=f'{name}_act1')(x)
    
    # Second conv
    x = layers.Conv2D(
        filters=channels,
        kernel_size=kernel_size,
        padding=padding,
        use_bias=use_bias,
        name=f'{name}_conv2'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    
    # Residual connection (identity if channels match)
    in_channels = residual.shape[-1]
    if in_channels != channels:
        residual = layers.Conv2D(
            filters=channels,
            kernel_size=1,
            padding='same',
            use_bias=use_bias,
            name=f'{name}_residual_conv'
        )(residual)
    
    x = layers.Add(name=f'{name}_add')([x, residual])
    x = layers.Activation(activation, name=f'{name}_final_act')(x)
    
    return x


def _downsample_block_v2(x: tf.Tensor, out_channels: int, kernel_size: int = 3,
                          stride: int = 2, activation: str = 'relu',
                          use_bias: bool = True, padding: str = 'same',
                          name: str = 'downsample') -> tf.Tensor:
    """Downsampling block with strided convolution."""
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=use_bias,
        name=f'{name}_conv'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Activation(activation, name=f'{name}_act')(x)
    return x


def _upsample_block_v2(x: tf.Tensor, out_channels: int, kernel_size: int = 3,
                        activation: str = 'relu', use_bias: bool = True,
                        padding: str = 'same', name: str = 'upsample') -> tf.Tensor:
    """
    Upsampling using resize (bilinear) + Conv2D.
    Avoids TILE ops that nearest-neighbor upsampling generates.
    Uses tf.image.resize which converts to RESIZE_BILINEAR in TFLite.
    """
    # Use resize instead of UpSampling2D to avoid TILE ops
    # This uses RESIZE_BILINEAR in TFLite which is supported
    x = layers.Lambda(
        lambda t: tf.image.resize(t, 
                                  [t.shape[1] * 2, t.shape[2] * 2], 
                                  method='bilinear'),
        name=f'{name}_resize'
    )(x)
    
    # Conv2D to process upsampled features
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        use_bias=use_bias,
        name=f'{name}_conv'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Activation(activation, name=f'{name}_act')(x)
    
    return x


def build_inference_model_v2(config: Dict[str, Any]) -> Model:
    """Build the standalone v2 model for inference (export)."""
    return build_tiny_unet_v2(config)


def get_v2_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get v2 model configuration summary for logging."""
    model = build_tiny_unet_v2(config)
    actual_params = count_parameters(model)
    
    v2_config = config.get('model', {}).get('tiny_unet_v2', {})
    
    encoder_config = v2_config.get('encoder', [
        {'resolution': [28, 28], 'channels': 8, 'num_blocks': 1},
        {'resolution': [14, 14], 'channels': 16, 'num_blocks': 1}
    ])
    
    bottleneck_config = v2_config.get('bottleneck', {
        'resolution': [7, 7], 'channels': 16, 'num_blocks': 1
    })
    
    decoder_config = v2_config.get('decoder', [
        {'resolution': [14, 14], 'channels': 16, 'num_blocks': 1},
        {'resolution': [28, 28], 'channels': 8, 'num_blocks': 1}
    ])
    
    num_residual_blocks = (encoder_config[0]['num_blocks'] + 
                          encoder_config[1]['num_blocks'] + 
                          bottleneck_config['num_blocks'] + 
                          decoder_config[0]['num_blocks'] + 
                          decoder_config[1]['num_blocks'])
    
    return {
        'input_shape': (28, 28, 12),
        'output_shape': (28, 28, 1),
        'residual_blocks': num_residual_blocks,
        'actual_params': actual_params,
        'architecture': 'Tiny U-Net v2 Compact-16',
        'max_channels': 16,
        'downsampling_stages': 2,
        'upsampling_method': 'nearest_conv',
        'skip_connection_type': 'additive',
        'conditioning': 'direct_spatial_input'
    }


# For backward compatibility with existing factory function
def build_inference_model(config: Dict[str, Any]) -> Model:
    """Build the standalone model for inference (export). Selects architecture based on config."""
    model_config = config.get('model', {})
    architecture = model_config.get('architecture', 'conditional_cnn')
    
    if architecture == 'tiny_unet_v2':
        return build_tiny_unet_v2(config)
    elif architecture == 'tiny_unet_v1':
        # Import v1 builder
        from src.model import build_tiny_unet_v1
        return build_tiny_unet_v1(config)
    else:
        # Default to legacy CNN
        from src.model import build_conditional_cnn
        return build_conditional_cnn(config)


if __name__ == "__main__":
    # Test the v2 model
    config = {
        'model': {
            'input_height': 28,
            'input_width': 28,
            'total_input_channels': 12,
            'num_classes': 10,
            'architecture': 'tiny_unet_v2',
            'tiny_unet_v2': {
                'initial_conv_filters': 8,
                'initial_conv_kernel': 3,
                'encoder': [
                    {'resolution': [28, 28], 'channels': 8, 'num_blocks': 1},
                    {'resolution': [14, 14], 'channels': 16, 'num_blocks': 1}
                ],
                'bottleneck': {
                    'resolution': [7, 7], 'channels': 16, 'num_blocks': 1
                },
                'decoder': [
                    {'resolution': [14, 14], 'channels': 16, 'num_blocks': 1},
                    {'resolution': [28, 28], 'channels': 8, 'num_blocks': 1}
                ],
                'output_channels': 1,
                'residual_block': {
                    'kernel_size': 3,
                    'activation': 'relu',
                    'use_bias': True,
                    'padding': 'same'
                },
                'downsample': {
                    'type': 'strided_conv',
                    'kernel_size': 3,
                    'stride': 2
                },
                'upsample': {
                    'type': 'nearest_conv',
                    'kernel_size': 3
                }
            },
            'use_bias': True,
            'padding': 'same'
        }
    }
    
    print("Building Tiny U-Net v2 Compact-16...")
    model = build_tiny_unet_v2(config)
    print_model_summary(model)
    
    # Test forward pass
    import numpy as np
    dummy_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
    output = model(dummy_input, training=False)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")
    
    # Check all layers
    print("\nLayer shapes:")
    for i, layer in enumerate(model.layers):
        print(f"  {i}: {layer.name} - in: {layer.input_shape} -> out: {layer.output_shape}")
    
    print("\nTiny U-Net v2 Compact-16 built successfully!")