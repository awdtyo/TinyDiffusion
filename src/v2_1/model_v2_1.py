"""
Tiny U-Net v2.1 — Extreme Memory Optimization for ESP32-WROOM-32 without PSRAM
Target: < 85 KB tensor arena, < 108 KB largest free block
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from typing import Dict, Any, Tuple
from src.utils import count_parameters, print_model_summary


def build_tiny_unet_v2_1(config: Dict[str, Any]) -> Model:
    """
    Build the Ultra-Compact Tiny U-Net v2.1 for diffusion.
    
    Architecture (configurable channel progression):
    Input: 28x28x12 (noisy image 1ch + 10 class maps + 1 timestep map)
    
    Stage                      Resolution    Channels
    -------------------------------------------------
    Input                      28x28         12
    Initial Conv               28x28         C1
    Encoder Level 1            28x28         C1   (1 residual block)
    Downsample                 14x14         C2
    Encoder Level 2            14x14         C2   (1 residual block)
    Downsample                 7x7           C3
    Bottleneck                 7x7           C3   (1 residual block)
    Upsample 1                 14x14         C2   (RESIZE_BILINEAR + Conv2D)
    Additive Skip              14x14         C2   (ADD)
    Decoder Level 1            14x14         C2   (1 residual block)
    Upsample 2                 28x28         C1   (RESIZE_BILINEAR + Conv2D)
    Additive Skip              28x28         C1   (ADD)
    Decoder Level 2            28x28         C1   (1 residual block)
    Output Conv                28x28         1
    
    Output: 28x28x1 (predicted v)
    
    Target channel configurations to test:
    - 4/6/8/4 (target ~80 KB arena)
    - 4/8/8/4 (target ~90 KB arena) 
    - 6/8/8/6 (target ~100 KB arena)
    """
    model_config = config.get('model', {})
    v21_config = model_config.get('tiny_unet_v2_1', {})
    
    input_height = model_config.get('input_height', 28)
    input_width = model_config.get('input_width', 28)
    total_input_channels = model_config.get('total_input_channels', 12)
    num_classes = model_config.get('num_classes', 10)
    
    # Channel progression - configurable
    channels = v21_config.get('channels', [4, 6, 8, 4])  # [C1, C2, C3, C4] = [encoder1, encoder2, bottleneck, decoder2]
    C1, C2, C3, C4 = channels[0], channels[1], channels[2], channels[3]
    
    initial_conv_filters = C1
    initial_conv_kernel = v21_config.get('initial_conv_kernel', 3)
    
    encoder_config = v21_config.get('encoder', [
        {'resolution': [28, 28], 'channels': C1, 'num_blocks': 1},
        {'resolution': [14, 14], 'channels': C2, 'num_blocks': 1}
    ])
    
    bottleneck_config = v21_config.get('bottleneck', {
        'resolution': [7, 7], 'channels': C3, 'num_blocks': 1
    })
    
    decoder_config = v21_config.get('decoder', [
        {'resolution': [14, 14], 'channels': C2, 'num_blocks': 1},
        {'resolution': [28, 28], 'channels': C4, 'num_blocks': 1}
    ])
    
    output_channels = v21_config.get('output_channels', 1)
    
    residual_config = v21_config.get('residual_block', {
        'kernel_size': 3,
        'activation': 'relu',
        'use_bias': True,
        'padding': 'same'
    })
    
    downsample_config = v21_config.get('downsample', {
        'type': 'strided_conv', 'kernel_size': 3, 'stride': 2
    })
    
    upsample_config = v21_config.get('upsample', {
        'type': 'bilinear_conv', 'kernel_size': 3
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
    # x: 28x28xC1
    
    # ==================== ENCODER ====================
    
    # Encoder Level 1: 28x28, C1 channels
    for i in range(encoder_config[0]['num_blocks']):
        x = _residual_block_v21(
            x, 
            channels=encoder_config[0]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            name=f'encoder1_block{i+1}'
        )
    skip1 = x  # 28x28xC1 - save for decoder skip connection
    
    # Downsample: 28x28 -> 14x14, C1 -> C2
    x = _downsample_block_v21(
        x,
        out_channels=encoder_config[1]['channels'],
        kernel_size=downsample_config.get('kernel_size', 3),
        stride=downsample_config.get('stride', 2),
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='downsample1'
    )
    # x: 14x14xC2
    
    # Encoder Level 2: 14x14, C2 channels
    for i in range(encoder_config[1]['num_blocks']):
        x = _residual_block_v21(
            x,
            channels=encoder_config[1]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            name=f'encoder2_block{i+1}'
        )
    skip2 = x  # 14x14xC2 - save for decoder skip connection
    
    # Downsample: 14x14 -> 7x7, C2 -> C3
    x = _downsample_block_v21(
        x,
        out_channels=bottleneck_config['channels'],
        kernel_size=downsample_config.get('kernel_size', 3),
        stride=downsample_config.get('stride', 2),
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='downsample2'
    )
    # x: 7x7xC3
    
    # Bottleneck: 7x7, C3 channels
    for i in range(bottleneck_config['num_blocks']):
        x = _residual_block_v21(
            x,
            channels=bottleneck_config['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            name=f'bottleneck_block{i+1}'
        )
    # x: 7x7xC3
    
    # ==================== DECODER ====================
    
    # Upsample 1: 7x7 -> 14x14 using RESIZE_BILINEAR + Conv2D
    x = _upsample_block_v21(
        x,
        out_channels=decoder_config[0]['channels'],
        kernel_size=upsample_config.get('kernel_size', 3),
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='upsample1'
    )
    # x: 14x14xC2
    
    # Additive skip connection from encoder level 2 (14x14xC2)
    x = layers.Add(name='skip_add1')([x, skip2])
    
    # Decoder Level 1: 14x14, C2 channels
    for i in range(decoder_config[0]['num_blocks']):
        x = _residual_block_v21(
            x,
            channels=decoder_config[0]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            name=f'decoder1_block{i+1}'
        )
    
    # Upsample 2: 14x14 -> 28x28 using RESIZE_BILINEAR + Conv2D
    x = _upsample_block_v21(
        x,
        out_channels=decoder_config[1]['channels'],
        kernel_size=upsample_config.get('kernel_size', 3),
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='upsample2'
    )
    # x: 28x28xC4
    
    # Additive skip connection from encoder level 1 (28x28xC1)
    # Note: if C4 != C1, need 1x1 conv projection
    if C4 != C1:
        skip1_proj = layers.Conv2D(
            filters=C4,
            kernel_size=1,
            padding='same',
            use_bias=use_bias,
            name='skip1_proj'
        )(skip1)
        x = layers.Add(name='skip_add2')([x, skip1_proj])
    else:
        x = layers.Add(name='skip_add2')([x, skip1])
    
    # Decoder Level 2: 28x28, C4 channels
    for i in range(decoder_config[1]['num_blocks']):
        x = _residual_block_v21(
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
    
    model = Model(inputs=inputs, outputs=outputs, name='TinyUNetV2_1_UltraCompact')
    
    return model


def _residual_block_v21(x: tf.Tensor, channels: int, kernel_size: int = 3,
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


def _downsample_block_v21(x: tf.Tensor, out_channels: int, kernel_size: int = 3,
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


def _upsample_block_v21(x: tf.Tensor, out_channels: int, kernel_size: int = 3,
                        activation: str = 'relu', use_bias: bool = True,
                        padding: str = 'same', name: str = 'upsample') -> tf.Tensor:
    """
    Upsampling using RESIZE_BILINEAR + Conv2D.
    Avoids TILE ops that nearest-neighbor upsampling generates.
    """
    # Use resize instead of UpSampling2D to avoid TILE ops
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


def build_inference_model_v2_1(config: Dict[str, Any]) -> Model:
    """Build the standalone v2.1 model for inference (export)."""
    return build_tiny_unet_v2_1(config)


def get_v2_1_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get v2.1 model configuration summary for logging."""
    model = build_tiny_unet_v2_1(config)
    actual_params = count_parameters(model)
    
    v21_config = config.get('model', {}).get('tiny_unet_v2_1', {})
    channels = v21_config.get('channels', [4, 6, 8, 4])
    
    encoder_config = v21_config.get('encoder', [
        {'resolution': [28, 28], 'channels': channels[0], 'num_blocks': 1},
        {'resolution': [14, 14], 'channels': channels[1], 'num_blocks': 1}
    ])
    
    bottleneck_config = v21_config.get('bottleneck', {
        'resolution': [7, 7], 'channels': channels[2], 'num_blocks': 1
    })
    
    decoder_config = v21_config.get('decoder', [
        {'resolution': [14, 14], 'channels': channels[1], 'num_blocks': 1},
        {'resolution': [28, 28], 'channels': channels[3], 'num_blocks': 1}
    ])
    
    num_residual_blocks = (encoder_config[0]['num_blocks'] + 
                          encoder_config[1]['num_blocks'] + 
                          bottleneck_config['num_blocks'] + 
                          decoder_config[0]['num_blocks'] + 
                          decoder_config[1]['num_blocks'])
    
    return {
        'input_shape': (28, 28, 12),
        'output_shape': (28, 28, 1),
        'channels': channels,
        'residual_blocks': num_residual_blocks,
        'actual_params': actual_params,
        'architecture': f'Tiny U-Net v2.1 UltraCompact ({channels[0]}/{channels[1]}/{channels[2]}/{channels[3]})',
        'max_channels': max(channels),
        'downsampling_stages': 2,
        'upsampling_method': 'bilinear_conv',
        'skip_connection_type': 'additive',
        'conditioning': 'direct_spatial_input'
    }


def build_inference_model(config: Dict[str, Any]) -> Model:
    """Build the standalone model for inference (export). Selects architecture based on config."""
    model_config = config.get('model', {})
    architecture = model_config.get('architecture', 'conditional_cnn')
    
    if architecture == 'tiny_unet_v2_1':
        return build_tiny_unet_v2_1(config)
    elif architecture == 'tiny_unet_v2':
        from src.v2.model_v2 import build_tiny_unet_v2
        return build_tiny_unet_v2(config)
    elif architecture == 'tiny_unet_v1':
        from src.model import build_tiny_unet_v1
        return build_tiny_unet_v1(config)
    else:
        from src.model import build_conditional_cnn
        return build_conditional_cnn(config)


if __name__ == "__main__":
    # Test the v2.1 model with different channel configurations
    import yaml
    import numpy as np
    
    channel_configs = [
        [4, 6, 8, 4],
        [4, 8, 8, 4],
        [6, 8, 8, 6],
    ]
    
    for channels in channel_configs:
        config = {
            'model': {
                'input_height': 28,
                'input_width': 28,
                'total_input_channels': 12,
                'num_classes': 10,
                'architecture': 'tiny_unet_v2_1',
                'tiny_unet_v2_1': {
                    'channels': channels,
                    'initial_conv_kernel': 3,
                    'encoder': [
                        {'resolution': [28, 28], 'channels': channels[0], 'num_blocks': 1},
                        {'resolution': [14, 14], 'channels': channels[1], 'num_blocks': 1}
                    ],
                    'bottleneck': {
                        'resolution': [7, 7], 'channels': channels[2], 'num_blocks': 1
                    },
                    'decoder': [
                        {'resolution': [14, 14], 'channels': channels[1], 'num_blocks': 1},
                        {'resolution': [28, 28], 'channels': channels[3], 'num_blocks': 1}
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
                        'type': 'bilinear_conv',
                        'kernel_size': 3
                    }
                }
            }
        }
        
        model = build_tiny_unet_v2_1(config)
        summary = get_v2_1_config_summary(config)
        print(f"\n{'='*60}")
        print(f"Channels: {channels} | Params: {summary['actual_params']:,}")
        print(f"Max channels: {summary['max_channels']}")
        
        # Test forward pass
        dummy = np.random.randn(1, 28, 28, 12).astype(np.float32)
        output = model(dummy, training=False)
        print(f"Output shape: {output.shape} | Range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")