"""
Tests for model architecture and dimensions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf
import pytest
from src.model import (build_inference_model, build_conditional_cnn, build_tiny_unet_v1, 
                       get_model_config_summary)
from src.utils import load_config, count_parameters


def test_model_build():
    """Test that model builds correctly."""
    config = load_config("configs/default.yaml")
    model = build_inference_model(config)
    
    assert model is not None
    assert isinstance(model, tf.keras.Model)


def test_model_input_shape():
    """Test model input shape."""
    config = load_config("configs/default.yaml")
    model = build_inference_model(config)
    
    # Input: [batch, 28, 28, 12]
    assert model.input_shape == (None, 28, 28, 12)


def test_model_output_shape():
    """Test model output shape."""
    config = load_config("configs/default.yaml")
    model = build_inference_model(config)
    
    # Output: [batch, 28, 28, 1]
    assert model.output_shape == (None, 28, 28, 1)


def test_model_parameter_count():
    """Test parameter count matches expected ~3,277."""
    config = load_config("configs/default.yaml")
    model = build_inference_model(config)
    
    param_count = count_parameters(model)
    # Allow small tolerance for implementation differences
    assert 3200 <= param_count <= 3400, f"Parameter count {param_count} not in expected range"


def test_model_forward_pass():
    """Test forward pass with random input."""
    config = load_config("configs/default.yaml")
    model = build_inference_model(config)
    
    # Random input
    test_input = np.random.randn(2, 28, 28, 12).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    
    assert output.shape == (2, 28, 28, 1)
    assert output.dtype == np.float32


def test_model_layers():
    """Test model has correct number of layers."""
    config = load_config("configs/default.yaml")
    model = build_inference_model(config)
    
    # Should have 5 Conv2D + 4 Activation layers = 9 layers (excluding Input)
    conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    assert len(conv_layers) == 5
    
    # Check filter counts
    expected_filters = [8, 8, 12, 8, 1]
    for i, (layer, expected) in enumerate(zip(conv_layers, expected_filters)):
        assert layer.filters == expected, f"Layer {i} has {layer.filters} filters, expected {expected}"


def test_model_config_summary():
    """Test config summary."""
    config = load_config("configs/default.yaml")
    summary = get_model_config_summary(config)
    
    assert 'input_shape' in summary
    assert 'output_shape' in summary
    assert 'conv_layers' in summary
    assert summary['conv_layers'] == 5


# ========================================================================
# Tiny U-Net v1 Tests
# ========================================================================

def test_tiny_unet_v1_build():
    """Test that Tiny U-Net v1 builds correctly."""
    config = load_config("configs/tiny_unet_v1.yaml")
    model = build_inference_model(config)
    
    assert model is not None
    assert isinstance(model, tf.keras.Model)
    assert model.name == 'TinyUNetV1'


def test_tiny_unet_v1_input_shape():
    """Test Tiny U-Net v1 input shape."""
    config = load_config("configs/tiny_unet_v1.yaml")
    model = build_inference_model(config)
    
    # Input: [batch, 28, 28, 12]
    assert model.input_shape == (None, 28, 28, 12)


def test_tiny_unet_v1_output_shape():
    """Test Tiny U-Net v1 output shape."""
    config = load_config("configs/tiny_unet_v1.yaml")
    model = build_inference_model(config)
    
    # Output: [batch, 28, 28, 1]
    assert model.output_shape == (None, 28, 28, 1)


def test_tiny_unet_v1_parameter_count():
    """Test Tiny U-Net v1 parameter count is within target range (20K-40K)."""
    config = load_config("configs/tiny_unet_v1.yaml")
    model = build_inference_model(config)
    
    param_count = count_parameters(model)
    # Target: 20K-40K parameters
    assert 20000 <= param_count <= 40000, f"Parameter count {param_count} not in expected range 20K-40K"


def test_tiny_unet_v1_forward_pass():
    """Test Tiny U-Net v1 forward pass with random input."""
    config = load_config("configs/tiny_unet_v1.yaml")
    model = build_inference_model(config)
    
    # Random input - batch size 1
    test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    
    assert output.shape == (1, 28, 28, 1)
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))
    
    # Random input - batch size 4
    test_input = np.random.randn(4, 28, 28, 12).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    
    assert output.shape == (4, 28, 28, 1)
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))


def test_tiny_unet_v1_residual_blocks():
    """Test Tiny U-Net v1 has residual blocks."""
    config = load_config("configs/tiny_unet_v1.yaml")
    model = build_inference_model(config)
    
    # Check for ResidualBlock layers
    residual_blocks = [l for l in model.layers if l.__class__.__name__ == 'ResidualBlock']
    # Expected: 5 residual blocks (encoder1:1, encoder2:1, bottleneck:1, decoder1:1, decoder2:1)
    assert len(residual_blocks) == 5, f"Expected 5 residual blocks, got {len(residual_blocks)}"


def test_tiny_unet_v1_skip_connections():
    """Test Tiny U-Net v1 has skip connections (concatenate layers in decoder)."""
    config = load_config("configs/tiny_unet_v1.yaml")
    model = build_inference_model(config)
    
    # Check for concatenate layers used for skip connections
    concat_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Concatenate)]
    # Expected: 2 skip connections (skip_concat1, skip_concat2) + 1 conditioning_concat
    assert len(concat_layers) >= 2, f"Expected at least 2 skip concatenate layers, got {len(concat_layers)}"


def test_tiny_unet_v1_downsample_upsample():
    """Test Tiny U-Net v1 has downsampling and upsampling blocks."""
    config = load_config("configs/tiny_unet_v1.yaml")
    model = build_inference_model(config)
    
    # Check for downsample and upsample blocks
    downsample_blocks = [l for l in model.layers if l.__class__.__name__ == 'DownsampleBlock']
    upsample_blocks = [l for l in model.layers if l.__class__.__name__ == 'UpsampleBlock']
    
    assert len(downsample_blocks) == 2, f"Expected 2 downsample blocks, got {len(downsample_blocks)}"
    assert len(upsample_blocks) == 2, f"Expected 2 upsample blocks, got {len(upsample_blocks)}"


def test_tiny_unet_v1_conditioning():
    """Test Tiny U-Net v1 has conditioning embeddings."""
    config = load_config("configs/tiny_unet_v1.yaml")
    model = build_inference_model(config)
    
    # Check for timestep and class embedding layers
    timestep_emb = [l for l in model.layers if l.__class__.__name__ == 'SinusoidalTimestepEmbedding']
    class_emb = [l for l in model.layers if l.__class__.__name__ == 'ClassEmbedding']
    
    # FiLM layers are nested inside ResidualBlock layers
    film_layers = []
    for layer in model.layers:
        if layer.__class__.__name__ == 'ResidualBlock':
            # Look inside the residual block for FiLM layers (use _layers for Layer submodules)
            for sublayer in getattr(layer, '_layers', []):
                if sublayer.__class__.__name__ == 'FiLMConditioning':
                    film_layers.append(sublayer)
    
    assert len(timestep_emb) == 1, "Expected 1 timestep embedding layer"
    assert len(class_emb) == 1, "Expected 1 class embedding layer"
    # 2 FiLM layers per residual block * 5 residual blocks = 10
    assert len(film_layers) == 10, f"Expected 10 FiLM layers, got {len(film_layers)}"


def test_tiny_unet_v1_config_summary():
    """Test Tiny U-Net v1 config summary."""
    config = load_config("configs/tiny_unet_v1.yaml")
    summary = get_model_config_summary(config)
    
    assert 'input_shape' in summary
    assert 'output_shape' in summary
    assert 'residual_blocks' in summary
    assert 'actual_params' in summary
    assert summary['architecture'] == 'Tiny U-Net v1'
    assert summary['residual_blocks'] == 5
    assert 20000 <= summary['actual_params'] <= 40000


def test_architecture_selection():
    """Test that architecture selection works correctly."""
    # Test legacy CNN
    config_cnn = load_config("configs/default.yaml")
    model_cnn = build_inference_model(config_cnn)
    assert model_cnn.name == 'TinyConditionalCNN'
    
    # Test Tiny U-Net v1
    config_unet = load_config("configs/tiny_unet_v1.yaml")
    model_unet = build_inference_model(config_unet)
    assert model_unet.name == 'TinyUNetV1'


def test_both_architectures_forward_pass():
    """Test both architectures produce correct output shapes."""
    import numpy as np
    
    test_input = np.random.randn(2, 28, 28, 12).astype(np.float32)
    
    # Legacy CNN
    config_cnn = load_config("configs/default.yaml")
    model_cnn = build_inference_model(config_cnn)
    output_cnn = model_cnn.predict(test_input, verbose=0)
    assert output_cnn.shape == (2, 28, 28, 1)
    
    # Tiny U-Net v1
    config_unet = load_config("configs/tiny_unet_v1.yaml")
    model_unet = build_inference_model(config_unet)
    output_unet = model_unet.predict(test_input, verbose=0)
    assert output_unet.shape == (2, 28, 28, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])