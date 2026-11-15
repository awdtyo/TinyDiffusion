"""
Tests for model architecture and dimensions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf
import pytest
from src.model import build_inference_model, get_model_config_summary
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])