"""
Tests for data preprocessing and utilities.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from src.utils import (
    normalize_to_neg_one_one, denormalize_to_zero_one, denormalize_to_uint8,
    one_hot_encode, broadcast_conditioning, prepare_model_input,
    quantize_int8, dequantize_int8
)
from src.dataset import MNISTDataset


def test_normalize_to_neg_one_one():
    """Test normalization to [-1, 1]."""
    # From [0, 1]
    x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    y = normalize_to_neg_one_one(x)
    assert np.allclose(y, [-1.0, 0.0, 1.0])
    
    # From [0, 255]
    x = np.array([0, 127, 255], dtype=np.float32)
    y = normalize_to_neg_one_one(x)
    assert np.allclose(y, [-1.0, 0.0, 1.0], atol=0.01)


def test_denormalize_to_zero_one():
    """Test denormalization to [0, 1]."""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = denormalize_to_zero_one(x)
    assert np.allclose(y, [0.0, 0.5, 1.0])


def test_denormalize_to_uint8():
    """Test denormalization to [0, 255] uint8."""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = denormalize_to_uint8(x)
    assert y.dtype == np.uint8
    assert np.array_equal(y, [0, 127, 255])


def test_one_hot_encode():
    """Test one-hot encoding."""
    labels = np.array([0, 1, 9, 5])
    onehot = one_hot_encode(labels, 10)
    
    assert onehot.shape == (4, 10)
    assert onehot[0, 0] == 1.0
    assert onehot[1, 1] == 1.0
    assert onehot[2, 9] == 1.0
    assert onehot[3, 5] == 1.0


def test_broadcast_conditioning():
    """Test conditioning broadcast."""
    class_onehot = np.eye(10)[[3, 7]]  # [2, 10]
    timestep = np.array([[0.5], [0.25]])  # [2, 1]
    
    conditioning = broadcast_conditioning(class_onehot, timestep, 28, 28)
    
    assert conditioning.shape == (2, 28, 28, 11)
    
    # Check class maps
    assert np.all(conditioning[0, :, :, 3] == 1.0)
    assert np.all(conditioning[0, :, :, 7] == 0.0)
    assert np.all(conditioning[1, :, :, 7] == 1.0)
    
    # Check timestep map
    assert np.all(conditioning[0, :, :, 10] == 0.5)
    assert np.all(conditioning[1, :, :, 10] == 0.25)


def test_prepare_model_input():
    """Test full model input preparation."""
    noisy_images = np.random.randn(2, 28, 28, 1).astype(np.float32)
    class_labels = np.array([3, 7])
    timesteps = np.array([50, 25])
    
    model_input = prepare_model_input(
        noisy_images, class_labels, timesteps,
        num_classes=10, normalize_timestep=True, num_timesteps=100
    )
    
    assert model_input.shape == (2, 28, 28, 12)
    
    # Check image channel
    assert np.allclose(model_input[:, :, :, 0], noisy_images[:, :, :, 0])
    
    # Check class conditioning
    assert np.all(model_input[0, 0, 0, 1:11] == [0,0,0,1,0,0,0,0,0,0])
    assert np.all(model_input[1, 0, 0, 1:11] == [0,0,0,0,0,0,0,1,0,0])


def test_quantize_dequantize():
    """Test INT8 quantization round-trip."""
    x = np.random.randn(100).astype(np.float32)
    scale = 0.01
    zero_point = -128
    
    x_q = quantize_int8(x, scale, zero_point)
    x_dq = dequantize_int8(x_q, scale, zero_point)
    
    # Check round-trip error
    error = np.abs(x - x_dq)
    assert np.max(error) < scale * 0.5 + 1e-6  # Within half quantization step


def test_quantize_clipping():
    """Test quantization clips to INT8 range."""
    x = np.array([1000.0, -1000.0], dtype=np.float32)
    scale = 1.0
    zero_point = 0
    
    x_q = quantize_int8(x, scale, zero_point)
    
    assert x_q[0] == 127
    assert x_q[1] == -128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])