"""
Tests for DDIM scheduler.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf
import pytest
from src.diffusion import NoiseSchedule
from src.scheduler import DDIMScheduler
from src.model import build_inference_model
from src.utils import load_config


def test_scheduler_creation():
    """Test scheduler creation."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    
    assert scheduler.default_steps == 8
    assert scheduler.eta == 0.0


def test_ddim_timesteps_8():
    """Test 8-step DDIM timesteps."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    
    timesteps = scheduler.get_ddim_timesteps(8)
    
    assert len(timesteps) == 8
    assert timesteps[0] == 99  # Start from max noise
    assert timesteps[-1] == 12  # End near clean
    assert np.all(np.diff(timesteps) < 0)  # Decreasing


def test_ddim_timesteps_16():
    """Test 16-step DDIM timesteps."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    
    timesteps = scheduler.get_ddim_timesteps(16)
    
    assert len(timesteps) == 16
    assert timesteps[0] == 99
    assert np.all(np.diff(timesteps) < 0)


def test_ddim_timesteps_32():
    """Test 32-step DDIM timesteps."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    
    timesteps = scheduler.get_ddim_timesteps(32)
    
    assert len(timesteps) == 32
    assert timesteps[0] == 99
    assert np.all(np.diff(timesteps) < 0)


def test_ddim_step_deterministic():
    """Test DDIM step with eta=0 is deterministic."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    
    x_t = np.random.randn(1, 28, 28, 1).astype(np.float32)
    noise_pred = np.random.randn(1, 28, 28, 1).astype(np.float32)
    
    # Run twice with same inputs
    x_t_prev1 = scheduler.ddim_step(x_t, noise_pred, 99, 87, eta=0.0)
    x_t_prev2 = scheduler.ddim_step(x_t, noise_pred, 99, 87, eta=0.0)
    
    assert np.allclose(x_t_prev1, x_t_prev2)


def test_ddim_step_output_shape():
    """Test DDIM step output shape."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    
    x_t = np.random.randn(2, 28, 28, 1).astype(np.float32)
    noise_pred = np.random.randn(2, 28, 28, 1).astype(np.float32)
    
    x_t_prev = scheduler.ddim_step(x_t, noise_pred, 99, 87, eta=0.0)
    
    assert x_t_prev.shape == (2, 28, 28, 1)


def test_ddim_step_final():
    """Test DDIM final step (t_prev=-1)."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    
    x_t = np.random.randn(1, 28, 28, 1).astype(np.float32)
    noise_pred = np.random.randn(1, 28, 28, 1).astype(np.float32)
    
    # Final step to t=-1 (alpha_bar=1)
    x_t_prev = scheduler.ddim_step(x_t, noise_pred, 12, -1, eta=0.0)
    
    assert x_t_prev.shape == (1, 28, 28, 1)
    # Output should be in valid range
    assert np.all(x_t_prev >= -1.0) and np.all(x_t_prev <= 1.0)


def test_scheduler_batch_step():
    """Test batch DDIM step."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    
    x_t = np.random.randn(4, 28, 28, 1).astype(np.float32)
    noise_pred = np.random.randn(4, 28, 28, 1).astype(np.float32)
    
    x_t_prev = scheduler.ddim_step_batch(x_t, noise_pred, 99, 87, eta=0.0)
    
    assert x_t_prev.shape == (4, 28, 28, 1)


def test_sampler_output_shape():
    """Test sampler output shape with dummy model."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    model = build_inference_model(config)
    
    # Use random model (no training)
    sample = scheduler.sample(model, class_label=5, num_steps=8, eta=0.0)
    
    assert sample.shape == (28, 28, 1)
    assert np.all(sample >= -1.0) and np.all(sample <= 1.0)


def test_sampler_batch():
    """Test batch sampling."""
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    model = build_inference_model(config)
    
    class_labels = np.array([0, 1, 2, 3, 4])
    samples = scheduler.sample_batch(model, class_labels, num_steps=8, eta=0.0)
    
    assert samples.shape == (5, 28, 28, 1)


def test_generate_class_grid():
    """Test class grid generation."""
    from src.scheduler import generate_class_grid
    
    config = load_config("../configs/default.yaml")
    schedule = NoiseSchedule(config)
    model = build_inference_model(config)
    
    grid = generate_class_grid(model, schedule, classes=[0, 1, 2], samples_per_class=2, num_steps=4)
    
    assert grid.shape == (3, 2, 28, 28, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])