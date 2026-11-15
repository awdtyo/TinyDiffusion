"""
Tests for diffusion process and noise schedule.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from src.diffusion import NoiseSchedule, forward_diffusion, get_schedule_arrays
from src.utils import load_config


def test_noise_schedule_creation():
    """Test noise schedule creation."""
    config = load_config("configs/default.yaml")
    schedule = NoiseSchedule(config)
    
    assert schedule.num_timesteps == 100
    assert schedule.schedule_type == 'linear'
    assert len(schedule.betas) == 100
    assert len(schedule.alphas) == 100
    assert len(schedule.alphas_cumprod) == 100


def test_beta_range():
    """Test beta values are in valid range."""
    config = load_config("configs/default.yaml")
    schedule = NoiseSchedule(config)
    
    assert np.all(schedule.betas > 0)
    assert np.all(schedule.betas < 1)
    assert schedule.betas[0] == pytest.approx(0.0001, rel=1e-3)
    assert schedule.betas[-1] == pytest.approx(0.02, rel=1e-3)


def test_alpha_cumprod_monotonic():
    """Test alpha_cumprod is monotonically decreasing."""
    config = load_config("configs/default.yaml")
    schedule = NoiseSchedule(config)
    
    diff = np.diff(schedule.alphas_cumprod)
    assert np.all(diff <= 0), "alpha_cumprod should be monotonically decreasing"


def test_forward_diffusion_shape():
    """Test forward diffusion output shape."""
    config = load_config("configs/default.yaml")
    schedule = NoiseSchedule(config)
    
    x_0 = np.random.randn(4, 28, 28, 1).astype(np.float32)
    timesteps = np.array([0, 25, 50, 99], dtype=np.int32)
    noise = np.random.randn(4, 28, 28, 1).astype(np.float32)
    
    x_t = forward_diffusion(x_0, timesteps, noise, schedule.num_timesteps)
    
    assert x_t.shape == (4, 28, 28, 1)


def test_forward_diffusion_t0():
    """Test forward diffusion at t=0 returns x_0 (approximately)."""
    config = load_config("configs/default.yaml")
    schedule = NoiseSchedule(config)
    
    x_0 = np.random.randn(1, 28, 28, 1).astype(np.float32)
    timesteps = np.array([0], dtype=np.int32)
    noise = np.random.randn(1, 28, 28, 1).astype(np.float32)
    
    x_t = forward_diffusion(x_0, timesteps, noise, schedule.num_timesteps)
    
    # At t=0, alpha_bar = 1 - beta_start = 0.9999, so x_t is very close to x_0
    # but not exactly equal due to beta_start=0.0001
    corr = np.corrcoef(x_t.flatten(), x_0.flatten())[0, 1]
    assert corr > 0.999  # Very high correlation


def test_forward_diffusion_t_max():
    """Test forward diffusion at t=max is mostly noise."""
    config = load_config("configs/default.yaml")
    schedule = NoiseSchedule(config)
    
    x_0 = np.random.randn(1, 28, 28, 1).astype(np.float32)
    timesteps = np.array([99], dtype=np.int32)
    noise = np.random.randn(1, 28, 28, 1).astype(np.float32)
    
    x_t = forward_diffusion(x_0, timesteps, noise, schedule.num_timesteps)
    
    # At t=99, alpha_bar ≈ 0.54, so x_t ≈ 0.73 * x_0 + 0.68 * noise
    # Correlation with noise should be moderate
    corr = np.corrcoef(x_t.flatten(), noise.flatten())[0, 1]
    assert corr > 0.7  # Moderate correlation with noise


def test_forward_diffusion_equation():
    """Test forward diffusion follows the exact equation."""
    config = load_config("configs/default.yaml")
    schedule = NoiseSchedule(config)
    
    x_0 = np.random.randn(2, 28, 28, 1).astype(np.float32)
    timesteps = np.array([10, 50], dtype=np.int32)
    noise = np.random.randn(2, 28, 28, 1).astype(np.float32)
    
    x_t = forward_diffusion(x_0, timesteps, noise, schedule.num_timesteps)
    
    # Manual computation
    sqrt_alpha_bar = schedule.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
    
    expected = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    assert np.allclose(x_t, expected, atol=1e-5)


def test_get_schedule_arrays():
    """Test schedule array generation."""
    arrays = get_schedule_arrays(100, 'linear', 0.0001, 0.02)
    
    assert 'betas' in arrays
    assert 'alphas' in arrays
    assert 'alphas_cumprod' in arrays
    assert 'sqrt_alphas_cumprod' in arrays
    assert 'sqrt_one_minus_alphas_cumprod' in arrays
    
    assert arrays['betas'].shape == (100,)
    assert arrays['alphas_cumprod'].shape == (100,)


def test_cosine_schedule():
    """Test cosine schedule creation."""
    config = load_config("configs/default.yaml")
    config['diffusion']['schedule_type'] = 'cosine'
    schedule = NoiseSchedule(config)
    
    assert schedule.schedule_type == 'cosine'
    assert len(schedule.betas) == 100
    assert np.all(schedule.betas > 0)
    assert np.all(schedule.betas < 1)


def test_timestep_embedding():
    """Test timestep normalization."""
    config = load_config("configs/default.yaml")
    schedule = NoiseSchedule(config)
    
    timesteps = np.array([0, 50, 99], dtype=np.int32)
    embedding = schedule.get_timestep_embedding(timesteps)
    
    assert embedding[0] == 0.0
    assert embedding[1] == pytest.approx(0.5, rel=1e-2)
    assert embedding[2] == pytest.approx(0.99, rel=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])