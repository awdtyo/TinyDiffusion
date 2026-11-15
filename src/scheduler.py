"""
DDIM Scheduler for TinyDiffusion-ESP32.
Implements deterministic and stochastic DDIM sampling.
"""
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List, Dict, Any
from src.diffusion import NoiseSchedule, get_schedule_arrays


class DDIMScheduler:
    """DDIM scheduler for fast sampling with fewer steps."""
    
    def __init__(self, config: Dict[str, Any], noise_schedule: NoiseSchedule):
        self.config = config
        self.noise_schedule = noise_schedule
        
        sampling_config = config.get('sampling', {})
        self.default_steps = sampling_config.get('default_steps', 8)
        self.supported_steps = sampling_config.get('supported_steps', [8, 16, 32])
        self.eta = sampling_config.get('eta', 0.0)
        self.use_ddim = sampling_config.get('use_ddim', True)
        
        # Pre-compute schedule arrays for efficiency
        self.schedule_arrays = get_schedule_arrays(
            noise_schedule.num_timesteps,
            noise_schedule.schedule_type,
            noise_schedule.beta_start,
            noise_schedule.beta_end,
            noise_schedule.cosine_s
        )
        
        # DDIM specific: compute alpha_bar for inference timesteps
        self._setup_ddim_timesteps()
    
    def _setup_ddim_timesteps(self) -> None:
        """Setup DDIM timestep sequences for different step counts."""
        self.ddim_timesteps = {}
        
        for num_steps in self.supported_steps:
            # Endpoint-inclusive DDIM timestep selection:
            # Select timesteps spanning from num_timesteps-1 down to 0,
            # including both endpoints. This ensures the first DDIM step
            # starts at the maximum training timestep where alpha_bar is
            # closest to zero (for proper Gaussian initialization).
            timesteps = np.linspace(
                self.noise_schedule.num_timesteps - 1,
                0,
                num_steps,
                dtype=np.int32
            )
            
            # Verify the sequence
            assert len(timesteps) == num_steps, f"Expected {num_steps} timesteps, got {len(timesteps)}"
            assert timesteps[0] == self.noise_schedule.num_timesteps - 1, f"First timestep must be {self.noise_schedule.num_timesteps - 1}, got {timesteps[0]}"
            assert timesteps[-1] == 0, f"Last timestep must be 0, got {timesteps[-1]}"
            assert np.all(np.diff(timesteps) < 0), "Timestep sequence must be strictly decreasing"
            
            self.ddim_timesteps[num_steps] = timesteps
    
    def get_ddim_timesteps(self, num_steps: int) -> np.ndarray:
        """Get DDIM timestep sequence for given number of steps."""
        if num_steps not in self.ddim_timesteps:
            # Generate on the fly using endpoint-inclusive approach
            timesteps = np.linspace(
                self.noise_schedule.num_timesteps - 1,
                0,
                num_steps,
                dtype=np.int32
            )
            # Verify
            assert len(timesteps) == num_steps
            assert timesteps[0] == self.noise_schedule.num_timesteps - 1
            assert timesteps[-1] == 0
            assert np.all(np.diff(timesteps) < 0)
            return timesteps
        return self.ddim_timesteps[num_steps]
    
    def ddim_step(self, x_t: np.ndarray, noise_pred: np.ndarray, 
                  t: int, t_prev: int, eta: float = None) -> np.ndarray:
        """
        Single DDIM denoising step.
        
        DDIM update rule (from Song et al. 2020):
        x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * epsilon_pred + sigma_t * noise
        
        Where:
        x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)
        sigma_t = eta * sqrt((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t / alpha_bar_{t-1})
        
        For eta=0 (deterministic): sigma_t = 0
        """
        if eta is None:
            eta = self.eta
        
        # Get schedule values
        alpha_bar_t = self.schedule_arrays['alphas_cumprod'][t]
        alpha_bar_t_prev = self.schedule_arrays['alphas_cumprod'][t_prev] if t_prev >= 0 else 1.0
        
        sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
        sqrt_alpha_bar_t_prev = np.sqrt(alpha_bar_t_prev)
        sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)
        
        # Predict x_0
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        
        # Clip x_0_pred to valid range for stability
        x_0_pred = np.clip(x_0_pred, -1.0, 1.0)
        
        # Compute sigma_t
        if eta > 0:
            sigma_t = eta * np.sqrt(
                (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t) * 
                (1.0 - alpha_bar_t / alpha_bar_t_prev)
            )
        else:
            sigma_t = 0.0
        
        # Direction pointing to x_t
        dir_xt = np.sqrt(1.0 - alpha_bar_t_prev - sigma_t**2) * noise_pred
        
        # Random noise for stochasticity
        if eta > 0:
            noise = np.random.randn(*x_t.shape).astype(np.float32)
            x_t_prev = sqrt_alpha_bar_t_prev * x_0_pred + dir_xt + sigma_t * noise
        else:
            x_t_prev = sqrt_alpha_bar_t_prev * x_0_pred + dir_xt
        
        return x_t_prev.astype(np.float32)
    
    def sample(self, model: tf.keras.Model, class_label: int, 
               num_steps: int = None, eta: float = None,
               return_intermediates: bool = False) -> np.ndarray:
        """
        Generate sample using DDIM sampling.
        
        Args:
            model: Trained noise prediction model
            class_label: Target digit class (0-9)
            num_steps: Number of DDIM steps (default: config default)
            eta: DDIM eta parameter (default: config default)
            return_intermediates: Whether to return intermediate steps
            
        Returns:
            Generated image [28, 28, 1] in [-1, 1] or list of intermediates
        """
        if num_steps is None:
            num_steps = self.default_steps
        if eta is None:
            eta = self.eta
        
        # Get timestep sequence
        timesteps = self.get_ddim_timesteps(num_steps)
        
        # Start from pure noise
        x_t = np.random.randn(1, 28, 28, 1).astype(np.float32)
        
        intermediates = [x_t.copy()] if return_intermediates else None
        
        # Class conditioning
        class_onehot = np.zeros((1, 10), dtype=np.float32)
        class_onehot[0, class_label] = 1.0
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            
            # Prepare model input
            # Timestep normalization
            t_normalized = np.array([[t / self.noise_schedule.num_timesteps]], dtype=np.float32)
            
            # Broadcast conditioning
            height, width = 28, 28
            class_maps = np.zeros((1, height, width, 10), dtype=np.float32)
            for c in range(10):
                class_maps[:, :, :, c] = class_onehot[:, c:c+1]
            
            timestep_maps = np.zeros((1, height, width, 1), dtype=np.float32)
            timestep_maps[:, :, :, 0] = t_normalized[0, 0]
            
            conditioning = np.concatenate([class_maps, timestep_maps], axis=-1)
            model_input = np.concatenate([x_t, conditioning], axis=-1)
            
            # Predict noise
            noise_pred = model.predict(model_input, verbose=0)
            
            # DDIM step
            x_t = self.ddim_step(x_t, noise_pred, t, t_prev, eta)
            
            if return_intermediates:
                intermediates.append(x_t.copy())
        
        if return_intermediates:
            return intermediates
        return x_t[0]
    
    def sample_batch(self, model: tf.keras.Model, class_labels: np.ndarray,
                     num_steps: int = None, eta: float = None) -> np.ndarray:
        """Generate batch of samples."""
        if num_steps is None:
            num_steps = self.default_steps
        if eta is None:
            eta = self.eta
        
        batch_size = len(class_labels)
        timesteps = self.get_ddim_timesteps(num_steps)
        
        # Start from pure noise
        x_t = np.random.randn(batch_size, 28, 28, 1).astype(np.float32)
        
        # Class conditioning
        class_onehot = np.zeros((batch_size, 10), dtype=np.float32)
        class_onehot[np.arange(batch_size), class_labels] = 1.0
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            
            # Prepare model input
            t_normalized = np.full((batch_size, 1), t / self.noise_schedule.num_timesteps, dtype=np.float32)
            
            height, width = 28, 28
            class_maps = np.zeros((batch_size, height, width, 10), dtype=np.float32)
            for c in range(10):
                # Broadcast (batch, 1) to (batch, height, width)
                class_maps[:, :, :, c] = class_onehot[:, c:c+1].reshape(batch_size, 1, 1)
            
            timestep_maps = np.zeros((batch_size, height, width, 1), dtype=np.float32)
            timestep_maps[:, :, :, 0] = t_normalized[:, 0].reshape(batch_size, 1, 1)
            
            conditioning = np.concatenate([class_maps, timestep_maps], axis=-1)
            model_input = np.concatenate([x_t, conditioning], axis=-1)
            
            # Predict noise
            noise_pred = model.predict(model_input, verbose=0)
            
            # DDIM step (batch version)
            x_t = self.ddim_step_batch(x_t, noise_pred, t, t_prev, eta)
        
        return x_t
    
    def ddim_step_batch(self, x_t: np.ndarray, noise_pred: np.ndarray,
                        t: int, t_prev: int, eta: float = None) -> np.ndarray:
        """Batch version of DDIM step."""
        if eta is None:
            eta = self.eta
        
        alpha_bar_t = self.schedule_arrays['alphas_cumprod'][t]
        alpha_bar_t_prev = self.schedule_arrays['alphas_cumprod'][t_prev] if t_prev >= 0 else 1.0
        
        sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
        sqrt_alpha_bar_t_prev = np.sqrt(alpha_bar_t_prev)
        sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)
        
        # Predict x_0
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        x_0_pred = np.clip(x_0_pred, -1.0, 1.0)
        
        # Compute sigma_t
        if eta > 0:
            sigma_t = eta * np.sqrt(
                (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t) * 
                (1.0 - alpha_bar_t / alpha_bar_t_prev)
            )
        else:
            sigma_t = 0.0
        
        dir_xt = np.sqrt(1.0 - alpha_bar_t_prev - sigma_t**2) * noise_pred
        
        if eta > 0:
            noise = np.random.randn(*x_t.shape).astype(np.float32)
            x_t_prev = sqrt_alpha_bar_t_prev * x_0_pred + dir_xt + sigma_t * noise
        else:
            x_t_prev = sqrt_alpha_bar_t_prev * x_0_pred + dir_xt
        
        return x_t_prev.astype(np.float32)


def ddim_sampling_reference(model: tf.keras.Model, class_label: int,
                            noise_schedule: NoiseSchedule,
                            num_steps: int = 8, eta: float = 0.0,
                            return_intermediates: bool = False) -> np.ndarray:
    """
    Reference DDIM sampling implementation matching the embedded firmware logic.
    This should be kept in sync with the C++ implementation.
    """
    scheduler = DDIMScheduler({'sampling': {'default_steps': num_steps, 'eta': eta}}, noise_schedule)
    return scheduler.sample(model, class_label, num_steps, eta, return_intermediates)


def generate_class_grid(model: tf.keras.Model, noise_schedule: NoiseSchedule,
                        classes: List[int] = None, samples_per_class: int = 5,
                        num_steps: int = 8, eta: float = 0.0) -> np.ndarray:
    """Generate a grid of samples for multiple classes."""
    if classes is None:
        classes = list(range(10))
    
    all_samples = []
    for class_id in classes:
        class_samples = []
        for _ in range(samples_per_class):
            sample = ddim_sampling_reference(model, class_id, noise_schedule, num_steps, eta)
            class_samples.append(sample)
        all_samples.append(np.stack(class_samples))
    
    return np.stack(all_samples)  # [num_classes, samples_per_class, 28, 28, 1]


if __name__ == "__main__":
    # Test scheduler
    config = {
        'diffusion': {
            'schedule_type': 'linear',
            'num_timesteps': 100,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'normalize_timestep': True
        },
        'sampling': {
            'default_steps': 8,
            'supported_steps': [8, 16, 32],
            'eta': 0.0,
            'use_ddim': True
        }
    }
    
    schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, schedule)
    
    print(f"DDIM timesteps for 8 steps: {scheduler.get_ddim_timesteps(8)}")
    print(f"DDIM timesteps for 16 steps: {scheduler.get_ddim_timesteps(16)}")
    print(f"DDIM timesteps for 32 steps: {scheduler.get_ddim_timesteps(32)}")
    
    # Test DDIM step
    x_t = np.random.randn(1, 28, 28, 1).astype(np.float32)
    noise_pred = np.random.randn(1, 28, 28, 1).astype(np.float32)
    x_t_prev = scheduler.ddim_step(x_t, noise_pred, 99, 87, 0.0)
    print(f"DDIM step output shape: {x_t_prev.shape}")
    
    print("Scheduler module test passed!")