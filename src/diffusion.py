"""
Diffusion process implementation for TinyDiffusion-ESP32.
Implements DDPM forward process and noise scheduling.
"""
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any, List
from src.utils import load_config


class NoiseSchedule:
    """Noise schedule for DDPM diffusion process."""
    
    def __init__(self, config: Dict[str, Any]):
        diff_config = config.get('diffusion', {})
        self.schedule_type = diff_config.get('schedule_type', 'linear')
        self.num_timesteps = diff_config.get('num_timesteps', 100)
        self.beta_start = diff_config.get('beta_start', 0.0001)
        self.beta_end = diff_config.get('beta_end', 0.02)
        self.cosine_s = diff_config.get('cosine_s', 0.008)
        self.normalize_timestep = diff_config.get('normalize_timestep', True)
        self.prediction_type = diff_config.get('prediction_type', 'epsilon')
        
        self._compute_schedule()
    
    def _compute_schedule(self) -> None:
        """Compute beta, alpha, and cumulative alpha schedules."""
        if self.schedule_type == 'linear':
            self.betas = np.linspace(self.beta_start, self.beta_end, 
                                     self.num_timesteps, dtype=np.float32)
        elif self.schedule_type == 'cosine':
            self.betas = self._cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        # Clamp betas for numerical stability
        self.betas = np.clip(self.betas, 1e-8, 1.0 - 1e-8)
        
        # Alpha = 1 - beta
        self.alphas = 1.0 - self.betas
        
        # Cumulative product of alphas: alpha_bar_t = prod_{s=1}^t alpha_s
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # For convenience
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior variance computation
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / 
                                   (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(
            np.maximum(self.posterior_variance, 1e-20))
        
        # Coefficients for posterior mean
        self.posterior_mean_coef1 = (self.betas * np.sqrt(self.alphas_cumprod_prev) / 
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / 
                                     (1.0 - self.alphas_cumprod))
    
    def _cosine_schedule(self) -> np.ndarray:
        """Cosine noise schedule from Improved DDPM paper."""
        steps = self.num_timesteps + 1
        x = np.linspace(0, self.num_timesteps, steps, dtype=np.float32)
        alphas_cumprod = np.cos(((x / self.num_timesteps) + self.cosine_s) / 
                                (1 + self.cosine_s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 1e-8, 0.999)
    
    def get_timestep_embedding(self, timesteps: np.ndarray) -> np.ndarray:
        """Normalize timesteps to [0, 1] for conditioning."""
        if self.normalize_timestep:
            return timesteps.astype(np.float32) / self.num_timesteps
        return timesteps.astype(np.float32)
    
    def __len__(self) -> int:
        return self.num_timesteps
    
    def __repr__(self) -> str:
        return (f"NoiseSchedule(type={self.schedule_type}, steps={self.num_timesteps}, "
                f"beta_range=[{self.beta_start}, {self.beta_end}])")


def forward_diffusion(x_0: np.ndarray, timesteps: np.ndarray, 
                      noise: np.ndarray, num_timesteps: int = 100,
                      schedule_type: str = 'linear',
                      beta_start: float = 0.0001,
                      beta_end: float = 0.02) -> np.ndarray:
    """
    Forward diffusion process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    
    Args:
        x_0: Clean images [batch, H, W, C] in [-1, 1]
        timesteps: Timestep indices [batch] in [0, num_timesteps-1]
        noise: Gaussian noise [batch, H, W, C] ~ N(0, 1)
        num_timesteps: Total number of diffusion steps
        schedule_type: 'linear' or 'cosine'
        beta_start: Start beta for linear schedule
        beta_end: End beta for linear schedule
        
    Returns:
        x_t: Noisy images at timestep t
    """
    # Compute schedule
    if schedule_type == 'linear':
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
    elif schedule_type == 'cosine':
        steps = num_timesteps + 1
        x = np.linspace(0, num_timesteps, steps, dtype=np.float32)
        alphas_cumprod = np.cos(((x / num_timesteps) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 1e-8, 0.999)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    # Get coefficients for each timestep
    sqrt_alphas_cumprod_t = np.sqrt(alphas_cumprod[timesteps])
    sqrt_one_minus_alphas_cumprod_t = np.sqrt(1.0 - alphas_cumprod[timesteps])
    
    # Reshape for broadcasting: [batch] -> [batch, 1, 1, 1]
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.reshape(-1, 1, 1, 1)
    
    # Forward diffusion equation
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    return x_t.astype(np.float32)


def forward_diffusion_v(x_0: np.ndarray, timesteps: np.ndarray,
                        noise: np.ndarray, num_timesteps: int = 100,
                        schedule_type: str = 'linear',
                        beta_start: float = 0.0001,
                        beta_end: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward diffusion for v-prediction training.
    
    Returns:
        x_t: Noisy images at timestep t
        v_true: V-prediction target [batch, H, W, C]
                v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x_0
    """
    if schedule_type == 'linear':
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
    elif schedule_type == 'cosine':
        steps = num_timesteps + 1
        x = np.linspace(0, num_timesteps, steps, dtype=np.float32)
        alphas_cumprod = np.cos(((x / num_timesteps) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 1e-8, 0.999)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    # Get coefficients for each timestep
    sqrt_alphas_cumprod_t = np.sqrt(alphas_cumprod[timesteps])
    sqrt_one_minus_alphas_cumprod_t = np.sqrt(1.0 - alphas_cumprod[timesteps])
    
    # Reshape for broadcasting: [batch] -> [batch, 1, 1, 1]
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.reshape(-1, 1, 1, 1)
    
    # Forward diffusion equation
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    # V-prediction target: v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x_0
    v_true = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_0
    
    return x_t.astype(np.float32), v_true.astype(np.float32)


def forward_diffusion_tf(x_0: tf.Tensor, timesteps: tf.Tensor, 
                         noise: tf.Tensor, 
                         sqrt_alphas_cumprod: tf.Tensor,
                         sqrt_one_minus_alphas_cumprod: tf.Tensor) -> tf.Tensor:
    """TensorFlow version of forward diffusion for training (epsilon prediction)."""
    # Gather coefficients for each timestep
    sqrt_alpha_bar_t = tf.gather(sqrt_alphas_cumprod, timesteps)
    sqrt_one_minus_alpha_bar_t = tf.gather(sqrt_one_minus_alphas_cumprod, timesteps)
    
    # Reshape for broadcasting
    sqrt_alpha_bar_t = tf.reshape(sqrt_alpha_bar_t, [-1, 1, 1, 1])
    sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, [-1, 1, 1, 1])
    
    # Forward diffusion
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    return x_t


def forward_diffusion_v_tf(x_0: tf.Tensor, timesteps: tf.Tensor,
                           noise: tf.Tensor,
                           sqrt_alphas_cumprod: tf.Tensor,
                           sqrt_one_minus_alphas_cumprod: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow version of forward diffusion for v-prediction training."""
    # Gather coefficients for each timestep
    sqrt_alpha_bar_t = tf.gather(sqrt_alphas_cumprod, timesteps)
    sqrt_one_minus_alpha_bar_t = tf.gather(sqrt_one_minus_alphas_cumprod, timesteps)
    
    # Reshape for broadcasting
    sqrt_alpha_bar_t = tf.reshape(sqrt_alpha_bar_t, [-1, 1, 1, 1])
    sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, [-1, 1, 1, 1])
    
    # Forward diffusion
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    
    # V-prediction target: v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x_0
    v_true = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x_0
    
    return x_t, v_true


def sample_timesteps(batch_size: int, num_timesteps: int) -> np.ndarray:
    """Sample random timesteps uniformly for training."""
    return np.random.randint(0, num_timesteps, size=batch_size, dtype=np.int32)


def get_schedule_arrays(num_timesteps: int, schedule_type: str = 'linear',
                        beta_start: float = 0.0001, beta_end: float = 0.02,
                        cosine_s: float = 0.008) -> Dict[str, np.ndarray]:
    """Get all schedule arrays for a given configuration."""
    if schedule_type == 'linear':
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
    elif schedule_type == 'cosine':
        steps = num_timesteps + 1
        x = np.linspace(0, num_timesteps, steps, dtype=np.float32)
        alphas_cumprod = np.cos(((x / num_timesteps) + cosine_s) / (1 + cosine_s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 1e-8, 0.999)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod
    }


def visualize_forward_diffusion(x_0: np.ndarray, schedule: NoiseSchedule, 
                                num_steps: int = 10, save_path: str = None) -> None:
    """Visualize forward diffusion process on a sample image."""
    import matplotlib.pyplot as plt
    from src.utils import denormalize_to_zero_one, ensure_dir
    
    # Select timesteps to visualize
    timesteps = np.linspace(0, schedule.num_timesteps - 1, num_steps, dtype=np.int32)
    
    # Add batch dimension if needed
    if x_0.ndim == 3:
        x_0 = x_0[np.newaxis, ...]
    
    noise = np.random.randn(*x_0.shape).astype(np.float32)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2.5))
    
    for i, t in enumerate(timesteps):
        t_batch = np.array([t], dtype=np.int32)
        x_t = forward_diffusion(x_0, t_batch, noise, schedule.num_timesteps, 
                                schedule.schedule_type, schedule.beta_start, schedule.beta_end)
        x_t_vis = denormalize_to_zero_one(x_t[0, ..., 0])
        axes[i].imshow(x_t_vis, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"t={t}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diffusion visualization to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test noise schedule
    config = {
        'diffusion': {
            'schedule_type': 'linear',
            'num_timesteps': 100,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'normalize_timestep': True
        }
    }
    
    schedule = NoiseSchedule(config)
    print(f"Schedule: {schedule}")
    print(f"Betas range: [{schedule.betas.min():.6f}, {schedule.betas.max():.6f}]")
    print(f"Alphas cumprod range: [{schedule.alphas_cumprod.min():.6f}, {schedule.alphas_cumprod.max():.6f}]")
    
    # Test forward diffusion
    x_0 = np.random.randn(1, 28, 28, 1).astype(np.float32)
    timesteps = np.array([0, 25, 50, 75, 99], dtype=np.int32)
    noise = np.random.randn(1, 28, 28, 1).astype(np.float32)
    
    x_t = forward_diffusion(x_0, timesteps, noise, schedule.num_timesteps, 
                           schedule.schedule_type, schedule.beta_start, schedule.beta_end)
    print(f"x_t shape: {x_t.shape}")
    print(f"x_t range: [{x_t.min():.3f}, {x_t.max():.3f}]")
    
    # Test timestep embedding
    t_emb = schedule.get_timestep_embedding(timesteps)
    print(f"Timestep embedding range: [{t_emb.min():.3f}, {t_emb.max():.3f}]")
    
    print("Diffusion module test passed!")