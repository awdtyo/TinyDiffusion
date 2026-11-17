#!/usr/bin/env python3
"""Phase 4: Evaluate Tiny U-Net v2.1 Configuration C [6,8,8,6] generation quality"""

import sys, os, json, numpy as np, tensorflow as tf
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, set_global_seed, ensure_dir, save_json, denormalize_to_zero_one
from src.v2_1.model_v2_1 import build_tiny_unet_v2_1
from src.diffusion import NoiseSchedule
from src.scheduler import DDIMScheduler

def main():
    import numpy as np, tensorflow as tf
    import sys, os, json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from src.utils import load_config, set_global_seed, ensure_dir, save_json, denormalize_to_zero_one
    from src.v2_1.model_v2_1 import build_tiny_unet_v2_1
    from src.diffusion import NoiseSchedule
    from src.scheduler import DDIMScheduler
    
    np.random.seed(42); tf.random.set_seed(42)
    
    # Build model with correct Configuration C [6,8,8,6]
    config = {
        'model': {
            'input_height': 28, 'input_width': 28, 'total_input_channels': 12,
            'num_classes': 10, 'architecture': 'tiny_unet_v2_1',
            'tiny_unet_v2_1': {
                'channels': [6, 8, 8, 6],
                'initial_conv_kernel': 3,
                'encoder': [
                    {'resolution': [28, 28], 'channels': 6, 'num_blocks': 1},
                    {'resolution': [14, 14], 'channels': 8, 'num_blocks': 1}
                ],
                'bottleneck': {'resolution': [7, 7], 'channels': 8, 'num_blocks': 1},
                'decoder': [
                    {'resolution': [14, 14], 'channels': 8, 'num_blocks': 1},
                    {'resolution': [28, 28], 'channels': 6, 'num_blocks': 1}
                ],
                'output_channels': 1,
                'residual_block': {'kernel_size': 3, 'activation': 'relu', 'use_bias': True, 'padding': 'same'},
                'downsample': {'type': 'strided_conv', 'kernel_size': 3, 'stride': 2},
                'upsample': {'type': 'bilinear_conv', 'kernel_size': 3}
            }
        }
    }
    
    cnn = build_tiny_unet_v2_1(config)
    cnn(tf.random.normal([1,28,28,12]), training=False)
    cnn.load_weights("artifacts/checkpoints_tiny_unet_v2_1_c/best_model.keras")
    cnn.compile(optimizer=tf.keras.optimizers.Adam(0.001,0.9,0.999,1e-7), loss='mse')
    print(f"Model: {cnn.name}, Params: {cnn.count_params():,}")
    
    noise_schedule = NoiseSchedule(load_config("configs/tiny_unet_v2_1.yaml"))
    scheduler = DDIMScheduler(load_config("configs/tiny_unet_v2_1.yaml"), noise_schedule)
    
    # Diffusion constants
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float32)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(1 - betas)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1 - alphas_cumprod)
    
    # Fixed noises for reproducibility
    np.random.seed(42)
    fixed_noises = []
    for class_idx in range(10):
        class_noises = []
        for sample_idx in range(5):
            noise = np.random.randn(1, 28, 28, 1).astype(np.float32)
            class_noises.append(noise)
        fixed_noises.append(class_noises)
    
    classes = list(range(10))
    samples_per_class = 5
    step_configs = [8, 16, 32, 50, 100]
    
    output_dir = "artifacts/evaluation_tiny_unet_v2_1_c"
    ensure_dir(output_dir)
    
    import matplotlib.pyplot as plt
    
    all_results = {}
    
    for steps in step_configs:
        print(f"\nGenerating samples for {steps} DDIM steps...")
        
        samples = []
        for class_idx, class_id in enumerate(range(10)):
            class_samples = []
            for sample_idx in range(5):
                noise = fixed_noises[class_idx][sample_idx]
                
                x_t = noise.copy()
                timesteps = scheduler.get_ddim_timesteps(steps)
                class_onehot = np.zeros((1, 10), dtype=np.float32)
                class_onehot[0, class_id] = 1.0
                
                for i, t in enumerate(timesteps):
                    t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                    t_normalized = np.array([[t / scheduler.noise_schedule.num_timesteps]], dtype=np.float32)
                    class_maps = np.zeros((1, 28, 28, 10), dtype=np.float32)
                    for c in range(10):
                        class_maps[:, :, :, c] = class_onehot[:, c:c+1]
                    timestep_maps = np.zeros((1, 28, 28, 1), dtype=np.float32)
                    timestep_maps[:, :, :, 0] = t_normalized[0, 0]
                    conditioning = np.concatenate([class_maps, timestep_maps], axis=-1)
                    model_input = np.concatenate([x_t, conditioning], axis=-1)
                    noise_pred = cnn.predict(model_input, verbose=0)
                    x_t = scheduler.ddim_step(x_t, noise_pred, t, t_prev, scheduler.eta)
                
                class_samples.append(x_t[0])
            samples.append(np.stack(class_samples))
        
        samples = np.stack(samples)
        all_results[steps] = samples
        
        # Save grid
        fig, axes = plt.subplots(10, 5, figsize=(10, 20))
        for c in range(10):
            for s in range(5):
                img = denormalize_to_zero_one(samples[c, s, ..., 0])
                axes[c, s].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[c, s].set_title(f"Cl{c}")
                axes[c, s].axis('off')
        plt.suptitle(f"Tiny U-Net v2.1 Config C - {steps} DDIM Steps", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/grid_{steps}steps.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Per-class images
        for c in range(10):
            fig, axes = plt.subplots(1, 5, figsize=(10, 2))
            for s in range(5):
                img = denormalize_to_zero_one(samples[c, s, ..., 0])
                axes[s].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[s].set_title(f"Sample {s+1}")
                axes[s].axis('off')
            plt.suptitle(f"Class {c} - {steps} steps")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/class_{c}_{steps}steps.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        pass
    
    # Cross-step comparison
    print("\nGenerating cross-step comparison...")
    fig, axes = plt.subplots(10, 5, figsize=(12, 24))
    for c in range(10):
        for s_idx, steps in enumerate(step_configs):
            img = denormalize_to_zero_one(all_results[steps][c, 0, ..., 0])
            axes[c, s_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[c, s_idx].set_title(f"Cl{c} - {steps} steps")
            axes[c, s_idx].axis('off')
    plt.suptitle("Cross-Step Comparison: Same Initial Noise, Different DDIM Steps", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cross_step_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nEvaluation complete!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    import numpy as np, tensorflow as tf
    import sys, os, json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from src.utils import load_config, set_global_seed, ensure_dir, save_json, denormalize_to_zero_one
    from src.v2_1.model_v2_1 import build_tiny_unet_v2_1
    from src.diffusion import NoiseSchedule
    from src.scheduler import DDIMScheduler
    
    np.random.seed(42); tf.random.set_seed(42)
    
    # Build model with correct Configuration C [6,8,8,6]
    config = {
        'model': {
            'input_height': 28, 'input_width': 28, 'total_input_channels': 12,
            'num_classes': 10, 'architecture': 'tiny_unet_v2_1',
            'tiny_unet_v2_1': {
                'channels': [6, 8, 8, 6],
                'initial_conv_kernel': 3,
                'encoder': [
                    {'resolution': [28, 28], 'channels': 6, 'num_blocks': 1},
                    {'resolution': [14, 14], 'channels': 8, 'num_blocks': 1}
                ],
                'bottleneck': {'resolution': [7, 7], 'channels': 8, 'num_blocks': 1},
                'decoder': [
                    {'resolution': [14, 14], 'channels': 8, 'num_blocks': 1},
                    {'resolution': [28, 28], 'channels': 6, 'num_blocks': 1}
                ],
                'output_channels': 1,
                'residual_block': {'kernel_size': 3, 'activation': 'relu', 'use_bias': True, 'padding': 'same'},
                'downsample': {'type': 'strided_conv', 'kernel_size': 3, 'stride': 2},
                'upsample': {'type': 'bilinear_conv', 'kernel_size': 3}
            }
        }
    }
    
    cnn = build_tiny_unet_v2_1(config)
    cnn(tf.random.normal([1,28,28,12]), training=False)
    cnn.load_weights("artifacts/checkpoints_tiny_unet_v2_1_c/best_model.keras")
    cnn.compile(optimizer=tf.keras.optimizers.Adam(0.001,0.9,0.999,1e-7), loss='mse')
    print(f"Model: {cnn.name}, Params: {cnn.count_params():,}")
    
    noise_schedule = NoiseSchedule(load_config("configs/tiny_unet_v2_1.yaml"))
    scheduler = DDIMScheduler(load_config("configs/tiny_unet_v2_1.yaml"), noise_schedule)
    
    # Diffusion constants
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float32)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(1 - betas)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1 - alphas_cumprod)
    
    # Fixed noises for reproducibility
    np.random.seed(42)
    fixed_noises = []
    for class_idx in range(10):
        class_noises = []
        for sample_idx in range(5):
            noise = np.random.randn(1, 28, 28, 1).astype(np.float32)
            class_noises.append(noise)
        fixed_noises.append(class_noises)
    
    classes = list(range(10))
    samples_per_class = 5
    step_configs = [8, 16, 32, 50, 100]
    
    output_dir = "artifacts/evaluation_tiny_unet_v2_1_c"
    ensure_dir(output_dir)
    
    import matplotlib.pyplot as plt
    
    all_results = {}
    
    for steps in step_configs:
        print(f"\nGenerating samples for {steps} DDIM steps...")
        
        samples = []
        for class_idx, class_id in enumerate(range(10)):
            class_samples = []
            for sample_idx in range(5):
                noise = fixed_noises[class_idx][sample_idx]
                
                x_t = noise.copy()
                timesteps = scheduler.get_ddim_timesteps(steps)
                class_onehot = np.zeros((1, 10), dtype=np.float32)
                class_onehot[0, class_id] = 1.0
                
                for i, t in enumerate(timesteps):
                    t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                    t_normalized = np.array([[t / scheduler.noise_schedule.num_timesteps]], dtype=np.float32)
                    class_maps = np.zeros((1, 28, 28, 10), dtype=np.float32)
                    for c in range(10):
                        class_maps[:, :, :, c] = class_onehot[:, c:c+1]
                    timestep_maps = np.zeros((1, 28, 28, 1), dtype=np.float32)
                    timestep_maps[:, :, :, 0] = t_normalized[0, 0]
                    conditioning = np.concatenate([class_maps, timestep_maps], axis=-1)
                    model_input = np.concatenate([x_t, conditioning], axis=-1)
                    noise_pred = cnn.predict(model_input, verbose=0)
                    x_t = scheduler.ddim_step(x_t, noise_pred, t, t_prev, scheduler.eta)
                
                class_samples.append(x_t[0])
            samples.append(np.stack(class_samples))
        
        samples = np.stack(samples)
        all_results[steps] = samples
        
        # Save grid
        fig, axes = plt.subplots(10, 5, figsize=(10, 20))
        for c in range(10):
            for s in range(5):
                img = denormalize_to_zero_one(samples[c, s, ..., 0])
                axes[c, s].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[c, s].set_title(f"Cl{c}")
                axes[c, s].axis('off')
        plt.suptitle(f"Tiny U-Net v2.1 Config C - {steps} DDIM Steps", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/grid_{steps}steps.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Per-class images
        for c in range(10):
            fig, axes = plt.subplots(1, 5, figsize=(10, 2))
            for s in range(5):
                img = denormalize_to_zero_one(samples[c, s, ..., 0])
                axes[s].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[s].set_title(f"Sample {s+1}")
                axes[s].axis('off')
            plt.suptitle(f"Class {c} - {steps} steps")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/class_{c}_{steps}steps.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        pass
    
    # Cross-step comparison
    print("\nGenerating cross-step comparison...")
    fig, axes = plt.subplots(10, 5, figsize=(12, 24))
    for c in range(10):
        for s_idx, steps in enumerate(step_configs):
            img = denormalize_to_zero_one(all_results[steps][c, 0, ..., 0])
            axes[c, s_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[c, s_idx].set_title(f"Cl{c} - {steps} steps")
            axes[c, s_idx].axis('off')
    plt.suptitle("Cross-Step Comparison: Same Initial Noise, Different DDIM Steps", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cross_step_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nEvaluation complete!")
    print(f"Results saved to {output_dir}")