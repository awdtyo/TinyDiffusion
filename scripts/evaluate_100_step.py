#!/usr/bin/env python3
"""
100-Step DDIM Evaluation for Tiny U-Net v1.
Generates samples and compares 8, 16, 32, 50, 100 DDIM steps.
"""
import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (load_config, set_global_seed, ensure_dir, 
                       denormalize_to_zero_one, save_json)
from src.model import build_inference_model
from src.diffusion import NoiseSchedule
from src.scheduler import DDIMScheduler, generate_class_grid


def load_model_and_scheduler(config_path, checkpoint_path):
    """Load model and scheduler from config and checkpoint."""
    config = load_config(config_path)
    set_global_seed(config.get('training', {}).get('seed', 42))
    
    model = build_inference_model(config)
    model.load_weights(checkpoint_path)
    
    noise_schedule = NoiseSchedule(config)
    scheduler = DDIMScheduler(config, noise_schedule)
    
    return model, noise_schedule, scheduler, config


def generate_fixed_noise_samples(model, scheduler, class_label, num_steps, 
                                  fixed_noise=None, fixed_seed=None):
    """Generate sample with optional fixed initial noise for reproducibility."""
    if fixed_seed is not None:
        np.random.seed(fixed_seed)
    
    if fixed_noise is None:
        x_t = np.random.randn(1, 28, 28, 1).astype(np.float32)
    else:
        x_t = fixed_noise.copy()
    
    timesteps = scheduler.get_ddim_timesteps(num_steps)
    
    class_onehot = np.zeros((1, 10), dtype=np.float32)
    class_onehot[0, class_label] = 1.0
    
    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        
        t_normalized = np.array([[t / scheduler.noise_schedule.num_timesteps]], dtype=np.float32)
        
        height, width = 28, 28
        class_maps = np.zeros((1, height, width, 10), dtype=np.float32)
        for c in range(10):
            class_maps[:, :, :, c] = class_onehot[:, c:c+1]
        
        timestep_maps = np.zeros((1, height, width, 1), dtype=np.float32)
        timestep_maps[:, :, :, 0] = t_normalized[0, 0]
        
        conditioning = np.concatenate([class_maps, timestep_maps], axis=-1)
        model_input = np.concatenate([x_t, conditioning], axis=-1)
        
        noise_pred = model.predict(model_input, verbose=0)
        x_t = scheduler.ddim_step(x_t, noise_pred, t, t_prev, scheduler.eta)
    
    return x_t[0]


def generate_grid_fixed_noise(model, scheduler, classes, samples_per_class, 
                               num_steps, fixed_noises=None):
    """Generate class grid with fixed initial noises for reproducibility."""
    all_samples = []
    noises_used = []
    
    for class_idx, class_id in enumerate(classes):
        class_samples = []
        for sample_idx in range(samples_per_class):
            if fixed_noises is not None:
                noise = fixed_noises[class_idx][sample_idx]
            else:
                np.random.seed(class_idx * 100 + sample_idx * 10 + num_steps)
                noise = np.random.randn(1, 28, 28, 1).astype(np.float32)
            
            sample = generate_fixed_noise_samples(
                model, scheduler, class_id, num_steps, fixed_noise=noise)
            class_samples.append(sample)
            noises_used.append(noise)
        all_samples.append(np.stack(class_samples))
    
    return np.stack(all_samples), noises_used


def visualize_grid(samples, class_labels=None, title="Generated Samples", 
                   save_path=None, samples_per_class=5):
    """Visualize generated samples in a grid."""
    num_classes, sppc = samples.shape[0], samples.shape[1]
    samples = samples.reshape(-1, 28, 28, 1)
    
    if class_labels is None:
        class_labels = np.repeat(range(num_classes), sppc)
    
    num_samples = len(samples)
    cols = min(10, num_samples)
    rows = int(np.ceil(num_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (sample, label) in enumerate(zip(samples, class_labels)):
        img = denormalize_to_zero_one(sample[..., 0])
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"Class: {label}")
        axes[i].axis('off')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def visualize_step_comparison(samples_dict, class_label, save_path=None):
    """Visualize same class across different step counts."""
    steps_list = sorted(samples_dict.keys())
    samples_per_step = len(samples_dict[steps_list[0]])
    
    fig, axes = plt.subplots(samples_per_step, len(steps_list),
                              figsize=(len(steps_list) * 2.5, samples_per_step * 2.5))
    
    if samples_per_step == 1:
        axes = axes.reshape(1, -1)
    
    for row in range(samples_per_step):
        for col, steps in enumerate(steps_list):
            sample = samples_dict[steps][row]
            img = denormalize_to_zero_one(sample[..., 0])
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title(f"{steps} steps")
            axes[row, col].axis('off')
    
    plt.suptitle(f"Class {class_label} - DDIM Steps Comparison", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def measure_latency(model, scheduler, num_runs=10, num_steps=8, warmup=2):
    """Measure inference latency."""
    # Warmup
    for _ in range(warmup):
        _ = scheduler.sample(model, 0, num_steps, 0.0)
    
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = scheduler.sample(model, 0, num_steps, 0.0)
        latencies.append(time.perf_counter() - start)
    
    latencies = np.array(latencies)
    return {
        'mean_ms': float(np.mean(latencies) * 1000),
        'std_ms': float(np.std(latencies) * 1000),
        'min_ms': float(np.min(latencies) * 1000),
        'max_ms': float(np.max(latencies) * 1000),
        'per_step_ms': float(np.mean(latencies) * 1000 / num_steps),
        'num_runs': num_runs,
        'warmup_runs': warmup
    }


def analyze_samples(samples):
    """Analyze pixel statistics of generated samples."""
    flat = samples.flatten()
    return {
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
        'range': float(np.max(flat) - np.min(flat)),
        'median': float(np.median(flat))
    }


def compute_diversity(samples):
    """Compute diversity metric: average pairwise distance within class."""
    # samples: [num_classes, samples_per_class, 28, 28, 1]
    num_classes = samples.shape[0]
    sppc = samples.shape[1]
    diversities = []
    
    for c in range(num_classes):
        class_samples = samples[c].reshape(sppc, -1)  # [sppc, 784]
        if sppc > 1:
            dists = []
            for i in range(sppc):
                for j in range(i+1, sppc):
                    dist = np.linalg.norm(class_samples[i] - class_samples[j])
                    dists.append(dist)
            diversities.append(float(np.mean(dists)) if dists else 0.0)
        else:
            diversities.append(0.0)
    
    return {
        'per_class': diversities,
        'mean': float(np.mean(diversities)),
        'std': float(np.std(diversities))
    }


def main():
    config_path = "configs/tiny_unet_v1.yaml"
    checkpoint_path = "artifacts/checkpoints_tiny_unet_v1/inference_model.keras"
    output_dir = "artifacts/evaluation_tiny_unet_v1/ddim_100"
    
    ensure_dir(output_dir)
    
    print("=" * 60)
    print("100-Step DDIM Evaluation for Tiny U-Net v1")
    print("=" * 60)
    
    # Load model and scheduler
    print(f"Loading model from {checkpoint_path}...")
    model, noise_schedule, scheduler, config = load_model_and_scheduler(
        config_path, checkpoint_path)
    
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Prediction type: {noise_schedule.prediction_type}")
    print(f"Training timesteps: {noise_schedule.num_timesteps}")
    print()
    
    # Configuration
    classes = list(range(10))
    samples_per_class = 5
    step_configs = [8, 16, 32, 50, 100]
    
    # Generate fixed initial noises for reproducibility
    np.random.seed(42)
    fixed_noises = []
    for class_idx in range(len(classes)):
        class_noises = []
        for sample_idx in range(samples_per_class):
            noise = np.random.randn(1, 28, 28, 1).astype(np.float32)
            class_noises.append(noise)
        fixed_noises.append(class_noises)
    
    # Generate samples for each step configuration
    all_results = {}
    latency_results = {}
    diversity_results = {}
    pixel_stats = {}
    
    for steps in step_configs:
        print(f"\nGenerating samples for {steps} DDIM steps...")
        
        # Generate class grid with fixed noises
        samples, noises_used = generate_grid_fixed_noise(
            model, scheduler, classes, samples_per_class, steps, fixed_noises)
        
        all_results[steps] = samples
        
        # Analyze
        stats = {}
        for c in classes:
            class_samples = samples[c]
            stats[c] = analyze_samples(class_samples)
        pixel_stats[steps] = stats
        
        # Diversity
        div = compute_diversity(samples)
        diversity_results[steps] = div
        
        # Save class grid
        visualize_grid(
            samples, 
            title=f"Tiny U-Net v1 - {steps} DDIM Steps (5 samples/class)",
            save_path=os.path.join(output_dir, f"class_grid_{steps}steps.png"),
            samples_per_class=samples_per_class
        )
        
        # Save individual class images
        for c in classes:
            class_samples = samples[c]
            visualize_grid(
                class_samples.reshape(1, samples_per_class, 28, 28, 1),
                title=f"Class {c} - {steps} steps",
                save_path=os.path.join(output_dir, f"class_{c}_{steps}steps.png"),
                samples_per_class=samples_per_class
            )
    
    # Step comparison for each class
    print("\nGenerating step comparison grids...")
    for c in classes:
        class_samples_dict = {}
        for steps in step_configs:
            class_samples_dict[steps] = all_results[steps][c]
        visualize_step_comparison(
            class_samples_dict, c,
            save_path=os.path.join(output_dir, f"step_comparison_class_{c}.png")
        )
    
    # Overall comparison grid (same initial noise across step configs)
    print("\nGenerating cross-step comparison grid...")
    # Show first sample of each class across all step configs
    comparison_samples = []
    for c in classes:
        row_samples = []
        for steps in step_configs:
            row_samples.append(all_results[steps][c][0])  # First sample
        comparison_samples.append(np.stack(row_samples))
    comparison_samples = np.stack(comparison_samples)  # [10, 5, 28, 28, 1]
    
    # Flatten for visualization: show class 0 across all steps, class 1 across all steps, etc.
    viz_samples = comparison_samples.reshape(-1, 28, 28, 1)
    step_labels = step_configs * 10
    class_labels_viz = np.repeat(classes, len(step_configs))
    titles = [f"Class {c} - {s} steps" for c, s in zip(class_labels_viz, step_labels)]
    
    fig, axes = plt.subplots(10, 5, figsize=(12, 24))
    for idx, (sample, title) in enumerate(zip(viz_samples, titles)):
        row, col = divmod(idx, 5)
        img = denormalize_to_zero_one(sample[..., 0])
        axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(title, fontsize=9)
        axes[row, col].axis('off')
    plt.suptitle("Cross-Step Comparison: Same Initial Noise, Different DDIM Steps", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_step_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/cross_step_comparison.png")
    
    # Measure latency for each configuration
    print("\nMeasuring inference latency...")
    for steps in step_configs:
        print(f"  Measuring {steps} steps...")
        latency = measure_latency(model, scheduler, num_runs=10, num_steps=steps, warmup=2)
        latency_results[steps] = latency
        print(f"    {latency['mean_ms']:.1f} ms total, {latency['per_step_ms']:.1f} ms/step")
    
    # Compile results
    results = {
        'model': {
            'name': model.name,
            'parameters': model.count_params(),
            'checkpoint': checkpoint_path
        },
        'config': {
            'num_timesteps': noise_schedule.num_timesteps,
            'schedule_type': noise_schedule.schedule_type,
            'beta_start': noise_schedule.beta_start,
            'beta_end': noise_schedule.beta_end,
            'prediction_type': noise_schedule.prediction_type
        },
        'evaluation': {
            'classes': classes,
            'samples_per_class': samples_per_class,
            'step_configs': step_configs,
            'fixed_seed': 42
        },
        'pixel_statistics': pixel_stats,
        'diversity': diversity_results,
        'latency': latency_results,
        'output_directory': output_dir
    }
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_100_step_results.json')
    save_json(results, results_path)
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model.name} ({model.count_params():,} params)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print()
    
    print("Latency Comparison:")
    for steps in step_configs:
        lat = latency_results[steps]
        print(f"  {steps:3d} steps: {lat['mean_ms']:7.1f} ms total, {lat['per_step_ms']:6.1f} ms/step")
    
    print()
    print("Diversity (mean pairwise distance within class):")
    for steps in step_configs:
        div = diversity_results[steps]
        print(f"  {steps:3d} steps: mean={div['mean']:.3f}, std={div['std']:.3f}")
    
    print()
    print("Pixel Statistics (mean/std across all classes):")
    for steps in step_configs:
        means = [pixel_stats[steps][c]['mean'] for c in classes]
        stds = [pixel_stats[steps][c]['std'] for c in classes]
        print(f"  {steps:3d} steps: mean={np.mean(means):.3f}, std={np.mean(stds):.3f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()