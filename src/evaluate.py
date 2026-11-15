"""
Evaluation pipeline for TinyDiffusion-ESP32.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple
from src.utils import (load_config, ensure_dir, denormalize_to_zero_one, 
                       denormalize_to_uint8, print_model_summary, count_parameters,
                       get_model_size_mb, save_json, Timer)
from src.dataset import MNISTDataset
from src.diffusion import NoiseSchedule
from src.model import build_inference_model
from src.scheduler import DDIMScheduler, generate_class_grid


class ModelEvaluator:
    """Comprehensive evaluation of the trained diffusion model."""
    
    def __init__(self, config: Dict[str, Any], model: tf.keras.Model, noise_schedule: NoiseSchedule):
        self.config = config
        self.model = model
        self.noise_schedule = noise_schedule
        self.eval_config = config.get('evaluation', {})
        self.scheduler = DDIMScheduler(config, noise_schedule)
        self.dataset = MNISTDataset(config)
    
    def evaluate_loss(self, dataset: tf.data.Dataset, num_batches: int = None) -> float:
        """Evaluate model loss on a dataset."""
        from src.diffusion import forward_diffusion_tf
        
        total_loss = 0.0
        total_samples = 0
        
        for i, (x_0, class_labels) in enumerate(dataset):
            if num_batches and i >= num_batches:
                break
            
            batch_size = x_0.shape[0]
            
            # Sample timesteps and noise
            timesteps = tf.random.uniform(
                (batch_size,), 0, self.noise_schedule.num_timesteps, dtype=tf.int32)
            noise = tf.random.normal(tf.shape(x_0), dtype=tf.float32)
            
            # Forward diffusion
            x_t = forward_diffusion_tf(
                x_0, timesteps, noise,
                self.scheduler.schedule_arrays['sqrt_alphas_cumprod'],
                self.scheduler.schedule_arrays['sqrt_one_minus_alphas_cumprod']
            )
            
            # Prepare conditioning
            class_onehot = tf.one_hot(class_labels, depth=10)
            t_normalized = tf.cast(timesteps, tf.float32) / self.noise_schedule.num_timesteps
            t_normalized = tf.reshape(t_normalized, [-1, 1])
            
            class_maps = tf.repeat(class_onehot[:, tf.newaxis, tf.newaxis, :], 28*28, axis=1)
            class_maps = tf.reshape(class_maps, (batch_size, 28, 28, 10))
            
            timestep_maps = tf.repeat(t_normalized[:, tf.newaxis, tf.newaxis, :], 28*28, axis=1)
            timestep_maps = tf.reshape(timestep_maps, (batch_size, 28, 28, 1))
            
            conditioning = tf.concat([class_maps, timestep_maps], axis=-1)
            model_input = tf.concat([x_t, conditioning], axis=-1)
            
            # Predict
            noise_pred = self.model(model_input, training=False)
            
            # MSE loss
            loss = tf.reduce_mean(tf.square(noise - noise_pred))
            total_loss += loss * batch_size
            total_samples += batch_size
        
        return float(total_loss / total_samples) if total_samples > 0 else 0.0
    
    def generate_samples(self, class_label: int, num_samples: int = 5,
                         num_steps: int = 8, eta: float = 0.0,
                         return_intermediates: bool = False) -> List[np.ndarray]:
        """Generate samples for a specific class."""
        samples = []
        for _ in range(num_samples):
            sample = self.scheduler.sample(
                self.model, class_label, num_steps, eta, return_intermediates)
            if return_intermediates:
                samples.append(sample)
            else:
                samples.append(sample)
        return samples
    
    def generate_class_grid(self, classes: List[int] = None, 
                            samples_per_class: int = 5,
                            num_steps: int = 8, eta: float = 0.0) -> np.ndarray:
        """Generate grid of samples for all classes."""
        if classes is None:
            classes = list(range(10))
        
        return generate_class_grid(
            self.model, self.noise_schedule, classes, samples_per_class, num_steps, eta)
    
    def compare_sampling_steps(self, class_label: int, 
                               steps_list: List[int] = None,
                               samples_per_step: int = 3) -> Dict[int, List[np.ndarray]]:
        """Compare generation quality across different sampling step counts."""
        if steps_list is None:
            steps_list = self.eval_config.get('compare_steps', [8, 16, 32])
        
        results = {}
        for steps in steps_list:
            step_samples = []
            for _ in range(samples_per_step):
                sample = self.scheduler.sample(self.model, class_label, steps, 0.0)
                step_samples.append(sample)
            results[steps] = step_samples
        return results
    
    def visualize_samples(self, samples: np.ndarray, 
                          class_labels: List[int] = None,
                          title: str = "Generated Samples",
                          save_path: str = None) -> None:
        """Visualize generated samples in a grid."""
        # samples shape: [num_classes, samples_per_class, 28, 28, 1] or [N, 28, 28, 1]
        if samples.ndim == 5:
            num_classes, samples_per_class = samples.shape[0], samples.shape[1]
            samples = samples.reshape(-1, 28, 28, 1)
            if class_labels is None:
                class_labels = np.repeat(range(num_classes), samples_per_class)
        elif samples.ndim == 4:
            if class_labels is None:
                class_labels = [f"Sample {i}" for i in range(len(samples))]
        else:
            raise ValueError(f"Unexpected samples shape: {samples.shape}")
        
        num_samples = len(samples)
        cols = min(8, num_samples)
        rows = int(np.ceil(num_samples / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
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
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def visualize_step_comparison(self, class_label: int,
                                  steps_list: List[int] = None,
                                  save_path: str = None) -> None:
        """Visualize comparison across sampling steps."""
        if steps_list is None:
            steps_list = self.eval_config.get('compare_steps', [8, 16, 32])
        
        results = self.compare_sampling_steps(class_label, steps_list, samples_per_step=4)
        
        num_steps = len(steps_list)
        samples_per_step = len(results[steps_list[0]])
        
        fig, axes = plt.subplots(samples_per_step, num_steps, 
                                 figsize=(num_steps * 2.5, samples_per_step * 2.5))
        
        if samples_per_step == 1:
            axes = axes.reshape(1, -1)
        
        for row in range(samples_per_step):
            for col, steps in enumerate(steps_list):
                sample = results[steps][row]
                img = denormalize_to_zero_one(sample[..., 0])
                axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
                if row == 0:
                    axes[row, col].set_title(f"{steps} steps")
                axes[row, col].axis('off')
        
        plt.suptitle(f"Class {class_label} - Sampling Steps Comparison")
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved step comparison to {save_path}")
        
        plt.show()
    
    def analyze_pixel_distribution(self, samples: np.ndarray) -> Dict[str, float]:
        """Analyze pixel value distribution of generated samples."""
        # Flatten all samples
        flat = samples.flatten()
        
        return {
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat)),
            'min': float(np.min(flat)),
            'max': float(np.max(flat)),
            'range': float(np.max(flat) - np.min(flat)),
            'median': float(np.median(flat))
        }
    
    def measure_inference_latency(self, num_runs: int = 10, 
                                  num_steps: int = 8) -> Dict[str, float]:
        """Measure inference latency on host."""
        import time
        
        # Warmup
        _ = self.scheduler.sample(self.model, 0, num_steps, 0.0)
        
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.scheduler.sample(self.model, 0, num_steps, 0.0)
            latencies.append(time.perf_counter() - start)
        
        latencies = np.array(latencies)
        
        return {
            'mean_ms': float(np.mean(latencies) * 1000),
            'std_ms': float(np.std(latencies) * 1000),
            'min_ms': float(np.min(latencies) * 1000),
            'max_ms': float(np.max(latencies) * 1000),
            'per_step_ms': float(np.mean(latencies) * 1000 / num_steps)
        }
    
    def test_reproducibility(self, class_label: int = 0, 
                             num_steps: int = 8, seed: int = 42) -> bool:
        """Test that generation is reproducible with fixed seed."""
        import random
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        sample1 = self.scheduler.sample(self.model, class_label, num_steps, 0.0)
        
        # Reset seeds
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        sample2 = self.scheduler.sample(self.model, class_label, num_steps, 0.0)
        
        return np.allclose(sample1, sample2)
    
    def run_full_evaluation(self, output_dir: str = "artifacts/evaluation") -> Dict[str, Any]:
        """Run complete evaluation suite."""
        ensure_dir(output_dir)
        
        results = {}
        
        # 1. Model info
        print("Collecting model info...")
        results['model'] = {
            'parameters': count_parameters(self.model),
            'size_mb': get_model_size_mb('artifacts/checkpoints/inference_model.keras') 
                       if os.path.exists('artifacts/checkpoints/inference_model.keras') else 0
        }
        print(f"Model params: {results['model']['parameters']:,}")
        
        # 2. Load test data for loss evaluation
        print("Evaluating validation loss...")
        _, val_ds, _ = self.dataset.prepare_datasets(
            validation_split=self.config.get('training', {}).get('validation_split', 0.1),
            batch_size=self.config.get('training', {}).get('batch_size', 64)
        )
        val_loss = self.evaluate_loss(val_ds, num_batches=50)
        results['val_loss'] = val_loss
        print(f"Validation loss: {val_loss:.6f}")
        
        # 3. Generate class grid
        print("Generating class grid...")
        classes = self.eval_config.get('classes', list(range(10)))
        samples_per_class = self.eval_config.get('samples_per_class', 5)
        num_steps = self.eval_config.get('compare_steps', [8, 16, 32])[0]
        
        class_grid = self.generate_class_grid(classes, samples_per_class, num_steps)
        results['class_grid_shape'] = class_grid.shape
        
        # Visualize
        self.visualize_samples(class_grid, 
                               title=f"Generated Digits ({num_steps} DDIM steps)",
                               save_path=os.path.join(output_dir, 'class_grid.png'))
        
        # 4. Step comparison for each class
        print("Comparing sampling steps...")
        for class_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            self.visualize_step_comparison(class_id, 
                                          save_path=os.path.join(output_dir, f'step_comparison_class_{class_id}.png'))
        
        # 5. Pixel distribution analysis
        print("Analyzing pixel distributions...")
        pixel_stats = {}
        for class_id in classes:
            class_samples = class_grid[class_id]
            pixel_stats[class_id] = self.analyze_pixel_distribution(class_samples)
        results['pixel_distribution'] = pixel_stats
        
        # 6. Inference latency
        print("Measuring inference latency...")
        for steps in [8, 16, 32]:
            latency = self.measure_inference_latency(num_runs=20, num_steps=steps)
            results[f'latency_{steps}_steps'] = latency
            print(f"  {steps} steps: {latency['mean_ms']:.1f} ms total, {latency['per_step_ms']:.1f} ms/step")
        
        # 7. Reproducibility test
        print("Testing reproducibility...")
        reproducible = self.test_reproducibility()
        results['reproducible'] = reproducible
        print(f"  Reproducible: {reproducible}")
        
        # Save results
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        save_json(results, results_path)
        print(f"Evaluation results saved to {results_path}")
        
        return results


def plot_training_history(history: Dict[str, List[float]], 
                          save_path: str = None) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.get('loss', []), label='Train Loss')
    axes[0].plot(history.get('val_loss', []), label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate if available
    if 'lr' in history:
        axes[1].plot(history['lr'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def visualize_forward_diffusion_samples(model: tf.keras.Model, noise_schedule: NoiseSchedule,
                                         save_path: str = None) -> None:
    """Visualize forward diffusion process on real MNIST samples."""
    from src.diffusion import visualize_forward_diffusion
    from src.dataset import MNISTDataset
    from src.utils import load_config
    
    config = load_config()
    dataset = MNISTDataset(config)
    dataset.load()
    
    # Get one sample per class
    samples = []
    for class_id in range(10):
        class_samples = dataset.get_class_samples(class_id, num_samples=1)
        samples.append(class_samples[0])
    
    samples = np.stack(samples)  # [10, 28, 28, 1]
    
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    
    for class_idx in range(10):
        x_0 = samples[class_idx:class_idx+1]
        visualize_forward_diffusion(x_0[0], noise_schedule, num_steps=10)
        # This creates its own figure, so we need a different approach
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test evaluation setup
    config = load_config("configs/default.yaml")
    noise_schedule = NoiseSchedule(config)
    model = build_inference_model(config)
    
    evaluator = ModelEvaluator(config, model, noise_schedule)
    
    # Test sample generation
    sample = evaluator.scheduler.sample(model, 7, 8, 0.0)
    print(f"Sample shape: {sample.shape}")
    print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
    
    # Test reproducibility
    reproducible = evaluator.test_reproducibility()
    print(f"Reproducible: {reproducible}")
    
    print("Evaluation module test passed!")