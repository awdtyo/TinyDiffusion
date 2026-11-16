#!/usr/bin/env python3
"""
Float32 TFLite Conversion for Tiny U-Net v1.
Converts the trained Tiny U-Net v1 model to float32 TensorFlow Lite.
"""
import sys
import os
import json
import hashlib
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, ensure_dir, save_json, get_tflite_tensor_details, print_model_summary
from src.model import build_inference_model
from src.diffusion import NoiseSchedule
from src.dataset import MNISTDataset
from src.scheduler import DDIMScheduler


def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def export_tflite_float32(model: tf.keras.Model, output_path: str) -> bytes:
    """Export model to float32 TensorFlow Lite."""
    ensure_dir(os.path.dirname(output_path))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = []  # No quantization
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Float32 TFLite model saved to {output_path} ({len(tflite_model)} bytes)")
    return tflite_model


def inspect_tflite_model(tflite_path: str) -> dict:
    """Inspect the converted TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    details = get_tflite_tensor_details(interpreter)
    
    # Get operator list
    ops = []
    for op in interpreter._get_ops_details():
        ops.append({
            'op_name': op['op_name'],
            'inputs': [int(x) for x in op['inputs']],
            'outputs': [int(x) for x in op['outputs']]
        })
    
    unique_ops = list(set(op['op_name'] for op in ops))
    
    # Check for Flex/custom ops
    flex_ops = [op for op in ops if 'FLEX' in op['op_name'] or 'CUSTOM' in op['op_name']]
    
    model_size = os.path.getsize(tflite_path)
    
    inspection = {
        'model_path': tflite_path,
        'model_size_bytes': model_size,
        'model_size_kb': model_size / 1024,
        'input_details': details['input'],
        'output_details': details['output'],
        'num_operators': len(ops),
        'unique_operators': unique_ops,
        'flex_custom_operators': flex_ops,
        'all_operators': ops
    }
    
    print(f"\n=== TFLite Model Inspection ===")
    print(f"Model size: {model_size} bytes ({model_size/1024:.1f} KB)")
    print(f"Input: {details['input']}")
    print(f"Output: {details['output']}")
    print(f"Number of operators: {len(ops)}")
    print(f"Unique operators: {unique_ops}")
    if flex_ops:
        print(f"Flex/Custom operators: {flex_ops}")
    else:
        print(f"Flex/Custom operators: None")
    
    return inspection


def run_parity_tests(keras_model: tf.keras.Model, tflite_path: str, 
                     noise_schedule: NoiseSchedule, config: dict) -> dict:
    """Run numerical parity tests between Keras and TFLite models."""
    
    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"\n=== TFLite Tensor Details ===")
    print(f"Input: {input_details['name']}, shape={input_details['shape']}, dtype={input_details['dtype']}")
    print(f"Output: {output_details['name']}, shape={output_details['shape']}, dtype={output_details['dtype']}")
    
    def run_tflite(input_data):
        """Run inference on TFLite model."""
        interpreter.set_tensor(input_details['index'], input_data.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details['index'])
    
    results = {}
    
    # Test A: Random tensor parity
    print("\n--- Test A: Random Tensor Parity ---")
    np.random.seed(12345)
    random_errors = []
    
    for i in range(10):
        test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
        
        keras_out = keras_model.predict(test_input, verbose=0)
        tflite_out = run_tflite(test_input)
        
        max_abs = np.max(np.abs(keras_out - tflite_out))
        mae = np.mean(np.abs(keras_out - tflite_out))
        mse = np.mean((keras_out - tflite_out) ** 2)
        
        # Cosine similarity
        keras_flat = keras_out.flatten()
        tflite_flat = tflite_out.flatten()
        cos_sim = np.dot(keras_flat, tflite_flat) / (np.linalg.norm(keras_flat) * np.linalg.norm(tflite_flat) + 1e-8)
        
        random_errors.append({
            'run': i,
            'max_abs_error': float(max_abs),
            'mae': float(mae),
            'mse': float(mse),
            'cosine_similarity': float(cos_sim),
            'keras_range': [float(keras_out.min()), float(keras_out.max())],
            'tflite_range': [float(tflite_out.min()), float(tflite_out.max())]
        })
        
        print(f"  Run {i}: max_abs={max_abs:.2e}, mae={mae:.2e}, mse={mse:.2e}, cos={cos_sim:.6f}")
    
    results['random_tensor_parity'] = random_errors
    results['random_avg_max_abs'] = float(np.mean([e['max_abs_error'] for e in random_errors]))
    results['random_avg_mae'] = float(np.mean([e['mae'] for e in random_errors]))
    results['random_avg_mse'] = float(np.mean([e['mse'] for e in random_errors]))
    
    # Test B: Real forward-noised MNIST samples
    print("\n--- Test B: Real Forward-Noised MNIST Samples ---")
    
    # Load MNIST test data
    dataset = MNISTDataset(config)
    _, _, x_test, y_test = dataset.load()
    x_test_proc = dataset.preprocess(x_test)
    
    # Get schedule arrays
    from src.diffusion import get_schedule_arrays
    schedule_arrays = get_schedule_arrays(
        noise_schedule.num_timesteps,
        noise_schedule.schedule_type,
        noise_schedule.beta_start,
        noise_schedule.beta_end,
        noise_schedule.cosine_s
    )
    
    timesteps_to_test = [0, 100, 250, 500, 673, 750, 875, 999]
    mnist_errors = []
    
    for t in timesteps_to_test:
        # Use fixed samples for reproducibility
        np.random.seed(t + 42)
        idx = np.random.choice(len(x_test_proc), 10, replace=False)
        x_0_batch = x_test_proc[idx]
        labels_batch = y_test[idx]
        
        noise = np.random.randn(*x_0_batch.shape).astype(np.float32)
        timesteps_batch = np.full(10, t, dtype=np.int32)
        
        # Forward diffusion
        from src.diffusion import forward_diffusion
        x_t = forward_diffusion(x_0_batch, timesteps_batch, noise, noise_schedule.num_timesteps,
                               noise_schedule.schedule_type, noise_schedule.beta_start, noise_schedule.beta_end)
        
        # Prepare conditioning
        from src.utils import prepare_model_input
        model_input = prepare_model_input(
            x_t, labels_batch, timesteps_batch,
            num_classes=10,
            normalize_timestep=noise_schedule.normalize_timestep,
            num_timesteps=noise_schedule.num_timesteps
        )
        
        # Run inference sample by sample (TFLite has fixed batch size 1)
        keras_outs = []
        tflite_outs = []
        for i in range(10):
            sample_input = model_input[i:i+1]
            keras_outs.append(keras_model.predict(sample_input, verbose=0))
            tflite_outs.append(run_tflite(sample_input))
        
        keras_out = np.concatenate(keras_outs, axis=0)
        tflite_out = np.concatenate(tflite_outs, axis=0)
        
        max_abs = np.max(np.abs(keras_out - tflite_out))
        mae = np.mean(np.abs(keras_out - tflite_out))
        mse = np.mean((keras_out - tflite_out) ** 2)
        
        keras_flat = keras_out.flatten()
        tflite_flat = tflite_out.flatten()
        cos_sim = np.dot(keras_flat, tflite_flat) / (np.linalg.norm(keras_flat) * np.linalg.norm(tflite_flat) + 1e-8)
        
        mnist_errors.append({
            'timestep': int(t),
            'max_abs_error': float(max_abs),
            'mae': float(mae),
            'mse': float(mse),
            'cosine_similarity': float(cos_sim),
            'keras_range': [float(keras_out.min()), float(keras_out.max())],
            'tflite_range': [float(tflite_out.min()), float(tflite_out.max())]
        })
        
        print(f"  t={t}: max_abs={max_abs:.2e}, mae={mae:.2e}, mse={mse:.2e}, cos={cos_sim:.6f}")
    
    results['mnist_parity'] = mnist_errors
    results['mnist_avg_max_abs'] = float(np.mean([e['max_abs_error'] for e in mnist_errors]))
    results['mnist_avg_mae'] = float(np.mean([e['mae'] for e in mnist_errors]))
    results['mnist_avg_mse'] = float(np.mean([e['mse'] for e in mnist_errors]))
    
    # Test C: Reconstructed x0 parity
    print("\n--- Test C: Reconstructed x0 Parity ---")
    
    x0_errors = []
    
    for t in timesteps_to_test:
        np.random.seed(t + 42)
        idx = np.random.choice(len(x_test_proc), 10, replace=False)
        x_0_batch = x_test_proc[idx]
        labels_batch = y_test[idx]
        
        noise = np.random.randn(*x_0_batch.shape).astype(np.float32)
        timesteps_batch = np.full(10, t, dtype=np.int32)
        
        # Forward diffusion
        from src.diffusion import forward_diffusion
        x_t = forward_diffusion(x_0_batch, timesteps_batch, noise, noise_schedule.num_timesteps,
                               noise_schedule.schedule_type, noise_schedule.beta_start, noise_schedule.beta_end)
        
        # Prepare conditioning
        from src.utils import prepare_model_input
        model_input = prepare_model_input(
            x_t, labels_batch, timesteps_batch,
            num_classes=10,
            normalize_timestep=noise_schedule.normalize_timestep,
            num_timesteps=noise_schedule.num_timesteps
        )
        
        # Run inference sample by sample
        keras_outs = []
        tflite_outs = []
        for i in range(10):
            sample_input = model_input[i:i+1]
            keras_outs.append(keras_model.predict(sample_input, verbose=0))
            tflite_outs.append(run_tflite(sample_input))
        
        keras_out = np.concatenate(keras_outs, axis=0)
        tflite_out = np.concatenate(tflite_outs, axis=0)
        
        # Reconstruct x0
        alpha_bar_t = schedule_arrays['alphas_cumprod'][t]
        sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)
        
        keras_x0 = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * keras_out
        tflite_x0 = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * tflite_out
        
        max_abs = np.max(np.abs(keras_x0 - tflite_x0))
        mse = np.mean((keras_x0 - tflite_x0) ** 2)
        
        x0_errors.append({
            'timestep': int(t),
            'max_abs_error': float(max_abs),
            'mse': float(mse),
            'keras_x0_range': [float(keras_x0.min()), float(keras_x0.max())],
            'tflite_x0_range': [float(tflite_x0.min()), float(tflite_x0.max())]
        })
        
        print(f"  t={t}: x0 max_abs={max_abs:.2e}, mse={mse:.2e}")
    
    results['x0_reconstruction_parity'] = x0_errors
    results['x0_avg_max_abs'] = float(np.mean([e['max_abs_error'] for e in x0_errors]))
    results['x0_avg_mse'] = float(np.mean([e['mse'] for e in x0_errors]))
    
    return results


def run_ddim_generation_comparison(keras_model: tf.keras.Model, tflite_path: str,
                                   noise_schedule: NoiseSchedule, config: dict,
                                   output_dir: str) -> dict:
    """Compare DDIM generation between Keras and TFLite models."""
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    def run_tflite(input_data):
        interpreter.set_tensor(input_details['index'], input_data.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details['index'])
    
    scheduler = DDIMScheduler(config, noise_schedule)
    
    classes = list(range(10))
    samples_per_class = 5
    step_configs = [8, 32, 100]
    
    # Fixed initial noises
    np.random.seed(42)
    fixed_noises = []
    for class_idx in range(len(classes)):
        class_noises = []
        for sample_idx in range(samples_per_class):
            noise = np.random.randn(1, 28, 28, 1).astype(np.float32)
            class_noises.append(noise)
        fixed_noises.append(class_noises)
    
    generation_results = {}
    
    for steps in step_configs:
        print(f"\n--- DDIM Generation: {steps} steps ---")
        
        keras_samples = []
        tflite_samples = []
        
        for class_idx, class_id in enumerate(classes):
            for sample_idx in range(samples_per_class):
                noise = fixed_noises[class_idx][sample_idx]
                
                # Generate with Keras
                keras_sample = scheduler.sample(keras_model, class_id, steps, 0.0, return_intermediates=False)
                keras_samples.append(keras_sample)
                
                # Generate with TFLite (need to implement TFLite version of sample)
                # For simplicity, use the same scheduler but with TFLite interpreter
                x_t = noise.copy()
                timesteps = scheduler.get_ddim_timesteps(steps)
                
                class_onehot = np.zeros((1, 10), dtype=np.float32)
                class_onehot[0, class_id] = 1.0
                
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
                    
                    noise_pred = run_tflite(model_input)
                    x_t = scheduler.ddim_step(x_t, noise_pred, t, t_prev, scheduler.eta)
                
                tflite_samples.append(x_t[0])
        
        keras_samples = np.stack(keras_samples).reshape(10, 5, 28, 28, 1)
        tflite_samples = np.stack(tflite_samples).reshape(10, 5, 28, 28, 1)
        
        # Compare
        max_abs = np.max(np.abs(keras_samples - tflite_samples))
        mae = np.mean(np.abs(keras_samples - tflite_samples))
        mse = np.mean((keras_samples - tflite_samples) ** 2)
        
        generation_results[steps] = {
            'max_abs_error': float(max_abs),
            'mae': float(mae),
            'mse': float(mse),
            'keras_range': [float(keras_samples.min()), float(keras_samples.max())],
            'tflite_range': [float(tflite_samples.min()), float(tflite_samples.max())]
        }
        
        print(f"  {steps} steps: max_abs={max_abs:.2e}, mae={mae:.2e}, mse={mse:.2e}")
        
        # Save visualization
        from src.utils import denormalize_to_zero_one
        import matplotlib.pyplot as plt
        
        # Comparison grid: Keras vs TFLite for each class
        fig, axes = plt.subplots(10, 10, figsize=(20, 20))
        for c in classes:
            for s in range(5):
                idx = c * 5 + s
                # Keras
                k_img = denormalize_to_zero_one(keras_samples[c, s, ..., 0])
                axes[c, s*2].imshow(k_img, cmap='gray', vmin=0, vmax=1)
                axes[c, s*2].set_title(f"Keras Cl{c} S{s}")
                axes[c, s*2].axis('off')
                
                # TFLite
                t_img = denormalize_to_zero_one(tflite_samples[c, s, ..., 0])
                axes[c, s*2+1].imshow(t_img, cmap='gray', vmin=0, vmax=1)
                axes[c, s*2+1].set_title(f"TFLite Cl{c} S{s}")
                axes[c, s*2+1].axis('off')
        
        plt.suptitle(f"Keras vs TFLite - {steps} DDIM Steps", fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'keras_vs_tflite_{steps}steps.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
        
        # Also save class grids separately
        for model_name, samples in [('keras', keras_samples), ('tflite', tflite_samples)]:
            fig, axes = plt.subplots(10, 5, figsize=(10, 20))
            for c in classes:
                for s in range(5):
                    img = denormalize_to_zero_one(samples[c, s, ..., 0])
                    axes[c, s].imshow(img, cmap='gray', vmin=0, vmax=1)
                    axes[c, s].set_title(f"Cl{c}")
                    axes[c, s].axis('off')
            plt.suptitle(f"{model_name.upper()} - {steps} DDIM Steps", fontsize=14)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'{model_name}_grid_{steps}steps.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    return generation_results


def measure_latency(model: tf.keras.Model, tflite_path: str, 
                    noise_schedule: NoiseSchedule, config: dict,
                    num_runs: int = 20, warmup: int = 5) -> dict:
    """Measure inference latency for Keras and TFLite models."""
    import time
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    def run_keras(input_data):
        return model.predict(input_data, verbose=0)
    
    def run_tflite(input_data):
        interpreter.set_tensor(input_details['index'], input_data.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details['index'])
    
    # Warmup
    test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
    for _ in range(warmup):
        _ = run_keras(test_input)
        _ = run_tflite(test_input)
    
    # Measure
    keras_times = []
    tflite_times = []
    
    for _ in range(num_runs):
        test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
        
        start = time.perf_counter()
        _ = run_keras(test_input)
        keras_times.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        _ = run_tflite(test_input)
        tflite_times.append(time.perf_counter() - start)
    
    keras_times = np.array(keras_times) * 1000  # ms
    tflite_times = np.array(tflite_times) * 1000  # ms
    
    results = {
        'keras': {
            'mean_ms': float(np.mean(keras_times)),
            'std_ms': float(np.std(keras_times)),
            'min_ms': float(np.min(keras_times)),
            'max_ms': float(np.max(keras_times)),
            'median_ms': float(np.median(keras_times))
        },
        'tflite': {
            'mean_ms': float(np.mean(tflite_times)),
            'std_ms': float(np.std(tflite_times)),
            'min_ms': float(np.min(tflite_times)),
            'max_ms': float(np.max(tflite_times)),
            'median_ms': float(np.median(tflite_times))
        }
    }
    
    print(f"\n=== Latency Comparison ({num_runs} runs, {warmup} warmup) ===")
    print(f"Keras:  mean={results['keras']['mean_ms']:.1f}ms, median={results['keras']['median_ms']:.1f}ms")
    print(f"TFLite: mean={results['tflite']['mean_ms']:.1f}ms, median={results['tflite']['median_ms']:.1f}ms")
    print(f"Speedup: {results['keras']['mean_ms'] / results['tflite']['mean_ms']:.2f}x")
    
    return results


def main():
    # Configuration
    config_path = "configs/tiny_unet_v1.yaml"
    checkpoint_path = "artifacts/checkpoints_tiny_unet_v1/inference_model.keras"
    output_dir = "artifacts/tflite_tiny_unet_v1"
    tflite_path = os.path.join(output_dir, "model_float32.tflite")
    report_path = os.path.join(output_dir, "conversion_report.json")
    eval_dir = os.path.join(output_dir, "evaluation")
    
    ensure_dir(output_dir)
    ensure_dir(eval_dir)
    
    print("=" * 60)
    print("Tiny U-Net v1 Float32 TFLite Conversion")
    print("=" * 60)
    
    # Load config and model
    config = load_config(config_path)
    
    print(f"Loading model from {checkpoint_path}...")
    keras_model = build_inference_model(config)
    
    # Build model
    dummy_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
    _ = keras_model(dummy_input, training=False)
    
    keras_model.load_weights(checkpoint_path)
    print(f"Model loaded: {keras_model.name} ({keras_model.count_params():,} params)")
    
    # Compute checkpoint hash
    checkpoint_hash = compute_file_hash(checkpoint_path)
    print(f"Checkpoint SHA256: {checkpoint_hash}")
    
    # Export to float32 TFLite
    print("\nConverting to float32 TFLite...")
    tflite_model_bytes = export_tflite_float32(keras_model, tflite_path)
    
    # Inspect converted model
    print("\nInspecting TFLite model...")
    inspection = inspect_tflite_model(tflite_path)
    
    # Run parity tests
    print("\nRunning numerical parity tests...")
    noise_schedule = NoiseSchedule(config)
    parity_results = run_parity_tests(keras_model, tflite_path, noise_schedule, config)
    
    # Run DDIM generation comparison
    print("\nRunning DDIM generation comparison...")
    generation_results = run_ddim_generation_comparison(
        keras_model, tflite_path, noise_schedule, config, eval_dir)
    
    # Measure latency
    print("\nMeasuring inference latency...")
    latency_results = measure_latency(keras_model, tflite_path, noise_schedule, config)
    
    # Compile full report
    report = {
        'conversion_metadata': {
            'timestamp': datetime.now().isoformat(),
            'source_checkpoint': checkpoint_path,
            'source_checkpoint_sha256': checkpoint_hash,
            'config_path': config_path,
            'tensorflow_version': tf.__version__,
            'keras_version': tf.keras.__version__,
            'output_model': tflite_path,
            'output_model_sha256': compute_file_hash(tflite_path) if os.path.exists(tflite_path) else None
        },
        'model_inspection': inspection,
        'parity_tests': parity_results,
        'generation_comparison': generation_results,
        'latency': latency_results,
        'evaluation_directory': eval_dir
    }
    
    # Save report
    save_json(report, report_path)
    print(f"\nConversion report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Source checkpoint: {checkpoint_path}")
    print(f"Checkpoint SHA256: {checkpoint_hash}")
    print(f"Output model: {tflite_path}")
    print(f"Model size: {inspection['model_size_bytes']} bytes ({inspection['model_size_kb']:.1f} KB)")
    print(f"Input: {inspection['input_details']}")
    print(f"Output: {inspection['output_details']}")
    print(f"Operators: {inspection['unique_operators']}")
    if inspection['flex_custom_operators']:
        print(f"Flex/Custom operators: {inspection['flex_custom_operators']}")
    else:
        print("Flex/Custom operators: None (all built-in)")
    
    print(f"\nParity Test A (Random): avg_max_abs={report['parity_tests']['random_avg_max_abs']:.2e}")
    print(f"Parity Test B (MNIST):  avg_max_abs={report['parity_tests']['mnist_avg_max_abs']:.2e}")
    print(f"Parity Test C (x0):     avg_max_abs={report['parity_tests']['x0_avg_max_abs']:.2e}")
    
    print(f"\nDDIM Generation (32 steps): max_abs={generation_results[32]['max_abs_error']:.2e}")
    
    print(f"\nLatency: Keras={report['latency']['keras']['mean_ms']:.1f}ms, TFLite={report['latency']['tflite']['mean_ms']:.1f}ms")
    print(f"Speedup: {report['latency']['keras']['mean_ms'] / report['latency']['tflite']['mean_ms']:.2f}x")
    
    # Success criteria check
    max_abs_32 = generation_results[32]['max_abs_error']
    if max_abs_32 < 1e-3:
        print("\n✅ FLOAT32 TFLite conversion SUCCESSFUL")
        print("   Numerical parity within tolerance")
        print("   DDIM generation preserved")
        print("   Ready for INT8 quantization phase")
    else:
        print("\n⚠️  FLOAT32 TFLite conversion COMPLETED WITH WARNINGS")
        print(f"   DDIM generation error: {max_abs_32:.2e}")
        print("   Review before proceeding to INT8 quantization")
    
    return report


if __name__ == "__main__":
    main()