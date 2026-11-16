#!/usr/bin/env python3
"""
Phase 7B: INT8 Quantization for Tiny U-Net v1.
Converts the FLOAT32 TFLite model to fully integer-quantized INT8 with proper
diffusion-specific calibration dataset.
"""
import sys
import os
import json
import hashlib
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, ensure_dir, save_json, get_tflite_tensor_details, run_tflite_inference
from src.model import build_inference_model
from src.diffusion import NoiseSchedule, forward_diffusion
from src.dataset import MNISTDataset
from src.scheduler import DDIMScheduler


def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def serialize_for_json(obj):
    """Recursively convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, type):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    else:
        return obj


def save_json_safe(data, filepath):
    """Save JSON with numpy type handling."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(serialize_for_json(data), f, indent=2)


def create_diffusion_calibration_dataset(config, noise_schedule, 
                                         num_forward_samples=400,
                                         num_trajectory_samples=100,
                                         seed=42):
    """
    Create representative calibration dataset for diffusion model.
    
    Includes:
    A. Forward-noised MNIST samples spanning all timesteps and classes
    B. Actual DDIM trajectory intermediate states from FLOAT32 model
    
    Returns a generator function for TFLite converter.
    """
    np.random.seed(seed)
    
    # Load MNIST training data
    dataset = MNISTDataset(config)
    x_train, y_train, _, _ = dataset.load()
    x_train_proc = dataset.preprocess(x_train)
    
    # Get schedule arrays
    from src.diffusion import get_schedule_arrays
    schedule_arrays = get_schedule_arrays(
        noise_schedule.num_timesteps,
        noise_schedule.schedule_type,
        noise_schedule.beta_start,
        noise_schedule.beta_end,
        noise_schedule.cosine_s
    )
    
    # Load FLOAT32 model for trajectory generation
    float_tflite_path = "artifacts/tflite_tiny_unet_v1/model_float32.tflite"
    interpreter = tf.lite.Interpreter(model_path=float_tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    def run_float_tflite(input_data):
        interpreter.set_tensor(input_details['index'], input_data.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details['index'])
    
    # Build Keras model for forward diffusion helper
    keras_model = build_inference_model(config)
    dummy = np.random.randn(1, 28, 28, 12).astype(np.float32)
    _ = keras_model(dummy)
    keras_model.load_weights('artifacts/checkpoints_tiny_unet_v1/inference_model.keras')
    
    scheduler = DDIMScheduler(config, noise_schedule)
    
    calibration_samples = []
    
    # ============================================================
    # A. Forward-noised MNIST samples
    # ============================================================
    print(f"Generating {num_forward_samples} forward-noised calibration samples...")
    
    timesteps_range = np.linspace(0, noise_schedule.num_timesteps - 1, 20, dtype=np.int32)
    
    for _ in range(num_forward_samples):
        # Random timestep from representative range
        t = np.random.choice(timesteps_range)
        
        # Random class (ensure all classes represented)
        class_id = np.random.randint(0, 10)
        
        # Random MNIST image
        idx = np.random.choice(len(x_train_proc))
        x_0 = x_train_proc[idx:idx+1]
        label = np.array([class_id])
        
        # Forward diffusion
        noise = np.random.randn(*x_0.shape).astype(np.float32)
        timestep_batch = np.array([t], dtype=np.int32)
        
        x_t = forward_diffusion(
            x_0, timestep_batch, noise,
            noise_schedule.num_timesteps,
            noise_schedule.schedule_type,
            noise_schedule.beta_start,
            noise_schedule.beta_end
        )
        
        # Prepare model input
        from src.utils import prepare_model_input
        model_input = prepare_model_input(
            x_t, label, timestep_batch,
            num_classes=10,
            normalize_timestep=noise_schedule.normalize_timestep,
            num_timesteps=noise_schedule.num_timesteps
        )
        
        calibration_samples.append(model_input.astype(np.float32))
    
    # ============================================================
    # B. DDIM trajectory intermediate states
    # ============================================================
    print(f"Generating {num_trajectory_samples} DDIM trajectory samples...")
    
    # Use 32-step DDIM (deployment target) for trajectory
    ddim_steps = 32
    timesteps = scheduler.get_ddim_timesteps(ddim_steps)
    
    for _ in range(num_trajectory_samples):
        # Random class
        class_id = np.random.randint(0, 10)
        class_onehot = np.zeros((1, 10), dtype=np.float32)
        class_onehot[0, class_id] = 1.0
        
        # Start from random noise
        x_t = np.random.randn(1, 28, 28, 1).astype(np.float32)
        
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
            
            # Add to calibration samples (this is the actual input distribution during generation)
            calibration_samples.append(model_input.astype(np.float32))
            
            # Run FLOAT32 inference to get next state
            noise_pred = run_float_tflite(model_input)
            x_t = scheduler.ddim_step(x_t, noise_pred, t, t_prev, scheduler.eta)
    
    print(f"Total calibration samples: {len(calibration_samples)}")
    
    # Print input statistics
    all_inputs = np.concatenate(calibration_samples, axis=0)
    print(f"Input stats: shape={all_inputs.shape}, "
          f"min={all_inputs.min():.3f}, max={all_inputs.max():.3f}, "
          f"mean={all_inputs.mean():.3f}, std={all_inputs.std():.3f}")
    
    # Per-channel stats
    for c in range(12):
        ch = all_inputs[..., c]
        print(f"  Channel {c}: min={ch.min():.3f}, max={ch.max():.3f}, "
              f"mean={ch.mean():.3f}, std={ch.std():.3f}")
    
    def gen():
        for sample in calibration_samples:
            yield [sample]
    
    return gen, calibration_samples


def export_int8_tflite(config, noise_schedule, float_model_path, 
                       output_path, calibration_gen):
    """Export fully quantized INT8 TFLite model."""
    ensure_dir(os.path.dirname(output_path))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(
        build_inference_model(config)
    )
    
    # Build model first
    model = build_inference_model(config)
    dummy = np.random.randn(1, 28, 28, 12).astype(np.float32)
    _ = model(dummy)
    model.load_weights('artifacts/checkpoints_tiny_unet_v1/inference_model.keras')
    
    # Create new converter from the trained model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable full INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    
    # Force INT8 for inputs and outputs
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Target ops: TFLITE_BUILTINS_INT8 (full integer)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Convert
    print("Converting to INT8 TFLite...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"INT8 TFLite model saved to {output_path} ({len(tflite_model)} bytes)")
    return tflite_model


def inspect_int8_model(tflite_path):
    """Inspect the converted INT8 TFLite model."""
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
    
    # Check for float operators
    float_ops = [op for op in ops if 'QUANTIZE' in op['op_name'] or 'DEQUANTIZE' in op['op_name'] 
                 or op['op_name'] in ['CONV_2D', 'TRANSPOSE_CONV', 'FULLY_CONNECTED', 'ADD', 'MUL']]
    
    model_size = os.path.getsize(tflite_path)
    
    inspection = {
        'model_path': tflite_path,
        'model_size_bytes': model_size,
        'model_size_kb': model_size / 1024,
        'input_details': details['input'],
        'output_details': details['output'],
        'num_operators': len(ops),
        'unique_operators': unique_ops,
        'all_operators': ops,
        'tensor_details': interpreter.get_tensor_details()
    }
    
    print(f"\n=== INT8 TFLite Model Inspection ===")
    print(f"Model size: {model_size} bytes ({model_size/1024:.1f} KB)")
    print(f"Input: {details['input']}")
    print(f"Output: {details['output']}")
    print(f"Number of operators: {len(ops)}")
    print(f"Unique operators: {unique_ops}")
    
    return inspection


def run_int8_parity_tests(float_tflite_path, int8_tflite_path, 
                          noise_schedule, config):
    """Run numerical parity tests between FLOAT32 and INT8 models."""
    
    # Load interpreters
    float_interpreter = tf.lite.Interpreter(model_path=float_tflite_path)
    float_interpreter.allocate_tensors()
    float_input_details = float_interpreter.get_input_details()[0]
    float_output_details = float_interpreter.get_output_details()[0]
    
    int8_interpreter = tf.lite.Interpreter(model_path=int8_tflite_path)
    int8_interpreter.allocate_tensors()
    int8_input_details = int8_interpreter.get_input_details()[0]
    int8_output_details = int8_interpreter.get_output_details()[0]
    
    print(f"\n=== Tensor Details ===")
    print(f"Float32 input: {float_input_details['name']}, shape={float_input_details['shape']}, dtype={float_input_details['dtype']}")
    print(f"Float32 output: {float_output_details['name']}, shape={float_output_details['shape']}, dtype={float_output_details['dtype']}")
    print(f"INT8 input: {int8_input_details['name']}, shape={int8_input_details['shape']}, dtype={int8_input_details['dtype']}, "
          f"scale={int8_input_details['quantization'][0]}, zp={int8_input_details['quantization'][1]}")
    print(f"INT8 output: {int8_output_details['name']}, shape={int8_output_details['shape']}, dtype={int8_output_details['dtype']}, "
          f"scale={int8_output_details['quantization'][0]}, zp={int8_output_details['quantization'][1]}")
    
    def run_float(input_data):
        float_interpreter.set_tensor(float_input_details['index'], input_data.astype(np.float32))
        float_interpreter.invoke()
        return float_interpreter.get_tensor(float_output_details['index'])
    
    def run_int8(input_data):
        # Quantize input
        scale, zp = int8_input_details['quantization']
        input_q = np.round(input_data / scale + zp).astype(np.int32)
        input_q = np.clip(input_q, -128, 127).astype(np.int8)
        
        int8_interpreter.set_tensor(int8_input_details['index'], input_q)
        int8_interpreter.invoke()
        output_q = int8_interpreter.get_tensor(int8_output_details['index'])
        
        # Dequantize output
        out_scale, out_zp = int8_output_details['quantization']
        output_dq = (output_q.astype(np.float32) - out_zp) * out_scale
        return output_dq
    
    results = {}
    
    # Test A: Random inputs
    print("\n--- Test A: Random Tensor Parity ---")
    np.random.seed(12345)
    random_errors = []
    
    for i in range(10):
        test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
        
        float_out = run_float(test_input)
        int8_out = run_int8(test_input)
        
        max_abs = np.max(np.abs(float_out - int8_out))
        mae = np.mean(np.abs(float_out - int8_out))
        mse = np.mean((float_out - int8_out) ** 2)
        
        float_flat = float_out.flatten()
        int8_flat = int8_out.flatten()
        cos_sim = np.dot(float_flat, int8_flat) / (np.linalg.norm(float_flat) * np.linalg.norm(int8_flat) + 1e-8)
        
        random_errors.append({
            'run': i,
            'max_abs_error': float(max_abs),
            'mae': float(mae),
            'mse': float(mse),
            'cosine_similarity': float(cos_sim),
            'float_range': [float(float_out.min()), float(float_out.max())],
            'int8_range': [float(int8_out.min()), float(int8_out.max())]
        })
        
        print(f"  Run {i}: max_abs={max_abs:.2e}, mae={mae:.2e}, mse={mse:.2e}, cos={cos_sim:.6f}")
    
    results['random_tensor_parity'] = random_errors
    results['random_avg_max_abs'] = float(np.mean([e['max_abs_error'] for e in random_errors]))
    results['random_avg_mae'] = float(np.mean([e['mae'] for e in random_errors]))
    results['random_avg_mse'] = float(np.mean([e['mse'] for e in random_errors]))
    
    # Test B: Forward-noised MNIST
    print("\n--- Test B: Forward-Noised MNIST Samples ---")
    
    dataset = MNISTDataset(config)
    x_test, y_test, _, _ = dataset.load()
    x_test_proc = dataset.preprocess(x_test)
    
    timesteps_to_test = [0, 100, 250, 500, 673, 750, 875, 999]
    mnist_errors = []
    
    for t in timesteps_to_test:
        np.random.seed(t + 42)
        idx = np.random.choice(len(x_test_proc), 10, replace=False)
        x_0_batch = x_test_proc[idx]
        labels_batch = y_test[idx]
        
        noise = np.random.randn(*x_0_batch.shape).astype(np.float32)
        timesteps_batch = np.full(10, t, dtype=np.int32)
        
        x_t = forward_diffusion(
            x_0_batch, timesteps_batch, noise,
            noise_schedule.num_timesteps,
            noise_schedule.schedule_type,
            noise_schedule.beta_start,
            noise_schedule.beta_end
        )
        
        from src.utils import prepare_model_input
        model_input = prepare_model_input(
            x_t, labels_batch, timesteps_batch,
            num_classes=10,
            normalize_timestep=noise_schedule.normalize_timestep,
            num_timesteps=noise_schedule.num_timesteps
        )
        
        # Run sample by sample
        float_outs = []
        int8_outs = []
        for i in range(10):
            sample = model_input[i:i+1]
            float_outs.append(run_float(sample))
            int8_outs.append(run_int8(sample))
        
        float_out = np.concatenate(float_outs, axis=0)
        int8_out = np.concatenate(int8_outs, axis=0)
        
        max_abs = np.max(np.abs(float_out - int8_out))
        mae = np.mean(np.abs(float_out - int8_out))
        mse = np.mean((float_out - int8_out) ** 2)
        
        float_flat = float_out.flatten()
        int8_flat = int8_out.flatten()
        cos_sim = np.dot(float_flat, int8_flat) / (np.linalg.norm(float_flat) * np.linalg.norm(int8_flat) + 1e-8)
        
        mnist_errors.append({
            'timestep': int(t),
            'max_abs_error': float(max_abs),
            'mae': float(mae),
            'mse': float(mse),
            'cosine_similarity': float(cos_sim),
            'float_range': [float(float_out.min()), float(float_out.max())],
            'int8_range': [float(int8_out.min()), float(int8_out.max())]
        })
        
        print(f"  t={t}: max_abs={max_abs:.2e}, mae={mae:.2e}, mse={mse:.2e}, cos={cos_sim:.6f}")
    
    results['mnist_parity'] = mnist_errors
    results['mnist_avg_max_abs'] = float(np.mean([e['max_abs_error'] for e in mnist_errors]))
    results['mnist_avg_mae'] = float(np.mean([e['mae'] for e in mnist_errors]))
    results['mnist_avg_mse'] = float(np.mean([e['mse'] for e in mnist_errors]))
    
    # Test C: Reconstructed x0 parity
    print("\n--- Test C: Reconstructed x0 Parity ---")
    
    from src.diffusion import get_schedule_arrays
    schedule_arrays = get_schedule_arrays(
        noise_schedule.num_timesteps,
        noise_schedule.schedule_type,
        noise_schedule.beta_start,
        noise_schedule.beta_end,
        noise_schedule.cosine_s
    )
    
    x0_errors = []
    
    for t in timesteps_to_test:
        np.random.seed(t + 42)
        idx = np.random.choice(len(x_test_proc), 10, replace=False)
        x_0_batch = x_test_proc[idx]
        labels_batch = y_test[idx]
        
        noise = np.random.randn(*x_0_batch.shape).astype(np.float32)
        timesteps_batch = np.full(10, t, dtype=np.int32)
        
        x_t = forward_diffusion(
            x_0_batch, timesteps_batch, noise,
            noise_schedule.num_timesteps,
            noise_schedule.schedule_type,
            noise_schedule.beta_start,
            noise_schedule.beta_end
        )
        
        from src.utils import prepare_model_input
        model_input = prepare_model_input(
            x_t, labels_batch, timesteps_batch,
            num_classes=10,
            normalize_timestep=noise_schedule.normalize_timestep,
            num_timesteps=noise_schedule.num_timesteps
        )
        
        float_outs = []
        int8_outs = []
        for i in range(10):
            sample = model_input[i:i+1]
            float_outs.append(run_float(sample))
            int8_outs.append(run_int8(sample))
        
        float_out = np.concatenate(float_outs, axis=0)
        int8_out = np.concatenate(int8_outs, axis=0)
        
        # Reconstruct x0
        alpha_bar_t = schedule_arrays['alphas_cumprod'][t]
        sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)
        
        float_x0 = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * float_out
        int8_x0 = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * int8_out
        
        max_abs = np.max(np.abs(float_x0 - int8_x0))
        mse = np.mean((float_x0 - int8_x0) ** 2)
        
        x0_errors.append({
            'timestep': int(t),
            'max_abs_error': float(max_abs),
            'mse': float(mse),
            'float_x0_range': [float(float_x0.min()), float(float_x0.max())],
            'int8_x0_range': [float(int8_x0.min()), float(int8_x0.max())]
        })
        
        print(f"  t={t}: x0 max_abs={max_abs:.2e}, mse={mse:.2e}")
    
    results['x0_reconstruction_parity'] = x0_errors
    results['x0_avg_max_abs'] = float(np.mean([e['max_abs_error'] for e in x0_errors]))
    results['x0_avg_mse'] = float(np.mean([e['mse'] for e in x0_errors]))
    
    return results


def run_int8_ddim_generation(int8_tflite_path, noise_schedule, config, 
                             output_dir, step_configs=[8, 16, 32, 50, 100]):
    """Run DDIM generation with INT8 model and compare with FLOAT32."""
    
    interpreter = tf.lite.Interpreter(model_path=int8_tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Load FLOAT32 for comparison
    float_interpreter = tf.lite.Interpreter(model_path="artifacts/tflite_tiny_unet_v1/model_float32.tflite")
    float_interpreter.allocate_tensors()
    float_input_details = float_interpreter.get_input_details()[0]
    float_output_details = float_interpreter.get_output_details()[0]
    
    def run_float(input_data):
        float_interpreter.set_tensor(float_input_details['index'], input_data.astype(np.float32))
        float_interpreter.invoke()
        return float_interpreter.get_tensor(float_output_details['index'])
    
    def run_int8(input_data):
        scale, zp = input_details['quantization']
        input_q = np.round(input_data / scale + zp).astype(np.int32)
        input_q = np.clip(input_q, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details['index'], input_q)
        interpreter.invoke()
        output_q = interpreter.get_tensor(output_details['index'])
        out_scale, out_zp = output_details['quantization']
        output_dq = (output_q.astype(np.float32) - out_zp) * out_scale
        return output_dq
    
    scheduler = DDIMScheduler(config, noise_schedule)
    
    classes = list(range(10))
    samples_per_class = 5
    
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
        int8_samples = []
        
        for class_idx, class_id in enumerate(classes):
            for sample_idx in range(samples_per_class):
                noise = fixed_noises[class_idx][sample_idx]
                
                # FLOAT32
                float_x = noise.copy()
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
                    model_input = np.concatenate([float_x, conditioning], axis=-1)
                    noise_pred = run_float(model_input)
                    float_x = scheduler.ddim_step(float_x, noise_pred, t, t_prev, scheduler.eta)
                
                keras_samples.append(float_x[0])
                
                # INT8
                int8_x = noise.copy()
                for i, t in enumerate(timesteps):
                    t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                    t_normalized = np.array([[t / scheduler.noise_schedule.num_timesteps]], dtype=np.float32)
                    class_maps = np.zeros((1, 28, 28, 10), dtype=np.float32)
                    for c in range(10):
                        class_maps[:, :, :, c] = class_onehot[:, c:c+1]
                    timestep_maps = np.zeros((1, 28, 28, 1), dtype=np.float32)
                    timestep_maps[:, :, :, 0] = t_normalized[0, 0]
                    conditioning = np.concatenate([class_maps, timestep_maps], axis=-1)
                    model_input = np.concatenate([int8_x, conditioning], axis=-1)
                    noise_pred = run_int8(model_input)
                    int8_x = scheduler.ddim_step(int8_x, noise_pred, t, t_prev, scheduler.eta)
                
                int8_samples.append(int8_x[0])
        
        keras_samples = np.stack(keras_samples).reshape(10, 5, 28, 28, 1)
        int8_samples = np.stack(int8_samples).reshape(10, 5, 28, 28, 1)
        
        max_abs = np.max(np.abs(keras_samples - int8_samples))
        mae = np.mean(np.abs(keras_samples - int8_samples))
        mse = np.mean((keras_samples - int8_samples) ** 2)
        
        generation_results[steps] = {
            'max_abs_error': float(max_abs),
            'mae': float(mae),
            'mse': float(mse),
            'keras_range': [float(keras_samples.min()), float(keras_samples.max())],
            'int8_range': [float(int8_samples.min()), float(int8_samples.max())]
        }
        
        print(f"  {steps} steps: max_abs={max_abs:.2e}, mae={mae:.2e}, mse={mse:.2e}")
        
        # Save visualization
        from src.utils import denormalize_to_zero_one
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(10, 10, figsize=(20, 20))
        for c in classes:
            for s in range(5):
                idx = c * 5 + s
                k_img = denormalize_to_zero_one(keras_samples[c, s, ..., 0])
                axes[c, s*2].imshow(k_img, cmap='gray', vmin=0, vmax=1)
                axes[c, s*2].set_title(f"F32 Cl{c} S{s}")
                axes[c, s*2].axis('off')
                
                t_img = denormalize_to_zero_one(int8_samples[c, s, ..., 0])
                axes[c, s*2+1].imshow(t_img, cmap='gray', vmin=0, vmax=1)
                axes[c, s*2+1].set_title(f"INT8 Cl{c} S{s}")
                axes[c, s*2+1].axis('off')
        
        plt.suptitle(f"FLOAT32 vs INT8 - {steps} DDIM Steps", fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'keras_float32_vs_int8_{steps}steps.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
        
        # Save individual grids
        for name, samples in [('float32', keras_samples), ('int8', int8_samples)]:
            fig, axes = plt.subplots(10, 5, figsize=(10, 20))
            for c in classes:
                for s in range(5):
                    img = denormalize_to_zero_one(samples[c, s, ..., 0])
                    axes[c, s].imshow(img, cmap='gray', vmin=0, vmax=1)
                    axes[c, s].set_title(f"Cl{c}")
                    axes[c, s].axis('off')
            plt.suptitle(f"{name.upper()} - {steps} DDIM Steps", fontsize=14)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'{name}_grid_{steps}steps.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    return generation_results


def measure_int8_latency(int8_tflite_path, num_runs=20, warmup=5):
    """Measure INT8 inference latency."""
    import time
    
    interpreter = tf.lite.Interpreter(model_path=int8_tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    scale, zp = input_details['quantization']
    
    # Warmup
    test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
    input_q = np.round(test_input / scale + zp).astype(np.int32)
    input_q = np.clip(input_q, -128, 127).astype(np.int8)
    
    for _ in range(warmup):
        interpreter.set_tensor(input_details['index'], input_q)
        interpreter.invoke()
    
    # Measure
    latencies = []
    for _ in range(num_runs):
        test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
        input_q = np.round(test_input / scale + zp).astype(np.int32)
        input_q = np.clip(input_q, -128, 127).astype(np.int8)
        
        start = time.perf_counter()
        interpreter.set_tensor(input_details['index'], input_q)
        interpreter.invoke()
        latencies.append(time.perf_counter() - start)
    
    latencies = np.array(latencies) * 1000  # ms
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies))
    }


def main():
    config_path = "configs/tiny_unet_v1.yaml"
    output_dir = "artifacts/tflite_tiny_unet_v1"
    eval_dir = os.path.join(output_dir, "evaluation_int8")
    float_tflite_path = os.path.join(output_dir, "model_float32.tflite")
    int8_tflite_path = os.path.join(output_dir, "model_int8.tflite")
    report_path = os.path.join(output_dir, "int8_conversion_metadata.json")
    
    ensure_dir(output_dir)
    ensure_dir(eval_dir)
    
    print("=" * 60)
    print("Phase 7B: INT8 Quantization for Tiny U-Net v1")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    noise_schedule = NoiseSchedule(config)
    
    # Verify FLOAT32 reference model exists
    if not os.path.exists(float_tflite_path):
        print(f"ERROR: FLOAT32 reference model not found at {float_tflite_path}")
        return
    
    # Create calibration dataset
    print("\nCreating diffusion-specific calibration dataset...")
    calibration_gen, calibration_samples = create_diffusion_calibration_dataset(
        config, noise_schedule,
        num_forward_samples=400,
        num_trajectory_samples=100,
        seed=42
    )
    
    # Save calibration stats
    all_inputs = np.concatenate(calibration_samples, axis=0)
    cal_stats = {
        'num_samples': len(calibration_samples),
        'forward_samples': 400,
        'trajectory_samples': 100,
        'shape': all_inputs.shape,
        'min': float(all_inputs.min()),
        'max': float(all_inputs.max()),
        'mean': float(all_inputs.mean()),
        'std': float(all_inputs.std()),
        'per_channel': {}
    }
    for c in range(12):
        ch = all_inputs[..., c]
        cal_stats['per_channel'][c] = {
            'min': float(ch.min()),
            'max': float(ch.max()),
            'mean': float(ch.mean()),
            'std': float(ch.std())
        }
    
    save_json(cal_stats, os.path.join(output_dir, "calibration_stats.json"))
    
    # Export INT8 TFLite
    print("\nExporting INT8 TFLite model...")
    # Need to recreate generator since it was consumed
    calibration_gen, _ = create_diffusion_calibration_dataset(
        config, noise_schedule,
        num_forward_samples=400,
        num_trajectory_samples=100,
        seed=42
    )
    tflite_model_bytes = export_int8_tflite(config, noise_schedule, 
                                            float_tflite_path, int8_tflite_path, calibration_gen)
    
    # Inspect INT8 model
    print("\nInspecting INT8 model...")
    inspection = inspect_int8_model(int8_tflite_path)
    
    # Run parity tests
    print("\nRunning numerical parity tests...")
    parity_results = run_int8_parity_tests(float_tflite_path, int8_tflite_path, 
                                           noise_schedule, config)
    
    # Run DDIM generation comparison
    print("\nRunning DDIM generation comparison...")
    generation_results = run_int8_ddim_generation(
        int8_tflite_path, noise_schedule, config, eval_dir,
        step_configs=[8, 16, 32, 50, 100]
    )
    
    # Measure latency
    print("\nMeasuring INT8 inference latency...")
    latency_results = measure_int8_latency(int8_tflite_path)
    
    # Compile full report
    report = {
        'conversion_metadata': {
            'timestamp': datetime.now().isoformat(),
            'source_checkpoint': 'artifacts/checkpoints_tiny_unet_v1/inference_model.keras',
            'source_checkpoint_sha256': compute_file_hash('artifacts/checkpoints_tiny_unet_v1/inference_model.keras'),
            'float_tflite_path': float_tflite_path,
            'float_tflite_sha256': compute_file_hash(float_tflite_path),
            'config_path': config_path,
            'tensorflow_version': tf.__version__,
            'keras_version': tf.keras.__version__,
            'output_model': int8_tflite_path,
            'output_model_sha256': compute_file_hash(int8_tflite_path) if os.path.exists(int8_tflite_path) else None
        },
        'calibration': cal_stats,
        'model_inspection': serialize_for_json(inspection),
        'parity_tests': parity_results,
        'generation_comparison': generation_results,
        'latency': latency_results,
        'evaluation_directory': eval_dir
    }
    
    # Save report
    save_json_safe(report, report_path)
    print(f"\nConversion report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("INT8 QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"INT8 model size: {inspection['model_size_bytes']} bytes ({inspection['model_size_kb']:.1f} KB)")
    print(f"Size reduction: {inspection['model_size_bytes'] / os.path.getsize(float_tflite_path) * 100:.1f}% of FLOAT32")
    print(f"Input: {inspection['input_details']}")
    print(f"Output: {inspection['output_details']}")
    print(f"Operators: {inspection['unique_operators']}")
    
    print(f"\nParity Test A (Random): avg_max_abs={report['parity_tests']['random_avg_max_abs']:.2e}")
    print(f"Parity Test B (MNIST):  avg_max_abs={report['parity_tests']['mnist_avg_max_abs']:.2e}")
    print(f"Parity Test C (x0):     avg_max_abs={report['parity_tests']['x0_avg_max_abs']:.2e}")
    
    print(f"\nDDIM Generation (32 steps): max_abs={generation_results[32]['max_abs_error']:.2e}")
    
    print(f"\nLatency: mean={report['latency']['mean_ms']:.1f}ms, median={report['latency']['median_ms']:.1f}ms")
    
    # Verdict
    max_abs_32 = generation_results[32]['max_abs_error']
    if max_abs_32 < 1e-3:
        verdict = "PASS"
    elif max_abs_32 < 1e-2:
        verdict = "CONDITIONAL"
    else:
        verdict = "FAIL"
    
    print(f"\nQuantization verdict: {verdict}")
    print(f"  DDIM 32-step max error: {max_abs_32:.2e}")
    
    return report


if __name__ == "__main__":
    main()