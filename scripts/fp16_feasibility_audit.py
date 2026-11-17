#!/usr/bin/env python3
"""
Phase 8.4: FP16 / Mixed-Precision Feasibility Audit
Investigates FP16 weights, full FP16, and mixed precision options for Tiny U-Net v1
"""

import sys
import os
import json
import hashlib
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, ensure_dir, save_json
from src.model import build_inference_model
from src.diffusion import NoiseSchedule
from src.scheduler import DDIMScheduler


def compute_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def inspect_tflite_model(model_path):
    """Inspect a TFLite model and return operator inventory and tensor info."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    ops = []
    for op in interpreter._get_ops_details():
        ops.append({
            'op_name': op['op_name'],
            'inputs': [int(x) for x in op['inputs']],
            'outputs': [int(x) for x in op['outputs']]
        })
    
    unique_ops = list(set(op['op_name'] for op in ops))
    
    # Check tensor dtypes
    tensor_dtypes = {}
    for detail in tensor_details:
        dt = str(detail['dtype'])
        tensor_dtypes[dt] = tensor_dtypes.get(dt, 0) + 1
    
    return {
        'model_path': model_path,
        'model_size': os.path.getsize(model_path),
        'model_sha256': compute_file_hash(model_path),
        'input_details': input_details,
        'output_details': output_details,
        'num_tensors': len(tensor_details),
        'tensor_dtypes': tensor_dtypes,
        'num_operators': len(ops),
        'unique_operators': unique_ops,
        'all_operators': ops
    }


def convert_fp16_weights(config, keras_model, output_path):
    """Convert model with FP16 weights but FLOAT32 activations."""
    ensure_dir(os.path.dirname(output_path))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # FP16 weights only
    converter.target_spec.supported_types = [tf.float16]
    # Keep built-in ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    print("Converting with FP16 weights...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved: {output_path} ({len(tflite_model)} bytes)")
    return tflite_model


def convert_full_fp16(config, keras_model, output_path):
    """Convert model with full FP16 (weights + activations where possible)."""
    ensure_dir(os.path.dirname(output_path))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    # For full FP16, we need to ensure the converter doesn't force some ops to float32
    # Note: This may not produce truly full FP16 if some ops lack FP16 kernels
    
    print("Converting with full FP16 attempt...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved: {output_path} ({len(tflite_model)} bytes)")
    return tflite_model


def convert_mixed_precision(config, keras_model, output_path):
    """Convert with mixed precision - selective FP16 where supported."""
    ensure_dir(os.path.dirname(output_path))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS  # Allow TF ops if needed
    ]
    
    print("Converting with mixed precision...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved: {output_path} ({len(tflite_model)} bytes)")
    return tflite_model


def run_parity_tests(float32_path, candidate_path, noise_schedule, config):
    """Compare candidate model against FLOAT32 reference."""
    
    float_interpreter = tf.lite.Interpreter(model_path=float32_path)
    float_interpreter.allocate_tensors()
    float_input = float_interpreter.get_input_details()[0]
    float_output = float_interpreter.get_output_details()[0]
    
    cand_interpreter = tf.lite.Interpreter(model_path=candidate_path)
    cand_interpreter.allocate_tensors()
    cand_input = cand_interpreter.get_input_details()[0]
    cand_output = cand_interpreter.get_output_details()[0]
    
    def run_float(input_data):
        float_interpreter.set_tensor(float_input['index'], input_data.astype(np.float32))
        float_interpreter.invoke()
        return float_interpreter.get_tensor(float_output['index'])
    
    def run_cand(input_data):
        # Handle different input dtypes
        if cand_input['dtype'] == np.float16:
            input_converted = input_data.astype(np.float16)
        else:
            input_converted = input_data.astype(np.float32)
        cand_interpreter.set_tensor(cand_input['index'], input_converted)
        cand_interpreter.invoke()
        out = cand_interpreter.get_tensor(cand_output['index'])
        return out.astype(np.float32) if out.dtype == np.float16 else out
    
    results = {}
    
    # Test A: Random tensors
    np.random.seed(12345)
    random_errors = []
    
    for i in range(10):
        test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
        float_out = run_float(test_input)
        cand_out = run_cand(test_input)
        
        max_abs = np.max(np.abs(float_out - cand_out))
        mae = np.mean(np.abs(float_out - cand_out))
        mse = np.mean((float_out - cand_out) ** 2)
        
        float_flat = float_out.flatten()
        cand_flat = cand_out.flatten()
        cos_sim = np.dot(float_flat, cand_flat) / (np.linalg.norm(float_flat) * np.linalg.norm(cand_flat) + 1e-8)
        
        random_errors.append({
            'run': i,
            'max_abs_error': float(max_abs),
            'mae': float(mae),
            'mse': float(mse),
            'cosine_similarity': float(cos_sim)
        })
    
    results['random_tensor_parity'] = random_errors
    results['random_avg_max_abs'] = float(np.mean([e['max_abs_error'] for e in random_errors]))
    
    # Test B: Forward-noised MNIST
    from src.dataset import MNISTDataset
    from src.diffusion import forward_diffusion, get_schedule_arrays
    from src.utils import prepare_model_input
    
    dataset = MNISTDataset(config)
    _, _, x_test, y_test = dataset.load()
    x_test_proc = dataset.preprocess(x_test)
    
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
        
        model_input = prepare_model_input(
            x_t, labels_batch, timesteps_batch,
            num_classes=10,
            normalize_timestep=noise_schedule.normalize_timestep,
            num_timesteps=noise_schedule.num_timesteps
        )
        
        float_outs = []
        cand_outs = []
        for i in range(10):
            sample = model_input[i:i+1]
            float_outs.append(run_float(sample))
            cand_outs.append(run_cand(sample))
        
        float_out = np.concatenate(float_outs, axis=0)
        cand_out = np.concatenate(cand_outs, axis=0)
        
        max_abs = np.max(np.abs(float_out - cand_out))
        mae = np.mean(np.abs(float_out - cand_out))
        mse = np.mean((float_out - cand_out) ** 2)
        
        mnist_errors.append({
            'timestep': int(t),
            'max_abs_error': float(max_abs),
            'mae': float(mae),
            'mse': float(mse)
        })
    
    results['mnist_parity'] = mnist_errors
    results['mnist_avg_max_abs'] = float(np.mean([e['max_abs_error'] for e in mnist_errors]))
    
    return results


def run_ddim_generation(candidate_path, noise_schedule, config, output_dir, step_configs=[8, 32]):
    """Run DDIM generation with candidate model."""
    
    cand_interpreter = tf.lite.Interpreter(model_path=candidate_path)
    cand_interpreter.allocate_tensors()
    cand_input = cand_interpreter.get_input_details()[0]
    cand_output = cand_interpreter.get_output_details()[0]
    
    def run_cand(input_data):
        if cand_input['dtype'] == np.float16:
            input_converted = input_data.astype(np.float16)
        else:
            input_converted = input_data.astype(np.float32)
        cand_interpreter.set_tensor(cand_input['index'], input_converted)
        cand_interpreter.invoke()
        out = cand_interpreter.get_tensor(cand_output['index'])
        return out.astype(np.float32) if out.dtype == np.float16 else out
    
    # Load FLOAT32 reference
    float_interpreter = tf.lite.Interpreter(model_path="artifacts/tflite_tiny_unet_v1/model_float32.tflite")
    float_interpreter.allocate_tensors()
    float_input = float_interpreter.get_input_details()[0]
    float_output = float_interpreter.get_output_details()[0]
    
    def run_float(input_data):
        float_interpreter.set_tensor(float_input['index'], input_data.astype(np.float32))
        float_interpreter.invoke()
        return float_interpreter.get_tensor(float_output['index'])
    
    scheduler = DDIMScheduler(config, noise_schedule)
    
    classes = list(range(10))
    samples_per_class = 5
    
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
        
        float_samples = []
        cand_samples = []
        
        for class_idx, class_id in enumerate(classes):
            for sample_idx in range(samples_per_class):
                noise = fixed_noises[class_idx][sample_idx]
                
                # FLOAT32 reference
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
                
                float_samples.append(float_x[0])
                
                # Candidate
                cand_x = noise.copy()
                for i, t in enumerate(timesteps):
                    t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                    t_normalized = np.array([[t / scheduler.noise_schedule.num_timesteps]], dtype=np.float32)
                    class_maps = np.zeros((1, 28, 28, 10), dtype=np.float32)
                    for c in range(10):
                        class_maps[:, :, :, c] = class_onehot[:, c:c+1]
                    timestep_maps = np.zeros((1, 28, 28, 1), dtype=np.float32)
                    timestep_maps[:, :, :, 0] = t_normalized[0, 0]
                    conditioning = np.concatenate([class_maps, timestep_maps], axis=-1)
                    model_input = np.concatenate([cand_x, conditioning], axis=-1)
                    noise_pred = run_cand(model_input)
                    cand_x = scheduler.ddim_step(cand_x, noise_pred, t, t_prev, scheduler.eta)
                
                cand_samples.append(cand_x[0])
        
        float_samples = np.stack(float_samples).reshape(10, 5, 28, 28, 1)
        cand_samples = np.stack(cand_samples).reshape(10, 5, 28, 28, 1)
        
        max_abs = np.max(np.abs(float_samples - cand_samples))
        mae = np.mean(np.abs(float_samples - cand_samples))
        mse = np.mean((float_samples - cand_samples) ** 2)
        
        generation_results[steps] = {
            'max_abs_error': float(max_abs),
            'mae': float(mae),
            'mse': float(mse),
            'float_range': [float(float_samples.min()), float(float_samples.max())],
            'cand_range': [float(cand_samples.min()), float(cand_samples.max())]
        }
        
        print(f"  {steps} steps: max_abs={max_abs:.2e}, mae={mae:.2e}, mse={mse:.2e}")
        
        # Save visualization
        from src.utils import denormalize_to_zero_one
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(10, 10, figsize=(20, 20))
        for c in classes:
            for s in range(5):
                idx = c * 5 + s
                f_img = denormalize_to_zero_one(float_samples[c, s, ..., 0])
                axes[c, s*2].imshow(f_img, cmap='gray', vmin=0, vmax=1)
                axes[c, s*2].set_title(f"F32 Cl{c} S{s}")
                axes[c, s*2].axis('off')
                
                c_img = denormalize_to_zero_one(cand_samples[c, s, ..., 0])
                axes[c, s*2+1].imshow(c_img, cmap='gray', vmin=0, vmax=1)
                axes[c, s*2+1].set_title(f"Cand Cl{c} S{s}")
                axes[c, s*2+1].axis('off')
        
        plt.suptitle(f"FLOAT32 vs Candidate - {steps} DDIM Steps", fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'float32_vs_candidate_{steps}steps.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    return generation_results


def main():
    output_dir = "artifacts/tflite_tiny_unet_v1/phase8_fp16"
    ensure_dir(output_dir)
    
    print("=" * 60)
    print("Phase 8.4: FP16 / Mixed-Precision Feasibility Audit")
    print("=" * 60)
    
    # Load config and model
    config = load_config("configs/tiny_unet_v1.yaml")
    noise_schedule = NoiseSchedule(config)
    
    keras_model = build_inference_model(config)
    dummy = np.random.randn(1, 28, 28, 12).astype(np.float32)
    _ = keras_model(dummy)
    keras_model.load_weights('artifacts/checkpoints_tiny_unet_v1/inference_model.keras')
    
    float32_path = "artifacts/tflite_tiny_unet_v1/model_float32.tflite"
    
    # Verify FLOAT32 reference
    print(f"\nVerifying FLOAT32 reference model...")
    float32_sha256 = compute_file_hash(float32_path)
    print(f"SHA256: {float32_sha256}")
    assert float32_sha256 == "10aef8894bc569e54b97acb641ff53def5e0e8f001c10b3f511009b18ac3c829"
    print("✓ SHA256 matches")
    
    # Inspect FLOAT32 baseline
    print("\n--- Inspecting FLOAT32 baseline ---")
    float32_info = inspect_tflite_model(float32_path)
    print(f"Size: {float32_info['model_size']} bytes")
    print(f"Input dtype: {float32_info['input_details'][0]['dtype']}")
    print(f"Output dtype: {float32_info['output_details'][0]['dtype']}")
    print(f"Tensor dtypes: {float32_info['tensor_dtypes']}")
    print(f"Operators: {float32_info['unique_operators']}")
    
    # Convert candidates
    print("\n" + "=" * 60)
    print("CONVERTING CANDIDATES")
    print("=" * 60)
    
    candidates = {}
    
    # 1. FP16 weights
    fp16w_path = os.path.join(output_dir, "model_fp16_weights.tflite")
    convert_fp16_weights(config, keras_model, fp16w_path)
    candidates['fp16_weights'] = fp16w_path
    
    # 2. Full FP16 (attempt)
    fp16_path = os.path.join(output_dir, "model_fp16_full.tflite")
    convert_full_fp16(config, keras_model, fp16_path)
    candidates['fp16_full'] = fp16_path
    
    # 3. Mixed precision
    mixed_path = os.path.join(output_dir, "model_mixed_precision.tflite")
    convert_mixed_precision(config, keras_model, mixed_path)
    candidates['mixed_precision'] = mixed_path
    
    # Inspect all candidates
    print("\n--- Inspecting Candidates ---")
    candidate_info = {}
    for name, path in candidates.items():
        info = inspect_tflite_model(path)
        candidate_info[name] = info
        print(f"\n{name}:")
        print(f"  Size: {info['model_size']} bytes")
        print(f"  Input dtype: {info['input_details'][0]['dtype']}")
        print(f"  Output dtype: {info['output_details'][0]['dtype']}")
        print(f"  Tensor dtypes: {info['tensor_dtypes']}")
        print(f"  Operators: {info['unique_operators']}")
    
    # Run parity tests
    print("\n" + "=" * 60)
    print("NUMERICAL PARITY TESTS")
    print("=" * 60)
    
    parity_results = {}
    generation_results = {}
    
    for name, path in candidates.items():
        print(f"\n--- Testing {name} ---")
        
        # Single-inference parity
        parity = run_parity_tests(float32_path, path, noise_schedule, config)
        parity_results[name] = parity
        print(f"  Random avg max_abs: {parity['random_avg_max_abs']:.2e}")
        print(f"  MNIST avg max_abs:  {parity['mnist_avg_max_abs']:.2e}")
        
        # DDIM generation
        gen = run_ddim_generation(path, noise_schedule, config, output_dir)
        generation_results[name] = gen
    
    # Compile full report
    report = {
        'audit_timestamp': datetime.now().isoformat(),
        'float32_baseline': float32_info,
        'candidates': candidate_info,
        'parity_results': parity_results,
        'generation_results': generation_results,
        'tflm_compatibility': {
            'tf_version': tf.__version__,
            'tflm_version': 'ESP-IDF component (TF ~2.15 equivalent)',
            'fp16_kernel_availability': 'Limited - most kernels are FLOAT32 only',
            'note': 'Full FP16 execution not supported in TFLM for this operator set'
        }
    }
    
    report_path = os.path.join(output_dir, "fp16_feasibility_report.json")
    save_json(report, report_path)
    print(f"\nReport saved to {report_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    print(f"{'Candidate':<20} {'Model Size':<12} {'Input':<8} {'Output':<8} {'Max Abs (rand)':<16} {'Max Abs (MNIST)':<16} {'32-step DDIM':<14}")
    print("-" * 100)
    
    for name, path in candidates.items():
        info = candidate_info[name]
        parity = parity_results[name]
        gen = generation_results[name]
        size_kb = info['model_size'] / 1024
        inp_dt = str(info['input_details'][0]['dtype']).split("'")[1]
        out_dt = str(info['output_details'][0]['dtype']).split("'")[1]
        rand_max = parity['random_avg_max_abs']
        mnist_max = parity['mnist_avg_max_abs']
        ddim_32 = gen[32]['max_abs_error']
        
        print(f"{name:<20} {size_kb:>6.1f} KB  {inp_dt:<8} {out_dt:<8} {rand_max:<16.2e} {mnist_max:<16.2e} {ddim_32:<14.2e}")
    
    # FLOAT32 baseline row
    f32_info = float32_info
    print(f"{'FLOAT32 (baseline)':<20} {f32_info['model_size']/1024:>6.1f} KB  {'float32':<8} {'float32':<8} {'N/A':<16} {'N/A':<16} {'N/A':<14}")
    
    return report


if __name__ == "__main__":
    import numpy as np
    main()