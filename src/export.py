"""
Model export utilities for TinyDiffusion-ESP32.
Exports Keras model to TensorFlow Lite (float32 and INT8).
"""
import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, List, Generator
from src.utils import (load_config, ensure_dir, save_json, get_tflite_tensor_details,
                       run_tflite_inference, compare_outputs, create_model_header,
                       verify_tflite_ops, print_model_summary)
from src.dataset import MNISTDataset
from src.diffusion import NoiseSchedule
from src.model import build_inference_model


def export_saved_model(model: tf.keras.Model, export_dir: str) -> None:
    """Export Keras model to SavedModel format."""
    ensure_dir(export_dir)
    model.save(export_dir, save_format='tf')
    print(f"SavedModel exported to {export_dir}")


def export_tflite_float32(model: tf.keras.Model, output_path: str) -> None:
    """Export model to float32 TensorFlow Lite."""
    ensure_dir(os.path.dirname(output_path))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = []  # No quantization
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Float32 TFLite model saved to {output_path} ({len(tflite_model)} bytes)")


def create_representative_dataset(config: Dict[str, Any], 
                                  noise_schedule: NoiseSchedule,
                                  num_samples: int = 500) -> Generator[List[np.ndarray], None, None]:
    """Create representative dataset for INT8 quantization calibration."""
    dataset = MNISTDataset(config)
    dataset.load()
    
    # Preprocess
    x_train = dataset.preprocess(dataset.x_train)
    y_train = dataset.y_train
    
    # Select random samples
    indices = np.random.choice(len(x_train), min(num_samples, len(x_train)), replace=False)
    x_calib = x_train[indices]
    y_calib = y_train[indices]
    
    # Generate model inputs for calibration
    for i in range(len(x_calib)):
        # Random timestep
        t = np.random.randint(0, noise_schedule.num_timesteps)
        
        # Prepare single sample input
        img = x_calib[i:i+1]  # [1, 28, 28, 1]
        label = y_calib[i:i+1]  # [1]
        
        # Add noise
        noise = np.random.randn(*img.shape).astype(np.float32)
        
        # Forward diffusion
        from src.diffusion import forward_diffusion
        x_t = forward_diffusion(img, np.array([t]), noise, noise_schedule.num_timesteps,
                               noise_schedule.schedule_type, noise_schedule.beta_start, noise_schedule.beta_end)
        
        # Prepare conditioning
        from src.utils import prepare_model_input
        model_input = prepare_model_input(
            x_t, label, np.array([t]),
            num_classes=10,
            normalize_timestep=noise_schedule.normalize_timestep,
            num_timesteps=noise_schedule.num_timesteps
        )
        
        yield [model_input.astype(np.float32)]


def export_tflite_int8(model: tf.keras.Model, config: Dict[str, Any],
                       noise_schedule: NoiseSchedule,
                       output_path: str,
                       representative_data: Generator = None) -> Dict[str, Any]:
    """Export model to fully quantized INT8 TensorFlow Lite."""
    ensure_dir(os.path.dirname(output_path))
    
    quant_config = config.get('quantization', {})
    representative_samples = quant_config.get('representative_samples', 500)
    target_ops = quant_config.get('target_ops', ['TFLITE_BUILTINS'])
    inference_input_type = quant_config.get('inference_input_type', 'int8')
    inference_output_type = quant_config.get('inference_output_type', 'int8')
    
    # Create representative dataset if not provided
    if representative_data is None:
        representative_data = create_representative_dataset(config, noise_schedule, representative_samples)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable full INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    
    # Force INT8 for inputs and outputs
    if inference_input_type == 'int8':
        converter.inference_input_type = tf.int8
    if inference_output_type == 'int8':
        converter.inference_output_type = tf.int8
    
    # Target ops
    supported_ops = []
    for op in target_ops:
        if op == 'TFLITE_BUILTINS':
            supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS)
        elif op == 'SELECT_TF_OPS':
            supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
    converter.target_spec.supported_ops = supported_ops
    
    # Convert
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"INT8 TFLite model saved to {output_path} ({len(tflite_model)} bytes)")
    
    # Verify the model
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    details = get_tflite_tensor_details(interpreter)
    
    return {
        'model_path': output_path,
        'size_bytes': len(tflite_model),
        'input_details': details['input'],
        'output_details': details['output'],
        'ops_verified': verify_tflite_ops(output_path)
    }


def verify_quantized_model(float_model_path: str, int8_model_path: str,
                          config: Dict[str, Any], noise_schedule: NoiseSchedule,
                          num_test_samples: int = 100) -> Dict[str, Any]:
    """Verify INT8 model matches float32 model outputs."""
    
    # Load interpreters
    float_interpreter = tf.lite.Interpreter(model_path=float_model_path)
    float_interpreter.allocate_tensors()
    
    int8_interpreter = tf.lite.Interpreter(model_path=int8_model_path)
    int8_interpreter.allocate_tensors()
    
    # Get quantization parameters
    int8_input_details = int8_interpreter.get_input_details()[0]
    int8_output_details = int8_interpreter.get_output_details()[0]
    
    input_scale, input_zero_point = int8_input_details['quantization']
    output_scale, output_zero_point = int8_output_details['quantization']
    
    print(f"Input quantization: scale={input_scale:.6f}, zero_point={input_zero_point}")
    print(f"Output quantization: scale={output_scale:.6f}, zero_point={output_zero_point}")
    
    # Create test dataset
    dataset = MNISTDataset(config)
    dataset.load()
    x_test = dataset.preprocess(dataset.x_test)
    y_test = dataset.y_test
    
    errors = []
    
    for i in range(min(num_test_samples, len(x_test))):
        # Random timestep
        t = np.random.randint(0, noise_schedule.num_timesteps)
        
        img = x_test[i:i+1]
        label = y_test[i:i+1]
        
        # Add noise
        noise = np.random.randn(*img.shape).astype(np.float32)
        
        # Forward diffusion
        from src.diffusion import forward_diffusion
        x_t = forward_diffusion(img, np.array([t]), noise, noise_schedule.num_timesteps,
                               noise_schedule.schedule_type, noise_schedule.beta_start, noise_schedule.beta_end)
        
        # Prepare model input
        from src.utils import prepare_model_input
        model_input_float = prepare_model_input(
            x_t, label, np.array([t]),
            num_classes=10,
            normalize_timestep=noise_schedule.normalize_timestep,
            num_timesteps=noise_schedule.num_timesteps
        )
        
        # Run float32 inference
        float_output = run_tflite_inference(float_interpreter, model_input_float.astype(np.float32))
        
        # Quantize input for INT8
        model_input_int8 = quantize_int8(model_input_float, input_scale, input_zero_point)
        
        # Run INT8 inference
        int8_output = run_tflite_inference(int8_interpreter, model_input_int8)
        
        # Compare
        error = compare_outputs(float_output, int8_output, output_scale, output_zero_point)
        errors.append(error)
    
    # Aggregate errors
    avg_mse = np.mean([e['mse'] for e in errors])
    avg_mae = np.mean([e['mae'] for e in errors])
    max_error = np.max([e['max_abs_error'] for e in errors])
    avg_rel_error = np.mean([e['relative_error'] for e in errors])
    
    results = {
        'num_samples': len(errors),
        'avg_mse': float(avg_mse),
        'avg_mae': float(avg_mae),
        'max_abs_error': float(max_error),
        'avg_relative_error': float(avg_rel_error),
        'input_quantization': {'scale': float(input_scale), 'zero_point': int(input_zero_point)},
        'output_quantization': {'scale': float(output_scale), 'zero_point': int(output_zero_point)}
    }
    
    print(f"Quantization verification:")
    print(f"  Avg MSE: {avg_mse:.6f}")
    print(f"  Avg MAE: {avg_mae:.6f}")
    print(f"  Max abs error: {max_error:.6f}")
    print(f"  Avg relative error: {avg_rel_error:.6f}")
    
    return results


def quantize_int8(data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize float32 data to INT8."""
    quantized = np.round(data / scale + zero_point).astype(np.int32)
    return np.clip(quantized, -128, 127).astype(np.int8)


def run_full_export_pipeline(config_path: str = "configs/default.yaml",
                             model_path: str = None) -> Dict[str, Any]:
    """Run complete model export pipeline."""
    config = load_config(config_path)
    
    # Load model
    if model_path:
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        # Build from config (for testing without trained weights)
        model = build_inference_model(config)
        print("WARNING: Using randomly initialized model for export test!")
    
    noise_schedule = NoiseSchedule(config)
    
    # Export paths
    export_config = config.get('export', {})
    saved_model_dir = export_config.get('saved_model_dir', 'artifacts/saved_model')
    float_tflite_path = export_config.get('float_tflite_path', 'artifacts/model_float32.tflite')
    int8_tflite_path = export_config.get('int8_tflite_path', 'artifacts/model_int8.tflite')
    model_header_path = export_config.get('model_header_path', 'esp32/include/model_data.h')
    conversion_report_path = export_config.get('conversion_report_path', 'artifacts/conversion_report.json')
    
    ensure_dir('artifacts')
    ensure_dir('esp32/include')
    
    report = {}
    
    # 1. Export SavedModel
    print("Exporting SavedModel...")
    export_saved_model(model, saved_model_dir)
    report['saved_model'] = saved_model_dir
    
    # 2. Export float32 TFLite
    print("Exporting float32 TFLite...")
    export_tflite_float32(model, float_tflite_path)
    report['float_tflite'] = {
        'path': float_tflite_path,
        'size_bytes': os.path.getsize(float_tflite_path)
    }
    
    # 3. Export INT8 TFLite
    print("Exporting INT8 TFLite...")
    int8_result = export_tflite_int8(model, config, noise_schedule, int8_tflite_path)
    report['int8_tflite'] = int8_result
    
    # 4. Verify quantization
    print("Verifying quantization...")
    verification = verify_quantized_model(float_tflite_path, int8_tflite_path, config, noise_schedule)
    report['verification'] = verification
    
    # 5. Generate C header
    print("Generating C header...")
    create_model_header(int8_tflite_path, model_header_path)
    report['model_header'] = model_header_path
    
    # Save conversion report
    save_json(report, conversion_report_path)
    print(f"Conversion report saved to {conversion_report_path}")
    
    return report


if __name__ == "__main__":
    # Test export pipeline with random model
    config = load_config("configs/default.yaml")
    noise_schedule = NoiseSchedule(config)
    model = build_inference_model(config)
    
    # Test float32 export
    ensure_dir('artifacts')
    export_tflite_float32(model, 'artifacts/test_float32.tflite')
    
    # Test INT8 export
    rep_data = create_representative_dataset(config, noise_schedule, 10)
    export_tflite_int8(model, config, noise_schedule, 'artifacts/test_int8.tflite', rep_data)
    
    # Verify ops
    ops_info = verify_tflite_ops('artifacts/test_int8.tflite')
    print(f"Ops verification: {ops_info}")
    
    print("Export module test passed!")