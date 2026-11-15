#!/usr/bin/env python3
"""
Verify TFLite model (float32 and INT8) correctness and compatibility.
"""
import sys
import os
import argparse
import numpy as np
import tensorflow as tf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (load_config, get_tflite_tensor_details, run_tflite_inference,
                       compare_outputs, verify_tflite_ops, quantize_int8, dequantize_int8)
from src.dataset import MNISTDataset
from src.diffusion import NoiseSchedule
from src.model import build_inference_model


def main():
    parser = argparse.ArgumentParser(description="Verify TFLite models")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file path')
    parser.add_argument('--float-model', type=str, default='artifacts/model_float32.tflite', help='Float32 TFLite model')
    parser.add_argument('--int8-model', type=str, default='artifacts/model_int8.tflite', help='INT8 TFLite model')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--check-ops', action='store_true', help='Check operator compatibility')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("=" * 60)
    print("TinyDiffusion-ESP32 TFLite Verification")
    print("=" * 60)
    
    # Load noise schedule for test input generation
    noise_schedule = NoiseSchedule(config)
    dataset = MNISTDataset(config)
    dataset.load()
    x_test = dataset.preprocess(dataset.x_test)
    y_test = dataset.y_test
    
    # 1. Verify float32 model
    print(f"\n1. Verifying float32 model: {args.float_model}")
    if os.path.exists(args.float_model):
        float_interpreter = tf.lite.Interpreter(model_path=args.float_model)
        float_interpreter.allocate_tensors()
        
        float_details = get_tflite_tensor_details(float_interpreter)
        print(f"   Input: {float_details['input']}")
        print(f"   Output: {float_details['output']}")
        
        if args.check_ops:
            ops_info = verify_tflite_ops(args.float_model)
            print(f"   Ops compatible: {ops_info['is_micro_compatible']}")
            print(f"   Used ops: {ops_info['unique_ops']}")
            if ops_info['unsupported_ops']:
                print(f"   Unsupported ops: {ops_info['unsupported_ops']}")
    else:
        print(f"   Float32 model not found!")
        return
    
    # 2. Verify INT8 model
    print(f"\n2. Verifying INT8 model: {args.int8_model}")
    if os.path.exists(args.int8_model):
        int8_interpreter = tf.lite.Interpreter(model_path=args.int8_model)
        int8_interpreter.allocate_tensors()
        
        int8_details = get_tflite_tensor_details(int8_interpreter)
        print(f"   Input: {int8_details['input']}")
        print(f"   Output: {int8_details['output']}")
        
        # Get quantization parameters
        input_quant = int8_details['input'][0]['quantization']
        output_quant = int8_details['output'][0]['quantization']
        print(f"   Input quantization: scale={input_quant[0]:.6f}, zero_point={input_quant[1]}")
        print(f"   Output quantization: scale={output_quant[0]:.6f}, zero_point={output_quant[1]}")
        
        if args.check_ops:
            ops_info = verify_tflite_ops(args.int8_model)
            print(f"   Ops compatible: {ops_info['is_micro_compatible']}")
            print(f"   Used ops: {ops_info['unique_ops']}")
            if ops_info['unsupported_ops']:
                print(f"   Unsupported ops: {ops_info['unsupported_ops']}")
    else:
        print(f"   INT8 model not found!")
        return
    
    # 3. Compare float32 vs INT8 outputs
    print(f"\n3. Comparing float32 vs INT8 outputs ({args.num_samples} samples)...")
    
    input_scale, input_zero_point = input_quant
    output_scale, output_zero_point = output_quant
    
    errors = []
    
    for i in range(min(args.num_samples, len(x_test))):
        t = np.random.randint(0, noise_schedule.num_timesteps)
        
        img = x_test[i:i+1]
        label = y_test[i:i+1]
        
        noise = np.random.randn(*img.shape).astype(np.float32)
        
        from src.diffusion import forward_diffusion
        x_t = forward_diffusion(img, np.array([t]), noise, noise_schedule.num_timesteps,
                               noise_schedule.schedule_type, noise_schedule.beta_start, noise_schedule.beta_end)
        
        from src.utils import prepare_model_input
        model_input_float = prepare_model_input(
            x_t, label, np.array([t]),
            num_classes=10,
            normalize_timestep=noise_schedule.normalize_timestep,
            num_timesteps=noise_schedule.num_timesteps
        )
        
        # Float inference
        float_output = run_tflite_inference(float_interpreter, model_input_float.astype(np.float32))
        
        # INT8 inference
        model_input_int8 = quantize_int8(model_input_float, input_scale, input_zero_point)
        int8_output = run_tflite_inference(int8_interpreter, model_input_int8)
        
        # Compare
        error = compare_outputs(float_output, int8_output, output_scale, output_zero_point)
        errors.append(error)
    
    avg_mse = np.mean([e['mse'] for e in errors])
    avg_mae = np.mean([e['mae'] for e in errors])
    max_abs_err = np.max([e['max_abs_error'] for e in errors])
    avg_rel_err = np.mean([e['relative_error'] for e in errors])
    
    print(f"   Average MSE: {avg_mse:.6f}")
    print(f"   Average MAE: {avg_mae:.6f}")
    print(f"   Max absolute error: {max_abs_err:.6f}")
    print(f"   Average relative error: {avg_rel_err:.6f}")
    
    # 4. Test INT8 inference with known input
    print(f"\n4. Testing INT8 inference with sample input...")
    test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
    test_input_int8 = quantize_int8(test_input, input_scale, input_zero_point)
    int8_result = run_tflite_inference(int8_interpreter, test_input_int8)
    dequant_result = dequantize_int8(int8_result, output_scale, output_zero_point)
    print(f"   INT8 output shape: {int8_result.shape}")
    print(f"   Dequantized range: [{dequant_result.min():.3f}, {dequant_result.max():.3f}]")
    
    # 5. Test float32 inference with same input
    float_result = run_tflite_inference(float_interpreter, test_input)
    print(f"   Float32 output range: [{float_result.min():.3f}, {float_result.max():.3f}]")
    
    diff = np.abs(float_result - dequant_result)
    print(f"   Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()