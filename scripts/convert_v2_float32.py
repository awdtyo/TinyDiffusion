#!/usr/bin/env python3
"""
Convert Tiny U-Net v2 to FLOAT32 TFLite and inspect.
"""

import sys
import os
import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.v2.model_v2 import build_tiny_unet_v2
from src.utils import load_config, ensure_dir


def main():
    config = load_config('configs/tiny_unet_v2.yaml')
    
    print("Building Tiny U-Net v2...")
    model = build_tiny_unet_v2(config)
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_params():,}")
    
    # Build model
    dummy = np.random.randn(1, 28, 28, 12).astype(np.float32)
    _ = model(dummy, training=False)
    
    # Convert to FLOAT32 TFLite
    print("\nConverting to FLOAT32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = []
    
    tflite_model = converter.convert()
    
    output_dir = "artifacts/tflite_tiny_unet_v2"
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "model_float32.tflite")
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved: {output_path} ({len(tflite_model)} bytes, {len(tflite_model)/1024:.1f} KB)")
    
    # Inspect
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    print("\n=== TFLite Model Inspection ===")
    print(f"Model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)")
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"Input: {input_details['name']}, shape={input_details['shape']}, dtype={input_details['dtype']}")
    print(f"Output: {output_details['name']}, shape={output_details['shape']}, dtype={output_details['dtype']}")
    
    # Operator inventory
    ops = []
    for op in interpreter._get_ops_details():
        ops.append(op['op_name'])
    
    unique_ops = sorted(set(ops))
    print(f"\nOperators ({len(ops)} total, {len(unique_ops)} unique):")
    for op in unique_ops:
        count = ops.count(op)
        print(f"  {op}: {count}")
    
    # Check for DELEGATE
    delegate_ops = [op for op in ops if 'DELEGATE' in op]
    if delegate_ops:
        print(f"\nWARNING: {len(delegate_ops)} DELEGATE ops found (XNNPACK)")
    else:
        print("\nNo DELEGATE ops found.")
    
    # Numerical parity test
    print("\n=== Numerical Parity Test ===")
    keras_model = model
    tf_interpreter = tf.lite.Interpreter(model_path=output_path)
    tf_interpreter.allocate_tensors()
    
    input_details = tf_interpreter.get_input_details()[0]
    output_details = tf_interpreter.get_output_details()[0]
    
    def run_tflite(input_data):
        tf_interpreter.set_tensor(input_details['index'], input_data.astype(np.float32))
        tf_interpreter.invoke()
        return tf_interpreter.get_tensor(output_details['index'])
    
    np.random.seed(42)
    max_errors = []
    for i in range(10):
        test_input = np.random.randn(1, 28, 28, 12).astype(np.float32)
        keras_out = keras_model.predict(test_input, verbose=0)
        tflite_out = run_tflite(test_input)
        
        max_abs = np.max(np.abs(keras_out - tflite_out))
        mae = np.mean(np.abs(keras_out - tflite_out))
        mse = np.mean((keras_out - tflite_out) ** 2)
        
        max_errors.append(max_abs)
        print(f"  Run {i}: max_abs={max_abs:.2e}, mae={mae:.2e}, mse={mse:.2e}")
    
    print(f"\nMax absolute error: {np.max(max_errors):.2e}")
    print(f"Avg max absolute error: {np.mean(max_errors):.2e}")


if __name__ == "__main__":
    import numpy as np
    main()