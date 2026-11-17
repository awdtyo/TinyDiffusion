#!/usr/bin/env python3
"""
Convert Tiny U-Net v2.1 configurations to FLOAT32 TFLite and test memory.
"""

import sys
import os
import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.v2_1.model_v2_1 import build_tiny_unet_v2_1
from src.utils import load_config, ensure_dir


def convert_and_test(channels, config):
    """Convert a channel configuration to TFLite and test."""
    print(f"\n{'='*60}")
    print(f"Testing channels: {channels}")
    print(f"{'='*60}")
    
    model = build_tiny_unet_v2_1(config)
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
    
    output_dir = f"artifacts/tflite_tiny_unet_v2_1_{channels[0]}{channels[1]}{channels[2]}{channels[3]}"
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "model_float32.tflite")
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved: {output_path} ({len(tflite_model)} bytes, {len(tflite_model)/1024:.1f} KB)")
    
    # Inspect
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    print(f"\nModel size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)")
    
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
    
    return {
        'channels': channels,
        'params': model.count_params(),
        'model_size': len(tflite_model),
        'model_path': output_path,
        'max_error': np.max(max_errors),
        'avg_error': np.mean(max_errors),
        'operators': unique_ops
    }


def main():
    import numpy as np
    
    channel_configs = [
        [4, 6, 8, 4],  # Smallest
        [4, 8, 8, 4],  # Medium
        [6, 8, 8, 6],  # Largest
    ]
    
    results = []
    
    for channels in channel_configs:
        # Create config for this channel config
        config = {
            'model': {
                'input_height': 28,
                'input_width': 28,
                'total_input_channels': 12,
                'num_classes': 10,
                'architecture': 'tiny_unet_v2_1',
                'tiny_unet_v2_1': {
                    'channels': channels,
                    'initial_conv_kernel': 3,
                    'encoder': [
                        {'resolution': [28, 28], 'channels': channels[0], 'num_blocks': 1},
                        {'resolution': [14, 14], 'channels': channels[1], 'num_blocks': 1}
                    ],
                    'bottleneck': {
                        'resolution': [7, 7], 'channels': channels[2], 'num_blocks': 1
                    },
                    'decoder': [
                        {'resolution': [14, 14], 'channels': channels[1], 'num_blocks': 1},
                        {'resolution': [28, 28], 'channels': channels[3], 'num_blocks': 1}
                    ],
                    'output_channels': 1,
                    'residual_block': {
                        'kernel_size': 3,
                        'activation': 'relu',
                        'use_bias': True,
                        'padding': 'same'
                    },
                    'downsample': {
                        'type': 'strided_conv',
                        'kernel_size': 3,
                        'stride': 2
                    },
                    'upsample': {
                        'type': 'bilinear_conv',
                        'kernel_size': 3
                    }
                }
            }
        }
        
        result = convert_and_test(channels, config)
        results.append(result)
    
    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON")
    print("=" * 100)
    print(f"{'Channels':<15} {'Params':<10} {'Model Size':<12} {'Max Error':<12} {'Avg Error':<12} {'Operators':<30}")
    print("-" * 100)
    
    for r in results:
        ch_str = str(r['channels'])
        ops_str = ', '.join(r['operators'][:5])
        if len(r['operators']) > 5:
            ops_str += f" (+{len(r['operators'])-5} more)"
        print(f"{ch_str:<15} {r['params']:<10,} {r['model_size']/1024:>6.1f} KB   {r['max_error']:<12.2e} {r['avg_error']:<12.2e} {ops_str:<30}")
    
    # Save results
    import json
    with open('artifacts/tflite_tiny_unet_v2_1/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to artifacts/tflite_tiny_unet_v2_1/comparison_results.json")


if __name__ == "__main__":
    import numpy as np
    main()