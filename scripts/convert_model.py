#!/usr/bin/env python3
"""
Convert trained model to TFLite (float32 and INT8) for ESP32 deployment.
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.export import run_full_export_pipeline
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Convert model to TFLite for ESP32")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default='artifacts/checkpoints/inference_model.keras', help='Model path')
    parser.add_argument('--float-output', type=str, default='artifacts/model_float32.tflite', help='Float32 TFLite output')
    parser.add_argument('--int8-output', type=str, default='artifacts/model_int8.tflite', help='INT8 TFLite output')
    parser.add_argument('--header-output', type=str, default='esp32/include/model_data.h', help='C header output')
    parser.add_argument('--source-output', type=str, default='esp32/src/model_data.cpp', help='C++ model source output')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Override export paths
    config['export']['float_tflite_path'] = args.float_output
    config['export']['int8_tflite_path'] = args.int8_output
    config['export']['model_header_path'] = args.header_output
    config['export']['model_source_path'] = args.source_output
    
    print("=" * 60)
    print("TinyDiffusion-ESP32 Model Conversion")
    print("=" * 60)
    print(f"Input model: {args.model}")
    print(f"Float32 output: {args.float_output}")
    print(f"INT8 output: {args.int8_output}")
    print(f"Header output: {args.header_output}")
    print(f"Source output: {args.source_output}")
    print("=" * 60)
    
    # Run export pipeline
    report = run_full_export_pipeline(args.config, args.model)
    
    print("\nConversion complete!")
    print(f"Float32 model size: {report['float_tflite']['size_bytes']} bytes")
    print(f"INT8 model size: {report['int8_tflite']['size_bytes']} bytes")
    print(f"Size reduction: {report['int8_tflite']['size_bytes'] / report['float_tflite']['size_bytes'] * 100:.1f}%")
    print(f"Verification MSE: {report['verification']['avg_mse']:.6f}")
    print(f"Verification MAE: {report['verification']['avg_mae']:.6f}")


if __name__ == "__main__":
    main()