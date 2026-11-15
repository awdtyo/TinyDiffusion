#!/usr/bin/env python3
"""
Generate C header file from TFLite model for ESP32 embedding.
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import create_model_header, create_model_source, load_config


def main():
    parser = argparse.ArgumentParser(description="Generate C header and source from TFLite model")
    parser.add_argument('--model', type=str, default='artifacts/model_int8.tflite', help='TFLite model path')
    parser.add_argument('--output', type=str, default='esp32/include/model_data.h', help='Output header path')
    parser.add_argument('--source', type=str, default='esp32/src/model_data.cpp', help='Output source path')
    parser.add_argument('--name', type=str, default='g_model', help='Model variable name')
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    print(f"Generating header from {args.model} -> {args.output}")
    create_model_header(args.model, args.output, args.name)
    print(f"Generating source from {args.model} -> {args.source}")
    create_model_source(args.model, args.source, args.name,
                        header_name=os.path.basename(args.output))
    
    # Verify generated files
    with open(args.output, 'r') as f:
        header_content = f.read()
    with open(args.source, 'r') as f:
        source_content = f.read()
    
    import re
    match = re.search(r'static_assert\(sizeof\(\w+\) == (\d+),', source_content)
    if not match:
        print("Error: generated source is missing its size static_assert")
        sys.exit(1)
    declared_size = int(match.group(1))
    actual_size = os.path.getsize(args.model)
    if declared_size != actual_size:
        print(f"Error: generated source declares {declared_size} bytes, "
              f"artifact is {actual_size} bytes")
        sys.exit(1)
    if f'{args.name}_len = sizeof({args.name});' not in source_content:
        print("Error: generated source does not derive length with sizeof()")
        sys.exit(1)
    if 'alignas(16)' not in header_content or 'alignas(16)' not in source_content:
        print("Error: generated files are missing 16-byte model alignment")
        sys.exit(1)
    print(f"Verified embedded model size: {declared_size} bytes (matches artifact)")
    
    print("Header/source generation complete!")


if __name__ == "__main__":
    main()