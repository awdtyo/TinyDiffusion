#!/usr/bin/env python3
"""
Generate C header file from TFLite model for ESP32 embedding.
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import create_model_header, load_config


def main():
    parser = argparse.ArgumentParser(description="Generate C header from TFLite model")
    parser.add_argument('--model', type=str, default='artifacts/model_int8.tflite', help='TFLite model path')
    parser.add_argument('--output', type=str, default='esp32/include/model_data.h', help='Output header path')
    parser.add_argument('--name', type=str, default='g_model', help='Model variable name')
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    print(f"Generating header from {args.model} -> {args.output}")
    create_model_header(args.model, args.output, args.name)
    
    # Verify header
    with open(args.output, 'r') as f:
        content = f.read()
    
    # Check size
    import re
    match = re.search(r'const unsigned int \w+_len = (\d+);', content)
    if match:
        size = int(match.group(1))
        print(f"Header declares model size: {size} bytes")
    
    print("Header generation complete!")


if __name__ == "__main__":
    main()