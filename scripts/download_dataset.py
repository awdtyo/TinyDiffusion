#!/usr/bin/env python3
"""
Download and prepare MNIST dataset for TinyDiffusion-ESP32.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import MNISTDataset
from src.utils import load_config, set_global_seed


def main():
    config = load_config("configs/default.yaml")
    set_global_seed(config.get('training', {}).get('seed', 42))
    
    dataset = MNISTDataset(config)
    x_train, y_train, x_test, y_test = dataset.load()
    
    # Preprocess and show stats
    x_train_proc = dataset.preprocess(x_train)
    x_test_proc = dataset.preprocess(x_test)
    
    print("\nDataset Statistics:")
    print(f"  Train images: {x_train_proc.shape}, range: [{x_train_proc.min():.3f}, {x_train_proc.max():.3f}]")
    print(f"  Test images: {x_test_proc.shape}, range: [{x_test_proc.min():.3f}, {x_test_proc.max():.3f}]")
    print(f"  Train labels: {y_train.shape}, classes: {sorted(set(y_train))}")
    print(f"  Test labels: {y_test.shape}, classes: {sorted(set(y_test))}")
    
    # Visualize samples
    dataset.visualize_samples(num_samples=16, save_path="artifacts/mnist_samples.png")
    
    print("\nDataset download and preparation complete!")


if __name__ == "__main__":
    main()