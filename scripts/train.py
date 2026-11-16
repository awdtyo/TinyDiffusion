#!/usr/bin/env python3
"""
Train the TinyDiffusion model for ESP32 deployment.
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.train import train_diffusion_model, DiffusionTrainer
from src.utils import load_config, set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Train TinyDiffusion model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_global_seed(config.get('training', {}).get('seed', 42))
    
    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    print("=" * 60)
    print("TinyDiffusion-ESP32 Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Prediction type: {config.get('diffusion', {}).get('prediction_type', 'epsilon')}")
    print(f"Resume from: {args.resume}")
    print("=" * 60)
    
    # Train
    model, history = train_diffusion_model(args.config, resume_from=args.resume)
    
    print("\nTraining complete!")
    print(f"Final train loss: {history['loss'][-1]:.6f}")
    if 'val_loss' in history:
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()