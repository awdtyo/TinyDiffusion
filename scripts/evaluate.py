#!/usr/bin/env python3
"""
Evaluate the trained TinyDiffusion model.
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluate import ModelEvaluator, plot_training_history
from src.model import build_inference_model
from src.diffusion import NoiseSchedule
from src.utils import load_config, set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyDiffusion model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default='artifacts/checkpoints/inference_model.keras', help='Model path')
    parser.add_argument('--output', type=str, default='artifacts/evaluation', help='Output directory')
    parser.add_argument('--steps', type=int, default=None, help='Sampling steps')
    parser.add_argument('--class-grid', action='store_true', help='Generate class grid')
    parser.add_argument('--step-comparison', action='store_true', help='Compare sampling steps')
    parser.add_argument('--latency', action='store_true', help='Measure inference latency')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_global_seed(config.get('training', {}).get('seed', 42))
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = build_inference_model(config)
    model.load_weights(args.model)
    
    noise_schedule = NoiseSchedule(config)
    evaluator = ModelEvaluator(config, model, noise_schedule)
    
    # Run evaluations
    if args.class_grid:
        print("Generating class grid...")
        classes = config.get('evaluation', {}).get('classes', list(range(10)))
        samples_per_class = config.get('evaluation', {}).get('samples_per_class', 5)
        steps = args.steps or config.get('sampling', {}).get('default_steps', 8)
        grid = evaluator.generate_class_grid(classes, samples_per_class, steps)
        evaluator.visualize_samples(grid, title=f"Generated Digits ({steps} steps)",
                                   save_path=os.path.join(args.output, 'class_grid.png'))
    
    if args.step_comparison:
        print("Comparing sampling steps...")
        steps_list = config.get('evaluation', {}).get('compare_steps', [8, 16, 32])
        for class_id in range(10):
            evaluator.visualize_step_comparison(class_id, steps_list,
                                               save_path=os.path.join(args.output, f'step_comp_{class_id}.png'))
    
    if args.latency:
        print("Measuring inference latency...")
        for steps in [8, 16, 32]:
            latency = evaluator.measure_inference_latency(num_runs=20, num_steps=steps)
            print(f"  {steps} steps: {latency['mean_ms']:.1f} ms total, {latency['per_step_ms']:.1f} ms/step")
    
    # Full evaluation if no specific flags
    if not any([args.class_grid, args.step_comparison, args.latency]):
        print("Running full evaluation...")
        results = evaluator.run_full_evaluation(args.output)
        print(f"Results saved to {args.output}")
    
    # Plot training history if available
    history_path = 'artifacts/logs/training_history.json'
    if os.path.exists(history_path):
        import json
        with open(history_path, 'r') as f:
            history = json.load(f)
        plot_training_history(history, save_path=os.path.join(args.output, 'training_history.png'))
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()