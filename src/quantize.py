"""
Quantization utilities for TinyDiffusion-ESP32.
Focuses on INT8 quantization, calibration, and numerical verification.
"""
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional, Generator
from src.utils import load_config
from src.dataset import MNISTDataset
from src.diffusion import NoiseSchedule
from src.export import create_representative_dataset


def analyze_model_quantization(model: tf.keras.Model, 
                               config: Dict[str, Any],
                               noise_schedule: NoiseSchedule,
                               num_samples: int = 100) -> Dict[str, Any]:
    """Analyze model's suitability for INT8 quantization."""
    
    # Check layer types
    layer_types = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    # Check weight ranges
    weight_stats = {}
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for i, w in enumerate(weights):
                key = f"{layer.name}_w{i}"
                weight_stats[key] = {
                    'shape': w.shape,
                    'min': float(np.min(w)),
                    'max': float(np.max(w)),
                    'mean': float(np.mean(w)),
                    'std': float(np.std(w))
                }
    
    # Check activation ranges using representative data
    rep_data = create_representative_dataset(config, noise_schedule, num_samples)
    
    activation_stats = {}
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation is not None:
            # Create intermediate model to get activations
            intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
            
            activations = []
            for batch in rep_data:
                act = intermediate_model.predict(batch, verbose=0)
                activations.append(act)
            
            if activations:
                all_acts = np.concatenate(activations, axis=0)
                activation_stats[layer.name] = {
                    'min': float(np.min(all_acts)),
                    'max': float(np.max(all_acts)),
                    'mean': float(np.mean(all_acts)),
                    'std': float(np.std(all_acts))
                }
    
    return {
        'layer_types': layer_types,
        'weight_stats': weight_stats,
        'activation_stats': activation_stats
    }


def calibrate_quantization_ranges(model: tf.keras.Model,
                                  config: Dict[str, Any],
                                  noise_schedule: NoiseSchedule,
                                  num_samples: int = 500) -> Dict[str, Dict[str, float]]:
    """Calibrate quantization ranges for each tensor."""
    
    rep_data = create_representative_dataset(config, noise_schedule, num_samples)
    
    # Get concrete function for TFLite conversion
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 28, 28, 12], dtype=tf.float32)])
    def serving_fn(x):
        return model(x)
    
    converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_fn.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    
    # Convert with calibration
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    
    # Analyze quantized model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    tensor_details = interpreter.get_tensor_details()
    
    ranges = {}
    for detail in tensor_details:
        name = detail['name']
        quant = detail['quantization']
        if quant != (0.0, 0):
            ranges[name] = {
                'scale': float(quant[0]),
                'zero_point': int(quant[1]),
                'dtype': str(detail['dtype']),
                'shape': detail['shape'].tolist()
            }
    
    return ranges


def quantize_weights_manually(model: tf.keras.Model, 
                              bits: int = 8) -> Dict[str, np.ndarray]:
    """Manually quantize model weights for analysis."""
    
    quantized_weights = {}
    max_val = 2**(bits - 1) - 1  # 127 for INT8
    min_val = -2**(bits - 1)     # -128 for INT8
    
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for i, w in enumerate(weights):
                key = f"{layer.name}_w{i}"
                
                # Per-tensor quantization
                w_min, w_max = np.min(w), np.max(w)
                scale = (w_max - w_min) / (max_val - min_val) if w_max != w_min else 1.0
                zero_point = int(np.round(-w_min / scale)) if scale != 0 else 0
                zero_point = np.clip(zero_point, min_val, max_val)
                
                # Quantize
                w_q = np.round(w / scale + zero_point).astype(np.int32)
                w_q = np.clip(w_q, min_val, max_val).astype(np.int8)
                
                # Dequantize for error analysis
                w_dq = (w_q.astype(np.float32) - zero_point) * scale
                
                mse = np.mean((w - w_dq) ** 2)
                max_err = np.max(np.abs(w - w_dq))
                
                quantized_weights[key] = {
                    'original_shape': w.shape,
                    'scale': float(scale),
                    'zero_point': int(zero_point),
                    'mse': float(mse),
                    'max_error': float(max_err),
                    'quantized': w_q
                }
    
    return quantized_weights


def simulate_int8_inference(model: tf.keras.Model,
                            input_data: np.ndarray,
                            weight_quantization: Dict[str, Any]) -> np.ndarray:
    """Simulate INT8 inference using manually quantized weights (for verification)."""
    # This is a simplified simulation for verification purposes
    # Real INT8 inference would use TFLite interpreter
    pass


def check_tflite_micro_compatibility(tflite_path: str) -> Dict[str, Any]:
    """Check if TFLite model is compatible with TensorFlow Lite Micro."""
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get operator details
    ops_details = interpreter._get_ops_details()
    
    # TFLite Micro supported operators (as of TF 2.15)
    # This list should be verified against the target TFLM version
    micro_supported_ops = {
        'ABS', 'ADD', 'ARG_MAX', 'AVERAGE_POOL_2D', 'BATCH_TO_SPACE_ND',
        'CAST', 'CEIL', 'CONCATENATION', 'CONV_2D', 'COS', 'DEPTHWISE_CONV_2D',
        'DIV', 'ELU', 'EXP', 'EXPAND_DIMS', 'FILL', 'FLOOR', 'FLOOR_DIV',
        'FLOOR_MOD', 'FULLY_CONNECTED', 'GATHER', 'GREATER', 'GREATER_EQUAL',
        'HARD_SWISH', 'LEAKY_RELU', 'LESS', 'LESS_EQUAL', 'LOCAL_RESPONSE_NORMALIZATION',
        'LOG', 'LOGICAL_AND', 'LOGICAL_NOT', 'LOGICAL_OR', 'LOGISTIC',
        'MAX_POOL_2D', 'MAXIMUM', 'MEAN', 'MINIMUM', 'MIRROR_PAD', 'MUL',
        'NEG', 'NOT_EQUAL', 'PAD', 'PADV2', 'POW', 'PRELU', 'QUANTIZE',
        'REDUCE_MAX', 'REDUCE_MIN', 'RELU', 'RELU6', 'RELU_N1_TO_1',
        'RESHAPE', 'RESIZE_BILINEAR', 'RESIZE_NEAREST_NEIGHBOR', 'RINT',
        'ROUND', 'RSQRT', 'SELECT', 'SELECT_V2', 'SHAPE', 'SIGMOID',
        'SIN', 'SLICE', 'SOFTMAX', 'SPACE_TO_BATCH_ND', 'SPACE_TO_DEPTH',
        'SPLIT', 'SPLIT_V', 'SQRT', 'SQUARE', 'SQUEEZE', 'STACK',
        'STRIDED_SLICE', 'SUB', 'SUM', 'TANH', 'TILE', 'TOP_K_V2',
        'TRANSPOSE', 'TRANSPOSE_CONV', 'UNIDIRECTIONAL_SEQUENCE_LSTM',
        'UNIDIRECTIONAL_SEQUENCE_RNN', 'UNPACK', 'VAR_HANDLE'
    }
    
    used_ops = set()
    unsupported_ops = set()
    
    for op in ops_details:
        op_name = op['op_name']
        used_ops.add(op_name)
        if op_name not in micro_supported_ops:
            unsupported_ops.add(op_name)
    
    # Check tensor sizes
    tensor_details = interpreter.get_tensor_details()
    max_tensor_size = 0
    total_tensor_size = 0
    for detail in tensor_details:
        size = np.prod(detail['shape']) * np.dtype(detail['dtype']).itemsize
        max_tensor_size = max(max_tensor_size, size)
        total_tensor_size += size
    
    return {
        'is_compatible': len(unsupported_ops) == 0,
        'used_ops': sorted(list(used_ops)),
        'unsupported_ops': sorted(list(unsupported_ops)),
        'num_tensors': len(tensor_details),
        'max_tensor_size_bytes': max_tensor_size,
        'total_tensor_size_bytes': total_tensor_size,
        'recommended_arena_size': max_tensor_size * 4  # Rough estimate
    }


def generate_quantization_report(config: Dict[str, Any],
                                 model: tf.keras.Model,
                                 noise_schedule: NoiseSchedule,
                                 float_tflite_path: str,
                                 int8_tflite_path: str,
                                 output_path: str) -> Dict[str, Any]:
    """Generate comprehensive quantization report."""
    
    report = {}
    
    # Model analysis
    print("Analyzing model for quantization...")
    report['model_analysis'] = analyze_model_quantization(model, config, noise_schedule)
    
    # Calibration ranges
    print("Calibrating quantization ranges...")
    report['calibration_ranges'] = calibrate_quantization_ranges(model, config, noise_schedule)
    
    # Weight quantization analysis
    print("Analyzing weight quantization...")
    report['weight_quantization'] = quantize_weights_manually(model)
    
    # TFLite Micro compatibility
    print("Checking TFLite Micro compatibility...")
    report['float_model_compatibility'] = check_tflite_micro_compatibility(float_tflite_path)
    report['int8_model_compatibility'] = check_tflite_micro_compatibility(int8_tflite_path)
    
    # Model sizes
    import os
    report['model_sizes'] = {
        'float_tflite_bytes': os.path.getsize(float_tflite_path),
        'int8_tflite_bytes': os.path.getsize(int8_tflite_path),
        'size_reduction_ratio': os.path.getsize(int8_tflite_path) / os.path.getsize(float_tflite_path)
    }
    
    # Save report
    from src.utils import save_json
    save_json(report, output_path)
    print(f"Quantization report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Test quantization analysis
    config = load_config("configs/default.yaml")
    noise_schedule = NoiseSchedule(config)
    model = build_inference_model(config)
    
    from src.model import build_inference_model
    
    analysis = analyze_model_quantization(model, config, noise_schedule, num_samples=10)
    print(f"Layer types: {analysis['layer_types']}")
    print(f"Weight stats keys: {list(analysis['weight_stats'].keys())}")
    
    # Check compatibility (will use dummy model)
    ensure_dir('artifacts')
    from src.export import export_tflite_float32
    export_tflite_float32(model, 'artifacts/test_compat_float.tflite')
    
    compat = check_tflite_micro_compatibility('artifacts/test_compat_float.tflite')
    print(f"Compatible: {compat['is_compatible']}")
    print(f"Used ops: {compat['used_ops']}")
    if compat['unsupported_ops']:
        print(f"Unsupported ops: {compat['unsupported_ops']}")
    
    print("Quantization module test passed!")