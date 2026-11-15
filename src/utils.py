"""
Utility functions for TinyDiffusion-ESP32.
"""
import os
import yaml
import json
import random
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, List


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # TensorFlow deterministic operations (may impact performance)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary as JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def count_parameters(model: tf.keras.Model) -> int:
    """Count total trainable parameters in a Keras model."""
    return model.count_params()


def get_model_size_mb(model_path: str) -> float:
    """Get model file size in MB."""
    return os.path.getsize(model_path) / (1024 * 1024)


def format_time(seconds: float) -> str:
    """Format time in human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        return f"{seconds / 60:.1f} min"


def print_model_summary(model: tf.keras.Model) -> None:
    """Print detailed model summary with parameter counts per layer."""
    print("=" * 80)
    print(f"Model: {model.name}")
    print("=" * 80)
    model.summary(expand_nested=True)
    print("=" * 80)
    total_params = count_parameters(model)
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print("=" * 80)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def normalize_to_neg_one_one(images: np.ndarray) -> np.ndarray:
    """Normalize images from [0, 1] or [0, 255] to [-1, 1]."""
    images = images.astype(np.float32)
    if images.max() > 1.0:
        images = images / 255.0
    return images * 2.0 - 1.0


def denormalize_to_zero_one(images: np.ndarray) -> np.ndarray:
    """Denormalize images from [-1, 1] to [0, 1]."""
    return (images + 1.0) / 2.0


def denormalize_to_uint8(images: np.ndarray) -> np.ndarray:
    """Denormalize images from [-1, 1] to [0, 255] uint8."""
    images = denormalize_to_zero_one(images)
    return np.clip(images * 255, 0, 255).astype(np.uint8)


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    return tf.keras.utils.to_categorical(labels, num_classes)


def broadcast_conditioning(class_onehot: np.ndarray, timestep: np.ndarray, 
                           height: int, width: int) -> np.ndarray:
    """
    Broadcast class conditioning and timestep to spatial maps.
    
    Args:
        class_onehot: [batch, num_classes] one-hot encoded labels
        timestep: [batch, 1] normalized timestep values
        height: spatial height
        width: spatial width
        
    Returns:
        conditioning: [batch, height, width, num_classes + 1]
    """
    batch_size = class_onehot.shape[0]
    num_classes = class_onehot.shape[1]
    
    # Broadcast class maps: [batch, num_classes] -> [batch, height, width, num_classes]
    class_maps = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
    for c in range(num_classes):
        class_maps[:, :, :, c] = class_onehot[:, c:c+1]
    
    # Broadcast timestep: [batch, 1] -> [batch, height, width, 1]
    timestep_maps = np.zeros((batch_size, height, width, 1), dtype=np.float32)
    timestep_maps[:, :, :, 0] = timestep[:, 0]
    
    # Concatenate
    conditioning = np.concatenate([class_maps, timestep_maps], axis=-1)
    return conditioning


def prepare_model_input(noisy_images: np.ndarray, class_labels: np.ndarray, 
                        timesteps: np.ndarray, num_classes: int = 10,
                        normalize_timestep: bool = True,
                        num_timesteps: int = 100) -> np.ndarray:
    """
    Prepare full model input by concatenating noisy image with conditioning.
    
    Args:
        noisy_images: [batch, height, width, 1] noisy images in [-1, 1]
        class_labels: [batch] integer class labels 0-9
        timesteps: [batch] integer timesteps 0 to num_timesteps-1
        num_classes: number of classes
        normalize_timestep: whether to normalize timestep to [0, 1]
        num_timesteps: total number of timesteps
        
    Returns:
        model_input: [batch, height, width, 12] concatenated input
    """
    batch_size = noisy_images.shape[0]
    height, width = noisy_images.shape[1], noisy_images.shape[2]
    
    # One-hot encode class labels
    class_onehot = one_hot_encode(class_labels, num_classes)
    
    # Normalize timestep
    timestep_normalized = timesteps.astype(np.float32).reshape(-1, 1)
    if normalize_timestep:
        timestep_normalized = timestep_normalized / num_timesteps
    
    # Broadcast conditioning
    conditioning = broadcast_conditioning(class_onehot, timestep_normalized, height, width)
    
    # Concatenate noisy image + conditioning
    model_input = np.concatenate([noisy_images, conditioning], axis=-1)
    return model_input


def quantize_int8(data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize float32 data to INT8 using scale and zero point."""
    quantized = np.round(data / scale + zero_point).astype(np.int32)
    return np.clip(quantized, -128, 127).astype(np.int8)


def dequantize_int8(data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize INT8 data to float32 using scale and zero point."""
    return (data.astype(np.float32) - zero_point) * scale


def get_tflite_tensor_details(interpreter: tf.lite.Interpreter) -> Dict[str, Any]:
    """Get input/output tensor details from TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return {
        'input': [{
            'name': d['name'],
            'index': d['index'],
            'shape': d['shape'].tolist(),
            'dtype': str(d['dtype']),
            'quantization': d['quantization'],
            'quantization_parameters': d.get('quantization_parameters', {})
        } for d in input_details],
        'output': [{
            'name': d['name'],
            'index': d['index'],
            'shape': d['shape'].tolist(),
            'dtype': str(d['dtype']),
            'quantization': d['quantization'],
            'quantization_parameters': d.get('quantization_parameters', {})
        } for d in output_details]
    }


def run_tflite_inference(interpreter: tf.lite.Interpreter, input_data: np.ndarray) -> np.ndarray:
    """Run inference on TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


def compare_outputs(float_output: np.ndarray, quant_output: np.ndarray, 
                    quant_scale: float, quant_zero_point: int) -> Dict[str, float]:
    """Compare float and quantized outputs."""
    quant_dequant = dequantize_int8(quant_output, quant_scale, quant_zero_point)
    
    mse = np.mean((float_output - quant_dequant) ** 2)
    mae = np.mean(np.abs(float_output - quant_dequant))
    max_abs_error = np.max(np.abs(float_output - quant_dequant))
    relative_error = np.mean(np.abs(float_output - quant_dequant) / (np.abs(float_output) + 1e-8))
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'max_abs_error': float(max_abs_error),
        'relative_error': float(relative_error)
    }


def create_model_header(tflite_path: str, header_path: str, model_name: str = "g_model") -> None:
    """Convert TFLite model to C header file for ESP32 embedding."""
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    model_size = len(model_data)
    
    header_content = f"""// TinyDiffusion-ESP32 Model Data
// Auto-generated from {tflite_path}
// Model size: {model_size} bytes
// DO NOT EDIT MANUALLY

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <cstdint>

extern const unsigned char {model_name}[] = {{
"""
    
    # Write bytes as hex array
    for i, byte in enumerate(model_data):
        if i % 12 == 0:
            header_content += "\n  "
        header_content += f"0x{byte:02x}, "
    
    header_content += f"""
}};

extern const unsigned int {model_name}_len = {model_size};

#endif // MODEL_DATA_H
"""
    
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    print(f"Model header generated: {header_path} ({model_size} bytes)")


def print_memory_info() -> None:
    """Print current memory usage (host-side)."""
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / 1024 / 1024:.1f} MB")
    print(f"VMS: {mem_info.vms / 1024 / 1024:.1f} MB")


class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.perf_counter()
        print(f"{self.name}: {format_time(self.end_time - self.start_time)}")
    
    @property
    def elapsed(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


def verify_tflite_ops(tflite_path: str) -> Dict[str, Any]:
    """Verify TFLite model operators and compatibility with TFLite Micro."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    ops = []
    for op in interpreter._get_ops_details():
        ops.append({
            'op_name': op['op_name'],
            'inputs': op['inputs'],
            'outputs': op['outputs']
        })
    
    # Check for unsupported ops in TFLite Micro
    # TFLite Micro supported ops (subset): CONV_2D, DEPTHWISE_CONV_2D, ADD, MUL, 
    # RESHAPE, CONCATENATION, PAD, MEAN, MAX_POOL_2D, AVERAGE_POOL_2D, 
    # SOFTMAX, LOGISTIC, TANH, RELU, RELU6, PRELU, LEAKY_RELU, ELU,
    # FULLY_CONNECTED, TRANSPOSE_CONV, QUANTIZE, DEQUANTIZE, etc.
    
    supported_ops = {
        'CONV_2D', 'DEPTHWISE_CONV_2D', 'ADD', 'MUL', 'RESHAPE', 
        'CONCATENATION', 'PAD', 'MEAN', 'MAX_POOL_2D', 'AVERAGE_POOL_2D',
        'SOFTMAX', 'LOGISTIC', 'TANH', 'RELU', 'RELU6', 'PRELU', 
        'LEAKY_RELU', 'ELU', 'FULLY_CONNECTED', 'TRANSPOSE_CONV',
        'QUANTIZE', 'DEQUANTIZE', 'STRIDED_SLICE', 'TRANSPOSE',
        'SQUEEZE', 'EXPAND_DIMS', 'CAST'
    }
    
    unsupported = []
    for op in ops:
        if op['op_name'] not in supported_ops:
            unsupported.append(op['op_name'])
    
    return {
        'total_ops': len(ops),
        'unique_ops': list(set(op['op_name'] for op in ops)),
        'unsupported_ops': list(set(unsupported)),
        'is_micro_compatible': len(unsupported) == 0
    }