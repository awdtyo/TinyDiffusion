"""
Tests for quantization and TFLite conversion.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf
import pytest
from src.model import build_inference_model
from src.diffusion import NoiseSchedule
from src.utils import load_config, verify_tflite_ops, get_tflite_tensor_details


def test_model_export_float32(tmp_path):
    """Test float32 TFLite export."""
    config = load_config("../configs/default.yaml")
    model = build_inference_model(config)
    
    # Export to temp file
    tflite_path = tmp_path / "model_float32.tflite"
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    
    tflite_path.write_bytes(tflite_model)
    
    # Verify it loads
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    details = get_tflite_tensor_details(interpreter)
    
    assert len(details['input']) == 1
    assert len(details['output']) == 1
    assert details['input'][0]['shape'] == [1, 28, 28, 12]
    assert details['output'][0]['shape'] == [1, 28, 28, 1]


def test_model_ops_compatibility():
    """Test model uses TFLite Micro compatible ops."""
    config = load_config("../configs/default.yaml")
    model = build_inference_model(config)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    
    # Check ops
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
        f.write(tflite_model)
        temp_path = f.name
    
    try:
        ops_info = verify_tflite_ops(temp_path)
        assert ops_info['is_micro_compatible'], f"Unsupported ops: {ops_info['unsupported_ops']}"
    finally:
        os.unlink(temp_path)


def test_input_output_quantization_params():
    """Test INT8 quantization parameters exist."""
    config = load_config("../configs/default.yaml")
    model = build_inference_model(config)
    
    # This test requires a quantized model, skip if not available
    quantized_path = "../artifacts/model_int8.tflite"
    if not os.path.exists(quantized_path):
        pytest.skip("Quantized model not available")
    
    interpreter = tf.lite.Interpreter(model_path=quantized_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Check quantization params exist
    assert input_details['quantization'] != (0.0, 0)
    assert output_details['quantization'] != (0.0, 0)
    
    input_scale, input_zp = input_details['quantization']
    output_scale, output_zp = output_details['quantization']
    
    assert input_scale > 0
    assert output_scale > 0


def test_quantized_model_inference():
    """Test quantized model runs inference."""
    quantized_path = "../artifacts/model_int8.tflite"
    if not os.path.exists(quantized_path):
        pytest.skip("Quantized model not available")
    
    interpreter = tf.lite.Interpreter(model_path=quantized_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Create test input
    test_input = np.random.randint(-128, 127, size=input_details['shape'], dtype=np.int8)
    
    interpreter.set_tensor(input_details['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])
    
    assert output.shape == tuple(output_details['shape'])
    assert output.dtype == np.int8


def test_model_size_reasonable():
    """Test model size is reasonable for ESP32."""
    quantized_path = "../artifacts/model_int8.tflite"
    if not os.path.exists(quantized_path):
        pytest.skip("Quantized model not available")
    
    size = os.path.getsize(quantized_path)
    
    # Should be under 50KB for ESP32
    assert size < 50000, f"Model too large: {size} bytes"
    
    # Should be at least a few KB
    assert size > 5000, f"Model too small: {size} bytes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])