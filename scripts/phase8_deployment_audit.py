#!/usr/bin/env python3
"""
Phase 8: ESP32 TFLite Micro Compatibility Audit & Memory Analysis.
Analyzes the INT8 and FLOAT32 TFLite models for ESP32 deployment feasibility.
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, ensure_dir, save_json


def check_tflite_micro_operators():
    """Define TFLite Micro operator support based on ESP-IDF TFLM component."""
    
    # Based on ESP-IDF 5.x / TFLite Micro included in ESP-IDF
    # This list should be verified against the actual TFLM version used
    micro_supported_ops = {
        # Core arithmetic
        'ADD': {'int8': True, 'float32': True, 'esp_nn': True},
        'MUL': {'int8': True, 'float32': True, 'esp_nn': True},
        'SUB': {'int8': True, 'float32': True, 'esp_nn': True},
        'DIV': {'int8': False, 'float32': True, 'esp_nn': False},
        
        # Convolution
        'CONV_2D': {'int8': True, 'float32': True, 'esp_nn': True},
        'DEPTHWISE_CONV_2D': {'int8': True, 'float32': True, 'esp_nn': True},
        'TRANSPOSE_CONV': {'int8': False, 'float32': True, 'esp_nn': False},  # Reference kernel only
        
        # Fully connected
        'FULLY_CONNECTED': {'int8': True, 'float32': True, 'esp_nn': True},
        
        # Activation
        'RELU': {'int8': True, 'float32': True, 'esp_nn': True},
        'RELU6': {'int8': True, 'float32': True, 'esp_nn': True},
        'RELU_N1_TO_1': {'int8': True, 'float32': True, 'esp_nn': True},
        'TANH': {'int8': False, 'float32': True, 'esp_nn': False},
        'SIGMOID': {'int8': False, 'float32': True, 'esp_nn': False},
        'LOGISTIC': {'int8': False, 'float32': True, 'esp_nn': False},
        'HARD_SWISH': {'int8': True, 'float32': True, 'esp_nn': True},
        'LEAKY_RELU': {'int8': True, 'float32': True, 'esp_nn': True},
        'PRELU': {'int8': True, 'float32': True, 'esp_nn': False},
        'ELU': {'int8': False, 'float32': True, 'esp_nn': False},
        
        # Pooling
        'MAX_POOL_2D': {'int8': True, 'float32': True, 'esp_nn': True},
        'AVERAGE_POOL_2D': {'int8': True, 'float32': True, 'esp_nn': True},
        
        # Normalization
        'LOCAL_RESPONSE_NORMALIZATION': {'int8': False, 'float32': True, 'esp_nn': False},
        'L2_NORMALIZATION': {'int8': False, 'float32': True, 'esp_nn': False},
        
        # Tensor manipulation
        'CONCATENATION': {'int8': True, 'float32': True, 'esp_nn': False},
        'SPLIT': {'int8': True, 'float32': True, 'esp_nn': False},
        'SPLIT_V': {'int8': True, 'float32': True, 'esp_nn': False},
        'RESHAPE': {'int8': True, 'float32': True, 'esp_nn': False},
        'EXPAND_DIMS': {'int8': True, 'float32': True, 'esp_nn': False},
        'SQUEEZE': {'int8': True, 'float32': True, 'esp_nn': False},
        'STRIDED_SLICE': {'int8': True, 'float32': True, 'esp_nn': False},
        'PAD': {'int8': True, 'float32': True, 'esp_nn': False},
        'PADV2': {'int8': True, 'float32': True, 'esp_nn': False},
        'MIRROR_PAD': {'int8': False, 'float32': True, 'esp_nn': False},
        'TRANSPOSE': {'int8': True, 'float32': True, 'esp_nn': False},
        'SPACE_TO_DEPTH': {'int8': True, 'float32': True, 'esp_nn': False},
        'DEPTH_TO_SPACE': {'int8': True, 'float32': True, 'esp_nn': False},
        'BATCH_TO_SPACE_ND': {'int8': False, 'float32': True, 'esp_nn': False},
        'SPACE_TO_BATCH_ND': {'int8': False, 'float32': True, 'esp_nn': False},
        
        # Element-wise
        'ABS': {'int8': True, 'float32': True, 'esp_nn': False},
        'NEG': {'int8': True, 'float32': True, 'esp_nn': False},
        'SQUARE': {'int8': False, 'float32': True, 'esp_nn': False},
        'SQRT': {'int8': False, 'float32': True, 'esp_nn': False},
        'RSQRT': {'int8': False, 'float32': True, 'esp_nn': False},
        'POW': {'int8': False, 'float32': True, 'esp_nn': False},
        
        # Comparison
        'GREATER': {'int8': True, 'float32': True, 'esp_nn': False},
        'GREATER_EQUAL': {'int8': True, 'float32': True, 'esp_nn': False},
        'LESS': {'int8': True, 'float32': True, 'esp_nn': False},
        'LESS_EQUAL': {'int8': True, 'float32': True, 'esp_nn': False},
        'EQUAL': {'int8': True, 'float32': True, 'esp_nn': False},
        'NOT_EQUAL': {'int8': True, 'float32': True, 'esp_nn': False},
        'LOGICAL_AND': {'int8': True, 'float32': True, 'esp_nn': False},
        'LOGICAL_OR': {'int8': True, 'float32': True, 'esp_nn': False},
        'LOGICAL_NOT': {'int8': True, 'float32': True, 'esp_nn': False},
        'SELECT': {'int8': True, 'float32': True, 'esp_nn': False},
        'SELECT_V2': {'int8': True, 'float32': True, 'esp_nn': False},
        'MAXIMUM': {'int8': True, 'float32': True, 'esp_nn': False},
        'MINIMUM': {'int8': True, 'float32': True, 'esp_nn': False},
        
        # Reduction
        'MEAN': {'int8': True, 'float32': True, 'esp_nn': False},
        'SUM': {'int8': True, 'float32': True, 'esp_nn': False},
        'REDUCE_MAX': {'int8': True, 'float32': True, 'esp_nn': False},
        'REDUCE_MIN': {'int8': True, 'float32': True, 'esp_nn': False},
        'REDUCE_PROD': {'int8': False, 'float32': True, 'esp_nn': False},
        'ARG_MAX': {'int8': True, 'float32': True, 'esp_nn': False},
        'ARG_MIN': {'int8': True, 'float32': True, 'esp_nn': False},
        
        # Quantization
        'QUANTIZE': {'int8': True, 'float32': False, 'esp_nn': False},
        'DEQUANTIZE': {'int8': True, 'float32': False, 'esp_nn': False},
        'REQUANTIZE': {'int8': True, 'float32': False, 'esp_nn': False},
        
        # Other
        'SHAPE': {'int8': True, 'float32': True, 'esp_nn': False},  # Returns int32
        'RANK': {'int8': True, 'float32': True, 'esp_nn': False},
        'SIZE': {'int8': True, 'float32': True, 'esp_nn': False},
        'PACK': {'int8': True, 'float32': True, 'esp_nn': False},
        'UNPACK': {'int8': True, 'float32': True, 'esp_nn': False},
        'STACK': {'int8': True, 'float32': True, 'esp_nn': False},
        'GATHER': {'int8': True, 'float32': True, 'esp_nn': False},
        'CAST': {'int8': True, 'float32': True, 'esp_nn': False},
        'FILL': {'int8': True, 'float32': True, 'esp_nn': False},
        'TILE': {'int8': True, 'float32': True, 'esp_nn': False},
        'SIN': {'int8': False, 'float32': True, 'esp_nn': False},  # Reference kernel
        'COS': {'int8': False, 'float32': True, 'esp_nn': False},  # Reference kernel
        'EXP': {'int8': False, 'float32': True, 'esp_nn': False},
        'LOG': {'int8': False, 'float32': True, 'esp_nn': False},
        'FLOOR': {'int8': True, 'float32': True, 'esp_nn': False},
        'CEIL': {'int8': True, 'float32': True, 'esp_nn': False},
        'ROUND': {'int8': True, 'float32': True, 'esp_nn': False},
        'RINT': {'int8': False, 'float32': True, 'esp_nn': False},
        'DELEGATE': {'int8': False, 'float32': False, 'esp_nn': False},  # XNNPACK delegate - desktop only
    }
    
    return micro_supported_ops


def analyze_model_operators(tflite_path, model_name, micro_ops):
    """Analyze operators in a TFLite model for ESP32 compatibility."""
    import tensorflow as tf
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    ops_details = interpreter._get_ops_details()
    tensor_details = interpreter.get_tensor_details()
    
    used_ops = []
    unsupported_ops = []
    int8_unsupported = []
    
    for op in ops_details:
        op_name = op['op_name']
        used_ops.append(op_name)
        
        if op_name not in micro_ops:
            unsupported_ops.append(op_name)
        else:
            # Check if INT8 is supported for this op
            if 'QUANTIZE' in op_name or 'DEQUANTIZE' in op_name:
                pass  # These are needed for INT8 but not for float32
    
    # Check tensor sizes
    max_tensor_size = 0
    total_tensor_size = 0
    tensor_info = []
    
    for detail in tensor_details:
        shape = detail['shape']
        dtype = detail['dtype']
        itemsize = np.dtype(dtype).itemsize
        size = int(np.prod(shape)) * itemsize if len(shape) > 0 else 0
        max_tensor_size = max(max_tensor_size, size)
        total_tensor_size += size
        
        tensor_info.append({
            'name': detail['name'],
            'index': detail['index'],
            'shape': shape,
            'dtype': str(dtype),
            'size_bytes': size,
            'quantization': detail['quantization'] if 'quantization' in detail else None
        })
    
    # Estimate tensor arena (rough heuristic: 4x max tensor + overhead)
    recommended_arena = max_tensor_size * 4 + 10240  # 10KB overhead
    
    return {
        'model_name': model_name,
        'used_operators': sorted(list(set(used_ops))),
        'unsupported_operators': unsupported_ops,
        'tensor_count': len(tensor_details),
        'max_tensor_size_bytes': max_tensor_size,
        'total_tensor_size_bytes': total_tensor_size,
        'recommended_arena_size': recommended_arena,
        'tensor_details': tensor_info
    }


def analyze_esp32_memory_budget():
    """Analyze ESP32 memory budget for deployment."""
    
    # ESP32-WROOM-32 SRAM: ~520 KB
    # Reserved: IRAM, DRAM, stack, heap, WiFi/BT buffers
    
    memory_budget = {
        'total_sram_kb': 520,
        'reserved_system_kb': 100,      # IRAM, system tasks, ISR
        'wifi_bt_buffers_kb': 50,       # If WiFi/BT used
        'stack_kb': 32,                 # Default task stack
        'heap_overhead_kb': 20,         # Memory allocator overhead
        'tensor_arena_kb': 70,          # Current config
        'display_buffer_kb': 1,         # 128*64/8 = 1KB
        'scheduler_buffers_kb': 4,      # DDIM timesteps, alpha_bar arrays
        'serial_buffers_kb': 4,         # Serial I/O buffers
        'model_storage_flash_kb': 0,    # In flash, not SRAM
    }
    
    used_sram = sum([
        memory_budget['reserved_system_kb'],
        memory_budget['wifi_bt_buffers_kb'],
        memory_budget['stack_kb'],
        memory_budget['heap_overhead_kb'],
        memory_budget['tensor_arena_kb'],
        memory_budget['display_buffer_kb'],
        memory_budget['scheduler_buffers_kb'],
        memory_budget['serial_buffers_kb'],
    ])
    
    free_sram = memory_budget['total_sram_kb'] - used_sram
    
    memory_budget['used_sram_kb'] = used_sram
    memory_budget['free_sram_kb'] = free_sram
    memory_budget['margin_percent'] = (free_sram / memory_budget['total_sram_kb']) * 100
    
    return memory_budget


def analyze_tensor_arena_requirements(int8_analysis, float32_analysis):
    """Analyze tensor arena requirements for both models."""
    
    # TFLite Micro tensor arena calculation
    # The arena must hold:
    # - All persistent tensors (weights, biases)
    # - All intermediate activation tensors
    # - Scratch buffers for operators
    # - Alignment padding
    
    int8_max_tensor = int8_analysis['max_tensor_size_bytes']
    float32_max_tensor = float32_analysis['max_tensor_size_bytes']
    
    # Common heuristic: arena >= max_tensor * 3-4
    int8_arena_min = int8_max_tensor * 4
    int8_arena_recommended = int8_arena_min + 16384  # 16KB safety margin
    
    float32_arena_min = float32_max_tensor * 4
    float32_arena_recommended = float32_arena_min + 16384
    
    # Convert to KB
    int8_arena_min_kb = int8_arena_min / 1024
    int8_arena_rec_kb = int8_arena_recommended / 1024
    float32_arena_min_kb = float32_arena_min / 1024
    float32_arena_rec_kb = float32_arena_recommended / 1024
    
    return {
        'int8': {
            'max_tensor_kb': int8_max_tensor / 1024,
            'min_arena_kb': int8_arena_min_kb,
            'recommended_arena_kb': int8_arena_rec_kb,
            'current_config_kb': 70,
            'fits_current': int8_arena_rec_kb <= 70
        },
        'float32': {
            'max_tensor_kb': float32_max_tensor / 1024,
            'min_arena_kb': float32_arena_min_kb,
            'recommended_arena_kb': float32_arena_rec_kb,
            'current_config_kb': 70,
            'fits_current': float32_arena_rec_kb <= 70
        }
    }


def analyze_flash_storage():
    """Analyze flash storage requirements."""
    flash_info = {
        'total_flash_kb': 4096,
        'partition_table_kb': 4,
        'bootloader_kb': 64,
        'app_partition_kb': 1280,  # Typical factory partition
        'spiffs_kb': 1024,         # Filesystem
        'nvs_kb': 20,
        'ota_data_kb': 8,
    }
    
    # Model sizes
    flash_info['float32_model_kb'] = 190.7
    flash_info['int8_model_kb'] = 109.1
    
    # Available for model in app partition
    flash_info['available_app_kb'] = flash_info['app_partition_kb'] - 200  # Code overhead
    flash_info['model_fits_app'] = True  # Both models are small
    
    return flash_info


def main():
    output_dir = "artifacts/tflite_tiny_unet_v1/phase8_audit"
    ensure_dir(output_dir)
    
    print("=" * 60)
    print("Phase 8: ESP32 TFLite Micro Compatibility Audit")
    print("=" * 60)
    
    # Paths
    float32_path = "artifacts/tflite_tiny_unet_v1/model_float32.tflite"
    int8_path = "artifacts/tflite_tiny_unet_v1/model_int8.tflite"
    
    # Check TFLite Micro operator support
    print("\n1. Loading TFLite Micro operator support matrix...")
    micro_ops = check_tflite_micro_operators()
    print(f"   {len(micro_ops)} operators defined in support matrix")
    
    # Analyze FLOAT32 model
    print("\n2. Analyzing FLOAT32 model operators...")
    float32_analysis = analyze_model_operators(float32_path, "FLOAT32", micro_ops)
    print(f"   Used operators: {float32_analysis['used_operators']}")
    print(f"   Unsupported: {float32_analysis['unsupported_operators']}")
    print(f"   Max tensor: {float32_analysis['max_tensor_size_bytes']/1024:.1f} KB")
    print(f"   Recommended arena: {float32_analysis['recommended_arena_size']/1024:.1f} KB")
    
    # Analyze INT8 model
    print("\n3. Analyzing INT8 model operators...")
    int8_analysis = analyze_model_operators(int8_path, "INT8", micro_ops)
    print(f"   Used operators: {int8_analysis['used_operators']}")
    print(f"   Unsupported: {int8_analysis['unsupported_operators']}")
    print(f"   Max tensor: {int8_analysis['max_tensor_size_bytes']/1024:.1f} KB")
    print(f"   Recommended arena: {int8_analysis['recommended_arena_size']/1024:.1f} KB")
    
    # Memory budget analysis
    print("\n4. ESP32 Memory Budget Analysis...")
    memory_budget = analyze_esp32_memory_budget()
    print(f"   Total SRAM: {memory_budget['total_sram_kb']} KB")
    print(f"   Used SRAM: {memory_budget['used_sram_kb']} KB")
    print(f"   Free SRAM: {memory_budget['free_sram_kb']} KB ({memory_budget['margin_percent']:.1f}% margin)")
    
    # Tensor arena requirements
    print("\n5. Tensor Arena Requirements...")
    arena_analysis = analyze_tensor_arena_requirements(int8_analysis, float32_analysis)
    print(f"   INT8 - Max tensor: {arena_analysis['int8']['max_tensor_kb']:.1f} KB")
    print(f"   INT8 - Min arena: {arena_analysis['int8']['min_arena_kb']:.1f} KB")
    print(f"   INT8 - Recommended: {arena_analysis['int8']['recommended_arena_kb']:.1f} KB")
    print(f"   INT8 - Current config: {arena_analysis['int8']['current_config_kb']} KB")
    print(f"   INT8 - Fits current: {arena_analysis['int8']['fits_current']}")
    print(f"   FLOAT32 - Max tensor: {arena_analysis['float32']['max_tensor_kb']:.1f} KB")
    print(f"   FLOAT32 - Recommended: {arena_analysis['float32']['recommended_arena_kb']:.1f} KB")
    print(f"   FLOAT32 - Fits current: {arena_analysis['float32']['fits_current']}")
    
    # Flash storage
    print("\n6. Flash Storage Analysis...")
    flash_info = analyze_flash_storage()
    print(f"   Total flash: {flash_info['total_flash_kb']} KB")
    print(f"   FLOAT32 model: {flash_info['float32_model_kb']} KB")
    print(f"   INT8 model: {flash_info['int8_model_kb']} KB")
    print(f"   Available in app: {flash_info['available_app_kb']} KB")
    
    # Compatibility matrix
    print("\n7. Operator Compatibility Matrix...")
    
    all_used_ops = set(float32_analysis['used_operators'] + int8_analysis['used_operators'])
    
    compat_matrix = []
    for op in sorted(all_used_ops):
        if op in micro_ops:
            m = micro_ops[op]
            compat_matrix.append({
                'operator': op,
                'float32_supported': m['float32'],
                'int8_supported': m['int8'],
                'esp_nn_accelerated': m['esp_nn'],
                'in_float32_model': op in float32_analysis['used_operators'],
                'in_int8_model': op in int8_analysis['used_operators'],
                'status': 'OK' if (m['float32'] or not op in float32_analysis['used_operators']) 
                              and (m['int8'] or not op in int8_analysis['used_operators']) else 'UNSUPPORTED'
            })
        else:
            compat_matrix.append({
                'operator': op,
                'float32_supported': False,
                'int8_supported': False,
                'esp_nn_accelerated': False,
                'in_float32_model': op in float32_analysis['used_operators'],
                'in_int8_model': op in int8_analysis['used_operators'],
                'status': 'UNKNOWN_OPERATOR'
            })
    
    unsupported_float32 = [c for c in compat_matrix if c['in_float32_model'] and not c['float32_supported']]
    unsupported_int8 = [c for c in compat_matrix if c['in_int8_model'] and not c['int8_supported']]
    esp_nn_accelerated = [c for c in compat_matrix if c['esp_nn_accelerated']]
    
    print(f"   Unsupported in FLOAT32: {[c['operator'] for c in unsupported_float32]}")
    print(f"   Unsupported in INT8: {[c['operator'] for c in unsupported_int8]}")
    print(f"   ESP-NN accelerated: {[c['operator'] for c in esp_nn_accelerated]}")
    
    # Deployment verdict
    print("\n8. Deployment Feasibility Verdict...")
    
    issues = []
    
    # INT8 quality issue (from Phase 7B)
    issues.append("CRITICAL: INT8 quantization FAILED - DDIM generation max error = 2.0")
    issues.append("   INT8 model produces completely different outputs than FLOAT32")
    
    # Operator issues
    if unsupported_int8:
        for c in unsupported_int8:
            issues.append(f"INT8 UNSUPPORTED: {c['operator']} - no INT8 kernel in TFLM")
    
    if unsupported_float32:
        for c in unsupported_float32:
            issues.append(f"FLOAT32 UNSUPPORTED: {c['operator']} - no float32 kernel in TFLM")
    
    # TRANSPOSE_CONV issue
    if 'TRANSPOSE_CONV' in int8_analysis['used_operators']:
        issues.append("TRANSPOSE_CONV: No INT8 support, reference kernel only (slow)")
        issues.append("TRANSPOSE_CONV: No ESP-NN acceleration available")
    
    # SIN/COS issue
    if 'SIN' in int8_analysis['used_operators'] or 'COS' in int8_analysis['used_operators']:
        issues.append("SIN/COS: No INT8 support, reference kernel only (slow)")
    
    # DEQUANTIZE/QUANTIZE in INT8 model
    if 'DEQUANTIZE' in int8_analysis['used_operators'] or 'QUANTIZE' in int8_analysis['used_operators']:
        issues.append("INT8 model contains QUANTIZE/DEQUANTIZE ops - not fully quantized")
    
    # Tensor arena
    if not arena_analysis['int8']['fits_current']:
        issues.append(f"INT8 tensor arena too small: need {arena_analysis['int8']['recommended_arena_kb']:.1f} KB, have {arena_analysis['int8']['current_config_kb']} KB")
    
    if not arena_analysis['float32']['fits_current']:
        issues.append(f"FLOAT32 tensor arena too small: need {arena_analysis['float32']['recommended_arena_kb']:.1f} KB, have {arena_analysis['float32']['current_config_kb']} KB")
    
    # DELEGATE issue
    if 'DELEGATE' in int8_analysis['used_operators'] or 'DELEGATE' in float32_analysis['used_operators']:
        issues.append("DELEGATE operator present (XNNPACK) - desktop only, not for TFLM")
    
    # Print all issues
    for issue in issues:
        print(f"   {issue}")
    
    # Final verdict
    critical_issues = [i for i in issues if 'CRITICAL' in i or 'UNSUPPORTED' in i]
    major_issues = [i for i in issues if 'no INT8 support' in i or 'reference kernel' in i or 'not fully quantized' in i or 'too small' in i or 'DELEGATE' in i]
    minor_issues = [i for i in issues if i not in critical_issues and i not in major_issues]
    
    if critical_issues:
        verdict = "NOT READY"
    elif major_issues:
        verdict = "CONDITIONALLY READY"
    else:
        verdict = "READY FOR FIRMWARE INTEGRATION"
    
    print(f"\n   FINAL VERDICT: {verdict}")
    print(f"   Critical issues: {len(critical_issues)}")
    print(f"   Major issues: {len(major_issues)}")
    print(f"   Minor issues: {len(minor_issues)}")
    
    # Compile full report
    report = {
        'audit_timestamp': '2026-11-16',
        'float32_analysis': float32_analysis,
        'int8_analysis': int8_analysis,
        'memory_budget': memory_budget,
        'arena_analysis': arena_analysis,
        'flash_analysis': flash_info,
        'compatibility_matrix': compat_matrix,
        'unsupported_float32': [c['operator'] for c in unsupported_float32],
        'unsupported_int8': [c['operator'] for c in unsupported_int8],
        'esp_nn_accelerated': [c['operator'] for c in esp_nn_accelerated],
        'issues': issues,
        'critical_issues': critical_issues,
        'major_issues': major_issues,
        'minor_issues': minor_issues,
        'verdict': verdict
    }
    
    # Save report
    report_path = os.path.join(output_dir, "deployment_audit.json")
    save_json(report, report_path)
    print(f"\nAudit report saved to {report_path}")
    
    # Also save markdown report
    md_path = os.path.join(output_dir, "deployment_report.md")
    with open(md_path, 'w') as f:
        f.write("# Phase 8: ESP32 Deployment Feasibility Report\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Critical issues: {len(critical_issues)}\n")
        f.write(f"- Major issues: {len(major_issues)}\n")
        f.write(f"- Minor issues: {len(minor_issues)}\n\n")
        f.write("## Critical Issues\n\n")
        for issue in critical_issues:
            f.write(f"- {issue}\n")
        f.write("\n## Major Issues\n\n")
        for issue in major_issues:
            f.write(f"- {issue}\n")
        f.write("\n## Minor Issues\n\n")
        for issue in minor_issues:
            f.write(f"- {issue}\n")
        f.write("\n## Compatibility Matrix\n\n")
        f.write("| Operator | Float32 | INT8 | ESP-NN | In F32 | In INT8 | Status |\n")
        f.write("|----------|---------|------|--------|--------|---------|--------|\n")
        for c in compat_matrix:
            f.write(f"| {c['operator']} | {'✅' if c['float32_supported'] else '❌'} | {'✅' if c['int8_supported'] else '❌'} | {'✅' if c['esp_nn_accelerated'] else '❌'} | {'✅' if c['in_float32_model'] else '❌'} | {'✅' if c['in_int8_model'] else '❌'} | {c['status']} |\n")
        f.write("\n## Memory Budget\n\n")
        f.write(f"- Total SRAM: {memory_budget['total_sram_kb']} KB\n")
        f.write(f"- Used SRAM: {memory_budget['used_sram_kb']} KB\n")
        f.write(f"- Free SRAM: {memory_budget['free_sram_kb']} KB ({memory_budget['margin_percent']:.1f}%)\n")
        f.write(f"- INT8 arena needed: {arena_analysis['int8']['recommended_arena_kb']:.1f} KB (current: {arena_analysis['int8']['current_config_kb']} KB)\n")
        f.write(f"- FLOAT32 arena needed: {arena_analysis['float32']['recommended_arena_kb']:.1f} KB (current: {arena_analysis['float32']['current_config_kb']} KB)\n")
        f.write("\n## Flash Storage\n\n")
        f.write(f"- FLOAT32 model: {flash_info['float32_model_kb']:.1f} KB\n")
        f.write(f"- INT8 model: {flash_info['int8_model_kb']:.1f} KB\n")
        f.write(f"- Available in app partition: {flash_info['available_app_kb']} KB\n")
    
    print(f"Markdown report saved to {md_path}")
    
    return report


if __name__ == "__main__":
    main()