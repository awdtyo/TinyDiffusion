/**
 * TinyDiffusion-ESP32 Model Runner Header
 * 
 * TensorFlow Lite Micro inference engine for the conditional diffusion model
 */

#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#include "config.h"
#include <cstdint>

// Forward declare TFLite Micro types
namespace tflite {
  class MicroInterpreter;
  class MicroMutableOpResolver;
  struct MicroAllocator;
}

// ==================== Model Runner Class ====================

class ModelRunner {
public:
  ModelRunner();
  ~ModelRunner();
  
  // Initialize model and interpreter
  bool begin();
  
  // Run single denoising step
  // input: [28, 28, 12] INT8 quantized
  // output: [28, 28, 1] INT8 quantized
  bool run_inference(const int8_t* input, int8_t* output);
  
  // Get model info
  void print_model_info();
  
  // Get tensor arena usage
  size_t get_arena_used_bytes() const;
  size_t get_arena_size() const;
  
  // Check if initialized
  bool is_initialized() const { return initialized_; }

private:
  // TFLite Micro interpreter
  tflite::MicroInterpreter* interpreter_;
  tflite::MicroMutableOpResolver* resolver_;
  
  // Tensor arena
  uint8_t* tensor_arena_;
  size_t tensor_arena_size_;
  
  // Input/output tensor indices
  int input_index_;
  int output_index_;
  
  // Quantization parameters
  float input_scale_;
  int32_t input_zero_point_;
  float output_scale_;
  int32_t output_zero_point_;
  
  // Initialization state
  bool initialized_;
  
  // Register required operators
  bool register_operators();
  
  // Validate model dimensions
  bool validate_model();
};

#endif // MODEL_RUNNER_H