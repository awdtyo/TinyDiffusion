/**
 * TinyDiffusion-ESP32 FLOAT32 Model Runner Header
 * 
 * TensorFlow Lite Micro inference for the conditional diffusion model (FLOAT32)
 */

#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#include "model_data.h"
#include <cstdint>
#include <cstddef>

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
  class MicroInterpreter;
  template <unsigned int tOpCount> class MicroMutableOpResolver;
}

// ==================== Model Runner Class ====================

class ModelRunner {
public:
  ModelRunner();
  ~ModelRunner();
  
  // Initialize model and interpreter
  bool begin();
  
  // Run single denoising step
  // input: [1, 28, 28, 12] FLOAT32 (9408 elements)
  // output: [1, 28, 28, 1] FLOAT32 (784 elements)
  bool run_inference(const float* input, float* output);
  
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
  tflite::MicroMutableOpResolver<19>* resolver_;
  
  // Tensor arena
  uint8_t* tensor_arena_;
  size_t tensor_arena_size_;
  
  // Input/output tensor indices
  int input_index_;
  int output_index_;
  
  // Initialization state
  bool initialized_;
  
  // Register required operators
  bool register_operators();
  
  // Validate the raw FlatBuffer before dereferencing any of its structures
  bool validate_model_buffer(const uint8_t* data, size_t length);
  
  // Validate model input/output tensor metadata after allocation
  bool validate_model();
};

#endif // MODEL_RUNNER_H