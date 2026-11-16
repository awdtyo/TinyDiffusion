/**
 * TinyDiffusion-ESP32 FLOAT32 Model Runner Implementation
 * 
 * TensorFlow Lite Micro inference for the conditional diffusion model (FLOAT32)
 */

#include "model_runner.h"
#include "model_data.h"
#include <Arduino.h>

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

using namespace tflite;

ModelRunner::ModelRunner() 
  : interpreter_(nullptr)
  , resolver_(nullptr)
  , tensor_arena_(nullptr)
  , tensor_arena_size_(0)
  , input_index_(-1)
  , output_index_(-1)
  , initialized_(false) {
}

ModelRunner::~ModelRunner() {
  delete interpreter_;
  delete resolver_;
  // tensor_arena_ is statically allocated in .bss, do not delete
}

bool ModelRunner::validate_model_buffer(const uint8_t* data, size_t length) {
  if (!data || length < 8) {
    Serial.println("ERROR: Model data is null or too small");
    return false;
  }
  
  // Check TFLite magic bytes (TFL3)
  if (data[4] != 'T' || data[5] != 'F' || data[6] != 'L' || data[7] != '3') {
    Serial.printf("ERROR: Invalid TFLite magic bytes: %c%c%c%c\n", 
                  data[4], data[5], data[6], data[7]);
    return false;
  }
  
  // Verify size matches
  if (length != g_model_len) {
    Serial.printf("ERROR: Model length mismatch: expected %u, got %zu\n", 
                  g_model_len, length);
    return false;
  }
  
  return true;
}

bool ModelRunner::register_operators() {
  resolver_ = new MicroMutableOpResolver<19>();
  if (!resolver_) {
    Serial.println("ERROR: Failed to create op resolver");
    return false;
  }
  
  // Register all operators required by the FLOAT32 model
  // Order matters for some TFLM versions - register in dependency order
  
  if (resolver_->AddConv2D() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register CONV_2D");
    return false;
  }
  if (resolver_->AddTransposeConv() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register TRANSPOSE_CONV");
    return false;
  }
  if (resolver_->AddFullyConnected() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register FULLY_CONNECTED");
    return false;
  }
  if (resolver_->AddAdd() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register ADD");
    return false;
  }
  if (resolver_->AddMul() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register MUL");
    return false;
  }
  if (resolver_->AddConcatenation() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register CONCATENATION");
    return false;
  }
  if (resolver_->AddSplit() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register SPLIT");
    return false;
  }
  if (resolver_->AddReshape() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register RESHAPE");
    return false;
  }
  if (resolver_->AddExpandDims() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register EXPAND_DIMS");
    return false;
  }
  if (resolver_->AddStridedSlice() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register STRIDED_SLICE");
    return false;
  }
  if (resolver_->AddArgMax() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register ARG_MAX");
    return false;
  }
  if (resolver_->AddCast() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register CAST");
    return false;
  }
  if (resolver_->AddSin() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register SIN");
    return false;
  }
  if (resolver_->AddCos() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register COS");
    return false;
  }
  if (resolver_->AddGather() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register GATHER");
    return false;
  }
  if (resolver_->AddShape() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register SHAPE");
    return false;
  }
  if (resolver_->AddPack() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register PACK");
    return false;
  }
  if (resolver_->AddRelu() != kTfLiteOk) {
    Serial.println("ERROR: Failed to register RELU");
    return false;
  }
  
  Serial.println("All 19 operators registered successfully");
  return true;
}

bool ModelRunner::validate_model() {
  if (!interpreter_) {
    Serial.println("ERROR: Interpreter not initialized");
    return false;
  }
  
  // Get input tensor
  TfLiteTensor* input = interpreter_->input(0);
  if (!input) {
    Serial.println("ERROR: No input tensor");
    return false;
  }
  
  input_index_ = 0;
  
  Serial.println("Input tensor:");
  Serial.printf("  Type: %d (expected 1=kTfLiteFloat32)\n", input->type);
  Serial.printf("  Dims: [");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.printf("%d", input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.printf(", ");
  }
  Serial.printf("]\n");
  Serial.printf("  Bytes: %d\n", input->bytes);
  
  if (input->type != kTfLiteFloat32) {
    Serial.printf("ERROR: Input type is %d, expected %d (FLOAT32)\n", input->type, kTfLiteFloat32);
    return false;
  }
  
  if (input->dims->size != 4 || 
      input->dims->data[0] != 1 || 
      input->dims->data[1] != 28 || 
      input->dims->data[2] != 28 || 
      input->dims->data[3] != 12) {
    Serial.println("ERROR: Input shape mismatch: expected [1, 28, 28, 12]");
    return false;
  }
  
  // Get output tensor
  TfLiteTensor* output = interpreter_->output(0);
  if (!output) {
    Serial.println("ERROR: No output tensor");
    return false;
  }
  
  output_index_ = 0;
  
  Serial.println("Output tensor:");
  Serial.printf("  Type: %d (expected 1=kTfLiteFloat32)\n", output->type);
  Serial.printf("  Dims: [");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.printf("%d", output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.printf(", ");
  }
  Serial.printf("]\n");
  Serial.printf("  Bytes: %d\n", output->bytes);
  
  if (output->type != kTfLiteFloat32) {
    Serial.printf("ERROR: Output type is %d, expected %d (FLOAT32)\n", output->type, kTfLiteFloat32);
    return false;
  }
  
  if (output->dims->size != 4 || 
      output->dims->data[0] != 1 || 
      output->dims->data[1] != 28 || 
      output->dims->data[2] != 28 || 
      output->dims->data[3] != 1) {
    Serial.println("ERROR: Output shape mismatch: expected [1, 28, 28, 1]");
    return false;
  }
  
  return true;
}

bool ModelRunner::begin() {
  Serial.println("=== TinyDiffusion FLOAT32 Model Runner ===");
  Serial.printf("Model size: %u bytes\n", g_model_len);
  
  // Validate model buffer
  if (!validate_model_buffer(g_model, g_model_len)) {
    return false;
  }
  
  // Register operators
  if (!register_operators()) {
    return false;
  }
  
  // Allocate tensor arena using heap_caps_malloc to get contiguous internal RAM
  // Due to heap fragmentation, largest free block is ~108 KB
  // Try MALLOC_CAP_32BIT for potentially better allocation
  tensor_arena_size_ = 110592;  // 108 KB
  tensor_arena_ = (uint8_t*)heap_caps_malloc(tensor_arena_size_, MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
  if (!tensor_arena_) {
    Serial.printf("ERROR: Failed to allocate tensor arena (%zu bytes) with 32BIT\n", tensor_arena_size_);
    // Try 8BIT
    tensor_arena_ = (uint8_t*)heap_caps_malloc(tensor_arena_size_, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!tensor_arena_) {
      Serial.printf("ERROR: Failed to allocate tensor arena (%zu bytes) with 8BIT\n", tensor_arena_size_);
      // Try smaller
      tensor_arena_size_ = 102400;  // 100 KB
      tensor_arena_ = (uint8_t*)heap_caps_malloc(tensor_arena_size_, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
      if (!tensor_arena_) {
        Serial.printf("ERROR: Failed to allocate fallback tensor arena (%zu bytes)\n", tensor_arena_size_);
        return false;
      }
      Serial.printf("WARNING: Using smaller tensor arena: %zu bytes (%.1f KB)\n", 
                    tensor_arena_size_, tensor_arena_size_ / 1024.0);
    } else {
      Serial.printf("Tensor arena allocated: %zu bytes (%.1f KB) via 8BIT\n", 
                    tensor_arena_size_, tensor_arena_size_ / 1024.0);
    }
  } else {
    Serial.printf("Tensor arena allocated: %zu bytes (%.1f KB) via 32BIT\n", 
                  tensor_arena_size_, tensor_arena_size_ / 1024.0);
  }
  
  // Create interpreter
  const Model* model = GetModel(g_model);
  if (!model) {
    Serial.println("ERROR: Failed to parse model flatbuffer");
    return false;
  }
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("WARNING: Model schema version %lu, runtime expects %d\n", 
                  (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
  }
  
  // Create interpreter
  interpreter_ = new MicroInterpreter(model, *resolver_, tensor_arena_, tensor_arena_size_);
  if (!interpreter_) {
    Serial.println("ERROR: Failed to create interpreter");
    return false;
  }
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter_->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: Failed to allocate tensors");
    return false;
  }
  
  Serial.println("Tensors allocated successfully");
  
  // Validate model I/O
  if (!validate_model()) {
    return false;
  }
  
  // Report memory usage
  size_t used = interpreter_->arena_used_bytes();
  size_t free = tensor_arena_size_ - used;
  Serial.printf("Arena usage: %zu bytes used, %zu bytes free (%.1f%% free)\n", 
                used, free, 100.0 * free / tensor_arena_size_);
  
  initialized_ = true;
  Serial.println("Model initialization complete!");
  return true;
}

bool ModelRunner::run_inference(const float* input, float* output) {
  if (!initialized_) {
    Serial.println("ERROR: Model not initialized");
    return false;
  }
  
  // Get input tensor
  TfLiteTensor* input_tensor = interpreter_->input(input_index_);
  if (!input_tensor) {
    Serial.println("ERROR: Input tensor not found");
    return false;
  }
  
  // Copy input data (9408 floats = 37632 bytes)
  const int input_elements = MODEL_INPUT_ELEMENTS;  // 9408
  if (input_tensor->bytes != input_elements * sizeof(float)) {
    Serial.printf("ERROR: Input tensor size mismatch: expected %d bytes, got %d\n", 
                  input_elements * sizeof(float), input_tensor->bytes);
    return false;
  }
  
  memcpy(input_tensor->data.f, input, input_tensor->bytes);
  
  // Invoke interpreter
  TfLiteStatus invoke_status = interpreter_->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed");
    return false;
  }
  
  // Get output tensor
  TfLiteTensor* output_tensor = interpreter_->output(output_index_);
  if (!output_tensor) {
    Serial.println("ERROR: Output tensor not found");
    return false;
  }
  
  // Copy output data (784 floats = 3136 bytes)
  const int output_elements = MODEL_OUTPUT_ELEMENTS;  // 784
  if (output_tensor->bytes != output_elements * sizeof(float)) {
    Serial.printf("ERROR: Output tensor size mismatch: expected %d bytes, got %d\n", 
                  output_elements * sizeof(float), output_tensor->bytes);
    return false;
  }
  
  memcpy(output, output_tensor->data.f, output_tensor->bytes);
  
  return true;
}

void ModelRunner::print_model_info() {
  if (!initialized_) {
    Serial.println("Model not initialized");
    return;
  }
  
  Serial.println("=== Model Info ===");
  Serial.printf("Input: [1, 28, 28, 12] float32 (%d elements)\n", MODEL_INPUT_ELEMENTS);
  Serial.printf("Output: [1, 28, 28, 1] float32 (%d elements)\n", MODEL_OUTPUT_ELEMENTS);
  Serial.printf("Arena size: %zu bytes (%.1f KB)\n", tensor_arena_size_, tensor_arena_size_ / 1024.0);
  Serial.printf("Arena used: %zu bytes\n", interpreter_->arena_used_bytes());
}

size_t ModelRunner::get_arena_used_bytes() const {
  if (!interpreter_) return 0;
  return interpreter_->arena_used_bytes();
}

size_t ModelRunner::get_arena_size() const {
  return tensor_arena_size_;
}