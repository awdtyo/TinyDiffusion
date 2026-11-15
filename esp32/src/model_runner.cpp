/**
 * TinyDiffusion-ESP32 Model Runner Implementation
 * 
 * TensorFlow Lite Micro inference for the conditional diffusion CNN
 */

#include "model_runner.h"
#include "config.h"
#include "model_data.h"

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"

// ESP-NN for optimized kernels
#include "esp_nn.h"

using namespace tflite;

// ==================== Model Runner Implementation ====================

ModelRunner::ModelRunner() 
  : interpreter_(nullptr)
  , resolver_(nullptr)
  , tensor_arena_(nullptr)
  , tensor_arena_size_(TENSOR_ARENA_SIZE)
  , input_index_(-1)
  , output_index_(-1)
  , input_scale_(0.0f)
  , input_zero_point_(0)
  , output_scale_(0.0f)
  , output_zero_point_(0)
  , initialized_(false) {
}

ModelRunner::~ModelRunner() {
  delete interpreter_;
  delete resolver_;
  if (tensor_arena_) {
    free(tensor_arena_);
  }
}

bool ModelRunner::register_operators() {
  // Create op resolver with only the operators needed by our model
  // Our model uses: CONV_2D, ADD, MUL, RELU (fused), RESHAPE, CONCATENATION
  // But in the exported graph, we only have CONV_2D operations
  
  resolver_ = new MicroMutableOpResolver<6>();
  
  // Conv2D is the main operator
  resolver_->AddConv2D();
  
  // For quantization support
  resolver_->AddQuantize();
  resolver_->AddDequantize();
  
  // For reshaping/concatenation if needed in graph
  resolver_->AddReshape();
  resolver_->AddConcatenation();
  
  // For element-wise ops
  resolver_->AddAdd();
  resolver_->AddMul();
  
  return resolver_ != nullptr;
}

bool ModelRunner::validate_model() {
  if (!interpreter_) return false;
  
  // Get input tensor
  TfLiteTensor* input = interpreter_->input(0);
  if (!input) {
    DEBUG_PRINTLN("Failed to get input tensor");
    return false;
  }
  
  input_index_ = 0;
  
  // Check input dimensions: [1, 28, 28, 12]
  if (input->dims->size != 4) {
    DEBUG_PRINTF("Input dims: %d, expected 4\n", input->dims->size);
    return false;
  }
  
  if (input->dims->data[1] != MODEL_INPUT_HEIGHT ||
      input->dims->data[2] != MODEL_INPUT_WIDTH ||
      input->dims->data[3] != MODEL_INPUT_CHANNELS) {
    DEBUG_PRINTF("Input shape mismatch: [%d, %d, %d], expected [%d, %d, %d]\n",
                 input->dims->data[1], input->dims->data[2], input->dims->data[3],
                 MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS);
    return false;
  }
  
  // Get input quantization
  input_scale_ = input->params.scale;
  input_zero_point_ = input->params.zero_point;
  
  DEBUG_PRINTF("Input: scale=%.6f, zp=%d\n", input_scale_, input_zero_point_);
  
  // Get output tensor
  TfLiteTensor* output = interpreter_->output(0);
  if (!output) {
    DEBUG_PRINTLN("Failed to get output tensor");
    return false;
  }
  
  output_index_ = 0;
  
  // Check output dimensions: [1, 28, 28, 1]
  if (output->dims->size != 4) {
    DEBUG_PRINTF("Output dims: %d, expected 4\n", output->dims->size);
    return false;
  }
  
  if (output->dims->data[1] != MODEL_OUTPUT_HEIGHT ||
      output->dims->data[2] != MODEL_OUTPUT_WIDTH ||
      output->dims->data[3] != MODEL_OUTPUT_CHANNELS) {
    DEBUG_PRINTF("Output shape mismatch: [%d, %d, %d], expected [%d, %d, %d]\n",
                 output->dims->data[1], output->dims->data[2], output->dims->data[3],
                 MODEL_OUTPUT_HEIGHT, MODEL_OUTPUT_WIDTH, MODEL_OUTPUT_CHANNELS);
    return false;
  }
  
  // Get output quantization
  output_scale_ = output->params.scale;
  output_zero_point_ = output->params.zero_point;
  
  DEBUG_PRINTF("Output: scale=%.6f, zp=%d\n", output_scale_, output_zero_point_);
  
  return true;
}

bool ModelRunner::begin() {
  if (initialized_) return true;
  
  DEBUG_PRINTLN("Initializing ModelRunner...");
  
  // Allocate tensor arena
  tensor_arena_ = (uint8_t*)heap_caps_malloc(tensor_arena_size_, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  if (!tensor_arena_) {
    DEBUG_PRINTLN("Failed to allocate tensor arena");
    return false;
  }
  
  DEBUG_PRINTF("Tensor arena: %zu bytes at %p\n", tensor_arena_size_, tensor_arena_);
  
  // Register operators
  if (!register_operators()) {
    DEBUG_PRINTLN("Failed to register operators");
    return false;
  }
  
  // Create interpreter
  const tflite::Model* model = tflite::GetModel(g_tiny_diffusion_model);
  if (!model) {
    DEBUG_PRINTLN("Failed to parse model");
    return false;
  }
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    DEBUG_PRINTF("Model schema version mismatch: %d vs %d\n", 
                 model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  
  interpreter_ = new MicroInterpreter(*model, *resolver_, tensor_arena_, tensor_arena_size_);
  if (!interpreter_) {
    DEBUG_PRINTLN("Failed to create interpreter");
    return false;
  }
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter_->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    DEBUG_PRINTLN("Failed to allocate tensors");
    return false;
  }
  
  DEBUG_PRINTF("Tensors allocated. Arena used: %zu bytes\n", interpreter_->arena_used_bytes());
  
  // Validate model
  if (!validate_model()) {
    DEBUG_PRINTLN("Model validation failed");
    return false;
  }
  
  initialized_ = true;
  DEBUG_PRINTLN("ModelRunner initialized successfully");
  
  return true;
}

bool ModelRunner::run_inference(const int8_t* input, int8_t* output) {
  if (!initialized_) {
    DEBUG_PRINTLN("ModelRunner not initialized");
    return false;
  }
  
  // Get input tensor
  TfLiteTensor* input_tensor = interpreter_->input(input_index_);
  if (!input_tensor) {
    DEBUG_PRINTLN("Failed to get input tensor");
    return false;
  }
  
  // Copy input data
  // Input size: 28 * 28 * 12 = 9408 bytes
  const size_t input_size = MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH * MODEL_INPUT_CHANNELS;
  memcpy(input_tensor->data.int8, input, input_size);
  
  // Run inference
  TfLiteStatus invoke_status = interpreter_->Invoke();
  if (invoke_status != kTfLiteOk) {
    DEBUG_PRINTLN("Inference failed");
    return false;
  }
  
  // Get output tensor
  TfLiteTensor* output_tensor = interpreter_->output(output_index_);
  if (!output_tensor) {
    DEBUG_PRINTLN("Failed to get output tensor");
    return false;
  }
  
  // Copy output data
  // Output size: 28 * 28 * 1 = 784 bytes
  const size_t output_size = MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS;
  memcpy(output, output_tensor->data.int8, output_size);
  
  return true;
}

void ModelRunner::print_model_info() {
  if (!initialized_) {
    DEBUG_PRINTLN("ModelRunner not initialized");
    return;
  }
  
  DEBUG_PRINTLN("=== Model Info ===");
  DEBUG_PRINTF("Model size: %u bytes\n", g_tiny_diffusion_model_len);
  DEBUG_PRINTF("Tensor arena: %zu bytes\n", tensor_arena_size_);
  DEBUG_PRINTF("Arena used: %zu bytes\n", interpreter_->arena_used_bytes());
  DEBUG_PRINTF("Input: [%d, %d, %d, %d] INT8 (scale=%.6f, zp=%d)\n",
               1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS,
               input_scale_, input_zero_point_);
  DEBUG_PRINTF("Output: [%d, %d, %d, %d] INT8 (scale=%.6f, zp=%d)\n",
               1, MODEL_OUTPUT_HEIGHT, MODEL_OUTPUT_WIDTH, MODEL_OUTPUT_CHANNELS,
               output_scale_, output_zero_point_);
  DEBUG_PRINTLN("==================");
}

size_t ModelRunner::get_arena_used_bytes() const {
  if (interpreter_) {
    return interpreter_->arena_used_bytes();
  }
  return 0;
}

size_t ModelRunner::get_arena_size() const {
  return tensor_arena_size_;
}