/**
 * TinyDiffusion-ESP32 Model Runner Implementation
 * 
 * TensorFlow Lite Micro inference for the conditional diffusion CNN
 */

#include "model_runner.h"
#include "config.h"
#include "model_data.h"

#include <Arduino.h>
#include <cstring>

// ESP32 specific
#include "esp_heap_caps.h"
#include "esp_random.h"

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

// FlatBuffers structural verification (header-only)
#include "flatbuffers/verifier.h"

// ESP-NN for optimized kernels
#include "esp_nn.h"

using namespace tflite;

// ==================== Constants ====================

namespace {
constexpr size_t kModelAlignment = 16;
constexpr size_t kTensorArenaAlignment = 16;

void print_hex_prefix(const uint8_t* data, size_t length, size_t count) {
  for (size_t i = 0; i < count && i < length; ++i) {
    Serial.printf("%02x%s", data[i], (i + 1 == count) ? "" : " ");
  }
}
}  // namespace

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
    heap_caps_free(tensor_arena_);
  }
}

bool ModelRunner::register_operators() {
  // Create op resolver with only the operators needed by our model
  // Our model uses: CONV_2D, QUANTIZE, DEQUANTIZE, RESHAPE, CONCATENATION, ADD, MUL
  resolver_ = new MicroMutableOpResolver<7>();
  
  if (!resolver_) {
    return false;
  }
  
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
  
  return true;
}

bool ModelRunner::validate_model_buffer(const uint8_t* data, size_t length) {
  if (data == nullptr) {
    Serial.println("VALIDATION FAILED: model pointer is null");
    return false;
  }

  if (length < 8) {
    Serial.printf("VALIDATION FAILED: model length %u is too small\n",
                  (unsigned)length);
    return false;
  }

  // The FlatBuffer file identifier is the first field after the root offset.
  if (memcmp(data + 4, "TFL3", 4) != 0) {
    Serial.print("VALIDATION FAILED: missing TFL3 identifier, got: ");
    print_hex_prefix(data + 4, 4, 4);
    Serial.println();
    return false;
  }

  // Validate the root table offset before calling GetModel(), because
  // GetModel() blindly dereferences it.
  const uint32_t root_offset = (uint32_t)data[0] |
                               ((uint32_t)data[1] << 8) |
                               ((uint32_t)data[2] << 16) |
                               ((uint32_t)data[3] << 24);
  if (root_offset < 4 || root_offset >= length || (root_offset & 3U) != 0) {
    Serial.printf("VALIDATION FAILED: invalid root table offset %u (len %u)\n",
                  (unsigned)root_offset, (unsigned)length);
    return false;
  }

  // Full FlatBuffer verification against the exact buffer bounds.  This
  // catches truncated arrays, bad offsets and invalid vectors before TFLite
  // Micro walks the graph.
  flatbuffers::Verifier verifier(data, length);
  if (!VerifyModelBuffer(verifier)) {
    Serial.println("VALIDATION FAILED: FlatBuffer structure is invalid "
                   "(model data truncated or corrupted)");
    return false;
  }

  const Model* model = GetModel(data);
  if (model == nullptr) {
    Serial.println("VALIDATION FAILED: GetModel() returned null");
    return false;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("VALIDATION FAILED: schema version %u, expected %d\n",
                  (unsigned)model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  const auto* subgraphs = model->subgraphs();
  if (subgraphs == nullptr || subgraphs->size() == 0) {
    Serial.println("VALIDATION FAILED: model has no subgraphs");
    return false;
  }

  const auto* buffers = model->buffers();
  if (buffers == nullptr) {
    Serial.println("VALIDATION FAILED: model has no buffer table");
    return false;
  }

  // Extra explicit bounds check on every inline weight buffer.  The verifier
  // already does this, but this makes the failure mode obvious in the log.
  const uint8_t* model_end = data + length;
  for (uint32_t i = 0; i < buffers->size(); ++i) {
    const Buffer* buffer = buffers->Get(i);
    if (buffer == nullptr || buffer->data() == nullptr) {
      continue;
    }
    const uint8_t* buffer_data = buffer->data()->data();
    const size_t buffer_size = buffer->data()->size();
    if (buffer_data < data || buffer_data > model_end ||
        buffer_size > (size_t)(model_end - buffer_data)) {
      Serial.printf("VALIDATION FAILED: buffer %u is outside model bounds\n",
                    (unsigned)i);
      return false;
    }
  }

  Serial.printf("Model schema version: %u\n", (unsigned)model->version());
  Serial.printf("Subgraphs: %u\n", (unsigned)subgraphs->size());
  Serial.printf("Buffers: %u\n", (unsigned)buffers->size());
  Serial.printf("Operator codes: %u\n", (unsigned)model->operator_codes()->size());

  return true;
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
    Serial.printf("Input dims: %d, expected 4\n", input->dims->size);
    return false;
  }
  
  if (input->dims->data[1] != MODEL_INPUT_HEIGHT ||
      input->dims->data[2] != MODEL_INPUT_WIDTH ||
      input->dims->data[3] != MODEL_INPUT_CHANNELS) {
    Serial.printf("Input shape mismatch: [%d, %d, %d], expected [%d, %d, %d]\n",
                  input->dims->data[1], input->dims->data[2], input->dims->data[3],
                  MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS);
    return false;
  }
  
  const size_t expected_input_bytes =
      MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH * MODEL_INPUT_CHANNELS;
  if (input->type != kTfLiteInt8 || input->bytes != expected_input_bytes) {
    Serial.printf("Input tensor mismatch: type=%d bytes=%u expected type=%d bytes=%u\n",
                  (int)input->type, (unsigned)input->bytes,
                  (int)kTfLiteInt8, (unsigned)expected_input_bytes);
    return false;
  }
  
  // Get input quantization
  input_scale_ = input->params.scale;
  input_zero_point_ = input->params.zero_point;
  
  Serial.printf("Input tensor: [%d, %d, %d, %d] INT8, %u bytes, scale=%.9f, zp=%ld\n",
                input->dims->data[0], input->dims->data[1], input->dims->data[2],
                input->dims->data[3], (unsigned)input->bytes,
                input_scale_, (long)input_zero_point_);
  
  // Get output tensor
  TfLiteTensor* output = interpreter_->output(0);
  if (!output) {
    DEBUG_PRINTLN("Failed to get output tensor");
    return false;
  }
  
  output_index_ = 0;
  
  // Check output dimensions: [1, 28, 28, 1]
  if (output->dims->size != 4) {
    Serial.printf("Output dims: %d, expected 4\n", output->dims->size);
    return false;
  }
  
  if (output->dims->data[1] != MODEL_OUTPUT_HEIGHT ||
      output->dims->data[2] != MODEL_OUTPUT_WIDTH ||
      output->dims->data[3] != MODEL_OUTPUT_CHANNELS) {
    Serial.printf("Output shape mismatch: [%d, %d, %d], expected [%d, %d, %d]\n",
                  output->dims->data[1], output->dims->data[2], output->dims->data[3],
                  MODEL_OUTPUT_HEIGHT, MODEL_OUTPUT_WIDTH, MODEL_OUTPUT_CHANNELS);
    return false;
  }
  
  const size_t expected_output_bytes =
      MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS;
  if (output->type != kTfLiteInt8 || output->bytes != expected_output_bytes) {
    Serial.printf("Output tensor mismatch: type=%d bytes=%u expected type=%d bytes=%u\n",
                  (int)output->type, (unsigned)output->bytes,
                  (int)kTfLiteInt8, (unsigned)expected_output_bytes);
    return false;
  }
  
  // Get output quantization
  output_scale_ = output->params.scale;
  output_zero_point_ = output->params.zero_point;
  
  Serial.printf("Output tensor: [%d, %d, %d, %d] INT8, %u bytes, scale=%.9f, zp=%ld\n",
                output->dims->data[0], output->dims->data[1], output->dims->data[2],
                output->dims->data[3], (unsigned)output->bytes,
                output_scale_, (long)output_zero_point_);
  
  return true;
}

bool ModelRunner::begin() {
  if (initialized_) return true;
  
  Serial.println("Initializing ModelRunner...");
  
  // The model lives in flash (PROGMEM) and is 16-byte aligned.  TFLite Micro
  // can read it directly from the ESP32's memory-mapped flash; copying it to
  // RAM is unnecessary and previously masked a truncated-array bug.
  const uint8_t* model_data = g_model;
  const size_t model_length = g_model_len;
  
  Serial.println("=== TFLite Model Diagnostics ===");
  Serial.printf("Model pointer: %p\n", (const void*)model_data);
  Serial.printf("Model byte length: %u\n", (unsigned)model_length);
  Serial.printf("Model alignment: address %% %u = %u\n",
                (unsigned)kModelAlignment,
                (unsigned)((uintptr_t)model_data % kModelAlignment));
  Serial.print("First 16 bytes: ");
  print_hex_prefix(model_data, model_length, 16);
  Serial.println();
  if (model_length >= 8) {
    Serial.printf("FlatBuffer identifier: %.4s\n", (const char*)(model_data + 4));
  }
  Serial.printf("Expected schema version: %d\n", TFLITE_SCHEMA_VERSION);
  
  // Safely validate identifier, root offset, schema and FlatBuffer structure.
  if (!validate_model_buffer(model_data, model_length)) {
    return false;
  }
  
  // Allocate the tensor arena in internal SRAM, 16-byte aligned.
  tensor_arena_ = (uint8_t*)heap_caps_aligned_alloc(
      kTensorArenaAlignment, tensor_arena_size_,
      MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  if (!tensor_arena_) {
    Serial.printf("Failed to allocate %u-byte tensor arena\n",
                  (unsigned)tensor_arena_size_);
    return false;
  }
  
  Serial.printf("Tensor arena: %p (%u bytes, address %% %u = %u)\n",
                (void*)tensor_arena_, (unsigned)tensor_arena_size_,
                (unsigned)kTensorArenaAlignment,
                (unsigned)((uintptr_t)tensor_arena_ % kTensorArenaAlignment));
  Serial.printf("Free internal heap before AllocateTensors: %u bytes\n",
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  Serial.printf("Largest free internal block: %u bytes\n",
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
  
  // Register operators
  if (!register_operators()) {
    Serial.println("Failed to register operators");
    return false;
  }
  
  const Model* model = GetModel(model_data);
  interpreter_ = new MicroInterpreter(model, *resolver_, tensor_arena_,
                                      tensor_arena_size_);
  if (!interpreter_) {
    Serial.println("Failed to create interpreter");
    return false;
  }
  
  Serial.println("Calling AllocateTensors()...");
  TfLiteStatus allocate_status = interpreter_->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.printf("Failed to allocate tensors (status=%d)\n", (int)allocate_status);
    return false;
  }
  
  Serial.printf("Tensors allocated. Arena used: %u bytes\n",
                (unsigned)interpreter_->arena_used_bytes());
  Serial.printf("Free internal heap after AllocateTensors: %u bytes\n",
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  Serial.printf("Largest free internal block: %u bytes\n",
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
  
  // Validate model
  if (!validate_model()) {
    Serial.println("Model validation failed");
    return false;
  }
  
  initialized_ = true;
  Serial.println("ModelRunner initialized successfully");
  
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
  if (input_tensor->type != kTfLiteInt8 || input_tensor->bytes != input_size) {
    Serial.println("Input tensor metadata changed unexpectedly");
    return false;
  }
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
  if (output_tensor->type != kTfLiteInt8 || output_tensor->bytes != output_size) {
    Serial.println("Output tensor metadata changed unexpectedly");
    return false;
  }
  memcpy(output, output_tensor->data.int8, output_size);
  
  return true;
}

void ModelRunner::print_model_info() {
  if (!initialized_) {
    DEBUG_PRINTLN("ModelRunner not initialized");
    return;
  }
  
  Serial.println("=== Model Info ===");
  Serial.printf("Model size: %u bytes\n", (unsigned)g_model_len);
  Serial.printf("Tensor arena: %u bytes\n", (unsigned)tensor_arena_size_);
  Serial.printf("Arena used: %u bytes\n", (unsigned)interpreter_->arena_used_bytes());
  Serial.printf("Input: [%d, %d, %d, %d] INT8 (scale=%.9f, zp=%ld)\n",
                1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS,
                input_scale_, (long)input_zero_point_);
  Serial.printf("Output: [%d, %d, %d, %d] INT8 (scale=%.9f, zp=%ld)\n",
                1, MODEL_OUTPUT_HEIGHT, MODEL_OUTPUT_WIDTH, MODEL_OUTPUT_CHANNELS,
                output_scale_, (long)output_zero_point_);
  Serial.println("==================");
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
