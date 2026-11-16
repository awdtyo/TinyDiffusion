/**
 * ESP32 Memory Feasibility Test - Stage 3: Early Allocation (Before Arduino Init)
 * 
 * Attempts to allocate tensor arena BEFORE Arduino/Serial initialization
 * to potentially get a larger contiguous block from less-fragmented heap.
 */

#include <Arduino.h>
#include "esp_heap_caps.h"

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model data
alignas(16) extern const unsigned char g_model[];
extern const unsigned int g_model_len;

using namespace tflite;

struct MemStats {
  size_t free_heap;
  size_t largest_block;
  size_t free_8bit;
  size_t largest_8bit;
};

void measure_memory(const char* label, MemStats& stats) {
  stats.free_heap = ESP.getFreeHeap();
  stats.largest_block = ESP.getMaxAllocHeap();
  stats.free_8bit = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  stats.largest_8bit = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  
  Serial.printf("\n=== %s ===\n", label);
  Serial.printf("Free heap:       %zu bytes (%.2f KB)\n", stats.free_heap, stats.free_heap / 1024.0);
  Serial.printf("Largest block:   %zu bytes (%.2f KB)\n", stats.largest_block, stats.largest_block / 1024.0);
  Serial.printf("Free 8BIT:       %zu bytes (%.2f KB)\n", stats.free_8bit, stats.free_8bit / 1024.0);
  Serial.printf("Largest 8BIT:    %zu bytes (%.2f KB)\n", stats.largest_8bit, stats.largest_8bit / 1024.0);
}

bool validate_model_buffer(const uint8_t* data, size_t length) {
  if (!data || length < 8) return false;
  if (data[4] != 'T' || data[5] != 'F' || data[6] != 'L' || data[7] != '3') return false;
  if (length != g_model_len) return false;
  return true;
}

bool register_operators(MicroMutableOpResolver<19>* resolver) {
  if (resolver->AddConv2D() != kTfLiteOk) return false;
  if (resolver->AddTransposeConv() != kTfLiteOk) return false;
  if (resolver->AddFullyConnected() != kTfLiteOk) return false;
  if (resolver->AddAdd() != kTfLiteOk) return false;
  if (resolver->AddMul() != kTfLiteOk) return false;
  if (resolver->AddConcatenation() != kTfLiteOk) return false;
  if (resolver->AddSplit() != kTfLiteOk) return false;
  if (resolver->AddReshape() != kTfLiteOk) return false;
  if (resolver->AddExpandDims() != kTfLiteOk) return false;
  if (resolver->AddStridedSlice() != kTfLiteOk) return false;
  if (resolver->AddArgMax() != kTfLiteOk) return false;
  if (resolver->AddCast() != kTfLiteOk) return false;
  if (resolver->AddSin() != kTfLiteOk) return false;
  if (resolver->AddCos() != kTfLiteOk) return false;
  if (resolver->AddGather() != kTfLiteOk) return false;
  if (resolver->AddShape() != kTfLiteOk) return false;
  if (resolver->AddPack() != kTfLiteOk) return false;
  if (resolver->AddRelu() != kTfLiteOk) return false;
  return true;
}

// Early allocation - BEFORE any Arduino initialization
uint8_t* g_early_tensor_arena = nullptr;
size_t g_early_arena_size = 0;

class EarlyAllocator {
public:
  EarlyAllocator() {
    // This runs before setup() but after some Arduino init
    // Try to allocate maximum possible arena
    size_t max_size = 120 * 1024;  // 120 KB
    g_early_arena_size = max_size;
    g_early_tensor_arena = (uint8_t*)heap_caps_malloc(max_size, MALLOC_CAP_8BIT);
    if (!g_early_tensor_arena) {
      g_early_arena_size = 110 * 1024;
      g_early_tensor_arena = (uint8_t*)heap_caps_malloc(g_early_arena_size, MALLOC_CAP_8BIT);
    }
    if (!g_early_tensor_arena) {
      g_early_arena_size = 108 * 1024;
      g_early_tensor_arena = (uint8_t*)heap_caps_malloc(g_early_arena_size, MALLOC_CAP_8BIT);
    }
  }
};

EarlyAllocator g_early_allocator;  // Global constructor runs early

void setup() {
  // Check if early allocation worked
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n");
  Serial.println("========================================");
  Serial.println("EARLY ALLOCATION TEST");
  Serial.println("========================================");
  
  if (g_early_tensor_arena) {
    Serial.printf("EARLY ALLOCATION SUCCESS: %zu bytes (%.2f KB) at %p\n", 
                  g_early_arena_size, g_early_arena_size / 1024.0, g_early_tensor_arena);
  } else {
    Serial.println("EARLY ALLOCATION FAILED");
    return;
  }
  
  MemStats initial;
  measure_memory("After Serial.begin()", initial);
  
  if (!validate_model_buffer(g_model, g_model_len)) {
    Serial.println("Model validation failed!");
    return;
  }
  
  Serial.printf("Model size: %u bytes\n", g_model_len);
  
  // Use early allocated arena
  Serial.println("\n\n========================================");
  Serial.println("TESTING WITH EARLY ALLOCATED ARENA");
  Serial.println("========================================");
  
  // Register operators
  MicroMutableOpResolver<19> resolver;
  if (!register_operators(&resolver)) {
    Serial.println("Operator registration failed!");
    return;
  }
  
  // Create interpreter with early allocated arena
  const Model* model = GetModel(g_model);
  if (!model) {
    Serial.println("Failed to parse model");
    return;
  }
  
  MicroInterpreter interpreter(model, resolver, g_early_tensor_arena, g_early_arena_size);
  
  MemStats before_allocate;
  measure_memory("Before AllocateTensors()", before_allocate);
  
  // Allocate tensors
  Serial.println("Calling AllocateTensors()...");
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  
  MemStats after_allocate;
  measure_memory("After AllocateTensors()", after_allocate);
  
  if (allocate_status != kTfLiteOk) {
    Serial.printf("FAILED: AllocateTensors() returned error %d\n", allocate_status);
    Serial.printf("Arena used: %zu bytes\n", interpreter.arena_used_bytes());
    Serial.printf("Arena size: %zu bytes\n", g_early_arena_size);
    return;
  }
  
  Serial.printf("SUCCESS: AllocateTensors() completed!\n");
  Serial.printf("Arena used: %zu bytes (%.2f KB)\n", interpreter.arena_used_bytes(), interpreter.arena_used_bytes() / 1024.0);
  Serial.printf("Arena remaining: %zu bytes (%.2f KB)\n", g_early_arena_size - interpreter.arena_used_bytes(), (g_early_arena_size - interpreter.arena_used_bytes()) / 1024.0);
  
  // Validate I/O
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  
  Serial.printf("Input:  type=%d, dims=[", input->type);
  for (int i = 0; i < input->dims->size; i++) {
    Serial.printf("%d%s", input->dims->data[i], (i < input->dims->size - 1) ? "," : "");
  }
  Serial.printf("], bytes=%d\n", input->bytes);
  
  Serial.printf("Output: type=%d, dims=[", output->type);
  for (int i = 0; i < output->dims->size; i++) {
    Serial.printf("%d%s", output->dims->data[i], (i < output->dims->size - 1) ? "," : "");
  }
  Serial.printf("], bytes=%d\n", output->bytes);
  
  Serial.println("\n\n=== TEST COMPLETE ===");
}

void loop() {
  delay(10000);
}