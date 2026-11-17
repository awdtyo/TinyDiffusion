/**
 * ESP32 Memory Feasibility Test for Tiny U-Net v2
 * Tests actual TFLM tensor arena allocation for v2 FLOAT32 model
 */

#include <Arduino.h>
#include "esp_heap_caps.h"

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model data - generated from model_float32.tflite
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

bool register_operators(MicroMutableOpResolver<4>* resolver) {
  if (resolver->AddConv2D() != kTfLiteOk) return false;
  if (resolver->AddAdd() != kTfLiteOk) return false;
  if (resolver->AddResizeBilinear() != kTfLiteOk) return false;
  if (resolver->AddRelu() != kTfLiteOk) return false;
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n");
  Serial.println("========================================");
  Serial.println("Tiny U-Net v2 ESP32 Memory Feasibility Test");
  Serial.println("========================================");
  
  MemStats initial;
  measure_memory("Initial (after Serial.begin)", initial);
  
  if (!validate_model_buffer(g_model, g_model_len)) {
    Serial.println("Model validation failed!");
    return;
  }
  
  Serial.printf("Model size: %u bytes (%.1f KB)\n", g_model_len, g_model_len / 1024.0);
  
  // Test different arena sizes
  size_t test_sizes_kb[] = {60, 70, 80, 90, 100, 110, 120};
  size_t num_tests = sizeof(test_sizes_kb) / sizeof(test_sizes_kb[0]);
  
  for (size_t i = 0; i < num_tests; i++) {
    size_t arena_size_kb = test_sizes_kb[i];
    size_t arena_size = arena_size_kb * 1024;
    
    Serial.printf("\n\n========================================\n");
    Serial.printf("TESTING WITH %zu KB ARENA\n", arena_size_kb);
    Serial.printf("========================================\n");
    
    MemStats before_arena;
    measure_memory("Before arena allocation", before_arena);
    
    uint8_t* tensor_arena = (uint8_t*)heap_caps_malloc(arena_size, MALLOC_CAP_8BIT);
    
    if (!tensor_arena) {
      Serial.printf("FAILED: Could not allocate %zu KB arena\n", arena_size_kb);
      continue;
    }
    
    MemStats after_arena;
    measure_memory("After arena allocation", after_arena);
    
    // Validate model
    if (!validate_model_buffer(g_model, g_model_len)) {
      Serial.println("Model validation failed!");
      heap_caps_free(tensor_arena);
      continue;
    }
    
    // Register operators
    MicroMutableOpResolver<4> resolver;
    if (!register_operators(&resolver)) {
      Serial.println("Operator registration failed!");
      heap_caps_free(tensor_arena);
      continue;
    }
    
    // Create interpreter
    const Model* model = GetModel(g_model);
    if (!model) {
      Serial.println("Failed to parse model");
      heap_caps_free(tensor_arena);
      continue;
    }
    
    MicroInterpreter interpreter(model, resolver, tensor_arena, arena_size);
    
    MemStats before_allocate;
    measure_memory("Before AllocateTensors()", before_allocate);
    
    // Allocate tensors
    Serial.println("Calling AllocateTensors()...");
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    
    MemStats after_allocate;
    measure_memory("After AllocateTensors()", after_allocate);
    
    if (allocate_status != kTfLiteOk) {
      Serial.printf("FAILED: AllocateTensors() returned error %d\n", (int)allocate_status);
      Serial.printf("  Arena used: %zu bytes (%.1f KB)\n", interpreter.arena_used_bytes(), interpreter.arena_used_bytes() / 1024.0);
      heap_caps_free(tensor_arena);
      continue;
    }
    
    Serial.printf("SUCCESS: AllocateTensors() completed!\n");
    Serial.printf("  Arena used: %zu bytes (%.1f KB)\n", interpreter.arena_used_bytes(), interpreter.arena_used_bytes() / 1024.0);
    Serial.printf("  Arena remaining: %zu bytes (%.1f KB)\n", arena_size - interpreter.arena_used_bytes(), (arena_size - interpreter.arena_used_bytes()) / 1024.0);
    Serial.printf("  Free heap after AllocateTensors: %zu bytes (%.1f KB)\n", after_allocate.free_heap, after_allocate.free_heap / 1024.0);
    Serial.printf("  Largest free block after: %zu bytes (%.1f KB)\n", after_allocate.largest_block, after_allocate.largest_block / 1024.0);
    
    // Test single inference
    Serial.println("Running test inference...");
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    if (input && output) {
      // Fill input with test data
      float test_val = 0.5f;
      for (int i = 0; i < input->bytes / sizeof(float); i++) {
        input->data.f[i] = test_val;
      }
      
      TfLiteStatus invoke_status = interpreter.Invoke();
      if (invoke_status == kTfLiteOk) {
        Serial.printf("Inference SUCCESS! Output range: [%.3f, %.3f]\n", 
                      output->data.f[0], output->data.f[output->bytes/sizeof(float)-1]);
      } else {
        Serial.println("Inference FAILED!");
      }
    }
    
    heap_caps_free(tensor_arena);
    delay(200);
  }
  
  Serial.println("\n\n=== ALL TESTS COMPLETE ===");
}

void loop() {
  delay(10000);
}