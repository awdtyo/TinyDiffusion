/**
 * TinyDiffusion-ESP32 FLOAT32 Test Harness
 * 
 * Deterministic single-inference test using fixed test vectors
 * for numerical parity validation against Python reference.
 */

#include <Arduino.h>
#include "model_runner.h"
#include "model_data.h"
#include "test_vectors.h"

// Test configuration
#define TEST_INPUT_ELEMENTS  9408
#define TEST_OUTPUT_ELEMENTS 784

// Expected reference values from Python
static constexpr float kExpectedSum = kExpectedOutputSum;
static constexpr float kExpectedMin = kExpectedOutputMin;
static constexpr float kExpectedMax = kExpectedOutputMax;

// Output buffer
float g_output[TEST_OUTPUT_ELEMENTS];

// Timing
uint32_t g_inference_start_us = 0;
uint32_t g_inference_end_us = 0;

// Global model runner
ModelRunner g_model_runner;

void print_memory_info() {
  Serial.printf("Free heap: %u bytes (%.2f KB)\n", ESP.getFreeHeap(), ESP.getFreeHeap() / 1024.0);
  Serial.printf("Largest free block: %u bytes (%.2f KB)\n", ESP.getMaxAllocHeap(), ESP.getMaxAllocHeap() / 1024.0);
  Serial.printf("Free PSRAM: %u bytes\n", ESP.getFreePsram());
}

void print_output_stats(const float* output, int elements) {
  float min_val = output[0];
  float max_val = output[0];
  float sum = 0.0f;
  
  for (int i = 0; i < elements; i++) {
    float v = output[i];
    if (v < min_val) min_val = v;
    if (v > max_val) max_val = v;
    sum += v;
  }
  
  float mean = sum / elements;
  
  Serial.printf("Output stats:\n");
  Serial.printf("  Min: %.6f\n", min_val);
  Serial.printf("  Max: %.6f\n", max_val);
  Serial.printf("  Mean: %.6f\n", mean);
  Serial.printf("  Sum: %.6f\n", sum);
  Serial.printf("  Expected Sum: %.6f\n", kExpectedSum);
  Serial.printf("  Sum diff: %.6f\n", sum - kExpectedSum);
}

bool compare_with_reference(const float* output, const float* reference, int elements) {
  float max_abs_error = 0.0f;
  float mae = 0.0f;
  float mse = 0.0f;
  int max_error_idx = 0;
  
  for (int i = 0; i < elements; i++) {
    float diff = output[i] - reference[i];
    float abs_diff = fabsf(diff);
    if (abs_diff > max_abs_error) {
      max_abs_error = abs_diff;
      max_error_idx = i;
    }
    mae += abs_diff;
    mse += diff * diff;
  }
  
  mae /= elements;
  mse /= elements;
  
  // Cosine similarity
  float dot = 0.0f, norm_out = 0.0f, norm_ref = 0.0f;
  for (int i = 0; i < elements; i++) {
    dot += output[i] * reference[i];
    norm_out += output[i] * output[i];
    norm_ref += reference[i] * reference[i];
  }
  float cos_sim = dot / (sqrtf(norm_out) * sqrtf(norm_ref) + 1e-8f);
  
  Serial.printf("Numerical comparison:\n");
  Serial.printf("  Max absolute error: %.6f (at index %d)\n", max_abs_error, max_error_idx);
  Serial.printf("  Mean absolute error: %.6f\n", mae);
  Serial.printf("  MSE: %.6f\n", mse);
  Serial.printf("  Cosine similarity: %.8f\n", cos_sim);
  
  // Print first few values for manual inspection
  Serial.printf("  First 10 output values: ");
  for (int i = 0; i < 10; i++) {
    Serial.printf("%.4f ", output[i]);
  }
  Serial.printf("\n");
  Serial.printf("  First 10 reference values: ");
  for (int i = 0; i < 10; i++) {
    Serial.printf("%.4f ", reference[i]);
  }
  Serial.printf("\n");
  
  return max_abs_error < 1e-4f && cos_sim > 0.9999f;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n");
  Serial.println("============================================");
  Serial.println("TinyDiffusion FLOAT32 ESP32 Test Harness");
  Serial.println("============================================");
  
  print_memory_info();
  
  // Initialize model
  Serial.println("\nInitializing model...");
  if (!g_model_runner.begin()) {
    Serial.println("FATAL: Model initialization failed!");
    while (1) { delay(1000); }
  }
  
  g_model_runner.print_model_info();
  print_memory_info();
  
  // Verify test vectors
  Serial.printf("\nTest vector sizes:\n");
  Serial.printf("  Input: %d elements (expected %d)\n", 
                (int)(sizeof(g_test_input) / sizeof(g_test_input[0])), TEST_INPUT_ELEMENTS);
  Serial.printf("  Reference output: %d elements (expected %d)\n", 
                (int)(sizeof(g_test_output_ref) / sizeof(g_test_output_ref[0])), TEST_OUTPUT_ELEMENTS);
  
  // Run inference
  Serial.println("\nRunning inference...");
  g_inference_start_us = micros();
  
  bool success = g_model_runner.run_inference(g_test_input, g_output);
  
  g_inference_end_us = micros();
  
  if (!success) {
    Serial.println("ERROR: Inference failed!");
    while (1) { delay(1000); }
  }
  
  uint32_t inference_time_us = g_inference_end_us - g_inference_start_us;
  Serial.printf("Inference time: %u us (%.2f ms)\n", 
                inference_time_us, inference_time_us / 1000.0);
  
  // Print output statistics
  print_output_stats(g_output, TEST_OUTPUT_ELEMENTS);
  print_memory_info();
  
  // Compare with reference
  Serial.println("\nComparing with Python reference...");
  bool parity_ok = compare_with_reference(g_output, g_test_output_ref, TEST_OUTPUT_ELEMENTS);
  
  Serial.println("\n============================================");
  if (parity_ok) {
    Serial.println("RESULT: PARITY TEST PASSED");
    Serial.println("FLOAT32 model output matches Python reference within tolerance.");
  } else {
    Serial.println("RESULT: PARITY TEST FAILED");
    Serial.println("Output differs from Python reference beyond acceptable tolerance.");
  }
  Serial.println("============================================\n");
}

void loop() {
  // Nothing to do in loop - single test run
  delay(10000);
}