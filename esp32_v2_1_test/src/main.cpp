/**
 * TinyDiffusion-ESP32 v2.1 FLOAT32 Test Harness with OLED
 * Tests the Configuration C [6,8,8,6] FLOAT32 model on ESP32-WROOM-32
 * with SSD1306 OLED display
 * 
 * Serial commands:
 *   G <digit>  - Generate digit (0-9) and display on OLED
 *   H          - Show help
 *   M          - Show memory info
 */

#include <Arduino.h>
#include <stdlib.h>
#include "esp_heap_caps.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "U8g2lib.h"
#include "Wire.h"
#include "config.h"

using namespace tflite;

// OLED display object
U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2(U8G2_R0, DISPLAY_SCL_PIN, DISPLAY_SDA_PIN, U8X8_PIN_NONE);

void oled_init() {
  Wire.begin(DISPLAY_SDA_PIN, DISPLAY_SCL_PIN);
  Wire.setClock(400000);
  u8g2.begin();
  u8g2.setFont(u8g2_font_6x10_tf);
  u8g2.setFontRefHeightExtendedText();
  u8g2.setDrawColor(1);
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_7x14_tf);
  u8g2.drawStr(10, 16, "TinyDiffusion");
  u8g2.drawStr(10, 32, "ESP32 + MNIST");
  u8g2.setFont(u8g2_font_5x7_tf);
  u8g2.drawStr(10, 48, "Conditional Diffusion");
  u8g2.drawStr(10, 56, "DDIM 8/16/32 steps");
  u8g2.sendBuffer();
}

void oled_draw_digit(const float* image, int class_label, 
                     uint32_t generation_time_ms, int sampling_steps) {
  u8g2.clearBuffer();
  
  const int src_w = MODEL_OUTPUT_WIDTH;
  const int src_h = MODEL_OUTPUT_HEIGHT;
  const int dst_w = src_w * DISPLAY_SCALE_FACTOR;
  const int dst_h = src_h * DISPLAY_SCALE_FACTOR;
  const int dst_x = (DISPLAY_WIDTH - dst_w) / 2;
  const int dst_y = 2;
  
  for (int sy = 0; sy < src_h; sy++) {
    for (int sx = 0; sx < src_w; sx++) {
      float val = image[sy * src_w + sx];
      val = (val + 1.0f) * 0.5f;
      if (val < 0.0f) val = 0.0f;
      if (val > 1.0f) val = 1.0f;
      const bool pixel_on = (val >= 0.5f);
      
      if (pixel_on) {
        int bx = dst_x + sx * DISPLAY_SCALE_FACTOR;
        int by = dst_y + sy * DISPLAY_SCALE_FACTOR;
        u8g2.drawBox(bx, by, DISPLAY_SCALE_FACTOR, DISPLAY_SCALE_FACTOR);
      }
    }
  }
  
  u8g2.setFont(u8g2_font_7x14_tf);
  char label_str[32];
  snprintf(label_str, sizeof(label_str), "Digit: %d", class_label);
  u8g2.drawStr(2, dst_y + dst_h + 16, label_str);
  
  u8g2.setFont(u8g2_font_5x7_tf);
  char info_str[64];
  snprintf(info_str, sizeof(info_str), "Time: %lu ms | Steps: %d",
           (unsigned long)generation_time_ms, sampling_steps);
  u8g2.drawStr(2, dst_y + dst_h + 28, info_str);
  
  char mem_str[64];
  snprintf(info_str, sizeof(info_str), "Free Heap: %lu KB",
           (unsigned long)(ESP.getFreeHeap() / 1024));
  u8g2.drawStr(2, dst_y + dst_h + 38, info_str);
  
  u8g2.sendBuffer();
}

void print_memory(const char* label) {
  Serial.printf("\n=== %s ===\n", label);
  Serial.printf("Free heap:       %zu bytes (%.2f KB)\n", ESP.getFreeHeap(), ESP.getFreeHeap() / 1024.0);
  Serial.printf("Largest block:   %zu bytes (%.2f KB)\n", ESP.getMaxAllocHeap(), ESP.getMaxAllocHeap() / 1024.0);
  Serial.printf("Free 8BIT:       %zu bytes (%.2f KB)\n", heap_caps_get_free_size(MALLOC_CAP_8BIT), heap_caps_get_free_size(MALLOC_CAP_8BIT) / 1024.0);
  Serial.printf("Largest 8BIT:    %zu bytes (%.2f KB)\n", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT), heap_caps_get_largest_free_block(MALLOC_CAP_8BIT) / 1024.0);
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

void print_help() {
  Serial.println("\n=== COMMANDS ===");
  Serial.println("  G <digit>  - Generate digit (0-9) and display on OLED");
  Serial.println("  H          - Show this help");
  Serial.println("  M          - Show memory info");
  Serial.println("================\n");
}

void process_command(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  
  if (cmd.startsWith("G ")) {
    int digit = cmd.substring(2).toInt();
    if (digit >= 0 && digit <= 9) {
      Serial.printf("Generating digit %d...\n", digit);
      
      // Re-initialize interpreter with current arena
      MicroMutableOpResolver<4> resolver;
      resolver.AddConv2D();
      resolver.AddAdd();
      resolver.AddResizeBilinear();
      resolver.AddRelu();
      
      const Model* model = GetModel(g_model);
      MicroInterpreter interpreter(model, resolver, tensor_arena, arena_size);
      
      TfLiteStatus allocate_status = interpreter.AllocateTensors();
      if (allocate_status != kTfLiteOk) {
        Serial.printf("AllocateTensors failed: %d\n", allocate_status);
        return;
      }
      
      TfLiteTensor* input = interpreter.input(0);
      TfLiteTensor* output = interpreter.output(0);
      
      // Fill with noise
      for (int i = 0; i < input->bytes / sizeof(float); i++) {
        input->data.f[i] = (float)(rand() % 2000 - 1000) / 1000.0f;
      }
      
      // We need to add class conditioning - for now just run inference
      uint32_t start = millis();
      TfLiteStatus status = interpreter->Invoke();
      uint32_t elapsed = millis() - start;
      
      if (status == kTfLiteOk) {
        TfLiteTensor* output = interpreter.output(0);
        
        // Display on OLED
        u8g2.clearBuffer();
        u8g2.setFont(u8g2_font_7x14_tf);
        char buf[32];
        snprintf(buf, sizeof(buf), "Digit: %d", digit);
        u8g2.drawStr(10, 16, buf);
        snprintf(buf, sizeof(buf), "Time: %lu ms", elapsed);
        u8g2.drawStr(10, 32, buf);
        u8g2.sendBuffer();
        
        Serial.printf("Generated digit %d in %lu ms\n", digit, elapsed);
        Serial.printf("Output[0]=%.3f\n", output->data.f[0]);
      } else {
        Serial.println("Inference FAILED!");
      }
    } else {
      Serial.println("Invalid digit. Use 0-9.");
    }
  } else if (cmd == "H") {
    print_help();
  } else if (cmd == "M") {
    print_memory("Current");
  } else if (cmd.length() > 0) {
    Serial.println("Unknown command. Type H for help.");
  }
}

void setup() {
  // Initialize OLED first
  Wire.begin(DISPLAY_SDA_PIN, DISPLAY_SCL_PIN);
  Wire.setClock(400000);
  
  u8g2.begin();
  u8g2.setFont(u8g2_font_6x10_tf);
  u8g2.setFontRefHeightExtendedText();
  u8g2.setDrawColor(1);
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_7x14_tf);
  u8g2.drawStr(10, 16, "TinyDiffusion");
  u8g2.drawStr(10, 32, "ESP32 + MNIST");
  u8g2.setFont(u8g2_font_5x7_tf);
  u8g2.drawStr(10, 48, "Conditional Diffusion");
  u8g2.drawStr(10, 56, "DDIM 8/16/32 steps");
  u8g2.sendBuffer();
  delay(1000);
  
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n");
  Serial.println("========================================");
  Serial.println("TinyDiffusion v2.1 Config C FLOAT32 Test");
  Serial.println("========================================");
  
  // Test with 70 KB arena (minimum that works)
  arena_size_kb = 70;
  arena_size = arena_size_kb * 1024;
  
  Serial.printf("\n\n========================================\n");
  Serial.printf("TESTING WITH %zu KB ARENA\n", arena_size_kb);
  Serial.printf("========================================\n");
  
  tensor_arena = (uint8_t*)heap_caps_malloc(arena_size * 1024, MALLOC_CAP_8BIT);
  if (!tensor_arena) {
    Serial.printf("FAILED: Could not allocate %zu KB arena\n", arena_size_kb);
    return;
  }
  
  MicroMutableOpResolver<4> resolver;
  resolver.AddConv2D();
  resolver.AddAdd();
  resolver.AddResizeBilinear();
  resolver.AddRelu();
  
  const Model* model = GetModel(g_model);
  if (!model) {
    Serial.println("Failed to parse model");
    return;
  }
  
  MicroInterpreter interpreter(model, resolver, tensor_arena, arena_size);
  
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.printf("FAILED: AllocateTensors() returned error %d\n", (int)allocate_status);
    return;
  }
  
  Serial.printf("SUCCESS: AllocateTensors() completed!\n");
  Serial.printf("  Arena used: %zu bytes (%.1f KB)\n", interpreter.arena_used_bytes(), interpreter.arena_used_bytes() / 1024.0);
  Serial.printf("  Arena remaining: %zu bytes (%.1f KB)\n", arena_size - interpreter.arena_used_bytes(), (arena_size - interpreter.arena_used_bytes()) / 1024.0);
  
  print_help();
  Serial.println("Ready. Type 'G <digit>' to generate a digit (0-9).");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    process_command(cmd);
  }
  
  delay(10);
}