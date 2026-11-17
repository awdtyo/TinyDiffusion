/**
 * TinyDiffusion-ESP32 Configuration Header
 * 
 * Hardware: ESP32 DevKit V1 (ESP32-WROOM-32)
 * Display: SSD1306 128x64 OLED via I2C
 * Inference: TensorFlow Lite Micro + ESP-NN
 */

#ifndef CONFIG_H
#define CONFIG_H

// ==================== Model Configuration ====================
#define MODEL_INPUT_HEIGHT        28
#define MODEL_INPUT_WIDTH         28
#define MODEL_INPUT_CHANNELS      12   // 1 image + 10 class maps + 1 timestep map
#define MODEL_OUTPUT_CHANNELS     1
#define MODEL_OUTPUT_HEIGHT       28
#define MODEL_OUTPUT_WIDTH        28

// Tensor arena size (bytes) - adjust based on model size
// Model ~12KB, tensors ~40KB, workspace ~10KB = ~62KB minimum
#define TENSOR_ARENA_SIZE         70000

// Default sampling steps for DDIM
#define DEFAULT_SAMPLING_STEPS    8
#define MAX_SAMPLING_STEPS        32

// ==================== Display Configuration ====================
#define DISPLAY_WIDTH             128
#define DISPLAY_HEIGHT            64
#define DISPLAY_I2C_ADDR          0x3C
#define DISPLAY_SDA_PIN           21
#define DISPLAY_SCL_PIN           22
#define DISPLAY_SCALE_FACTOR      2   // 28x28 -> 56x56 on 128x64 display

// Display buffer size
#define DISPLAY_BUFFER_SIZE       (DISPLAY_WIDTH * DISPLAY_HEIGHT / 8)

// ==================== Diffusion Configuration ====================
#define NUM_TRAINING_TIMESTEPS    100
#define BETA_START                0.0001f
#define BETA_END                  0.02f

// DDIM eta parameter (0 = deterministic, 1 = stochastic)
#define DDIM_ETA                  0.0f

// ==================== Serial Configuration ====================
#define SERIAL_BAUD_RATE          115200
#define SERIAL_BUFFER_SIZE        256

// ==================== Memory Configuration ====================
// ESP32 has ~520KB SRAM, no PSRAM
// Reserve ~100KB for system, ~70KB for tensor arena, ~10KB for buffers
// Leaves ~340KB for other allocations

// ==================== Timing Configuration ====================
#define RNG_SAMPLES_PER_BYTE      1   // Hardware RNG samples per byte

// ==================== Feature Flags ====================
#define ENABLE_SERIAL_COMMANDS    1
#define ENABLE_PERFORMANCE_METRICS 1
#define ENABLE_MODEL_INFO         1
#define USE_HW_RNG                1

// ==================== Debug Configuration ====================
#if defined(CORE_DEBUG_LEVEL) && CORE_DEBUG_LEVEL > 0
  #define DEBUG_PRINT(x)   Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
  #define DEBUG_PRINTF(...) Serial.printf(__VA_ARGS__)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
  #define DEBUG_PRINTF(...)
#endif

// ==================== Error Handling ====================
#define CHECK_TFLITE(x) do { \
  TfLiteStatus status = (x); \
  if (status != kTfLiteOk) { \
    DEBUG_PRINTF("TFLite error at %s:%d: %d\\n", __FILE__, __LINE__, status); \
    return false; \
  } \
} while(0)

#endif // CONFIG_H