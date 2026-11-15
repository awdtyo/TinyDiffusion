/**
 * TinyDiffusion-ESP32 Main Firmware
 * 
 * ESP32 DevKit V1 + SSD1306 OLED
 * TensorFlow Lite Micro + ESP-NN
 * Conditional diffusion model for MNIST digit generation
 * 
 * Hardware:
 * - ESP32 DevKit V1 (ESP32-WROOM-32)
 * - SSD1306 128x64 OLED (I2C: SDA=21, SCL=22)
 * 
 * Serial Commands (115200 baud):
 *   G <digit>  - Generate specific digit (0-9)
 *   R          - Generate random digit
 *   INFO       - Show model and memory info
 *   HELP       - Show commands
 */

#include <Arduino.h>
#include <cstring>
#include <esp_random.h>
#include "config.h"
#include "model_data.h"
#include "model_runner.h"
#include "diffusion.h"
#include "oled.h"

// ==================== Global Objects ====================
ModelRunner model_runner;

// ==================== Buffers ====================
// Reused buffers to avoid dynamic allocation during generation
static int8_t model_input_int8[MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH * MODEL_INPUT_CHANNELS];  // 9408
static int8_t model_output_int8[MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS];  // 784
static float model_output_float[MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS];  // 784
static float x_t_float[MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS];  // 784
static float noise_pred_float[MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS];  // 784
static float x_t_prev_float[MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS];  // 784

// DDIM timestep buffer
static int ddim_timesteps[MAX_SAMPLING_STEPS];

// ==================== State ====================
static int current_sampling_steps = DEFAULT_SAMPLING_STEPS;
static float current_eta = DDIM_ETA;

// ==================== Helper Functions ====================

/**
 * Generate random Gaussian noise using hardware RNG
 * Box-Muller transform
 */
float generate_gaussian_noise() {
  static bool has_spare = false;
  static float spare;
  
  if (has_spare) {
    has_spare = false;
    return spare;
  }
  
  #if USE_HW_RNG
    uint32_t u1 = esp_random();
    uint32_t u2 = esp_random();
    float f1 = (u1 & 0xFFFFFF) / 16777216.0f;
    float f2 = (u2 & 0xFFFFFF) / 16777216.0f;
    if (f1 == 0.0f) f1 = 1e-6f;
    float z0 = sqrtf(-2.0f * logf(f1)) * cosf(2.0f * M_PI * f2);
    spare = sqrtf(-2.0f * logf(f1)) * sinf(2.0f * M_PI * f2);
    has_spare = true;
    return z0;
  #else
    // Deterministic fallback - not truly random
    return 0.0f;
  #endif
}

/**
 * Initialize x_t with pure Gaussian noise
 */
void init_noise_image(float* x_t) {
  const int size = MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS;
  for (int i = 0; i < size; i++) {
    x_t[i] = generate_gaussian_noise();
  }
}

/**
 * Prepare model input from x_t, class_label, and timestep
 * 
 * Input layout: [28, 28, 12] = [image(1) + class_maps(10) + timestep(1)]
 * All quantized to INT8
 */
void prepare_model_input(const float* x_t, int class_label, int timestep, int8_t* model_input) {
  const int H = MODEL_INPUT_HEIGHT;
  const int W = MODEL_INPUT_WIDTH;
  const int num_pixels = H * W;
  
  // Channel layout:
  // 0: noisy image
  // 1-10: class one-hot maps (10 channels)
  // 11: timestep map
  
  // 1. Noisy image channel (quantized)
  for (int i = 0; i < num_pixels; i++) {
    float val = x_t[i];
    // Quantize: [-1, 1] -> INT8
    int8_t q = (int8_t)roundf(val / model_runner.get_input_scale() + model_runner.get_input_zero_point());
    model_input[i * MODEL_INPUT_CHANNELS + 0] = q;
  }
  
  // 2. Class conditioning maps (10 channels)
  for (int c = 0; c < 10; c++) {
    int8_t class_val = (c == class_label) ? 127 : -128;  // Approximate one-hot in INT8
    for (int i = 0; i < num_pixels; i++) {
      model_input[i * MODEL_INPUT_CHANNELS + 1 + c] = class_val;
    }
  }
  
  // 3. Timestep map (normalized to [0, 1])
  float t_norm = (float)timestep / NUM_TRAINING_TIMESTEPS;
  int8_t t_q = (int8_t)roundf((t_norm * 2.0f - 1.0f) / model_runner.get_input_scale() + model_runner.get_input_zero_point());
  for (int i = 0; i < num_pixels; i++) {
    model_input[i * MODEL_INPUT_CHANNELS + 11] = t_q;
  }
}

/**
 * Dequantize model output to float
 */
void dequantize_output(const int8_t* output_int8, float* output_float) {
  const int size = MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS;
  for (int i = 0; i < size; i++) {
    output_float[i] = (output_int8[i] - model_runner.get_output_zero_point()) * model_runner.get_output_scale();
  }
}

/**
 * One-shot inference smoke test.
 *
 * Runs exactly one forward pass with an all-zero (zero-point) input.  This
 * verifies that AllocateTensors(), the resolver, and the kernel can execute
 * before any sampling loop is attempted.
 */
bool run_dummy_inference_test() {
  const size_t input_size = MODEL_INPUT_SIZE;
  const size_t output_size = MODEL_OUTPUT_SIZE;
  
  memset(model_input_int8,
         (int)model_runner.get_input_zero_point(),
         input_size);
  
  if (!model_runner.run_inference(model_input_int8, model_output_int8)) {
    Serial.println("ERROR: Dummy inference failed");
    return false;
  }
  
  int min_value = 127;
  int max_value = -128;
  int64_t sum = 0;
  for (size_t i = 0; i < output_size; ++i) {
    const int value = model_output_int8[i];
    if (value < min_value) min_value = value;
    if (value > max_value) max_value = value;
    sum += value;
  }
  
  Serial.printf("Dummy inference OK: output INT8 min=%d max=%d mean=%.3f\n",
                min_value, max_value, (double)sum / (double)output_size);
  return true;
}

/**
 * Generate a single digit using DDIM sampling
 * 
 * @param class_label Target digit (0-9)
 * @param num_steps DDIM steps (8, 16, 32)
 * @param eta DDIM eta
 * @param output_buffer Output float image [28*28]
 * @return Generation time in milliseconds
 */
uint32_t generate_digit_local(int class_label, int num_steps, float eta, float* output_buffer) {
  uint32_t start_time = millis();
  
  // Get DDIM timesteps
  int n_steps = get_ddim_timesteps(num_steps, ddim_timesteps);
  
  // Initialize x_T with pure noise
  init_noise_image(x_t_float);
  
  // Show progress on OLED
  oled_draw_progress(0.0f, "Generating...");
  
  // Denoising loop
  for (int step = 0; step < n_steps; step++) {
    int t = ddim_timesteps[step];
    int t_prev = (step + 1 < n_steps) ? ddim_timesteps[step + 1] : -1;
    
    // Update progress
    float progress = (float)(step + 1) / n_steps;
    oled_draw_progress(progress, "Denoising...");
    
    // Prepare model input
    prepare_model_input(x_t_float, class_label, t, model_input_int8);
    
    // Run inference
    if (!model_runner.run_inference(model_input_int8, model_output_int8)) {
      DEBUG_PRINTLN("Inference failed!");
      return 0;
    }
    
    // Dequantize predicted noise
    dequantize_output(model_output_int8, noise_pred_float);
    
    // DDIM step
    ddim_step(x_t_float, noise_pred_float, t, t_prev, eta, x_t_prev_float);
    
    // Copy for next iteration
    memcpy(x_t_float, x_t_prev_float, sizeof(x_t_float));
  }
  
  // Final result is in x_t_float
  memcpy(output_buffer, x_t_float, sizeof(x_t_float));
  
  uint32_t end_time = millis();
  return end_time - start_time;
}

/**
 * Process serial commands
 */
void process_serial_command(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  
  if (cmd.startsWith("G ")) {
    // Generate specific digit
    int digit = cmd.substring(2).toInt();
    if (digit >= 0 && digit <= 9) {
      DEBUG_PRINTF("Generating digit %d...\n", digit);
      
      uint32_t gen_time = generate_digit_local(digit, current_sampling_steps, current_eta, model_output_float);
      
      if (gen_time > 0) {
        oled_draw_digit(model_output_float, digit, gen_time, current_sampling_steps);
        Serial.printf("OK: Generated digit %d in %lu ms (%d steps)\n", digit, gen_time, current_sampling_steps);
      } else {
        Serial.println("ERROR: Generation failed");
      }
    } else {
      Serial.println("ERROR: Digit must be 0-9");
    }
  }
  else if (cmd == "R") {
    // Generate random digit
    int digit = random(0, 10);
    DEBUG_PRINTF("Generating random digit %d...\n", digit);
    
    uint32_t gen_time = generate_digit_local(digit, current_sampling_steps, current_eta, model_output_float);
    
    if (gen_time > 0) {
      oled_draw_digit(model_output_float, digit, gen_time, current_sampling_steps);
      Serial.printf("OK: Generated random digit %d in %lu ms\n", digit, gen_time);
    } else {
      Serial.println("ERROR: Generation failed");
    }
  }
  else if (cmd == "INFO") {
    // Show info
    Serial.println("=== TinyDiffusion-ESP32 Info ===");
    Serial.printf("Model size: %u bytes\n", g_model_len);
    Serial.printf("Tensor arena: %zu bytes\n", model_runner.get_arena_size());
    Serial.printf("Arena used: %zu bytes\n", model_runner.get_arena_used_bytes());
    Serial.printf("Free heap: %lu bytes\n", ESP.getFreeHeap());
    Serial.printf("Largest free block: %lu bytes\n", ESP.getMaxAllocHeap());
    Serial.printf("Sampling steps: %d\n", current_sampling_steps);
    Serial.printf("DDIM eta: %.2f\n", current_eta);
    Serial.printf("Input quant: scale=%.6f, zp=%ld\n", model_runner.get_input_scale(), model_runner.get_input_zero_point());
    Serial.printf("Output quant: scale=%.6f, zp=%ld\n", model_runner.get_output_scale(), model_runner.get_output_zero_point());
    Serial.println("================================");
  }
  else if (cmd.startsWith("STEPS ")) {
    // Set sampling steps
    int steps = cmd.substring(6).toInt();
    if (steps == 8 || steps == 16 || steps == 32) {
      current_sampling_steps = steps;
      Serial.printf("OK: Sampling steps set to %d\n", steps);
    } else {
      Serial.println("ERROR: Steps must be 8, 16, or 32");
    }
  }
  else if (cmd.startsWith("ETA ")) {
    // Set DDIM eta
    float eta = cmd.substring(4).toFloat();
    if (eta >= 0.0f && eta <= 1.0f) {
      current_eta = eta;
      Serial.printf("OK: DDIM eta set to %.2f\n", eta);
    } else {
      Serial.println("ERROR: Eta must be 0.0-1.0");
    }
  }
  else if (cmd == "HELP" || cmd == "?") {
    Serial.println("=== TinyDiffusion-ESP32 Commands ===");
    Serial.println("G <0-9>    - Generate specific digit");
    Serial.println("R          - Generate random digit");
    Serial.println("STEPS <8|16|32> - Set DDIM sampling steps");
    Serial.println("ETA <0-1>  - Set DDIM eta (0=deterministic)");
    Serial.println("INFO       - Show model and memory info");
    Serial.println("HELP       - Show this help");
    Serial.println("=====================================");
  }
  else if (cmd.length() > 0) {
    Serial.println("ERROR: Unknown command. Type HELP for commands.");
  }
}

// ==================== Arduino Setup/Loop ====================

void setup() {
  // Initialize serial
  Serial.begin(SERIAL_BAUD_RATE);
  while (!Serial && millis() < 2000) {
    delay(10);
  }
  
  Serial.println("\n=== TinyDiffusion-ESP32 Starting ===");
  
  // Initialize display
  if (!oled_init()) {
    Serial.println("ERROR: Display initialization failed");
  } else {
    Serial.println("Display initialized");
  }
  
  // Initialize model runner
  Serial.println("Loading model...");
  if (!model_runner.begin()) {
    Serial.println("ERROR: Model initialization failed!");
    oled_draw_text("MODEL ERROR", "Check Serial", "Reset to retry");
    while (1) delay(1000);
  }
  
  Serial.println("Model loaded successfully");
  model_runner.print_model_info();
  
  // Smoke-test one forward pass before enabling the sampling loop.
  if (!run_dummy_inference_test()) {
    oled_draw_text("INFER ERROR", "Check Serial", "Reset to retry");
    while (1) delay(1000);
  }
  
  // Seed random for non-crypto uses
  randomSeed(esp_random());
  
  // Show ready state
  oled_draw_text("Ready", "Send: G <0-9>", "or HELP");
  
  Serial.println("=== Ready ===");
  Serial.println("Commands: G <digit>, R, STEPS, ETA, INFO, HELP");
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    process_serial_command(cmd);
  }
  
  // Small delay to prevent watchdog issues
  delay(10);
}
