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
#include <cmath>
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

// Temporary per-step sampling diagnostics. Set to 0 once the numerical path
// has been validated on hardware.
#define PRINT_STEP_STATS 1
// Print the final image as a 28x28 numeric grid (diagnostic only).
#define PRINT_FINAL_NUMERIC_GRID 1

// ==================== Deterministic noise (replay support) ====================
// `SEED <n>` with n != 0 replaces esp_random() with a xorshift32 stream so the
// same x_T can be reproduced exactly in the Python reference.  `SEED 0`
// restores the hardware RNG.  The algorithm below is mirrored by
// scripts/replay_reference.py.
static bool g_noise_deterministic = false;
static uint32_t g_noise_seed = 1;
static uint32_t g_noise_state = 1;
static bool g_has_spare = false;
static float g_spare = 0.0f;

static inline uint32_t xorshift32(uint32_t* state) {
  uint32_t s = *state;
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  *state = s;
  return s;
}

// ==================== Helper Functions ====================

/**
 * Generate random Gaussian noise (Box-Muller transform).
 * Uses xorshift32 when a replay seed is active, otherwise the hardware RNG.
 */
float generate_gaussian_noise() {
  if (g_has_spare) {
    g_has_spare = false;
    return g_spare;
  }

  uint32_t u1;
  uint32_t u2;
  if (g_noise_deterministic) {
    u1 = xorshift32(&g_noise_state);
    u2 = xorshift32(&g_noise_state);
  } else {
#if USE_HW_RNG
    u1 = esp_random();
    u2 = esp_random();
#else
    u1 = 0;
    u2 = 0;
#endif
  }

  float f1 = (u1 & 0xFFFFFF) / 16777216.0f;
  float f2 = (u2 & 0xFFFFFF) / 16777216.0f;
  if (f1 == 0.0f) f1 = 1e-6f;

  const float radius = sqrtf(-2.0f * logf(f1));
  const float z0 = radius * cosf(2.0f * M_PI * f2);
  g_spare = radius * sinf(2.0f * M_PI * f2);
  g_has_spare = true;
  return z0;
}

/**
 * Initialize x_t with pure Gaussian noise
 */
void init_noise_image(float* x_t) {
  const int size = MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS;
  if (g_noise_deterministic) {
    g_noise_state = g_noise_seed;
    g_has_spare = false;
    Serial.printf("Noise RNG: deterministic, seed=%lu, xorshift32+BoxMuller\n",
                  (unsigned long)g_noise_seed);
  } else {
    g_has_spare = false;
    Serial.println("Noise RNG: hardware esp_random(), BoxMuller");
  }
  for (int i = 0; i < size; i++) {
    x_t[i] = generate_gaussian_noise();
  }
}

/**
 * Quantize one float conditioning value exactly like Python's quantize_int8():
 *   q = round(value / scale + zero_point), then clip to [-128, 127]
 */
static inline int8_t quantize_model_input(float value) {
  const float scale = model_runner.get_input_scale();
  const int32_t zero_point = model_runner.get_input_zero_point();
  long q = lroundf(value / scale + (float)zero_point);
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

struct FloatStats {
  float min_value;
  float max_value;
  float mean;
  float stddev;
};

static FloatStats compute_stats(const float* data, int count) {
  FloatStats s = {0.0f, 0.0f, 0.0f, 0.0f};
  if (count <= 0) return s;

  s.min_value = data[0];
  s.max_value = data[0];
  double sum = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < count; i++) {
    const float v = data[i];
    if (v < s.min_value) s.min_value = v;
    if (v > s.max_value) s.max_value = v;
    sum += (double)v;
    sum_sq += (double)v * (double)v;
  }
  s.mean = (float)(sum / (double)count);
  double variance = sum_sq / (double)count - (double)s.mean * (double)s.mean;
  if (variance < 0.0) variance = 0.0;
  s.stddev = (float)sqrt(variance);
  return s;
}

static void print_tensor_stats(const char* label, const float* data, int count) {
  const FloatStats s = compute_stats(data, count);
  Serial.printf("  %-10s min=%+.4f max=%+.4f mean=%+.4f std=%.4f\n",
                label, s.min_value, s.max_value, s.mean, s.stddev);
}

/**
 * Prepare model input from x_t, class_label, and timestep
 * 
 * Input layout: [1, 28, 28, 12] NHWC
 *   0:    noisy image, trained in [-1, 1]
 *   1-10: spatially broadcast one-hot class maps in {0.0, 1.0}
 *   11:   spatially broadcast timestep t / NUM_TRAINING_TIMESTEPS
 * All channels are quantized with the input tensor's runtime scale/zp.
 */
void prepare_model_input(const float* x_t, int class_label, int timestep, int8_t* model_input) {
  const int num_pixels = MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH;

  // Same 0/1 representation used by tf.one_hot() during training.
  const int8_t class_on = quantize_model_input(1.0f);
  const int8_t class_off = quantize_model_input(0.0f);

  // Timestep normalization must match NoiseSchedule.get_timestep_embedding():
  // t / num_timesteps, without any extra [-1, 1] rescaling.
  const float t_norm = (float)timestep / (float)NUM_TRAINING_TIMESTEPS;
  const int8_t t_q = quantize_model_input(t_norm);

  for (int i = 0; i < num_pixels; i++) {
    int8_t* pixel = &model_input[i * MODEL_INPUT_CHANNELS];

    // 1. Noisy image channel
    pixel[0] = quantize_model_input(x_t[i]);

    // 2. Class conditioning maps (10 channels), 0/1 as in training
    for (int c = 0; c < 10; c++) {
      pixel[1 + c] = (c == class_label) ? class_on : class_off;
    }

    // 3. Timestep map
    pixel[11] = t_q;
  }
}

/**
 * Print the runtime quantization parameters and compare them with the values
 * compiled into model_data.cpp.  The firmware always runs with the tensor
 * metadata; the header constants exist only as a cross-check.
 */
static void print_quantization_metadata() {
  Serial.printf("  runtime tensors: input scale=%.9f zp=%ld | output scale=%.9f zp=%ld\n",
                model_runner.get_input_scale(),
                (long)model_runner.get_input_zero_point(),
                model_runner.get_output_scale(),
                (long)model_runner.get_output_zero_point());
  Serial.printf("  embedded header: input scale=%.9f zp=%ld | output scale=%.9f zp=%ld\n",
                g_model_input_scale, (long)g_model_input_zero_point,
                g_model_output_scale, (long)g_model_output_zero_point);
}

/**
 * Dump all 12 input channels for representative pixels: the float value that
 * the training pipeline would use, the unclipped rounding result, the clipped
 * INT8 code, and the value actually written to the tensor buffer.
 */
static void print_input_quantization(const float* x_t, int class_label, int timestep) {
  const float scale = model_runner.get_input_scale();
  const int32_t zp = model_runner.get_input_zero_point();
  const float t_norm = (float)timestep / (float)NUM_TRAINING_TIMESTEPS;

  Serial.printf("  input tensor NHWC [1,%d,%d,%d], pixel stride=%d\n",
                MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS,
                MODEL_INPUT_CHANNELS);
  Serial.printf("  formula: q = round(v / %.9f) + (%ld), clipped to [-128,127]\n",
                scale, (long)zp);
  Serial.printf("  timestep %d -> v = %d / %d = %.6f\n",
                timestep, timestep, NUM_TRAINING_TIMESTEPS, t_norm);

  const int probes[2] = {0,
                         (MODEL_INPUT_HEIGHT / 2) * MODEL_INPUT_WIDTH +
                             MODEL_INPUT_WIDTH / 2};
  for (int p = 0; p < 2; p++) {
    const int idx = probes[p];
    const int y = idx / MODEL_INPUT_WIDTH;
    const int x = idx % MODEL_INPUT_WIDTH;
    Serial.printf("  pixel(y=%d,x=%d) buffer offset %d:\n", y, x,
                  idx * MODEL_INPUT_CHANNELS);
    for (int c = 0; c < MODEL_INPUT_CHANNELS; c++) {
      float v;
      if (c == 0) {
        v = x_t[idx];
      } else if (c == 11) {
        v = t_norm;
      } else {
        v = ((c - 1) == class_label) ? 1.0f : 0.0f;
      }

      const long raw = lroundf(v / scale + (float)zp);
      int q = (int)raw;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      const int stored = model_input_int8[idx * MODEL_INPUT_CHANNELS + c];
      Serial.printf("    ch%02d v=%+.6f raw=%+ld q=%+4d stored=%+4d %s\n",
                    c, v, raw, q, stored, (q == stored) ? "OK" : "MISMATCH");
    }
  }

  // Image channel range and its dequantized reconstruction.
  const int num_pixels = MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH;
  int imin = 0;
  int imax = 0;
  for (int i = 1; i < num_pixels; i++) {
    if (x_t[i] < x_t[imin]) imin = i;
    if (x_t[i] > x_t[imax]) imax = i;
  }
  const int qmin = model_input_int8[imin * MODEL_INPUT_CHANNELS];
  const int qmax = model_input_int8[imax * MODEL_INPUT_CHANNELS];
  Serial.printf("  image min v=%+.4f (pixel %d) q=%+d dequant=%+.4f\n",
                x_t[imin], imin, qmin, (qmin - zp) * scale);
  Serial.printf("  image max v=%+.4f (pixel %d) q=%+d dequant=%+.4f\n",
                x_t[imax], imax, qmax, (qmax - zp) * scale);
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
  const int image_size =
      MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS;
  
  // Get DDIM timesteps
  int n_steps = get_ddim_timesteps(num_steps, ddim_timesteps);
  
  // Initialize x_T with pure noise
  init_noise_image(x_t_float);
  
  // Show progress on OLED
  oled_draw_progress(0.0f, "Generating...");
  
#if PRINT_STEP_STATS
  Serial.printf("\n=== DDIM start: class=%d steps=%d eta=%.2f ===\n",
                class_label, n_steps, (double)eta);
  Serial.printf("Timesteps:");
  for (int i = 0; i < n_steps; i++) {
    Serial.printf(" %d", ddim_timesteps[i]);
  }
  Serial.println();
  print_tensor_stats("x_T", x_t_float, image_size);
  print_quantization_metadata();
#endif
  
  // Denoising loop
  for (int step = 0; step < n_steps; step++) {
    int t = ddim_timesteps[step];
    int t_prev = (step + 1 < n_steps) ? ddim_timesteps[step + 1] : -1;
    
    // Update progress
    float progress = (float)(step + 1) / n_steps;
    oled_draw_progress(progress, "Denoising...");
    
    // Prepare model input
    prepare_model_input(x_t_float, class_label, t, model_input_int8);
    
#if PRINT_STEP_STATS
    Serial.printf("Step %d/%d: t=%d t_prev=%d\n", step + 1, n_steps, t, t_prev);
    Serial.printf("  input q: image=%d class_ch0=%d class_ch1=%d timestep=%d (target=%d)\n",
                  model_input_int8[0], model_input_int8[1],
                  model_input_int8[2], model_input_int8[11], class_label);
    if (step == 0) {
      print_input_quantization(x_t_float, class_label, t);
    }
    print_tensor_stats("x_t", x_t_float, image_size);
#endif
    
    // Run inference
    if (!model_runner.run_inference(model_input_int8, model_output_int8)) {
      Serial.println("ERROR: Inference failed");
      return 0;
    }
    
    // Dequantize predicted noise
    dequantize_output(model_output_int8, noise_pred_float);
    
#if PRINT_STEP_STATS
    print_tensor_stats("noise_pred", noise_pred_float, image_size);
#endif
    
    // DDIM step
    ddim_step(x_t_float, noise_pred_float, t, t_prev, eta, x_t_prev_float);
    
#if PRINT_STEP_STATS
    print_tensor_stats("x_next", x_t_prev_float, image_size);
#endif
    
    // Copy for next iteration
    memcpy(x_t_float, x_t_prev_float, sizeof(x_t_float));
  }
  
  // Final result is in x_t_float
  memcpy(output_buffer, x_t_float, sizeof(x_t_float));
  
#if PRINT_STEP_STATS
  int above_threshold = 0;
  for (int i = 0; i < image_size; i++) {
    if (x_t_float[i] >= 0.5f) above_threshold++;
  }
  print_tensor_stats("FINAL", x_t_float, image_size);
  Serial.printf("FINAL pixels >= 0.5: %d / %d (output_buffer=%p)\n",
                above_threshold, image_size, (void*)output_buffer);

  // ASCII rendering of exactly what oled_draw_digit() thresholds: a '#'
  // means the OLED pixel is lit, '+' is a negative value, ' ' a clear pixel.
  Serial.println("FINAL bitmap as OLED renders it ('#'=lit, '+'<0, ' '>=0):");
  for (int y = 0; y < MODEL_OUTPUT_HEIGHT; y++) {
    char row[MODEL_OUTPUT_WIDTH + 1];
    for (int x = 0; x < MODEL_OUTPUT_WIDTH; x++) {
      const float v = x_t_float[y * MODEL_OUTPUT_WIDTH + x];
      row[x] = (v >= 0.5f) ? '#' : ((v < 0.0f) ? '+' : ' ');
    }
    row[MODEL_OUTPUT_WIDTH] = '\0';
    Serial.println(row);
  }

#if PRINT_FINAL_NUMERIC_GRID
  // Exact float values handed to the OLED, so the bitmap can be verified
  // against the numbers instead of a photograph.  One write per row keeps the
  // UART stream compact.
  Serial.println("FINAL numeric grid (row-major, 2 decimals):");
  for (int y = 0; y < MODEL_OUTPUT_HEIGHT; y++) {
    char line[MODEL_OUTPUT_WIDTH * 7 + 1];
    int used = 0;
    for (int x = 0; x < MODEL_OUTPUT_WIDTH; x++) {
      used += snprintf(line + used, sizeof(line) - used, "%+6.2f",
                       x_t_float[y * MODEL_OUTPUT_WIDTH + x]);
    }
    line[used] = '\0';
    Serial.println(line);
  }
#endif
#endif
  
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
        Serial.println("OLED: rendering final post-DDIM sampled image");
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
      Serial.println("OLED: rendering final post-DDIM sampled image");
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
    Serial.printf("Noise RNG: %s seed=%lu\n",
                  g_noise_deterministic ? "deterministic" : "hardware",
                  (unsigned long)g_noise_seed);
    Serial.println("================================");
  }
  else if (cmd.startsWith("SEED ")) {
    // Deterministic noise for Python <-> firmware replay.
    const long seed = cmd.substring(5).toInt();
    if (seed > 0) {
      g_noise_seed = (uint32_t)seed;
      g_noise_deterministic = true;
      Serial.printf("OK: deterministic noise seed=%lu\n", (unsigned long)g_noise_seed);
    } else {
      g_noise_deterministic = false;
      Serial.println("OK: noise RNG restored to esp_random()");
    }
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
    Serial.println("SEED <n>   - Deterministic noise seed for replay (0 = hardware RNG)");
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
