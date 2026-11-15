/**
 * TinyDiffusion-ESP32 Diffusion Header
 * 
 * DDIM sampling implementation for ESP32
 * Matches Python reference implementation in src/scheduler.py
 */

#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "config.h"
#include <cstdint>

// ==================== Diffusion Schedule Constants ====================
// Pre-computed noise schedule (beta, alpha, alpha_cumprod)
// These are computed at compile time from the training configuration

// Beta schedule (linear)
static constexpr float kBetaStart = BETA_START;
static constexpr float kBetaEnd = BETA_END;
static constexpr int kNumTimesteps = NUM_TRAINING_TIMESTEPS;

// Pre-computed schedule arrays (generated from Python)
// These must match the training configuration exactly

// For memory efficiency, we store only what's needed for DDIM:
// - alphas_cumprod for the DDIM timesteps
// - sqrt(alphas_cumprod) and sqrt(1 - alphas_cumprod) for those timesteps

// DDIM timestep sequences for different step counts
// Generated from: np.arange(0, 100, 100//steps)[::-1]
static constexpr int kDDIMTimesteps8[8] = {99, 87, 74, 62, 49, 37, 24, 12};
static constexpr int kDDIMTimesteps16[16] = {99, 93, 87, 81, 74, 68, 62, 56, 49, 43, 37, 31, 24, 18, 12, 6};
static constexpr int kDDIMTimesteps32[32] = {99, 96, 93, 90, 87, 84, 81, 78, 74, 71, 68, 65, 62, 59, 56, 53, 
                                              49, 46, 43, 40, 37, 34, 31, 28, 24, 21, 18, 15, 12, 9, 6, 3};

// Pointers to timestep arrays
static constexpr int* kDDIMTimestepsPtr[] = {nullptr, nullptr, kDDIMTimesteps8, nullptr, nullptr, 
                                              nullptr, nullptr, nullptr, nullptr, nullptr,
                                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                              kDDIMTimesteps16};
static constexpr int kDDIMTimestepsLengths[] = {0, 0, 8, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 16};

// Alphas cumprod for all 100 timesteps (linear schedule)
// Computed from: betas = linspace(0.0001, 0.02, 100); alphas = 1 - betas; cumprod(alphas)
// These values MUST match the Python training exactly
extern const float g_alphas_cumprod[NUM_TRAINING_TIMESTEPS];
extern const float g_sqrt_alphas_cumprod[NUM_TRAINING_TIMESTEPS];
extern const float g_sqrt_one_minus_alphas_cumprod[NUM_TRAINING_TIMESTEPS];

// ==================== DDIM Sampling Functions ====================

/**
 * Perform a single DDIM denoising step
 * 
 * @param x_t Current noisy image [28, 28, 1] (float32, range [-1, 1])
 * @param noise_pred Predicted noise [28, 28, 1] (float32)
 * @param t Current timestep index
 * @param t_prev Previous timestep index (-1 for final step)
 * @param eta DDIM eta parameter (0 = deterministic)
 * @param x_t_prev Output: denoised image [28, 28, 1]
 */
void ddim_step(const float* x_t, const float* noise_pred, 
               int t, int t_prev, float eta, float* x_t_prev);

/**
 * Generate a digit using DDIM sampling
 * 
 * @param class_label Target digit class (0-9)
 * @param num_steps Number of DDIM steps (8, 16, or 32)
 * @param eta DDIM eta parameter
 * @param output_buffer Output image buffer [28*28] float32
 * @return True on success
 */
bool generate_digit(int class_label, int num_steps, float eta, float* output_buffer);

/**
 * Get DDIM timestep array for given step count
 * 
 * @param num_steps Number of steps (8, 16, 32)
 * @param timesteps Output array (must hold at least num_steps ints)
 * @return Number of timesteps
 */
int get_ddim_timesteps(int num_steps, int* timesteps);

/**
 * Get alpha_bar (alphas_cumprod) for a timestep
 * 
 * @param t Timestep index
 * @return alpha_bar_t
 */
inline float get_alpha_bar(int t) {
  if (t < 0) return 1.0f;
  if (t >= NUM_TRAINING_TIMESTEPS) return 0.0f;
  return g_alphas_cumprod[t];
}

/**
 * Get sqrt(alpha_bar) for a timestep
 * 
 * @param t Timestep index
 * @return sqrt(alpha_bar_t)
 */
inline float get_sqrt_alpha_bar(int t) {
  if (t < 0) return 1.0f;
  if (t >= NUM_TRAINING_TIMESTEPS) return 0.0f;
  return g_sqrt_alphas_cumprod[t];
}

/**
 * Get sqrt(1 - alpha_bar) for a timestep
 * 
 * @param t Timestep index
 * @return sqrt(1 - alpha_bar_t)
 */
inline float get_sqrt_one_minus_alpha_bar(int t) {
  if (t < 0) return 0.0f;
  if (t >= NUM_TRAINING_TIMESTEPS) return 1.0f;
  return g_sqrt_one_minus_alphas_cumprod[t];
}

#endif // DIFFUSION_H