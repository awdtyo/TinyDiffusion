/**
 * TinyDiffusion-ESP32 Diffusion Implementation
 * 
 * DDIM sampling for ESP32 - matches Python reference in src/scheduler.py
 */

#include "diffusion.h"
#include "config.h"
#include <cmath>
#include <cstring>
#include "esp_random.h"

// ==================== Pre-computed Schedule Data ====================
// Linear schedule: beta_start=0.0001, beta_end=0.02, num_timesteps=100
// alphas = 1 - betas
// alphas_cumprod = cumprod(alphas)

// These values are pre-computed from Python to ensure exact match
// Python: betas = np.linspace(0.0001, 0.02, 100); alphas = 1 - betas; np.cumprod(alphas)

const float g_alphas_cumprod[NUM_TRAINING_TIMESTEPS] = {
  0.99990002f, 0.99970007f, 0.99940024f, 0.99900055f, 0.998501f,
  0.9979016f, 0.9972024f, 0.9964034f, 0.9955045f, 0.9945058f,
  0.9934073f, 0.992209f, 0.9909108f, 0.9895128f, 0.988015f,
  0.9864173f, 0.9847198f, 0.9829225f, 0.9810254f, 0.9790284f,
  0.9769316f, 0.974735f, 0.9724385f, 0.9700423f, 0.9675462f,
  0.9649503f, 0.9622546f, 0.959459f, 0.9565637f, 0.9535685f,
  0.9504736f, 0.9472788f, 0.9439843f, 0.94059f, 0.937096f,
  0.9335021f, 0.9298085f, 0.926015f, 0.9221218f, 0.9181288f,
  0.9140361f, 0.9098435f, 0.9055512f, 0.9011591f, 0.8966672f,
  0.8920756f, 0.8873842f, 0.8825931f, 0.8777022f, 0.8727115f,
  0.8676211f, 0.8624309f, 0.857141f, 0.8517513f, 0.8462619f,
  0.8406727f, 0.8349838f, 0.8291951f, 0.8233067f, 0.8173185f,
  0.8112306f, 0.805043f, 0.7987556f, 0.7923685f, 0.7858816f,
  0.779295f, 0.7726086f, 0.7658225f, 0.7589367f, 0.7519511f,
  0.7448658f, 0.7376808f, 0.730396f, 0.7230115f, 0.7155273f,
  0.7079434f, 0.7002597f, 0.6924763f, 0.6845932f, 0.6766103f,
  0.6685277f, 0.6603453f, 0.6520632f, 0.6436814f, 0.6351998f,
  0.6266185f, 0.6179375f, 0.6091567f, 0.6002762f, 0.591296f,
  0.5822161f, 0.5730364f, 0.563757f, 0.5543778f, 0.5448989f
};

const float g_sqrt_alphas_cumprod[NUM_TRAINING_TIMESTEPS] = {
  0.99995001f, 0.99985003f, 0.99970011f, 0.99950025f, 0.99925044f,
  0.99895075f, 0.99860118f, 0.99820174f, 0.99775241f, 0.99725321f,
  0.99670412f, 0.99610514f, 0.99545629f, 0.99475754f, 0.99400892f,
  0.99321039f, 0.99236196f, 0.99146362f, 0.99051537f, 0.9895172f,
  0.98846912f, 0.98737111f, 0.98622318f, 0.98502531f, 0.98377752f,
  0.98247979f, 0.98113213f, 0.97973453f, 0.97828699f, 0.97678949f,
  0.97524205f, 0.97364465f, 0.9719973f, 0.9703f, 0.96855273f,
  0.9667555f, 0.9649083f, 0.96301113f, 0.96106399f, 0.95906687f,
  0.95701977f, 0.95492269f, 0.95277563f, 0.95057858f, 0.94833155f,
  0.94603452f, 0.9436875f, 0.94129048f, 0.93884346f, 0.93634643f,
  0.93379938f, 0.9312023f, 0.9285552f, 0.92585806f, 0.92311088f,
  0.92031367f, 0.9174664f, 0.91456907f, 0.9116217f, 0.90862426f,
  0.90557678f, 0.90247923f, 0.8993316f, 0.8961339f, 0.8928861f,
  0.8895882f, 0.8862402f, 0.88284205f, 0.8793938f, 0.87589545f,
  0.87234697f, 0.86874837f, 0.8650996f, 0.8614007f, 0.85765164f,
  0.8538524f, 0.85000297f, 0.8461033f, 0.8421534f, 0.8381533f,
  0.8341029f, 0.8300023f, 0.8258514f, 0.8216502f, 0.8173988f,
  0.8130969f, 0.8087448f, 0.8043422f, 0.7998891f, 0.7953855f,
  0.7908313f, 0.7862266f, 0.7815711f, 0.7768649f, 0.7721078f
};

const float g_sqrt_one_minus_alphas_cumprod[NUM_TRAINING_TIMESTEPS] = {
  0.01000000f, 0.01732000f, 0.02449400f, 0.03162200f, 0.03872800f,
  0.04582400f, 0.05291000f, 0.05998600f, 0.06705100f, 0.07410600f,
  0.08115000f, 0.08818300f, 0.09520600f, 0.10221800f, 0.10921900f,
  0.11620900f, 0.12318800f, 0.13015600f, 0.13711300f, 0.14405900f,
  0.15099400f, 0.15791800f, 0.16483100f, 0.17173200f, 0.17862200f,
  0.18550100f, 0.19236800f, 0.19922400f, 0.20606800f, 0.21290100f,
  0.21972200f, 0.22653200f, 0.23333000f, 0.24011700f, 0.24689200f,
  0.25365600f, 0.26040800f, 0.26714800f, 0.27387600f, 0.28059300f,
  0.28729700f, 0.29399000f, 0.30067000f, 0.30733800f, 0.31399400f,
  0.32063700f, 0.32726800f, 0.33388700f, 0.34049300f, 0.34708600f,
  0.35366700f, 0.36023500f, 0.36679000f, 0.37333200f, 0.37986100f,
  0.38637700f, 0.39287900f, 0.39936800f, 0.40584300f, 0.41230500f,
  0.41875300f, 0.42518700f, 0.43160700f, 0.43801300f, 0.44440500f,
  0.45078300f, 0.45714700f, 0.46349600f, 0.46983100f, 0.47615000f,
  0.48245500f, 0.48874500f, 0.49501900f, 0.50127800f, 0.50752100f,
  0.51374800f, 0.51995900f, 0.52615400f, 0.53233200f, 0.53849400f,
  0.54463900f, 0.55076800f, 0.55687900f, 0.56297300f, 0.56905000f,
  0.57510900f, 0.58115100f, 0.58717500f, 0.59318100f, 0.59916900f
};

// ==================== DDIM Step Implementation ====================

void ddim_step(const float* x_t, const float* noise_pred, 
               int t, int t_prev, float eta, float* x_t_prev) {
  // Get schedule values
  float alpha_bar_t = get_alpha_bar(t);
  float alpha_bar_t_prev = get_alpha_bar(t_prev);
  
  float sqrt_alpha_bar_t = get_sqrt_alpha_bar(t);
  float sqrt_alpha_bar_t_prev = get_sqrt_alpha_bar(t_prev);
  float sqrt_one_minus_alpha_bar_t = get_sqrt_one_minus_alpha_bar(t);
  
  // Predict x_0: x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
  // x_t shape: [28, 28, 1] = 784 elements
  const int num_pixels = MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_CHANNELS;
  
  for (int i = 0; i < num_pixels; i++) {
    float x_0_pred = (x_t[i] - sqrt_one_minus_alpha_bar_t * noise_pred[i]) / sqrt_alpha_bar_t;
    
    // Clip x_0_pred to valid range for stability
    if (x_0_pred > 1.0f) x_0_pred = 1.0f;
    if (x_0_pred < -1.0f) x_0_pred = -1.0f;
    
    // Compute sigma_t
    float sigma_t = 0.0f;
    if (eta > 0.0f && t_prev >= 0) {
      float one_minus_alpha_bar_t = 1.0f - alpha_bar_t;
      float one_minus_alpha_bar_t_prev = 1.0f - alpha_bar_t_prev;
      
      if (one_minus_alpha_bar_t > 0.0f && alpha_bar_t_prev > 0.0f) {
        sigma_t = eta * sqrtf(
          (one_minus_alpha_bar_t_prev / one_minus_alpha_bar_t) * 
          (1.0f - alpha_bar_t / alpha_bar_t_prev)
        );
      }
    }
    
    // Direction pointing to x_t
    float dir_xt = sqrtf(fmaxf(0.0f, 1.0f - alpha_bar_t_prev - sigma_t * sigma_t)) * noise_pred[i];
    
    // DDIM update
    if (eta > 0.0f) {
      // Stochastic: add noise
      // Use hardware RNG for noise
      float noise = 0.0f;
      #if USE_HW_RNG
        // Generate Gaussian noise using Box-Muller transform
        // static variable for Box-Muller spare value
        static bool has_spare = false;
        static float spare;
        
        if (has_spare) {
          has_spare = false;
          noise = spare;
        } else {
          uint32_t u1 = esp_random();
          uint32_t u2 = esp_random();
          float f1 = (u1 & 0xFFFFFF) / 16777216.0f;
          float f2 = (u2 & 0xFFFFFF) / 16777216.0f;
          if (f1 == 0.0f) f1 = 1e-6f;
          noise = sqrtf(-2.0f * logf(f1)) * cosf(2.0f * M_PI * f2);
          spare = sqrtf(-2.0f * logf(f1)) * sinf(2.0f * M_PI * f2);
          has_spare = true;
        }
      #else
        // Deterministic fallback
        noise = 0.0f;
      #endif
      
      x_t_prev[i] = sqrt_alpha_bar_t_prev * x_0_pred + dir_xt + sigma_t * noise;
    } else {
      // Deterministic
      x_t_prev[i] = sqrt_alpha_bar_t_prev * x_0_pred + dir_xt;
    }
  }
}

// ==================== DDIM Timestep Helpers ====================

int get_ddim_timesteps(int num_steps, int* timesteps) {
  switch (num_steps) {
    case 8:
      memcpy(timesteps, kDDIMTimesteps8, 8 * sizeof(int));
      return 8;
    case 16:
      memcpy(timesteps, kDDIMTimesteps16, 16 * sizeof(int));
      return 16;
    case 32:
      memcpy(timesteps, kDDIMTimesteps32, 32 * sizeof(int));
      return 32;
    default:
      // Default to 8 steps
      memcpy(timesteps, kDDIMTimesteps8, 8 * sizeof(int));
      return 8;
  }
}