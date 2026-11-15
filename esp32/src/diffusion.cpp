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

// The bound is intentionally omitted here: the header declares [100], so a
// short initializer list becomes a conflicting-declaration error instead of
// silently zero-filling the missing timesteps.
const float g_alphas_cumprod[] = {
  0.999899983f, 0.99959904f, 0.999097168f, 0.998394787f, 0.997492194f,
  0.996389866f, 0.995088518f, 0.993588865f, 0.991891742f, 0.989998162f,
  0.987909138f, 0.985625982f, 0.983150005f, 0.980482578f, 0.97762531f,
  0.974579811f, 0.971347928f, 0.967931569f, 0.964332581f, 0.960553169f,
  0.95659548f, 0.952461839f, 0.948154628f, 0.943676233f, 0.939029336f,
  0.934216619f, 0.929240704f, 0.924104571f, 0.918811023f, 0.913363099f,
  0.907763958f, 0.90201658f, 0.896124303f, 0.890090406f, 0.883918226f,
  0.87761116f, 0.871172667f, 0.864606321f, 0.85791564f, 0.851104319f,
  0.844175994f, 0.837134361f, 0.829983234f, 0.822726369f, 0.81536752f,
  0.807910621f, 0.800359488f, 0.792718053f, 0.784990251f, 0.777180016f,
  0.769291282f, 0.761327922f, 0.753293991f, 0.745193422f, 0.737030149f,
  0.728808165f, 0.720531404f, 0.712203801f, 0.703829288f, 0.695411742f,
  0.686955094f, 0.678463221f, 0.669939935f, 0.661389053f, 0.652814388f,
  0.644219637f, 0.635608613f, 0.626984894f, 0.618352115f, 0.609713912f,
  0.601073861f, 0.59243542f, 0.583801985f, 0.575177073f, 0.566563904f,
  0.557965875f, 0.549386203f, 0.54082799f, 0.532294393f, 0.523788393f,
  0.515313089f, 0.506871343f, 0.498465985f, 0.490099818f, 0.481775552f,
  0.473495811f, 0.465263218f, 0.457080215f, 0.448949277f, 0.440872729f,
  0.432852834f, 0.424891859f, 0.41699186f, 0.409154952f, 0.401383072f,
  0.393678129f, 0.386041969f, 0.378476322f, 0.370982885f, 0.36356324f
};

const float g_sqrt_alphas_cumprod[] = {
  0.999949992f, 0.99979949f, 0.999548495f, 0.999197066f, 0.998745322f,
  0.998193324f, 0.997541249f, 0.996789277f, 0.995937645f, 0.994986534f,
  0.993936181f, 0.992787004f, 0.99153924f, 0.990193188f, 0.988749385f,
  0.987208068f, 0.985569835f, 0.983835161f, 0.982004344f, 0.980078161f,
  0.978056967f, 0.975941539f, 0.973732293f, 0.971430004f, 0.969035268f,
  0.9665488f, 0.963971317f, 0.961303592f, 0.958546281f, 0.955700338f,
  0.952766478f, 0.949745536f, 0.946638405f, 0.94344604f, 0.940169275f,
  0.936809003f, 0.933366299f, 0.929842114f, 0.926237345f, 0.922553182f,
  0.918790519f, 0.91495049f, 0.911034167f, 0.907042623f, 0.902977049f,
  0.89883846f, 0.894628108f, 0.890347183f, 0.885996759f, 0.881578147f,
  0.87709254f, 0.87254107f, 0.867925107f, 0.863245845f, 0.858504593f,
  0.853702605f, 0.84884119f, 0.843921661f, 0.838945329f, 0.833913505f,
  0.82882756f, 0.823688805f, 0.818498611f, 0.81325829f, 0.807969272f,
  0.802632928f, 0.797250688f, 0.791823804f, 0.786353707f, 0.780841768f,
  0.775289536f, 0.769698262f, 0.764069378f, 0.758404315f, 0.752704382f,
  0.74697113f, 0.741205931f, 0.735410094f, 0.729585111f, 0.723732293f,
  0.717853129f, 0.711948991f, 0.706021249f, 0.700071275f, 0.694100559f,
  0.688110292f, 0.682102084f, 0.676077068f, 0.670036793f, 0.663982451f,
  0.657915533f, 0.651837289f, 0.645749092f, 0.639652193f, 0.633548021f,
  0.62743777f, 0.621322751f, 0.615204275f, 0.609083652f, 0.602962077f
};

const float g_sqrt_one_minus_alphas_cumprod[] = {
  0.0100008296f, 0.020023996f, 0.0300471559f, 0.040065106f, 0.0500779971f,
  0.0600843914f, 0.0700819716f, 0.0800695643f, 0.0900458694f, 0.100009188f,
  0.109958455f, 0.119891688f, 0.129807532f, 0.139704764f, 0.149581715f,
  0.159437105f, 0.169269234f, 0.179076612f, 0.188858196f, 0.198612258f,
  0.208337516f, 0.218032479f, 0.227695793f, 0.237326294f, 0.246922389f,
  0.25648272f, 0.266006202f, 0.275491238f, 0.284936786f, 0.294341475f,
  0.303703874f, 0.313023031f, 0.322297513f, 0.33152616f, 0.340707749f,
  0.349841177f, 0.358925253f, 0.367958814f, 0.376940787f, 0.385870039f,
  0.394745499f, 0.403566152f, 0.412330896f, 0.421038747f, 0.429688811f,
  0.438280016f, 0.446811497f, 0.455282271f, 0.463691443f, 0.47203812f,
  0.480321467f, 0.488540769f, 0.496695101f, 0.50478369f, 0.512805879f,
  0.520760834f, 0.5286479f, 0.53646642f, 0.544215679f, 0.551895142f,
  0.559504151f, 0.567042112f, 0.574508548f, 0.581902862f, 0.589224577f,
  0.596473277f, 0.603648424f, 0.610749602f, 0.617776573f, 0.624728799f,
  0.631605983f, 0.638407826f, 0.645134091f, 0.65178442f, 0.658358634f,
  0.664856493f, 0.671277761f, 0.677622318f, 0.683890045f, 0.690080881f,
  0.696194589f, 0.702231228f, 0.70819062f, 0.714073002f, 0.719878078f,
  0.725606084f, 0.731256962f, 0.73683089f, 0.742327929f, 0.747748137f,
  0.753091693f, 0.758358896f, 0.763549745f, 0.768664479f, 0.773703396f,
  0.778666735f, 0.783554733f, 0.788367748f, 0.793106019f, 0.797769845f
};

// Catch a truncated initializer list at compile time.  The previous arrays
// contained 95/95/90 values, so the remaining entries silently became 0.0f.
static_assert(sizeof(g_alphas_cumprod) / sizeof(g_alphas_cumprod[0]) ==
                  NUM_TRAINING_TIMESTEPS,
              "g_alphas_cumprod must contain exactly NUM_TRAINING_TIMESTEPS values");
static_assert(sizeof(g_sqrt_alphas_cumprod) / sizeof(g_sqrt_alphas_cumprod[0]) ==
                  NUM_TRAINING_TIMESTEPS,
              "g_sqrt_alphas_cumprod must contain exactly NUM_TRAINING_TIMESTEPS values");
static_assert(sizeof(g_sqrt_one_minus_alphas_cumprod) /
                      sizeof(g_sqrt_one_minus_alphas_cumprod[0]) ==
                  NUM_TRAINING_TIMESTEPS,
              "g_sqrt_one_minus_alphas_cumprod must contain exactly "
              "NUM_TRAINING_TIMESTEPS values");

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