/**
 * TinyDiffusion-ESP32 OLED Display Implementation
 * 
 * SSD1306 128x64 OLED via U8g2 library
 */

#include "oled.h"
#include "config.h"
#include <U8g2lib.h>
#include <Wire.h>

// ==================== Display Object ====================
// Software I2C for flexible pin assignment
U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2(U8G2_R0, 
                                          DISPLAY_SCL_PIN,  // clock
                                          DISPLAY_SDA_PIN,  // data
                                          U8X8_PIN_NONE);   // reset

// ==================== Implementation ====================

bool oled_init() {
  // Initialize I2C
  Wire.begin(DISPLAY_SDA_PIN, DISPLAY_SCL_PIN);
  Wire.setClock(400000);  // 400kHz I2C
  
  // Initialize display
  u8g2.begin();
  u8g2.setFont(u8g2_font_6x10_tf);  // Small readable font
  u8g2.setFontRefHeightExtendedText();
  u8g2.setDrawColor(1);  // White on black
  u8g2.clearBuffer();
  
  // Show startup screen
  oled_draw_startup();
  u8g2.sendBuffer();
  
  return true;
}

void oled_clear() {
  u8g2.clearBuffer();
}

void oled_send_buffer() {
  u8g2.sendBuffer();
}

void oled_draw_startup() {
  u8g2.clearBuffer();
  
  // Title
  u8g2.setFont(u8g2_font_7x14_tf);
  u8g2.drawStr(10, 16, "TinyDiffusion");
  u8g2.drawStr(10, 32, "ESP32 + MNIST");
  
  // Info
  u8g2.setFont(u8g2_font_5x7_tf);
  u8g2.drawStr(10, 48, "Conditional Diffusion");
  u8g2.drawStr(10, 56, "DDIM 8/16/32 steps");
  
  u8g2.sendBuffer();
}

void oled_draw_text(const char* line1, const char* line2, const char* line3) {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  
  int y = 16;
  if (line1) {
    u8g2.drawStr(0, y, line1);
    y += 14;
  }
  if (line2) {
    u8g2.drawStr(0, y, line2);
    y += 14;
  }
  if (line3) {
    u8g2.drawStr(0, y, line3);
  }
  
  u8g2.sendBuffer();
}

void oled_draw_progress(float progress, const char* label) {
  u8g2.clearBuffer();
  
  // Label
  u8g2.setFont(u8g2_font_6x10_tf);
  if (label) {
    u8g2.drawStr(0, 14, label);
  }
  
  // Progress bar background
  int bar_x = 10;
  int bar_y = 30;
  int bar_w = DISPLAY_WIDTH - 20;
  int bar_h = 10;
  
  u8g2.drawFrame(bar_x, bar_y, bar_w, bar_h);
  
  // Progress bar fill
  int fill_w = (int)(bar_w * progress);
  if (fill_w > 0) {
    u8g2.drawBox(bar_x, bar_y, fill_w, bar_h);
  }
  
  // Percentage text
  char pct_str[16];
  snprintf(pct_str, sizeof(pct_str), "%d%%", (int)(progress * 100));
  u8g2.drawStr(DISPLAY_WIDTH / 2 - 10, 50, pct_str);
  
  u8g2.sendBuffer();
}

void oled_draw_digit(const float* image, int class_label, 
                     uint32_t generation_time_ms, int sampling_steps) {
  u8g2.clearBuffer();
  
  // Draw scaled digit image (28x28 -> 56x56)
  const int src_w = MODEL_OUTPUT_WIDTH;
  const int src_h = MODEL_OUTPUT_HEIGHT;
  const int dst_w = src_w * DISPLAY_SCALE_FACTOR;  // 56
  const int dst_h = src_h * DISPLAY_SCALE_FACTOR;  // 56
  const int dst_x = (DISPLAY_WIDTH - dst_w) / 2;   // 36
  const int dst_y = 2;  // Top margin
  
  // Draw each pixel as a scaled block
  for (int sy = 0; sy < src_h; sy++) {
    for (int sx = 0; sx < src_w; sx++) {
      float val = image[sy * src_w + sx];
      // Normalize from [-1, 1] to [0, 1]
      val = (val + 1.0f) * 0.5f;
      if (val < 0.0f) val = 0.0f;
      if (val > 1.0f) val = 1.0f;
      
      // Threshold for monochrome display
      // Use dithering for better visual quality
      bool pixel_on = false;
      if (val > 0.5f) {
        pixel_on = true;
      } else if (val > 0.25f) {
        // Ordered dithering pattern
        pixel_on = ((sx + sy) & 1) == 0;
      } else if (val > 0.1f) {
        pixel_on = ((sx + sy * 2) & 3) == 0;
      }
      
      if (pixel_on) {
        // Draw scaled block
        int bx = dst_x + sx * DISPLAY_SCALE_FACTOR;
        int by = dst_y + sy * DISPLAY_SCALE_FACTOR;
        u8g2.drawBox(bx, by, DISPLAY_SCALE_FACTOR, DISPLAY_SCALE_FACTOR);
      }
    }
  }
  
  // Draw class label
  u8g2.setFont(u8g2_font_7x14_tf);
  char label_str[32];
  snprintf(label_str, sizeof(label_str), "Digit: %d", class_label);
  u8g2.drawStr(2, dst_y + dst_h + 16, label_str);
  
  // Draw generation info
  u8g2.setFont(u8g2_font_5x7_tf);
  char info_str[64];
  snprintf(info_str, sizeof(info_str), "Time: %lu ms | Steps: %d",
           (unsigned long)generation_time_ms, sampling_steps);
  u8g2.drawStr(2, dst_y + dst_h + 28, info_str);
  
  // Draw memory info
  char mem_str[64];
  snprintf(mem_str, sizeof(mem_str), "Free Heap: %lu KB",
           (unsigned long)(ESP.getFreeHeap() / 1024));
  u8g2.drawStr(2, dst_y + dst_h + 38, mem_str);
  
  u8g2.sendBuffer();
}