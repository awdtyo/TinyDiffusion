/**
 * TinyDiffusion-ESP32 OLED Display Header
 * 
 * SSD1306 128x64 OLED via U8g2 library
 */

#ifndef OLED_H
#define OLED_H

#include "config.h"
#include <U8g2lib.h>

// ==================== Display Object ====================
// U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE, DISPLAY_SCL_PIN, DISPLAY_SDA_PIN);
// For software I2C (more flexible pin assignment):
extern U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2;

// ==================== Display Functions ====================

/**
 * Initialize OLED display
 * @return true on success
 */
bool oled_init();

/**
 * Clear display buffer
 */
void oled_clear();

/**
 * Draw generated digit image (28x28 scaled to 56x56)
 * @param image Float32 image [28*28] in range [-1, 1]
 * @param class_label Digit class (0-9)
 * @param generation_time_ms Generation time in milliseconds
 * @param sampling_steps Number of DDIM steps used
 */
void oled_draw_digit(const float* image, int class_label, 
                     uint32_t generation_time_ms, int sampling_steps);

/**
 * Draw text message on display
 * @param line1 First line
 * @param line2 Second line (optional)
 * @param line3 Third line (optional)
 */
void oled_draw_text(const char* line1, const char* line2 = nullptr, const char* line3 = nullptr);

/**
 * Draw progress bar (for generation progress)
 * @param progress Progress 0.0 to 1.0
 * @param label Label text
 */
void oled_draw_progress(float progress, const char* label);

/**
 * Send buffer to display
 */
void oled_send_buffer();

/**
 * Draw startup screen
 */
void oled_draw_startup();

#endif // OLED_H