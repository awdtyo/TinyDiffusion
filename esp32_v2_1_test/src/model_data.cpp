#include "model_data.h"
#include <Arduino.h>

// Model data definitions - stored in flash (PROGMEM) for ESP32.
// alignas(16) is required by TFLite Micro / FlatBuffers.
alignas(16) const unsigned char g_model[] PROGMEM = {
#include "model_data.inc"
};

// Derive the length from the array itself so the two can never disagree.
const unsigned int g_model_len = sizeof(g_model);

// Compile-time size check
static_assert(sizeof(g_model) == 53336, "Model size mismatch: expected 53336 bytes");
