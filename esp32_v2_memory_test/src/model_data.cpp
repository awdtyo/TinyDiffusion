#include "model_data.h"
#include <Arduino.h>

alignas(16) const unsigned char g_model[] PROGMEM = {
#include "model_data.inc"
};

const unsigned int g_model_len = sizeof(g_model);

static_assert(sizeof(g_model) == 39572, "Model size mismatch: expected 39572 bytes");
