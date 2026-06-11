#include <Arduino.h>
#include <Wire.h>
#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "suspension_model.h"

// ─── Hard Limits (see AGENTS.md §9) ───────────────────────

constexpr int kTensorArenaSize = 10 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// ─── Observation Scaling (MUST match Python _OBS_SCALE) ───

constexpr float OBS_SCALE[2] = {150.0f, 50.0f};

// ─── Action -> Resistance (MUST match np.linspace(10000, 10, 10)) ──

static const float RESISTANCE_LEVELS[10] = {
    10000.0f, 8901.1f, 7802.2f, 6703.3f, 5604.4f,
     4505.6f, 3406.7f, 2307.8f, 1208.9f,   10.0f
};

// ─── Pin Assignment ───────────────────────────────────────

constexpr int PIN_ADC_VOLTAGE = 34;
constexpr int PIN_PWM         = 25;
constexpr int PIN_SDA = 21;
constexpr int PIN_SCL = 22;
constexpr int MPU6050_ADDR = 0x68;

// ─── Task Handles ─────────────────────────────────────────

TaskHandle_t inference_task_handle = NULL;
TaskHandle_t debug_task_handle     = NULL;

// ─── TFLite Globals ───────────────────────────────────────

static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor  = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// ─── Telemetry ────────────────────────────────────────────

static volatile float   tele_voltage       = 0.0f;
static volatile float   tele_accel         = 0.0f;
static volatile int     tele_action        = 0;
static volatile float   tele_inference_us  = 0.0f;
static volatile bool    tele_ready         = false;

// ─── High-pass filter state for acceleration ──────────────

static float accel_dc_bias = 9.81f;

static void setup_adc();
static void setup_pwm();
static void setup_imu();
static void setup_tflite();
static float read_voltage();
static float read_accel_z();
static int   run_inference(float voltage, float accel);
static void  apply_action(int action);

// ─── Forward declare TFLite micro error reporter ──────────

static tflite::MicroErrorReporter micro_error_reporter;

// ============================================================
//  Inference Task  -  Core 1, 10 ms period
// ============================================================

void inference_task(void* pvParameters) {
    // Boot-time inference measurement
    input_tensor->data.f[0] = 0.0f;
    input_tensor->data.f[1] = 0.0f;
    uint32_t t0 = micros();
    TfLiteStatus status = interpreter->Invoke();
    uint32_t t1 = micros();
    if (status != kTfLiteOk) {
        Serial.println("FATAL: boot inference failed");
        while (1) { delay(100); }
    }
    Serial.printf("Boot inference: %lu us\n", t1 - t0);
    Serial.printf("Arena used: %zu bytes\n", interpreter->arena_used_bytes());
    Serial.printf("Input type: %s\n", TfLiteTypeGetName(input_tensor->type));

    const TickType_t period = pdMS_TO_TICKS(10);
    TickType_t last_wake = xTaskGetTickCount();

    while (true) {
        vTaskDelayUntil(&last_wake, period);

        float voltage = read_voltage();
        float accel   = read_accel_z();

        uint32_t t_start = micros();
        int action = run_inference(voltage, accel);
        uint32_t t_end   = micros();

        apply_action(action);

        tele_voltage      = voltage;
        tele_accel        = accel;
        tele_action       = action;
        tele_inference_us = (float)(t_end - t_start);
        tele_ready        = true;
    }
}

// ============================================================
//  Debug Task  -  Core 0, 1 Hz
// ============================================================

void debug_task(void* pvParameters) {
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));

        if (tele_ready) {
            Serial.printf("V=%.2f  A=%.3f  act=%d  R=%.1f  t=%.0f us\n",
                (double)tele_voltage,
                (double)tele_accel,
                tele_action,
                (double)RESISTANCE_LEVELS[tele_action],
                (double)tele_inference_us);
        }
    }
}

// ============================================================
//  Arduino Entry Points
// ============================================================

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== DQN Suspension Controller ===");

    Wire.begin(PIN_SDA, PIN_SCL);
    setup_adc();
    setup_pwm();
    setup_tflite();
    setup_imu();

    xTaskCreatePinnedToCore(
        inference_task,
        "inference",
        16384,
        NULL,
        2,
        &inference_task_handle,
        1
    );

    xTaskCreatePinnedToCore(
        debug_task,
        "debug",
        4096,
        NULL,
        1,
        &debug_task_handle,
        0
    );

    Serial.println("System ready.");
}

void loop() {
    vTaskDelete(NULL);
}

// ============================================================
//  Hardware Setup
// ============================================================

static void setup_adc() {
    analogReadResolution(12);
    analogSetPinAttenuation(PIN_ADC_VOLTAGE, ADC_11db);
}

static void setup_pwm() {
    ledcSetup(0, 5000, 8);
    ledcAttachPin(PIN_PWM, 0);
    ledcWrite(0, 0);
}

static void setup_imu() {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x6B);
    Wire.write(0x00);
    Wire.endTransmission();

    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1C);
    Wire.write(0x08);
    Wire.endTransmission();

    delay(100);
    Serial.println("MPU-6050 ready");
}

static void setup_tflite() {
    const tflite::Model* model = tflite::GetModel(suspension_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model schema mismatch: expected %u, got %u\n",
            TFLITE_SCHEMA_VERSION, model->version());
        while (1) { delay(100); }
    }

    static tflite::MicroInterpreter interp(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter = &interp;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.printf("AllocateTensors() failed (arena=%d)\n", kTensorArenaSize);
        while (1) { delay(100); }
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
    Serial.printf("TFLite init OK (arena=%d B)\n", kTensorArenaSize);
}

// ============================================================
//  Sensors
// ============================================================

static float read_voltage() {
    int raw = analogRead(PIN_ADC_VOLTAGE);
    float pin_voltage = (float)raw * (3.3f / 4095.0f);

    // ADC input comes through a voltage divider.
    // Calibrate this ratio for your specific hardware.
    // Example: for a 1:100 divider, gain = 101.0f
    const float DIVIDER_GAIN = 101.0f;
    return pin_voltage * DIVIDER_GAIN;
}

static float read_accel_z() {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU6050_ADDR, 6, true);

    if (Wire.available() < 6) return 0.0f;

    int16_t ax = (Wire.read() << 8) | Wire.read();
    int16_t ay = (Wire.read() << 8) | Wire.read();
    int16_t az = (Wire.read() << 8) | Wire.read();

    float az_g = (float)az / 8192.0f;
    float az_ms2 = az_g * 9.81f;

    accel_dc_bias = 0.999f * accel_dc_bias + 0.001f * az_ms2;
    float dynamic_accel = az_ms2 - accel_dc_bias;

    return dynamic_accel;
}

// ============================================================
//  Inference
// ============================================================

static int run_inference(float voltage, float accel) {
    // Normalize using training-time scaling
    float obs0 = voltage / OBS_SCALE[0];
    float obs1 = accel   / OBS_SCALE[1];

    // Handle quantization if model uses int8
    if (input_tensor->type == kTfLiteInt8) {
        input_tensor->data.int8[0] = (int8_t)(obs0 / input_tensor->params.scale
                                    + input_tensor->params.zero_point);
        input_tensor->data.int8[1] = (int8_t)(obs1 / input_tensor->params.scale
                                    + input_tensor->params.zero_point);
    } else {
        input_tensor->data.f[0] = obs0;
        input_tensor->data.f[1] = obs1;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        return 0;
    }

    // Argmax over 10 Q-values
    float q[10];
    if (output_tensor->type == kTfLiteInt8) {
        for (int i = 0; i < 10; i++) {
            q[i] = ((float)output_tensor->data.int8[i]
                    - output_tensor->params.zero_point)
                   * output_tensor->params.scale;
        }
    } else {
        float* raw = output_tensor->data.f;
        for (int i = 0; i < 10; i++) q[i] = raw[i];
    }

    int best = 0;
    float best_q = q[0];
    for (int i = 1; i < 10; i++) {
        if (q[i] > best_q) {
            best_q = q[i];
            best = i;
        }
    }
    return best;
}

// ============================================================
//  Actuator
// ============================================================

static void apply_action(int action) {
    if (action < 0 || action > 9) action = 0;

    float r_ext = RESISTANCE_LEVELS[action];

    float duty_f = 255.0f * (1.0f - (r_ext - 10.0f) / (10000.0f - 10.0f));
    int duty = (int)(duty_f + 0.5f);
    if (duty < 0) duty = 0;
    if (duty > 255) duty = 255;

    ledcWrite(0, duty);
}
