import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.quarter_car_env import QuarterCarEnv

MODEL_DIR = "models"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_suspension_dqn.weights.h5")
FLOAT32_PATH = os.path.join(MODEL_DIR, "model_float32.tflite")
INT8_PATH = os.path.join(MODEL_DIR, "model_int8.tflite")
HEADER_PATH = "esp32/suspension_inference/src/suspension_model.h"
REPRESENTATIVE_DATA_PATH = os.path.join(os.path.dirname(__file__), "representative_data.npy")

STATE_SIZE = 3
ACTION_SIZE = 10

_OBS_SCALE = np.array([150.0, 3.0, 50.0], dtype=np.float32)

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(STATE_SIZE,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(ACTION_SIZE, activation='linear')
    ])
    return model

def generate_representative_dataset(num_samples=500):
    env = QuarterCarEnv()
    samples = []
    state, _ = env.reset()
    while len(samples) < num_samples:
        action = env.action_space.sample()
        next_state, _, terminated, truncated, _ = env.step(action)
        rel_vel = next_state[1]
        obs = np.array([
            env.Ke * rel_vel,
            rel_vel,
            next_state[2]
        ], dtype=np.float32)
        samples.append(obs)
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    arr = np.array(samples, dtype=np.float32)
    np.save(REPRESENTATIVE_DATA_PATH, arr)
    normalized = arr / _OBS_SCALE
    return normalized

def representative_dataset():
    data = np.load(REPRESENTATIVE_DATA_PATH)
    data = data.astype(np.float32) / _OBS_SCALE
    for i in range(len(data)):
        yield [data[i:i+1]]

def convert_to_tflite(model):
    model.save_weights(WEIGHTS_PATH.replace("best", "tmp"))
    saved_model_dir = os.path.join(MODEL_DIR, "tmp_savedmodel")
    model.save(saved_model_dir)
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        float32_tflite = converter.convert()
        with open(FLOAT32_PATH, "wb") as f:
            f.write(float32_tflite)
        print(f"  float32 TFLite: {FLOAT32_PATH}  ({len(float32_tflite) / 1024:.1f} KB)")

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        int8_tflite = converter.convert()
        with open(INT8_PATH, "wb") as f:
            f.write(int8_tflite)
        print(f"  int8 TFLite:    {INT8_PATH}  ({len(int8_tflite) / 1024:.1f} KB)")
    finally:
        import shutil
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        tmp_weights = WEIGHTS_PATH.replace("best", "tmp")
        if os.path.exists(tmp_weights):
            os.remove(tmp_weights)
    return float32_tflite, int8_tflite

def generate_c_header(tflite_data, header_path):
    hex_lines = []
    hex_lines.append("#ifndef SUSPENSION_MODEL_H")
    hex_lines.append("#define SUSPENSION_MODEL_H")
    hex_lines.append("")
    hex_lines.append("#ifdef __cplusplus")
    hex_lines.append('extern "C" {')
    hex_lines.append("#endif")
    hex_lines.append("")
    hex_lines.append("const unsigned char suspension_model_tflite[] = {")
    for i in range(0, len(tflite_data), 12):
        chunk = tflite_data[i:i+12]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        hex_lines.append(f"  {hex_str},")
    hex_lines.append("};")
    hex_lines.append(f"const unsigned int suspension_model_tflite_len = {len(tflite_data)};")
    hex_lines.append("")
    hex_lines.append("#ifdef __cplusplus")
    hex_lines.append("}")
    hex_lines.append("#endif")
    hex_lines.append("")
    hex_lines.append("#endif  // SUSPENSION_MODEL_H")
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w") as f:
        f.write("\n".join(hex_lines) + "\n")
    header_kb = os.path.getsize(header_path) / 1024
    print(f"  C header:       {header_path}  ({header_kb:.1f} KB)")

def verify_accuracy(float32_tflite, int8_tflite, num_samples=5):
    interpreter_f32 = tf.lite.Interpreter(model_content=float32_tflite)
    interpreter_f32.allocate_tensors()
    input_details_f32 = interpreter_f32.get_input_details()
    output_details_f32 = interpreter_f32.get_output_details()

    interpreter_i8 = tf.lite.Interpreter(model_content=int8_tflite)
    interpreter_i8.allocate_tensors()
    input_details_i8 = interpreter_i8.get_input_details()
    output_details_i8 = interpreter_i8.get_output_details()

    data = np.load(REPRESENTATIVE_DATA_PATH)
    data_norm = data.astype(np.float32) / _OBS_SCALE

    matches = 0
    total = min(num_samples, len(data))
    print(f"\n  Accuracy comparison ({total} samples, float32 vs int8):")
    for i in range(total):
        sample = data_norm[i:i+1].astype(np.float32)

        interpreter_f32.set_tensor(input_details_f32[0]['index'], sample)
        interpreter_f32.invoke()
        out_f32 = interpreter_f32.get_tensor(output_details_f32[0]['index'])
        action_f32 = int(np.argmax(out_f32[0]))

        input_i8 = sample
        if input_details_i8[0]['dtype'] == np.int8:
            scale = input_details_i8[0]['quantization_parameters']['scales'][0]
            zero_point = input_details_i8[0]['quantization_parameters']['zero_points'][0]
            input_i8 = np.round(sample / scale + zero_point).astype(np.int8)
            input_i8 = np.clip(input_i8, -128, 127).astype(np.int8)

        interpreter_i8.set_tensor(input_details_i8[0]['index'], input_i8)
        interpreter_i8.invoke()
        out_i8 = interpreter_i8.get_tensor(output_details_i8[0]['index'])

        if output_details_i8[0]['dtype'] == np.int8:
            scale = output_details_i8[0]['quantization_parameters']['scales'][0]
            zero_point = output_details_i8[0]['quantization_parameters']['zero_points'][0]
            out_i8 = (out_i8.astype(np.float32) - zero_point) * scale

        action_i8 = int(np.argmax(out_i8[0]))
        match = action_f32 == action_i8
        if match:
            matches += 1
        print(f"    [{i}] float32 action={action_f32}  int8 action={action_i8}  {'✅' if match else '❌'}")

    match_rate = matches / total * 100
    print(f"\n  int8 action match rate: {match_rate:.1f}%")
    if match_rate < 90:
        print("  ⚠️  WARNING: Match rate < 90%. Retrain with wider dataset or skip quantisation.")
    else:
        print("  ✅ Match rate >= 90% — int8 quantisation is safe.")
    return match_rate

def main():
    best_weights = WEIGHTS_PATH
    if not os.path.exists(best_weights):
        alt = best_weights.replace("best_", "final_")
        if os.path.exists(alt):
            best_weights = alt
            print(f"Best model not found, using: {best_weights}")
        else:
            print(f"ERROR: No trained model found at {best_weights} or {alt}")
            print("Run train.py first to train a model.")
            sys.exit(1)

    print("=" * 50)
    print("TFLite Conversion — Suspension DQN Model")
    print("=" * 50)

    print("\n[1/5] Building model architecture...")
    model = build_model()
    model.build(input_shape=(None, STATE_SIZE))
    model.compile()
    print(f"  Parameters: {model.count_params()}")

    print("[2/5] Loading weights...")
    dummy_weights = WEIGHTS_PATH.replace("best_", "tmp_best_")
    if os.path.exists(dummy_weights):
        os.remove(dummy_weights)
    try:
        model.load_weights(best_weights)
    except Exception as e:
        print(f"  ERROR loading weights: {e}")
        sys.exit(1)
    if os.path.exists(dummy_weights):
        os.remove(dummy_weights)
    print(f"  Loaded: {best_weights}")

    print("[3/5] Generating representative dataset...")
    if not os.path.exists(REPRESENTATIVE_DATA_PATH):
        print(f"  Generating {500} samples from QuarterCarEnv...")
        generate_representative_dataset(500)
    else:
        print(f"  Using existing: {REPRESENTATIVE_DATA_PATH}")
    data = np.load(REPRESENTATIVE_DATA_PATH)
    print(f"  Dataset: {len(data)} samples, shape {data.shape}")

    print("[4/5] Converting to TFLite...")
    float32_tflite, int8_tflite = convert_to_tflite(model)

    print("[5/5] Generating C header...")
    generate_c_header(int8_tflite, HEADER_PATH)

    print("\n" + "-" * 50)
    print("Verification")
    print("-" * 50)
    verify_accuracy(float32_tflite, int8_tflite, num_samples=5)

    print("\n✅ Conversion complete.")
    print(f"   float32: {os.path.getsize(FLOAT32_PATH)/1024:.1f} KB  ({FLOAT32_PATH})")
    print(f"   int8:    {os.path.getsize(INT8_PATH)/1024:.1f} KB  ({INT8_PATH})")
    print(f"   header:  {os.path.getsize(HEADER_PATH)/1024:.1f} KB  ({HEADER_PATH})")

if __name__ == "__main__":
    main()
