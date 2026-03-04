import numpy as np
import tensorflow as tf
import math

# ── 1. Training data: sin(x) for x in [0, 2π] ────────────────────────────────
X = np.linspace(0, 2 * math.pi, 1000).astype(np.float32).reshape(-1, 1)
Y = np.sin(X).astype(np.float32)

# ── 2. Build model ────────────────────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(1,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, Y, epochs=500, verbose=0)
print(f"Float model MAE: {np.mean(np.abs(model.predict(X, verbose=0) - Y)):.5f}")

# ── 3. Representative dataset for full-integer quantization ───────────────────
def representative_dataset():
    for x in X:
        yield [x.reshape(1, 1)]

# ── 4. Convert with correct int8 quantization ─────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open("sine_int8_fixed.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved sine_int8_fixed.tflite")

# ── 5. Verify the new model ───────────────────────────────────────────────────
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
ii = interp.get_input_details()[0]
oi = interp.get_output_details()[0]
in_scale  = ii["quantization"][0];  in_zp  = ii["quantization"][1]
out_scale = oi["quantization"][0];  out_zp = oi["quantization"][1]

errors = []
for x_f in np.linspace(0, 2*math.pi, 100):
    q_in = np.clip(round(x_f / in_scale) + in_zp, -128, 127)
    interp.set_tensor(ii["index"], np.array([[q_in]], dtype=np.int8))
    interp.invoke()
    q_out = int(interp.get_tensor(oi["index"]).flatten()[0])
    f_out = (q_out - out_zp) * out_scale
    errors.append(abs(f_out - math.sin(x_f)))

print(f"Fixed model max error: {max(errors):.5f}")
print(f"Fixed model mean error: {np.mean(errors):.5f}")