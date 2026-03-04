from ai_edge_litert.interpreter import Interpreter
import numpy as np
import math
import matplotlib.pyplot as plt

MODEL_PATH = "sine_int8_fixed.tflite"


# ─────────────────────────────────────────────────────────────
# Quantization helpers (TFLite-compatible)
# ─────────────────────────────────────────────────────────────

def quantize_multiplier(real_multiplier):
    if real_multiplier == 0:
        return 0, 0

    significand, exponent = math.frexp(real_multiplier)
    q31 = int(round(significand * (1 << 31)))

    if q31 == (1 << 31):
        q31 //= 2
        exponent += 1

    # TFLite defines shift as NEGATIVE exponent
    shift = -exponent
    return q31, shift


def requantize(acc, multiplier, shift):
    acc = acc.astype(np.int64)

    # SaturatingRoundingDoublingHighMul
    prod = acc * np.int64(multiplier)
    nudge = np.where(prod >= 0, (1 << 30), -(1 << 30))
    prod = (prod + nudge) >> 31

    if shift > 0:
        # RoundingDivideByPOT
        mask = (1 << shift) - 1
        remainder = prod & mask
        threshold = (mask >> 1)
        prod = (prod >> shift) + (remainder > threshold)

    elif shift < 0:
        prod = prod << (-shift)

    return prod.astype(np.int32)


# ─────────────────────────────────────────────────────────────
# Fully Connected Forward
# ─────────────────────────────────────────────────────────────

def fc_forward(ti, tw, tb, tds, iidx, widx, bidx, oidx):

    iscale = float(tds[iidx]["quantization_parameters"]["scales"][0])
    izp    = int(tds[iidx]["quantization_parameters"]["zero_points"][0])

    oscale = float(tds[oidx]["quantization_parameters"]["scales"][0])
    ozp    = int(tds[oidx]["quantization_parameters"]["zero_points"][0])

    wscales = tds[widx]["quantization_parameters"]["scales"]

    x = ti.flatten().astype(np.int32) - izp
    w = tw.astype(np.int32)
    b = tb.astype(np.int32)

    acc = np.dot(w, x) + b

    # Per-channel requantization
    res = np.zeros_like(acc, dtype=np.int32)

    for i in range(len(acc)):
        real_multiplier = (iscale * wscales[i]) / oscale
        multiplier, shift = quantize_multiplier(real_multiplier)

        r = requantize(np.array([acc[i]]), multiplier, shift)[0]
        res[i] = r

    res = res + ozp

    return res.clip(-128, 127).astype(np.int8)

# ─────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────

interp_ref = Interpreter(model_path=MODEL_PATH)
interp_ref.allocate_tensors()

tds = {t["index"]: t for t in interp_ref.get_tensor_details()}
ops = interp_ref._get_ops_details()

ii = interp_ref.get_input_details()[0]
oi = interp_ref.get_output_details()[0]

constants = {}
for t in interp_ref.get_tensor_details():
    if "Const" in t["name"] or "pseudo_qconst" in t["name"]:
        constants[t["index"]] = interp_ref.get_tensor(t["index"]).copy()

# Use full op execution order
ops_sequence = ops

in_scale  = float(tds[ii["index"]]["quantization_parameters"]["scales"][0])
in_zp     = int(tds[ii["index"]]["quantization_parameters"]["zero_points"][0])
out_scale = float(tds[oi["index"]]["quantization_parameters"]["scales"][0])
out_zp    = int(tds[oi["index"]]["quantization_parameters"]["zero_points"][0])


# ─────────────────────────────────────────────────────────────
# Sweep all 256 inputs
# ─────────────────────────────────────────────────────────────

f_ins, tfl_fs, our_fs, sin_vs, diffs = [], [], [], [], []

for q_in in range(-128, 128):

    f_in = (q_in - in_zp) * in_scale

    # TFLite reference
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    interp.set_tensor(ii["index"], np.array([[q_in]], dtype=np.int8))
    interp.invoke()

    tfl_q = int(interp.get_tensor(oi["index"]).flatten()[0])
    tfl_f = (tfl_q - out_zp) * out_scale

    # Manual forward
    acts = {ii["index"]: np.array([[q_in]], dtype=np.int8)}

    for op in ops_sequence:

        if op["op_name"] == "FULLY_CONNECTED":
            iidx = int(op["inputs"][0])
            widx = int(op["inputs"][1])
            bidx = int(op["inputs"][2])
            oidx = int(op["outputs"][0])

            acts[oidx] = fc_forward(
                acts[iidx],
                constants[widx],
                constants[bidx],
                tds,
                iidx,
                widx,
                bidx,
                oidx
            )

        elif op["op_name"] == "RELU":
            iidx = int(op["inputs"][0])
            oidx = int(op["outputs"][0])

            zp = int(tds[iidx]["quantization_parameters"]["zero_points"][0])
            acts[oidx] = np.maximum(acts[iidx], zp).astype(np.int8)

    our_q = int(acts[oi["index"]].flatten()[0])
    our_f = (our_q - out_zp) * out_scale

    f_ins.append(f_in)
    tfl_fs.append(tfl_f)
    our_fs.append(our_f)
    sin_vs.append(math.sin(f_in))
    diffs.append(our_q - tfl_q)


# ─────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────

f_ins  = np.array(f_ins)
tfl_fs = np.array(tfl_fs)
our_fs = np.array(our_fs)
sin_vs = np.array(sin_vs)
diffs  = np.array(diffs)

mismatches = int(np.sum(diffs != 0))

print(f"\nTotal mismatches: {mismatches}/256")
print(f"Max |diff|: {np.abs(diffs).max()}  Mean |diff|: {np.abs(diffs).mean():.2f}")


# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("TFLite int8 Manual Verification", fontsize=14, fontweight="bold")

ax = axes[0]
ax.plot(f_ins, sin_vs,  "k--", linewidth=1.5, label="sin(x)")
ax.plot(f_ins, tfl_fs,  "b-",  linewidth=2,   label="TFLite")
ax.plot(f_ins, our_fs,  "r-",  linewidth=1.5, label="Manual")
ax.legend()
ax.grid(True)

ax2 = axes[1]
ax2.bar(f_ins, diffs)
ax2.set_title(f"Differences ({mismatches}/256 mismatches)")
ax2.axhline(0, color="black")

plt.tight_layout()
plt.show()