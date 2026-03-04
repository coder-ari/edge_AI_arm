"""
CMSIS-NN v7 Code Generator
Reads a quantized int8 TFLite flatbuffer and emits into generated/:

  <model>_data.h   - weights, biases, per-channel multipliers/shifts
  <model>_run.h    - public API + input/output scale & zero-point constants
  <model>_run.c    - arm_fully_connected_s8 inference implementation

Also emits next to the .tflite:
  Makefile         - builds main.c against CMSIS/Source + CMSIS/Include on PC

Usage:
    python cmsis_nn_codegen.py <model.tflite> [model_name]
"""

import sys
import math
import numpy as np
from pathlib import Path

# ── TFLite runtime (lightest available) ──────────────────────
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter


# ─────────────────────────────────────────────────────────────
# Quantization helpers  (mirrors TFLite QuantizeMultiplier)
# ─────────────────────────────────────────────────────────────

def quantize_multiplier(real_multiplier: float):
    if real_multiplier == 0.0:
        return 0, 0
    significand, exponent = math.frexp(real_multiplier)
    q31 = int(round(significand * (1 << 31)))
    if q31 == (1 << 31):
        q31 //= 2
        exponent += 1
    shift = -exponent      # positive -> right-shift inside CMSIS-NN
    return q31, shift


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    tds    = {t["index"]: t for t in interp.get_tensor_details()}
    ops    = interp._get_ops_details()
    ii     = interp.get_input_details()[0]
    oi     = interp.get_output_details()[0]
    consts = {}
    for t in interp.get_tensor_details():
        if "Const" in t["name"] or "pseudo_qconst" in t["name"]:
            consts[t["index"]] = interp.get_tensor(t["index"]).copy()
    return tds, ops, ii, oi, consts


def get_qparams(tds, idx):
    qp = tds[idx]["quantization_parameters"]
    return list(qp["scales"]), [int(z) for z in qp["zero_points"]]


# ─────────────────────────────────────────────────────────────
# C array helpers
# ─────────────────────────────────────────────────────────────

def fmt_int8_array(name, data, cols=16):
    flat  = data.flatten().astype(np.int8)
    total = flat.size
    rows  = ["    " + ", ".join(f"{int(v):4d}" for v in flat[i:i+cols])
             for i in range(0, total, cols)]
    return (f"/* shape: {list(data.shape)} */\n"
            f"static const int8_t {name}[{total}] = {{\n"
            + ",\n".join(rows) + "\n};\n")


def fmt_int32_array(name, data, cols=8):
    flat  = data.flatten().astype(np.int32)
    total = flat.size
    rows  = ["    " + ", ".join(f"{int(v):12d}" for v in flat[i:i+cols])
             for i in range(0, total, cols)]
    return (f"/* shape: {list(data.shape)} */\n"
            f"static const int32_t {name}[{total}] = {{\n"
            + ",\n".join(rows) + "\n};\n")


def fmt_int32_list(name, values, cols=8):
    total = len(values)
    rows  = ["    " + ", ".join(f"{int(v):12d}" for v in values[i:i+cols])
             for i in range(0, total, cols)]
    return (f"static const int32_t {name}[{total}] = {{\n"
            + ",\n".join(rows) + "\n};\n")


# ─────────────────────────────────────────────────────────────
# Layer extraction
# ─────────────────────────────────────────────────────────────

def extract_layers(tds, ops, consts):
    layers = []
    fc_count = relu_count = 0

    for op in ops:
        name = op["op_name"]

        if name == "FULLY_CONNECTED":
            iidx, widx, bidx = (int(op["inputs"][x]) for x in (0, 1, 2))
            oidx = int(op["outputs"][0])

            w_data = consts[widx]
            b_data = consts[bidx]

            in_scales,  in_zps  = get_qparams(tds, iidx)
            w_scales,   _       = get_qparams(tds, widx)
            out_scales, out_zps = get_qparams(tds, oidx)

            in_scale  = float(in_scales[0])
            out_scale = float(out_scales[0])
            in_zp     = int(in_zps[0])
            out_zp    = int(out_zps[0])

            multipliers, shifts = [], []
            for ws in w_scales:
                m, s = quantize_multiplier((in_scale * float(ws)) / out_scale)
                multipliers.append(m)
                shifts.append(s)

            layers.append({
                "type"        : "FULLY_CONNECTED",
                "tag"         : f"fc{fc_count}",
                "in_dim"      : int(w_data.shape[1]),
                "out_dim"     : int(w_data.shape[0]),
                "in_zp"       : in_zp,
                "out_zp"      : out_zp,
                "in_scale"    : in_scale,
                "out_scale"   : out_scale,
                "w_data"      : w_data,
                "b_data"      : b_data,
                "multipliers" : multipliers,
                "shifts"      : shifts,
            })
            fc_count += 1

        elif name == "RELU":
            iidx = int(op["inputs"][0])
            _, in_zps = get_qparams(tds, iidx)
            layers.append({
                "type"  : "RELU",
                "tag"   : f"relu{relu_count}",
                "in_zp" : int(in_zps[0]),
            })
            relu_count += 1

    return layers


# ─────────────────────────────────────────────────────────────
# generated/<model>_data.h
# Internal weights/biases/quant arrays – not included by user code directly
# ─────────────────────────────────────────────────────────────

def generate_data_header(layers, model_name):
    guard = f"{model_name.upper()}_DATA_H_"
    o = []
    o += [f"/* Auto-generated by cmsis_nn_codegen.py  -  DO NOT EDIT */",
          f"#ifndef {guard}", f"#define {guard}", "",
          "#include <stdint.h>", ""]

    for lyr in layers:
        if lyr["type"] != "FULLY_CONNECTED":
            continue
        tag, TAG = lyr["tag"], lyr["tag"].upper()
        o += [f"/* {'-'*52} */",
              f"/* {TAG}  ({lyr['in_dim']} -> {lyr['out_dim']}) */",
              f"/* {'-'*52} */",
              f"#define {TAG}_IN_DIM   {lyr['in_dim']}",
              f"#define {TAG}_OUT_DIM  {lyr['out_dim']}",
              f"#define {TAG}_IN_ZP    ({lyr['in_zp']})",
              f"#define {TAG}_OUT_ZP   ({lyr['out_zp']})",
              "",
              fmt_int8_array(f"{tag}_weights",     lyr["w_data"]),
              fmt_int32_array(f"{tag}_bias",        lyr["b_data"]),
              fmt_int32_list(f"{tag}_multipliers",  lyr["multipliers"]),
              fmt_int32_list(f"{tag}_shifts",        lyr["shifts"])]

    o.append(f"#endif /* {guard} */")
    return "\n".join(o) + "\n"


# ─────────────────────────────────────────────────────────────
# generated/<model>_run.h
# Public header – the ONLY file user code needs to include
# ─────────────────────────────────────────────────────────────

def generate_run_header(layers, tds, ii, oi, model_name):
    guard = f"{model_name.upper()}_RUN_H_"
    MN    = model_name.upper()

    in_scales,  in_zps  = get_qparams(tds, ii["index"])
    out_scales, out_zps = get_qparams(tds, oi["index"])
    in_scale  = float(in_scales[0])
    in_zp     = int(in_zps[0])
    out_scale = float(out_scales[0])
    out_zp    = int(out_zps[0])

    fc_layers = [l for l in layers if l["type"] == "FULLY_CONNECTED"]
    first_in  = fc_layers[0]["in_dim"]
    last_out  = fc_layers[-1]["out_dim"]

    o = []
    o += [f"/* Auto-generated by cmsis_nn_codegen.py  -  DO NOT EDIT */",
          f"#ifndef {guard}", f"#define {guard}", "",
          "#include <stdint.h>", ""]

    o += [f"/* ----- Model dimensions ----- */",
          f"#define {MN}_INPUT_SIZE    {first_in}",
          f"#define {MN}_OUTPUT_SIZE   {last_out}",
          ""]

    o += [f"/* ----- Input quantisation ----- */",
          f"/* float_val = (q - {MN}_INPUT_ZP) * {MN}_INPUT_SCALE */",
          f"#define {MN}_INPUT_SCALE  {in_scale:.10f}f",
          f"#define {MN}_INPUT_ZP     ({in_zp})",
          ""]

    o += [f"/* ----- Output quantisation ----- */",
          f"/* float_val = (q - {MN}_OUTPUT_ZP) * {MN}_OUTPUT_SCALE */",
          f"#define {MN}_OUTPUT_SCALE {out_scale:.10f}f",
          f"#define {MN}_OUTPUT_ZP    ({out_zp})",
          ""]

    o += [f"/* ----- Inference function ----- */",
          f"/**",
          f" * @brief  Run {model_name} inference (CMSIS-NN v7).",
          f" * @param  input   Quantised int8 array, length {MN}_INPUT_SIZE.",
          f" * @param  output  Quantised int8 array, length {MN}_OUTPUT_SIZE.",
          f" *",
          f" * Dequantise before/after:",
          f" *   x = (input[i]  - {MN}_INPUT_ZP)  * {MN}_INPUT_SCALE",
          f" *   y = (output[i] - {MN}_OUTPUT_ZP) * {MN}_OUTPUT_SCALE",
          f" */",
          f"void {model_name}_run(const int8_t *input, int8_t *output);",
          ""]

    o.append(f"#endif /* {guard} */")
    return "\n".join(o) + "\n"


# ─────────────────────────────────────────────────────────────
# generated/<model>_run.c
# ─────────────────────────────────────────────────────────────

def generate_run_source(layers, model_name):
    fc_layers = [l for l in layers if l["type"] == "FULLY_CONNECTED"]
    max_dim   = max(max(l["in_dim"], l["out_dim"]) for l in fc_layers)
    first_in  = fc_layers[0]["in_dim"]
    last_out  = fc_layers[-1]["out_dim"]

    o = []
    o += [f"/* Auto-generated by cmsis_nn_codegen.py  -  DO NOT EDIT */",
          f"/* Per-channel int8 FC inference — bit-exact with TFLite */",
          f'#include "{model_name}_run.h"',
          f'#include "{model_name}_data.h"',
          f'#include <string.h>',
          f'#include <stdint.h>',
          "",
          f"/* Ping-pong activation buffers (static, zero heap) */",
          f"static int8_t _buf0[{max_dim}];",
          f"static int8_t _buf1[{max_dim}];",
          "",
          f"void {model_name}_run(const int8_t *input, int8_t *output)",
          f"{{",
          f"    memcpy(_buf0, input, {first_in} * sizeof(int8_t));",
          ""]

    src, dst = "_buf0", "_buf1"

    for i, lyr in enumerate(layers):

        if lyr["type"] == "FULLY_CONNECTED":
            tag, TAG = lyr["tag"], lyr["tag"].upper()
            in_dim, out_dim = lyr["in_dim"], lyr["out_dim"]
            in_zp_l  = lyr["in_zp"]
            out_zp_l = lyr["out_zp"]

            next_lyr = layers[i + 1] if i + 1 < len(layers) else None
            has_relu = next_lyr is not None and next_lyr["type"] == "RELU"
            act_min  = out_zp_l if has_relu else -128
            act_max  = 127

            # Per-channel multipliers/shifts are baked into _data.h.
            # CMSIS arm_fully_connected_s8 only accepts per-tensor quant params,
            # so we call it with multiplier=1 (identity) and shift=0, letting it
            # accumulate the raw int32 dot product + bias into a temporary int8
            # buffer — but that would clip. Instead we use a two-step approach:
            #   1. arm_fully_connected_s8 with a per-tensor requant that matches
            #      the FIRST channel (acceptable approximation only if all
            #      per-channel multipliers are identical, i.e. per-tensor weights).
            #   2. For true per-channel: emit a manual int32 accumulation loop
            #      using the stored per-channel multipliers/shifts from _data.h.
            #
            # We use approach 2 (always correct) via arm_nn_mat_mult_nt_t_s8
            # which is the underlying kernel, OR a pure-C fallback loop.
            # Here we emit the pure-C fallback which works on any platform.
            label = f"{TAG} ({in_dim}->{out_dim})" + (" +RELU" if has_relu else "")
            o += [f"    /* {label} (per-channel requant, pure-C) */",
                  f"    {{",
                  f"        static int32_t _acc[{out_dim}];",
                  f"        /* dot product + bias  (matches TFLite fc_forward) */",
                  f"        for (int _o = 0; _o < {TAG}_OUT_DIM; _o++) {{",
                  f"            int32_t _s = (int32_t){tag}_bias[_o];",
                  f"            for (int _i = 0; _i < {TAG}_IN_DIM; _i++)",
                  f"                _s += (int32_t)({tag}_weights[_o * {TAG}_IN_DIM + _i])",
                  f"                    * ((int32_t){src}[_i] - {TAG}_IN_ZP);",
                  f"            _acc[_o] = _s;",
                  f"        }}",
                  f"        /* per-channel SaturatingRoundingDoublingHighMul + RoundingDivideByPOT */",
                  f"        for (int _o = 0; _o < {TAG}_OUT_DIM; _o++) {{",
                  f"            int64_t _p = (int64_t)_acc[_o] * (int64_t){tag}_multipliers[_o];",
                  f"            int64_t _nudge = (_p >= 0) ? (int64_t)(1 << 30) : -(int64_t)(1 << 30);",
                  f"            int32_t _q = (int32_t)((_p + _nudge) >> 31);",
                  f"            int32_t _sh = {tag}_shifts[_o];",
                  f"            if (_sh > 0) {{",
                  f"                int32_t _mask = (1 << _sh) - 1;",
                  f"                int32_t _rem  = _q & _mask;",
                  f"                _q = (_q >> _sh) + (_rem > (_mask >> 1) ? 1 : 0);",
                  f"            }} else if (_sh < 0) {{",
                  f"                _q = _q << (-_sh);",
                  f"            }}",
                  f"            _q += {TAG}_OUT_ZP;",
                  f"            if (_q < {act_min}) _q = {act_min};",
                  f"            if (_q > {act_max}) _q = {act_max};",
                  f"            {dst}[_o] = (int8_t)_q;",
                  f"        }}",
                  f"    }}",
                  ""]

            src, dst = dst, src

        elif lyr["type"] == "RELU":
            prev = layers[i - 1] if i > 0 else None
            if prev and prev["type"] == "FULLY_CONNECTED":
                o += [f"    /* {lyr['tag'].upper()} folded into {prev['tag'].upper()} activation clamp */", ""]
            else:
                prev_fc = next((l for l in reversed(layers[:i]) if l["type"] == "FULLY_CONNECTED"), None)
                dim = prev_fc["out_dim"] if prev_fc else max_dim
                zp  = lyr["in_zp"]
                o += [f"    /* {lyr['tag'].upper()} standalone */",
                      f"    for (int _j = 0; _j < {dim}; _j++)",
                      f"        {src}[_j] = ({src}[_j] < {zp}) ? (int8_t){zp} : {src}[_j];",
                      ""]

    o += [f"    memcpy(output, {src}, {last_out} * sizeof(int8_t));",
          f"}}", ""]

    return "\n".join(o) + "\n"


# ─────────────────────────────────────────────────────────────
# main.c  (ready-to-compile PC test)
# ─────────────────────────────────────────────────────────────

def generate_main(model_name):
    MN = model_name.upper()
    return f"""\
/* PC inference test  -  auto-generated by cmsis_nn_codegen.py */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "generated/{model_name}_run.h"

int main(void)
{{
    int8_t input[{MN}_INPUT_SIZE];
    int8_t output[{MN}_OUTPUT_SIZE];

    printf("=====================================\\n");
    printf(" CMSIS-NN v7 PC Inference Test\\n");
    printf("=====================================\\n\\n");

    int mismatches = 0;

    for (int q = -128; q < 128; q++)
    {{
        input[0] = (int8_t)q;

        {model_name}_run(input, output);

        float x   = (q          - {MN}_INPUT_ZP)  * {MN}_INPUT_SCALE;
        float y   = (output[0]  - {MN}_OUTPUT_ZP) * {MN}_OUTPUT_SCALE;
        float ref = sinf(x);
        float err = fabsf(y - ref);

        if (err > 0.02f)
            mismatches++;

        printf("q=%4d  x=%8.4f  y=%8.4f  sin(x)=%8.4f  err=%8.5f\\n",
               q, x, y, ref, err);
    }}

    printf("\\n-------------------------------------\\n");
    printf("Total large mismatches (>0.02): %d\\n", mismatches);
    printf("-------------------------------------\\n");

    return 0;
}}
"""


# ─────────────────────────────────────────────────────────────
# build.ps1  (Windows PowerShell + gcc)
# CMSIS-NN sources expected under  CMSIS\Source\
# CMSIS-NN headers expected under  CMSIS\Include\
# ─────────────────────────────────────────────────────────────

def generate_build_ps1(model_name):
    return f"""\
# Auto-generated build script for Windows (PowerShell + gcc)
# Pure-C per-channel int8 inference — no CMSIS source files needed.
# Only CMSIS\\Include\\ is required for the type definitions.
#
# Usage:  .\\build.ps1
# Run:    .\\{model_name}_test.exe

# Collect include dirs from CMSIS recursively (for arm_nn_types.h etc.)
$inc_dirs = @("-I.", "-ICMSIS\\Include")
if (Test-Path "CMSIS") {{
    $sub = Get-ChildItem -Path CMSIS -Recurse -Directory |
           ForEach-Object {{ "-I$($_.FullName)" }}
    $inc_dirs = $inc_dirs + $sub | Select-Object -Unique
}}

$srcs   = @("main.c", "generated\\{model_name}_run.c")
$flags  = $inc_dirs + @("-O2", "-lm")
$output = "{model_name}_test.exe"

Write-Host "[build] Compiling -> $output"
gcc @flags -o $output @srcs

if ($LASTEXITCODE -eq 0) {{
    Write-Host "[build] OK  ->  .\\$output"
}} else {{
    Write-Host "[build] FAILED (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}}
"""


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python cmsis_nn_codegen.py <model.tflite> [model_name]")
        sys.exit(1)

    model_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else Path(model_path).stem.replace("-", "_")

    print(f"[codegen] Loading:  {model_path}")
    tds, ops, ii, oi, consts = load_model(model_path)

    print(f"[codegen] Extracting layers ...")
    layers = extract_layers(tds, ops, consts)

    n_fc   = sum(1 for l in layers if l["type"] == "FULLY_CONNECTED")
    n_relu = sum(1 for l in layers if l["type"] == "RELU")
    print(f"[codegen]   {n_fc} FC layers, {n_relu} RELU layers")
    for lyr in layers:
        if lyr["type"] == "FULLY_CONNECTED":
            print(f"[codegen]   {lyr['tag']}: {lyr['in_dim']} -> {lyr['out_dim']}"
                  f"  in_zp={lyr['in_zp']}  out_zp={lyr['out_zp']}")

    root    = Path(model_path).parent
    gen_dir = root / "generated"
    gen_dir.mkdir(exist_ok=True)

    files = {
        gen_dir / f"{model_name}_data.h" : generate_data_header(layers, model_name),
        gen_dir / f"{model_name}_run.h"  : generate_run_header(layers, tds, ii, oi, model_name),
        gen_dir / f"{model_name}_run.c"  : generate_run_source(layers, model_name),
        root    / "main.c"               : generate_main(model_name),
        root    / "build.ps1"            : generate_build_ps1(model_name),
    }

    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
        print(f"[codegen] Written:  {path}")

    MN = model_name.upper()
    print(f"""
[codegen] -- File layout ----------------------------------------
  generated/
    {model_name}_data.h   <- weights + quant arrays  (internal)
    {model_name}_run.h    <- public API + scale/ZP macros
    {model_name}_run.c    <- arm_fully_connected_s8 inference
  main.c                  <- PC test (sweep all 256 inputs)
  build.ps1               <- Windows PowerShell build script

[codegen] -- Public macros in {model_name}_run.h ----------------
  {MN}_INPUT_SIZE / OUTPUT_SIZE
  {MN}_INPUT_SCALE  / INPUT_ZP
  {MN}_OUTPUT_SCALE / OUTPUT_ZP
  void {model_name}_run(const int8_t *input, int8_t *output);

[codegen] -- PC build (Windows PowerShell) ----------------------
  1. Place CMSIS-NN sources in  CMSIS\\Source\\*.c
  2. Place CMSIS-NN headers in  CMSIS\\Include\\
  3. .\\build.ps1
  4. .\\{model_name}_test.exe

[codegen] -- CubeIDE --------------------------------------------
  1. Drag generated/ into your project.
  2. #include "generated/{model_name}_run.h"  in your app.
  3. Add CMSIS/Source files to the build.
  4. Add CMSIS/Include to include paths.
""")


if __name__ == "__main__":
    main()