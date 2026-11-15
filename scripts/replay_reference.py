#!/usr/bin/env python3
"""
Replay the ESP32 DDIM sampler in the Python reference implementation.

The firmware's `SEED <n>` command replaces esp_random() with a xorshift32
stream feeding Box-Muller, so x_T can be reproduced bit-for-bit here.  This
script builds that x_T, runs the same 8/16/32-step DDIM loop through the INT8
TFLite model with the tensor's runtime quantization parameters, and optionally
compares each intermediate statistic with a captured firmware log.

Usage:
    PYTHONPATH=. python scripts/replay_reference.py --seed 12345 --class 0 \
        --log /tmp/opencode/serial_seed.log
"""
import argparse
import re
import sys

import numpy as np
import tensorflow as tf

from src.diffusion import NoiseSchedule
from src.scheduler import DDIMScheduler
from src.utils import load_config

MASK32 = 0xFFFFFFFF


def xorshift32(state: int) -> int:
    s = state & MASK32
    s ^= (s << 13) & MASK32
    s ^= s >> 17
    s ^= (s << 5) & MASK32
    return s & MASK32


def firmware_noise(seed: int, count: int) -> np.ndarray:
    """Exact replica of main.cpp generate_gaussian_noise() (float32 Box-Muller)."""
    state = seed
    spare = np.float32(0.0)
    has_spare = False
    out = np.empty(count, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)
    for i in range(count):
        if has_spare:
            out[i] = spare
            has_spare = False
            continue
        state = xorshift32(state)
        u1 = state & 0xFFFFFF
        state = xorshift32(state)
        u2 = state & 0xFFFFFF

        f1 = np.float32(np.float32(u1) / np.float32(16777216.0))
        f2 = np.float32(np.float32(u2) / np.float32(16777216.0))
        if f1 == 0.0:
            f1 = np.float32(1e-6)
        radius = np.float32(np.sqrt(np.float32(-2.0) * np.log(f1)))
        out[i] = np.float32(radius * np.cos(np.float32(two_pi * f2)))
        spare = np.float32(radius * np.sin(np.float32(two_pi * f2)))
        has_spare = True
    return out


def stats(a: np.ndarray) -> tuple:
    flat = np.asarray(a, dtype=np.float64).ravel()
    return (float(flat.min()), float(flat.max()), float(flat.mean()), float(flat.std()))


def ascii_bitmap(img: np.ndarray) -> str:
    rows = []
    for y in range(img.shape[0]):
        rows.append("".join("#" if v >= 0.5 else ("+" if v < 0.0 else " ")
                            for v in img[y]))
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--class", dest="cls", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--model", default="artifacts/model_int8.tflite")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--log", default=None, help="firmware serial log to compare")
    args = ap.parse_args()

    cfg = load_config(args.config)
    schedule = NoiseSchedule(cfg)
    scheduler = DDIMScheduler(cfg, schedule)

    interp = tf.lite.Interpreter(model_path=args.model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    iscale, izp = inp["quantization"]
    oscale, ozp = out["quantization"]
    if inp["shape"].tolist() != [1, 28, 28, 12]:
        raise SystemExit(f"unexpected input shape {inp['shape'].tolist()}")
    print(f"model input  NHWC {inp['shape'].tolist()} scale={iscale!r} zp={izp}")
    print(f"model output {out['shape'].tolist()} scale={oscale!r} zp={ozp}")

    timesteps = scheduler.get_ddim_timesteps(args.steps)
    print(f"timesteps {timesteps.tolist()}")

    x = firmware_noise(args.seed, 28 * 28).reshape(1, 28, 28, 1)
    print(f"x_T        {fmt(stats(x))}  (firmware seed={args.seed})")

    log_steps = {}
    if args.log:
        log_steps = parse_firmware_log(args.log)
        print(f"parsed {len(log_steps)} steps from {args.log}")

    records = []
    for step, t in enumerate(timesteps):
        t_prev = int(timesteps[step + 1]) if step + 1 < len(timesteps) else -1

        model_input = np.concatenate(
            [x,
             np.broadcast_to(np.eye(10, dtype=np.float32)[args.cls],
                             (28, 28, 10))[None],
             np.full((1, 28, 28, 1), np.float32(t) / np.float32(schedule.num_timesteps),
                     dtype=np.float32)],
            axis=-1).astype(np.float32)
        q = np.clip(np.round(model_input / iscale + izp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        noise_pred = ((interp.get_tensor(out["index"]).astype(np.float32) - ozp)
                      * oscale).astype(np.float32)

        sx = stats(x)
        sp = stats(noise_pred)
        x_next = scheduler.ddim_step(x, noise_pred, int(t), t_prev, 0.0)
        sn = stats(x_next)

        records.append({"step": step + 1, "t": int(t), "t_prev": t_prev,
                        "x_t": sx, "noise_pred": sp, "x_next": sn})

        print(f"Step {step + 1}/{len(timesteps)}: t={t} t_prev={t_prev}")
        print(f"  x_t        {fmt(sx)}")
        print(f"  noise_pred {fmt(sp)}")
        print(f"  x_next     {fmt(sn)}")
        if step == 0:
            print(f"  input q: image={q[0, 0, 0, 0]} class_ch0={q[0, 0, 0, 1]} "
                  f"class_ch1={q[0, 0, 0, 2]} timestep={q[0, 0, 0, 11]}")
        x = x_next

    img = x[0, :, :, 0]
    print(f"FINAL      {fmt(stats(img))}")
    print(f"FINAL pixels >= 0.5: {int((img >= 0.5).sum())} / 784")
    print("FINAL bitmap:")
    print(ascii_bitmap(img))

    if log_steps:
        print("\n=== firmware vs python (max |delta| per statistic) ===")
        first_divergence = None
        for rec in records:
            fw = log_steps.get(rec["step"])
            if not fw:
                continue
            for key in ("x_t", "noise_pred", "x_next"):
                deltas = [abs(a - b) for a, b in zip(rec[key], fw[key])]
                worst = max(deltas)
                print(f"  step {rec['step']} {key:10s} "
                      f"min {deltas[0]:.4f} max {deltas[1]:.4f} "
                      f"mean {deltas[2]:.4f} std {deltas[3]:.4f}")
                if worst > 0.05 and first_divergence is None:
                    first_divergence = (rec["step"], key, worst)
        print("first divergence > 0.05:",
              "none" if first_divergence is None else first_divergence)

    return 0


def fmt(s: tuple) -> str:
    return (f"min={s[0]:+.4f} max={s[1]:+.4f} mean={s[2]:+.4f} std={s[3]:.4f}")


STAT_RE = re.compile(
    r"^\s*(x_t|noise_pred|x_next)\s+min=([-+0-9.]+) max=([-+0-9.]+) "
    r"mean=([-+0-9.]+) std=([0-9.]+)", re.M)


def parse_firmware_log(path: str) -> dict:
    """Extract per-step statistics from a captured serial log."""
    with open(path, "r", errors="replace") as f:
        text = f.read()
    steps = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"Step (\d+)/\d+: t=(-?\d+)", line)
        if m:
            current = int(m.group(1))
            steps.setdefault(current, {})
            continue
        m = STAT_RE.match(line)
        if m and current is not None:
            steps[current][m.group(1)] = tuple(float(m.group(i)) for i in range(2, 6))
    return steps


if __name__ == "__main__":
    sys.exit(main())
