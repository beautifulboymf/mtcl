#!/usr/bin/env python
# egl_l1_smoketest.py -- L1 of the EGL safety ladder: the SMALLEST driver-touching test.
#
# What it does, in order (each step announced BEFORE it runs, flushed to stdout AND to
# a timestamped log, so a host death mid-test leaves a forensic record of the exact
# call in flight):
#   A. import mujoco               -> loads libEGL userspace (L0: no kernel contact)
#   B. mujoco.GLContext(64,64)     -> eglInitialize + context creation (FIRST driver ioctl)
#   C. make_current                -> bind context
#   D. glGetString(GL_RENDERER)    -> must say NVIDIA/A100, NOT llvmpipe/softpipe
#                                     (a mesa string means EGL fell back to CPU and the
#                                     test proved nothing about the NVIDIA driver)
#   E. clear + read 1 pixel        -> one real render round-trip
#   F. free                        -> clean teardown
#
# Refuses to run unless: RLINF_ALLOW_EGL=1 (i.e. gpu_render_env.sh was sourced and its
# gates passed), the vendor json points at our bundle, and the target GPU is FULLY idle
# (3 samples: mem<=400MiB, util<=2) -- the one condition the two host crashes shared
# was a SATURATED card (Bug 4905391); this test exists to exercise the safe pattern.
#
# Usage:
#   source /share/fanruochen-local/dev/gpu_render_env.sh   # arms EGL + the key
#   python opd_distill/scripts/egl_l1_smoketest.py --gpu 6
import argparse
import datetime
import os
import subprocess
import sys
import time

LOG = "/share/fanruochen-local/outputs/egl_l1_smoketest.log"


def say(msg: str) -> None:
    line = f"[egl-l1 {datetime.datetime.now():%F %T.%f}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def fail(msg: str) -> None:
    say(f"ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6, help="absolute GPU index (default 6)")
    args = ap.parse_args()

    say(f"=== L1 EGL smoketest start (gpu={args.gpu}, pid={os.getpid()}) ===")

    # gate 1: the key (proves gpu_render_env.sh ran and ITS gates passed)
    if os.environ.get("RLINF_ALLOW_EGL") != "1":
        fail("RLINF_ALLOW_EGL != 1 -- source /share/fanruochen-local/dev/gpu_render_env.sh first")
    vendor = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "")
    if "nvidia-gl-535.179/10_nvidia.json" not in vendor:
        fail(f"vendor json is '{vendor}' -- not our isolated bundle; refusing")
    say(f"gate 1 OK: key present, vendor={vendor}")

    # gate 2: target GPU fully idle, 3 samples 2s apart
    for i in range(3):
        out = subprocess.run(
            ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        mem, util = (int(x) for x in out.replace(" ", "").split(","))
        say(f"gate 2 sample {i + 1}: GPU{args.gpu} mem={mem}MiB util={util}%")
        if mem > 400 or util > 2:
            fail(f"GPU{args.gpu} not fully idle -- the crash condition was a busy card; refusing")
        if i < 2:
            time.sleep(2)

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)

    say("STEP A: import mujoco (userspace .so load only -- no kernel contact yet)")
    import mujoco  # noqa: E402

    say("STEP B: mujoco.GLContext(64, 64) -- eglInitialize + context creation, FIRST driver ioctl")
    ctx = mujoco.GLContext(64, 64)
    say("STEP B done: context created")

    say("STEP C: make_current")
    ctx.make_current()
    say("STEP C done")

    say("STEP D: glGetString(GL_RENDERER)")
    from OpenGL import GL  # noqa: E402

    renderer = (GL.glGetString(GL.GL_RENDERER) or b"?").decode()
    vendor_gl = (GL.glGetString(GL.GL_VENDOR) or b"?").decode()
    say(f"STEP D done: GL_RENDERER='{renderer}' GL_VENDOR='{vendor_gl}'")
    if "NVIDIA" not in vendor_gl.upper() and "NVIDIA" not in renderer.upper():
        fail(
            f"renderer is '{renderer}' -- EGL fell back to a software backend; this run "
            "proved NOTHING about the NVIDIA driver. Check the bundle/env and retry."
        )

    say("STEP E: clear to green + read 1 pixel (one real GPU render round-trip)")
    GL.glClearColor(0.0, 1.0, 0.0, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glFinish()
    px = GL.glReadPixels(0, 0, 1, 1, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
    say(f"STEP E done: pixel={list(px[:3])} (expect [0, 255, 0])")

    say("STEP F: teardown (ctx.free)")
    ctx.free()
    say(f"=== L1_PASS renderer='{renderer}' gpu={args.gpu} ===")


if __name__ == "__main__":
    main()
