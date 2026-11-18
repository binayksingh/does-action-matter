#!/usr/bin/env python3
"""Efficiency table: parameters, GFLOPs, latency and throughput of each forecaster.

Measured exactly as in evaluation: fp32, batch size 1, model resolution (Cityscapes
256 x 512), one-hot context of 4 frames x 19 classes. Weights do not affect speed, so
freshly initialised models are used (no checkpoint needed).

  Params (M)         trainable parameters
  dParams (M)        extra parameters over the no-action model
  GFLOPs / step      floating-point operations of one forecasting step
                     (torch.utils.flop_counter; one multiply-add = 2 FLOPs)
  Latency (ms)       median GPU time of one step (short term) and of the 3-step
                     autoregressive rollout (mid term), CUDA events, after warm-up
  FPS                1000 / step latency
  Peak mem (MB)      peak GPU memory of the mid-term rollout

The flow-warping baseline is timed on the CPU (OpenCV Farneback at 512 px width plus
warping at full resolution), on one real Cityscapes val sample, as in baselines.py.

Run on a GPU:  python scripts/efficiency.py
Output: printed table + RUNS_DIR/forecast/efficiency.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import flow, paths as P  # noqa: E402

# ============================================================ SETTINGS
SIZE_HW = (256, 512)          # model resolution used for Cityscapes
CONTEXT, NUM_CLASSES = 4, 19
BASE, DEPTH = 64, 4           # as in train.py
MODES = ("none", "concat", "film")
WARMUP, ITERS = 20, 100
TIME_FLOW_BASELINE = True     # CPU timing of Farneback flow warping on one real sample
FLOW_ITERS = 20
OUT_FILE = P.RUNS_DIR / "forecast" / "efficiency.json"
# ======================================================================


def gpu_time_ms(fn, device) -> float:
    import torch
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def measure_model(mode: str, device) -> dict:
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from wamad.models.forecaster import ForecastNet

    m = ForecastNet(num_classes=NUM_CLASSES, context=CONTEXT, action_mode=mode, base=BASE, depth=DEPTH)
    m = m.to(device).eval()
    H, W = SIZE_HW
    lab = torch.randint(0, NUM_CLASSES, (1, CONTEXT, H, W), device=device)
    ctx = torch.nn.functional.one_hot(lab, NUM_CLASSES).permute(0, 1, 4, 2, 3).float()
    a_past = torch.randn(1, CONTEXT, 2, device=device)
    a_next = torch.randn(1, 2, device=device)
    a_fut = torch.randn(1, 3, 2, device=device)

    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            m(ctx, a_past, a_next)
        flops = fc.get_total_flops()
        step = gpu_time_ms(lambda: m(ctx, a_past, a_next), device)
        torch.cuda.reset_peak_memory_stats(device)
        roll = gpu_time_ms(lambda: m.rollout(ctx, a_past, a_fut), device)
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
    return {"params_M": params / 1e6, "gflops_step": flops / 1e9, "latency_step_ms": step,
            "latency_rollout3_ms": roll, "fps_step": 1000.0 / step, "peak_mem_MB": peak}


def measure_flow() -> dict:
    from wamad.data import cityscapes as cs
    from wamad.protocol import HORIZONS
    paths = cs.CityscapesPaths(P.CITYSCAPES_ROOT, P.PSEUDO_ROOT)
    s = cs.build_index(P.CITYSCAPES_ROOT, "val")[0]
    d = cs.load_sample(paths, s, HORIZONS["mid"], load_rgb=True, load_pseudo_targets=False)
    rgb, lab = d["ctx_rgb"], d["ctx_labels"]
    out = {}
    for name, steps in (("step", 1), ("rollout3", 3)):
        flow.flow_baseline(rgb, lab, steps, work_width=512)          # warm-up
        t = []
        for _ in range(FLOW_ITERS):
            t0 = time.perf_counter()
            flow.flow_baseline(rgb, lab, steps, work_width=512)
            t.append((time.perf_counter() - t0) * 1000)
        out[f"latency_{name}_ms"] = statistics.median(t)
    out["fps_step"] = 1000.0 / out["latency_step_ms"]
    out["input_hw"] = list(rgb.shape[1:3])
    return out


def main() -> int:
    import torch
    if not torch.cuda.is_available():
        print("needs a GPU (latency is reported on the GPU)")
        return 1
    device = torch.device("cuda")
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "precision": "fp32",
           "batch": 1, "size_hw": list(SIZE_HW), "models": {}}
    for mode in MODES:
        res["models"][mode] = measure_model(mode, device)
    base = res["models"]["none"]["params_M"]
    for r in res["models"].values():
        r["dparams_M"] = r["params_M"] - base
    if TIME_FLOW_BASELINE:
        try:
            res["flow_warp_cpu"] = measure_flow()
        except Exception as e:  # data path problems must not lose the GPU numbers
            print(f"(flow baseline timing skipped: {e})")

    print(f"\n{res['gpu']} | fp32 | batch 1 | {SIZE_HW[0]}x{SIZE_HW[1]}")
    print(f"{'model':8s} {'Params(M)':>9s} {'dParams':>8s} {'GFLOPs':>8s} {'step ms':>8s} "
          f"{'3-step ms':>9s} {'FPS':>7s} {'mem MB':>7s}")
    for mode, r in res["models"].items():
        print(f"{mode:8s} {r['params_M']:9.2f} {r['dparams_M']:+8.3f} {r['gflops_step']:8.1f} "
              f"{r['latency_step_ms']:8.2f} {r['latency_rollout3_ms']:9.2f} {r['fps_step']:7.1f} {r['peak_mem_MB']:7.0f}")
    if "flow_warp_cpu" in res:
        f = res["flow_warp_cpu"]
        print(f"{'flow(CPU)':8s} {'-':>9s} {'-':>8s} {'-':>8s} {f['latency_step_ms']:8.2f} "
              f"{f['latency_rollout3_ms']:9.2f} {f['fps_step']:7.1f} {'-':>7s}")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(res, indent=2))
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
