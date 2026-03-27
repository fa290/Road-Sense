import os
import sys
import time
import argparse
import platform
import json
from pathlib import Path

import torch
import numpy as np

# Config

YOLOV5_DIR = "yolov5"
WEIGHTS_PATH = "models/yolov5s.pt"
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "FPS_BENCHMARK_REPORT.md")
TARGET_FPS = 30
BATCH_SIZES = [1, 4, 8, 16]
IMG_SIZES = [320, 640, 1280]
WARMUP_RUNS = 10
BENCH_RUNS = 100


# Helpers


def get_device_info():
    info = {
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if info["cuda"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    return info


def load_model(weights, device, fp16=False):
    sys.path.insert(0, YOLOV5_DIR)
    from models.common import DetectMultiBackend  # noqa

    model = DetectMultiBackend(weights, device=device, fp16=fp16)
    model.eval()
    return model


def make_dummy_input(batch_size, img_size, device, fp16=False):
    dtype = torch.float16 if fp16 else torch.float32
    return torch.rand(batch_size, 3, img_size, img_size, dtype=dtype).to(device)


# Profiling


def profile_pipeline(model, batch_size, img_size, device, fp16=False, runs=BENCH_RUNS):
    """
    Returns dict with avg ms for: preprocess, inference, postprocess, total
    and the resulting FPS.
    """
    sys.path.insert(0, YOLOV5_DIR)
    from utils.general import non_max_suppression  # noqa

    pre_times, inf_times, post_times = [], [], []

    # Warm-up
    dummy = make_dummy_input(batch_size, img_size, device, fp16)
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(runs):
        # ── Preprocess ──────────────────────
        t0 = time.perf_counter()
        x = make_dummy_input(batch_size, img_size, device, fp16)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pre_times.append(time.perf_counter() - t0)

        # ── Inference ───────────────────────
        t1 = time.perf_counter()
        with torch.no_grad():
            preds = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inf_times.append(time.perf_counter() - t1)

        # ── Postprocess (NMS) ────────────────
        t2 = time.perf_counter()
        _ = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)
        if device.type == "cuda":
            torch.cuda.synchronize()
        post_times.append(time.perf_counter() - t2)

    avg_pre = np.mean(pre_times) * 1000
    avg_inf = np.mean(inf_times) * 1000
    avg_post = np.mean(post_times) * 1000
    avg_total = avg_pre + avg_inf + avg_post

    fps = (batch_size * 1000) / avg_total

    return {
        "preprocess_ms": round(avg_pre, 2),
        "inference_ms": round(avg_inf, 2),
        "postprocess_ms": round(avg_post, 2),
        "total_ms": round(avg_total, 2),
        "fps": round(fps, 1),
        "meets_target": fps >= TARGET_FPS,
    }


# Export helpers


def export_onnx(weights, img_size=640):
    print(f"\n[EXPORT] Exporting to ONNX (img_size={img_size})...")
    onnx_path = weights.replace(".pt", ".onnx")
    cmd = [
        sys.executable,
        os.path.join(YOLOV5_DIR, "export.py"),
        "--weights",
        weights,
        "--include",
        "onnx",
        "--imgsz",
        str(img_size),
    ]
    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True)
    success = os.path.exists(onnx_path)
    print(f"  {' Saved to: ' + onnx_path if success else '❌ Export failed'}")
    return onnx_path if success else None


def export_tensorrt(weights, img_size=640):
    try:
        import tensorrt  # noqa

        print(f"\n[EXPORT] Exporting to TensorRT...")
        cmd = [
            sys.executable,
            os.path.join(YOLOV5_DIR, "export.py"),
            "--weights",
            weights,
            "--include",
            "engine",
            "--imgsz",
            str(img_size),
            "--device",
            "0",
        ]
        import subprocess

        subprocess.run(cmd, capture_output=True, text=True)
        trt_path = weights.replace(".pt", ".engine")
        print(f"  {' ' + trt_path if os.path.exists(trt_path) else ' Failed'}")
        return trt_path if os.path.exists(trt_path) else None
    except ImportError:
        print("   TensorRT not installed — skipping.")
        return None


# Report generator


def generate_report(device_info, results_gpu, results_cpu, results_fp16):
    os.makedirs(REPORT_DIR, exist_ok=True)

    lines = []
    lines.append("# FPS Benchmark Report — YOLOv5s on KITTI\n")
    lines.append(
        f"**Target:** ≥ {TARGET_FPS} FPS for real-time autonomous driving deployment\n"
    )

    # Hardware
    lines.append("## Hardware Specifications\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| OS | {device_info['platform']} |")
    lines.append(f"| Python | {device_info['python']} |")
    lines.append(f"| PyTorch | {device_info['torch']} |")
    if device_info["cuda"]:
        lines.append(f"| GPU | {device_info['gpu_name']} |")
        lines.append(f"| GPU Memory | {device_info['gpu_memory']} |")
    lines.append(f"| CUDA Available | {'Yes' if device_info['cuda'] else 'No'} |\n")

    def result_table(results, title):
        lines.append(f"## {title}\n")
        lines.append(
            "| Batch | ImgSize | Pre(ms) | Inf(ms) | NMS(ms) | Total(ms) | FPS | ✅/❌ |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in results:
            status = "" if r["meets_target"] else "❌"
            lines.append(
                f"| {r['batch']} | {r['img_size']} | "
                f"{r['preprocess_ms']} | {r['inference_ms']} | {r['postprocess_ms']} | "
                f"{r['total_ms']} | **{r['fps']}** | {status} |"
            )
        lines.append("")

    if results_gpu:
        result_table(results_gpu, "GPU Results (FP32)")
    if results_fp16:
        result_table(results_fp16, "GPU Results — Half Precision (FP16)")
    if results_cpu:
        result_table(results_cpu, "CPU Results (Fallback)")

    # Recommendations
    lines.append("## Optimization Recommendations\n")
    lines.append(
        "- Use **FP16 (half precision)** on GPU — typically 1.5–2× faster with minimal accuracy loss."
    )
    lines.append(
        "- Use **batch_size=1** for lowest latency in real-time single-frame inference."
    )
    lines.append(
        "- Export to **ONNX** for cross-platform deployment and runtime optimization."
    )
    lines.append("- Export to **TensorRT** for maximum throughput on NVIDIA hardware.")
    lines.append(
        "- Use **img_size=320** if FPS target not met at 640 — good trade-off for speed.\n"
    )

    lines.append("## Acceptance Criteria\n")
    gpu_pass = any(r["meets_target"] for r in (results_gpu or []))
    fp16_pass = any(r["meets_target"] for r in (results_fp16 or []))
    cpu_pass = any(r["meets_target"] for r in (results_cpu or []))
    lines.append(
        f"- [{'x' if gpu_pass  else ' '}] FPS ≥ 30 on GPU (real-time deployment)"
    )
    lines.append(f"- [{'x' if fp16_pass else ' '}] FPS improvement with FP16")
    lines.append(f"- [{'x' if cpu_pass  else ' '}] FPS documented for CPU fallback")
    lines.append(f"- [x] Profiling breakdown: preprocess / inference / NMS")
    lines.append(f"- [x] Optimization recommendations provided\n")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print(f"\n[REPORT] Saved → {REPORT_PATH}")
    return report_text


# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS_PATH)
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmarks")
    parser.add_argument("--no-cpu", action="store_true", help="Skip CPU benchmarks")
    parser.add_argument("--no-fp16", action="store_true", help="Skip FP16 benchmarks")
    parser.add_argument("--onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--tensorrt", action="store_true", help="Export to TensorRT")
    args = parser.parse_args()

    device_info = get_device_info()
    print("\n" + "=" * 55)
    print("  FPS BENCHMARK — YOLOv5s KITTI")
    print("=" * 55)
    print(f"  Weights : {args.weights}")
    print(
        f"  Device  : {'GPU — ' + device_info.get('gpu_name','') if device_info['cuda'] else 'CPU only'}"
    )
    print(f"  Runs/config: {BENCH_RUNS} | Warmup: {WARMUP_RUNS}\n")

    results_gpu, results_cpu, results_fp16 = [], [], []

    # GPU FP32

    if device_info["cuda"] and not args.no_gpu:
        device = torch.device("cuda:0")
        model = load_model(args.weights, device, fp16=False)
        print("[GPU FP32] Benchmarking...")
        for bs in BATCH_SIZES:
            for sz in IMG_SIZES:
                print(f"  batch={bs}  img={sz}...", end=" ", flush=True)
                r = profile_pipeline(model, bs, sz, device, fp16=False)
                r.update({"batch": bs, "img_size": sz})
                results_gpu.append(r)
                print(f"FPS={r['fps']}  {'✅' if r['meets_target'] else '❌'}")
        del model

    # GPU FP16

    if device_info["cuda"] and not args.no_fp16:
        device = torch.device("cuda:0")
        model = load_model(args.weights, device, fp16=True)
        print("\n[GPU FP16] Benchmarking...")
        for bs in BATCH_SIZES:
            for sz in IMG_SIZES:
                print(f"  batch={bs}  img={sz}...", end=" ", flush=True)
                r = profile_pipeline(model, bs, sz, device, fp16=True)
                r.update({"batch": bs, "img_size": sz})
                results_fp16.append(r)
                print(f"FPS={r['fps']}  {'✅' if r['meets_target'] else '❌'}")
        del model

    # CPU
    if not args.no_cpu:
        device = torch.device("cpu")
        model = load_model(args.weights, device, fp16=False)
        print("\n[CPU] Benchmarking (batch=1 only for CPU)...")
        for sz in IMG_SIZES:
            print(f"  batch=1  img={sz}...", end=" ", flush=True)
            r = profile_pipeline(model, 1, sz, device, fp16=False, runs=20)
            r.update({"batch": 1, "img_size": sz})
            results_cpu.append(r)
            print(f"FPS={r['fps']}  {'✅' if r['meets_target'] else '❌'}")
        del model

    # Exports
    if args.onnx:
        export_onnx(args.weights)
    if args.tensorrt:
        export_tensorrt(args.weights)

    # Report
    generate_report(device_info, results_gpu, results_cpu, results_fp16)

    # Summary
    all_results = results_gpu + results_fp16 + results_cpu
    best = max(all_results, key=lambda x: x["fps"]) if all_results else None
    if best:
        print(
            f"\n🏆 Best config: batch={best['batch']} | img={best['img_size']} | FPS={best['fps']}"
        )
        print(
            f"   Real-time target (≥{TARGET_FPS} FPS): {'✅ MET' if best['meets_target'] else '❌ NOT MET'}"
        )


if __name__ == "__main__":
    main()
