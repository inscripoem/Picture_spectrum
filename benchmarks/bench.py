#!/usr/bin/env python3
"""
Unified benchmark script for Picture Spectrum.
Tests all versions across multiple image sizes.
"""

import subprocess
import time
import json
import os
import sys
import platform
from pathlib import Path

# Benchmark configuration
# Note: only samples/lena.jpg is tracked in git; other samples must be provided
# locally (see .gitignore) for their respective benchmark cases to run.
IMAGES = {
    "small": "samples/QQ20250606-150451.png",   # 248x110
    "medium": "samples/lena.jpg",                # 512x512
    "large": "samples/1080.png",                 # 1920x1080
    "xlarge": "samples/4k.jpg",                  # 3840x2160 (JPEG, renamed from 4k.png)
}

# Rust implementation lives in archive/rust/ (see archive/rust/README_rust.md)
RUST_BIN = "archive/rust/target/release/picture_spectrum"

PARAMS = {
    "sample_rate": 44100,
    "start_freq": 10000,
    "spacing_ms": 50,
    "freq_step": 10,
}

REPEAT = 3
WARMUP = 2


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def get_system_info():
    info = {"python": platform.python_version()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    try:
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        info["gpu"] = gpu_out
    except Exception:
        info["gpu"] = "N/A"
    return info


def run_timed(cmd, timeout=600):
    """Run command and return elapsed time."""
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, timeout=timeout)
    return time.perf_counter() - t0


def bench_cmd(label, cmd, repeat=REPEAT, warmup=WARMUP):
    """Benchmark a command with warm-up and multiple runs."""
    print(f"  {label}...", end="", flush=True)
    for _ in range(warmup):
        run_timed(cmd)
    times = [run_timed(cmd) for _ in range(repeat)]
    times.sort()
    median = times[len(times) // 2]
    print(f" {median:.3f}s (min={min(times):.3f}, max={max(times):.3f})")
    return {"median": median, "min": min(times), "max": max(times), "runs": times}


def make_output_path(prefix, idx):
    return f"/tmp/bench_{prefix}_{idx}.wav"


def bench_python_original(image, idx):
    """Benchmark original Python algorithm."""
    output = make_output_path("py_orig", idx)
    cmd = [
        "uv", "run", "python", "python/bench_original.py",
        "--image", image,
        "--sample-rate", str(PARAMS["sample_rate"]),
        "--start-freq", str(PARAMS["start_freq"]),
        "--spacing-ms", str(PARAMS["spacing_ms"]),
        "--freq-step", str(PARAMS["freq_step"]),
        "--output", output,
    ]
    result = bench_cmd("Python (original)", cmd)
    try:
        os.unlink(output)
    except OSError:
        pass
    return result


def bench_python_script(script, image, idx):
    """Benchmark a Python script."""
    output = make_output_path("py_opt", idx)
    cmd = [
        "uv", "run", "python", script,
        "--image", image,
        "--sample-rate", str(PARAMS["sample_rate"]),
        "--start-freq", str(PARAMS["start_freq"]),
        "--spacing-ms", str(PARAMS["spacing_ms"]),
        "--freq-step", str(PARAMS["freq_step"]),
        "--output", output,
    ]
    result = bench_cmd(f"Python ({script})", cmd)
    try:
        os.unlink(output)
    except OSError:
        pass
    return result


def bench_rust(image, idx, method="balanced"):
    """Benchmark Rust release version."""
    output = make_output_path("rust", idx)
    cmd = [
        RUST_BIN,
        "-i", image,
        "-r", str(PARAMS["sample_rate"]),
        "-f", str(PARAMS["start_freq"]),
        "-s", str(PARAMS["spacing_ms"]),
        "-z", str(PARAMS["freq_step"]),
        "-m", method,
        "-o", output,
    ]
    result = bench_cmd(f"Rust ({method})", cmd)
    try:
        os.unlink(output)
    except OSError:
        pass
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Version label for results")
    parser.add_argument("--python-script", help="Python script to benchmark")
    parser.add_argument("--rust", action="store_true", help="Benchmark Rust version")
    parser.add_argument("--images", nargs="+", default=["small", "medium"])
    parser.add_argument("--all-images", action="store_true", help="Run all image sizes")
    args = parser.parse_args()

    if args.all_images:
        args.images = list(IMAGES.keys())

    results = {
        "version": args.label,
        "commit": get_git_hash(),
        "system": get_system_info(),
        "params": PARAMS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
    }

    for i, size in enumerate(args.images):
        if size not in IMAGES:
            print(f"Unknown: {size}")
            continue
        image = IMAGES[size]
        if not os.path.exists(image):
            print(f"Not found: {image}")
            continue

        print(f"\n{'='*50}")
        print(f"  {size}: {image}")
        print(f"{'='*50}")

        entry = {}

        if args.python_script:
            entry["python"] = bench_python_script(args.python_script, image, i)

        if args.rust:
            entry["rust"] = bench_rust(image, i)

        results["benchmarks"][size] = entry

    # Save
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"result_{args.label}_{results['commit']}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
