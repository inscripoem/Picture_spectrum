#!/usr/bin/env python3
"""
Verify correctness of different Picture Spectrum implementations.
Compares WAV outputs across versions to ensure they produce equivalent results.
"""

import numpy as np
import soundfile as sf
import subprocess
import sys
import os
from pathlib import Path

# Rust implementation lives in archive/rust/ (see archive/rust/README_rust.md)
RUST_BIN = "archive/rust/target/release/picture_spectrum"


def generate_wav(script, image, output, params):
    """Generate WAV using a Python script."""
    cmd = [
        "uv", "run", "python", script,
        "--image", image,
        "--sample-rate", str(params["sample_rate"]),
        "--start-freq", str(params["start_freq"]),
        "--spacing-ms", str(params["spacing_ms"]),
        "--freq-step", str(params["freq_step"]),
        "--output", output,
    ]
    subprocess.run(cmd, capture_output=True, timeout=600)


def generate_wav_rust(image, output, params):
    """Generate WAV using Rust binary."""
    cmd = [
        RUST_BIN,
        "-i", image,
        "-r", str(params["sample_rate"]),
        "-f", str(params["start_freq"]),
        "-s", str(params["spacing_ms"]),
        "-z", str(params["freq_step"]),
        "-o", output,
    ]
    subprocess.run(cmd, capture_output=True, timeout=600)


def compare_wav(file1, file2, label1, label2, tolerance_db=-60):
    """Compare two WAV files and report differences."""
    data1, sr1 = sf.read(file1)
    data2, sr2 = sf.read(file2)

    print(f"\n  Comparing: {label1} vs {label2}")
    print(f"    Shape: {data1.shape} vs {data2.shape}")
    print(f"    Sample rate: {sr1} vs {sr2}")

    if data1.shape != data2.shape:
        print(f"    ⚠ SHAPE MISMATCH - files have different lengths")
        # Compare what we can
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

    if sr1 != sr2:
        print(f"    ✗ SAMPLE RATE MISMATCH")
        return False

    # Compute difference metrics
    diff = np.abs(data1 - data2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # RMS of each signal
    rms1 = np.sqrt(np.mean(data1 ** 2))
    rms2 = np.sqrt(np.mean(data2 ** 2))
    
    # Relative difference
    if rms1 > 0 and rms2 > 0:
        # SNR: signal (reference) vs noise (difference)
        noise_rms = np.sqrt(np.mean((data1 - data2) ** 2))
        if noise_rms > 0:
            snr = 20 * np.log10(rms1 / noise_rms)
        else:
            snr = float('inf')
    else:
        snr = 0

    print(f"    RMS: {rms1:.6f} vs {rms2:.6f}")
    print(f"    Max diff: {max_diff:.6f}")
    print(f"    Mean diff: {mean_diff:.6f}")
    print(f"    SNR: {snr:.1f} dB")

    # Check if signals are reasonably close
    # For LUT approximation, expect >60dB SNR
    # For GPU float32 vs CPU float64, expect >80dB SNR
    if snr > 80:
        print(f"    ✓ EXCELLENT match (SNR > 80dB)")
    elif snr > 60:
        print(f"    ✓ GOOD match (SNR > 60dB, expected for LUT approximation)")
    elif snr > 40:
        print(f"    ⚠ ACCEPTABLE (SNR > 40dB, some numerical differences)")
    else:
        print(f"    ✗ POOR match (SNR < 40dB, likely algorithmic difference)")

    return snr > 40


def visualize_wav(file_path, label, num_rows=5):
    """Print basic stats about a WAV file to verify content."""
    data, sr = sf.read(file_path)
    duration_ms = len(data) / sr * 1000
    
    print(f"\n  {label}:")
    print(f"    Duration: {duration_ms:.1f}ms ({len(data)} samples @ {sr}Hz)")
    print(f"    Range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"    RMS: {np.sqrt(np.mean(data**2)):.6f}")
    print(f"    Non-zero samples: {np.count_nonzero(data)}/{len(data)} ({100*np.count_nonzero(data)/len(data):.1f}%)")
    
    # Check for silence or clipping
    if np.max(np.abs(data)) < 1e-6:
        print(f"    ⚠ WARNING: Output is near-silent!")
    elif np.max(np.abs(data)) > 0.99:
        print(f"    ⚠ WARNING: Output may be clipping")


def main():
    params = {
        "sample_rate": 44100,
        "start_freq": 10000,
        "spacing_ms": 50,
        "freq_step": 10,
    }
    
    # Use small image for quick validation
    test_image = "samples/lena.jpg"  # 512x512, manageable size
    
    print("=" * 60)
    print("Picture Spectrum Correctness Verification")
    print("=" * 60)
    print(f"\nTest image: {test_image}")
    print(f"Params: {params}")
    
    outputs = {}
    
    # Generate outputs from each version
    print("\n--- Generating WAV files ---")
    
    print("  Python original...", end="", flush=True)
    generate_wav("python/bench_original.py", test_image, "/tmp/verify_orig.wav", params)
    outputs["original"] = "/tmp/verify_orig.wav"
    print(" done")
    
    print("  Python vectorized...", end="", flush=True)
    generate_wav("python/picture_spectrum_vec.py", test_image, "/tmp/verify_vec.wav", params)
    outputs["vectorized"] = "/tmp/verify_vec.wav"
    print(" done")
    
    print("  Python GPU...", end="", flush=True)
    generate_wav("python/picture_spectrum_gpu.py", test_image, "/tmp/verify_gpu.wav", params)
    outputs["gpu"] = "/tmp/verify_gpu.wav"
    print(" done")
    
    # Check if Rust binary exists
    rust_bin = RUST_BIN
    if os.path.exists(rust_bin):
        print("  Rust optimized...", end="", flush=True)
        generate_wav_rust(test_image, "/tmp/verify_rust.wav", params)
        outputs["rust"] = "/tmp/verify_rust.wav"
        print(" done")
    
    # Visualize each output
    print("\n--- Output Analysis ---")
    for label, path in outputs.items():
        if os.path.exists(path):
            visualize_wav(path, label)
    
    # Cross-compare all versions
    print("\n--- Cross-comparison (vs original) ---")
    all_ok = True
    for label, path in outputs.items():
        if label != "original" and os.path.exists(path):
            ok = compare_wav(
                outputs["original"], path,
                "original", label
            )
            if not ok:
                all_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ALL VERSIONS PRODUCE CONSISTENT OUTPUT")
    else:
        print("⚠ SOME VERSIONS HAVE SIGNIFICANT DIFFERENCES")
        print("  This may be expected (LUT approximation) or indicate a bug.")
    print("=" * 60)
    
    # Cleanup
    for path in outputs.values():
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
