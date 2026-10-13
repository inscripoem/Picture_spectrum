#!/usr/bin/env python3
"""
GPU-accelerated Python picture spectrum generator using CuPy.

Pre-computes sin and cos basis matrices on GPU once, then processes image rows
in batches. Each row applies phase modulation using the trigonometric identity:
    sin(x + φ) = sin(x)cos(φ) + cos(x)sin(φ)

Per-row VRAM usage: 2 × width × duration_samples × 8 bytes (~68MB for 4K).
Batch of 64 rows: ~4.4GB — well within 8GB RTX 3070.
"""

import argparse
import time

import cv2
import cupy as cp
import numpy as np
import soundfile as sf

BATCH_SIZE = 64


def picture_spectrum_gpu(img, fs, q, s, z, batch_size=BATCH_SIZE):
    """Generate spectrum using GPU-accelerated basis matrix multiplication.

    Args:
        img: Grayscale image as 2-D numpy array (uint8).
        fs: Sample rate (Hz).
        q: Start frequency (Hz).
        s: Spacing duration (ms).
        z: Frequency step (Hz).
        batch_size: Number of image rows to process per GPU batch.

    Returns:
        1-D numpy array of waveform samples.
    """
    hi, wi = img.shape

    Ts = 1.0 / fs
    duration_samples = int(s / Ts / 1000)

    # --- Pre-compute sin and cos basis matrices on GPU ---
    # basis[col, n] = sin/cos(2π × (col × z + q) × n × Ts)
    cols = cp.arange(wi, dtype=cp.float64)
    n = cp.arange(duration_samples, dtype=cp.float64)
    freqs = cols * z + q
    angle = 2.0 * cp.pi * cp.outer(freqs, n) * Ts  # (W, N)
    sin_basis = cp.sin(angle)                        # (W, N)
    cos_basis = cp.cos(angle)                        # (W, N)

    # --- Pre-allocate output on CPU ---
    wave = np.zeros((hi + 1) * duration_samples, dtype=np.float64)

    # --- Process rows in batches ---
    num_batches = (hi + batch_size - 1) // batch_size

    for b in range(num_batches):
        row_start = b * batch_size
        row_end = min(row_start + batch_size, hi)

        # Transfer pixel batch to GPU, normalize
        pixel_batch = img[row_start:row_end, :].astype(np.float64) / 256.0
        pixel_gpu = cp.asarray(pixel_batch)  # (batch_rows, W)

        # Compute phase modulation for each pixel
        phases = pixel_gpu * (cp.pi / 180.0)  # (batch_rows, W)
        cos_phases = cp.cos(phases)            # (batch_rows, W)
        sin_phases = cp.sin(phases)            # (batch_rows, W)

        # sin(2πft + φ) = sin(2πft)cos(φ) + cos(2πft)sin(φ)
        # For each row: weighted_basis = sin_basis * cos(φ) + cos_basis * sin(φ)
        # Then: row_wave = pixel_values @ weighted_basis / wi

        # Compute (batch_rows, W) element-wise multiplied with (W, N) basis
        # Using einsum for efficient batched computation
        term1 = cp.einsum('bw,wn->bn', pixel_gpu * cos_phases, sin_basis)
        term2 = cp.einsum('bw,wn->bn', pixel_gpu * sin_phases, cos_basis)
        batch_wave_gpu = (term1 + term2) / wi  # (batch_rows, N)

        # Transfer results back to CPU and write directly via reshape
        batch_wave = cp.asnumpy(batch_wave_gpu)  # (batch_rows, N)
        n_rows = row_end - row_start
        # Place rows in normal order (will reverse entire array at end)
        wave[(row_start + 1) * duration_samples : (row_start + 1 + n_rows) * duration_samples] = batch_wave.ravel()

    # Reverse entire array to match original behavior (reverses both row order AND sample order within rows)
    wave = wave[::-1].copy()

    return wave


def print_gpu_info():
    """Print GPU device info and initial memory state."""
    dev = cp.cuda.Device()
    mem = dev.mem_info
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    print(f"GPU: {name}")
    print(f"VRAM: {mem[1] / 1e9:.1f} GB total, {mem[0] / 1e9:.1f} GB free")


def main():
    parser = argparse.ArgumentParser(description="Picture Spectrum (CuPy GPU)")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--start-freq", type=int, default=10000)
    parser.add_argument("--spacing-ms", type=int, default=50)
    parser.add_argument("--freq-step", type=int, default=10)
    parser.add_argument("--output", help="Output WAV path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Rows per GPU batch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    hi, wi = img.shape
    pixels = hi * wi

    print_gpu_info()
    print(f"Image: {wi}x{hi} ({pixels:,} pixels)")
    print(f"Batch size: {args.batch_size} rows")
    print()

    # Warm-up: run a tiny computation to trigger CUDA context init
    _ = cp.sin(cp.array([1.0]))

    t1 = time.perf_counter()

    wav = picture_spectrum_gpu(
        img, args.sample_rate, args.start_freq,
        args.spacing_ms, args.freq_step,
        batch_size=args.batch_size,
    )

    output = args.output or f"{args.spacing_ms}ms_{args.freq_step}hz_gpu.wav"
    sf.write(output, wav, args.sample_rate, subtype='PCM_16')

    t2 = time.perf_counter()
    elapsed = t2 - t1

    # GPU memory stats
    mem_pool = cp.get_default_memory_pool()
    peak_bytes = mem_pool.total_bytes()

    print(f"Completed in {elapsed:.3f}s ({pixels / elapsed:,.0f} pixels/s)")
    print(f"GPU peak memory: {peak_bytes / 1e6:.1f} MB")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
