#!/usr/bin/env python3
"""
Vectorized Python picture spectrum generator.

Pre-computes sin and cos basis matrices once, then each row applies phase modulation
using the trigonometric identity: sin(x + φ) = sin(x)cos(φ) + cos(x)sin(φ)

Uses einsum for efficient computation without creating intermediate matrices.
"""

import argparse
import time

import cv2
import numpy as np
import soundfile as sf


def picture_spectrum_vec(img, fs, q, s, z):
    """Generate spectrum using vectorized basis matrix multiplication."""
    hi, wi = img.shape

    Ts = 1.0 / fs
    duration_samples = int(s / Ts / 1000)

    # Pre-compute sin and cos basis matrices: shape (width, duration_samples)
    cols = np.arange(wi)
    n = np.arange(duration_samples)
    freqs = cols * z + q
    angle = 2.0 * np.pi * np.outer(freqs, n) * Ts
    sin_basis = np.sin(angle)
    cos_basis = np.cos(angle)

    # Pre-allocate output: hi rows + 1 trailing zero-row
    wave = np.zeros((hi + 1) * duration_samples, dtype=np.float64)

    for row in range(hi):
        pixel_values = img[row, :].astype(np.float64) / 256.0
        phases = pixel_values * (np.pi / 180.0)
        cos_phases = np.cos(phases)
        sin_phases = np.sin(phases)

        # Efficient computation using einsum (no intermediate matrix)
        # sin(2πft + φ) = sin(2πft)cos(φ) + cos(2πft)sin(φ)
        term1 = np.einsum('w,wn->n', pixel_values * cos_phases, sin_basis)
        term2 = np.einsum('w,wn->n', pixel_values * sin_phases, cos_basis)
        row_wave = (term1 + term2) / wi

        wave[(row + 1) * duration_samples : (row + 2) * duration_samples] = row_wave

    # Reverse entire array to match original behavior
    # Use np.flip which returns a view (no copy needed for read-only use)
    wave = np.flip(wave).copy()

    return wave


def main():
    parser = argparse.ArgumentParser(description="Picture Spectrum (Vectorized Python)")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--start-freq", type=int, default=10000)
    parser.add_argument("--spacing-ms", type=int, default=50)
    parser.add_argument("--freq-step", type=int, default=10)
    parser.add_argument("--output", help="Output WAV path")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    hi, wi = img.shape
    pixels = hi * wi

    t1 = time.perf_counter()

    wav = picture_spectrum_vec(
        img, args.sample_rate, args.start_freq,
        args.spacing_ms, args.freq_step,
    )

    output = args.output or f"{args.spacing_ms}ms_{args.freq_step}hz_{int(t1 * 1000)}.wav"
    sf.write(output, wav, args.sample_rate, subtype='PCM_16')

    t2 = time.perf_counter()
    elapsed = t2 - t1
    print(f"Image: {wi}x{hi} ({pixels:,} pixels)")
    print(f"Completed in {elapsed:.3f}s ({pixels / elapsed:,.0f} pixels/s)")


if __name__ == "__main__":
    main()
