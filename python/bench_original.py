#!/usr/bin/env python3
"""
Original Python algorithm with CLI support for benchmarking.
Same algorithm as main.py, just with argparse for benchmark script compatibility.
"""

import argparse
import cv2
import numpy as np
import soundfile as sf
import time


def sin_wave(A, f, fs, phi, t):
    """Generate sine wave (original algorithm)."""
    Ts = 1 / fs
    n = t / Ts / 1000
    n = np.arange(n)
    y = A * np.sin(2 * np.pi * f * n * Ts + phi * (np.pi / 180))
    return y


def picture_spectrum(img, hz, q, s, z):
    """Generate spectrum (corrected algorithm)."""
    img_cv = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    size = img_cv.shape
    hi = size[0]
    wi = size[1]
    
    # Pre-allocate output: initial zero wave + hi rows
    duration_samples = int(s / (1/hz) / 1000)
    wave = np.zeros((hi + 1) * duration_samples)
    
    for ho in range(hi):
        data = np.zeros(duration_samples)  # Reset for each row
        for wo in range(wi):
            a = img_cv[ho][wo] / 256
            f = wo * z + q
            data = data + sin_wave(a, f, hz, a, s)
        data = data / wi
        wave[(ho + 1) * duration_samples : (ho + 2) * duration_samples] = data
    
    # Reverse to match original stacking order
    wave = wave[::-1].copy()
    return wave


def main():
    parser = argparse.ArgumentParser(description="Picture Spectrum (Original Python)")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--start-freq", type=int, default=10000)
    parser.add_argument("--spacing-ms", type=int, default=50)
    parser.add_argument("--freq-step", type=int, default=10)
    parser.add_argument("--output", help="Output WAV path")
    args = parser.parse_args()

    t1 = time.perf_counter()

    wav = picture_spectrum(args.image, args.sample_rate, args.start_freq,
                          args.spacing_ms, args.freq_step)

    output = args.output or f"{args.spacing_ms}ms_{args.freq_step}hz_{int(t1*1000)}.wav"
    sf.write(output, wav, args.sample_rate, subtype='PCM_16')

    t2 = time.perf_counter()
    print(f"Completed in {t2 - t1:.3f}s")


if __name__ == "__main__":
    main()
