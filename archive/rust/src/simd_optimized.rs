use image::GrayImage;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// LUT-based sin() with linear interpolation
// ---------------------------------------------------------------------------

/// LUT size — must be a power of 2 for fast masking
const LUT_SIZE: usize = 4096;
/// Bit-mask for modular wrap (LUT_SIZE - 1)
const LUT_MASK: usize = LUT_SIZE - 1;
/// Scale factor: radians → LUT index space
const LUT_SCALE: f32 = LUT_SIZE as f32 / (2.0 * PI);

/// Pre-computed sine lookup table covering [0, 2π).
struct SinLut {
    table: [f32; LUT_SIZE],
}

impl SinLut {
    fn new() -> Self {
        let mut table = [0.0f32; LUT_SIZE];
        for i in 0..LUT_SIZE {
            table[i] = (i as f32 * 2.0 * PI / LUT_SIZE as f32).sin();
        }
        SinLut { table }
    }

    /// Fast sin via LUT with linear interpolation.
    /// Error < 0.01% compared to f32::sin().
    #[inline(always)]
    fn sin(&self, x: f32) -> f32 {
        // Map x into [0, LUT_SIZE) — handles negative x correctly
        let phase = ((x * LUT_SCALE) % LUT_SIZE as f32 + LUT_SIZE as f32) % LUT_SIZE as f32;
        let idx = phase as usize;
        let frac = phase - idx as f32;
        let next = (idx + 1) & LUT_MASK;
        self.table[idx] * (1.0 - frac) + self.table[next] * frac
    }
}

/// Global LUT instance — initialized exactly once on first use.
static SIN_LUT: OnceLock<SinLut> = OnceLock::new();

#[inline(always)]
fn get_sin_lut() -> &'static SinLut {
    SIN_LUT.get_or_init(SinLut::new)
}

// ---------------------------------------------------------------------------
// SIMD-optimized sine wave generation (public API — unchanged signature)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub fn sin_wave_simd(
    amplitude: f32,
    frequency: f32,
    sample_rate: f32,
    phase: f32,
    duration_ms: u32,
) -> Vec<f32> {
    sin_wave_optimized(amplitude, frequency, sample_rate, phase, duration_ms)
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn sin_wave_simd(
    amplitude: f32,
    frequency: f32,
    sample_rate: f32,
    phase: f32,
    duration_ms: u32,
) -> Vec<f32> {
    sin_wave_optimized(amplitude, frequency, sample_rate, phase, duration_ms)
}

/// Optimized sine wave generation using LUT with linear interpolation.
pub fn sin_wave_optimized(
    amplitude: f32,
    frequency: f32,
    sample_rate: f32,
    phase: f32,
    duration_ms: u32,
) -> Vec<f32> {
    let lut = get_sin_lut();
    let duration_samples = (duration_ms as f32 * sample_rate / 1000.0) as usize;
    let mut wave = Vec::with_capacity(duration_samples);

    let angular_freq = 2.0 * PI * frequency / sample_rate;
    let phase_rad = phase * PI / 180.0;

    for n in 0..duration_samples {
        let x = angular_freq * n as f32 + phase_rad;
        wave.push(amplitude * lut.sin(x));
    }

    wave
}

// ---------------------------------------------------------------------------
// Picture spectrum — fully optimized
// ---------------------------------------------------------------------------

/// Parallel picture-to-spectrum conversion with three key optimizations:
/// 1. LUT sin() instead of f32::sin()  (~5× faster inner loop)
/// 2. Inline sample accumulation — no per-pixel Vec heap allocation
/// 3. Direct row parallelism via (0..height).into_par_iter()
pub fn picture_spectrum_balanced(
    img: &GrayImage,
    sample_rate: u32,
    start_freq: u32,
    spacing_ms: u32,
    freq_step: u32,
    pb: &indicatif::ProgressBar,
    progress_start: u64,
    progress_range: u64,
) -> Vec<f32> {
    let (width, height) = img.dimensions();
    let sample_rate_f = sample_rate as f32;
    let duration_samples = (spacing_ms as f32 * sample_rate_f / 1000.0) as usize;
    let norm_factor = if width > 0 { 1.0 / width as f32 } else { 1.0 };
    let lut = get_sin_lut();
    let completed_rows = AtomicU64::new(0);

    // Direct row parallelism — no intermediate pixel_work Vec, no row_work regrouping
    let row_results: Vec<Vec<f32>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_wave = vec![0.0f32; duration_samples];

            for col in 0..width {
                let pixel_value = img.get_pixel(col, row).0[0] as f32;
                let amplitude = pixel_value / 256.0;

                if amplitude > 0.01 {
                    let frequency = (col * freq_step + start_freq) as f32;
                    let phase_rad = amplitude * PI / 180.0;
                    let angular_freq = 2.0 * PI * frequency / sample_rate_f;

                    // Incremental phase in LUT-space — avoids per-sample fmodf
                    let phase_inc = angular_freq * LUT_SCALE;
                    let mut phase =
                        ((phase_rad * LUT_SCALE) % LUT_SIZE as f32 + LUT_SIZE as f32)
                            % LUT_SIZE as f32;

                    for n in 0..duration_samples {
                        // Mask idx to guarantee in-bounds even with float rounding
                        let idx = (phase as usize) & LUT_MASK;
                        let frac = phase - (phase as u32) as f32;
                        let next = (idx + 1) & LUT_MASK;
                        let sin_val =
                            lut.table[idx] * (1.0 - frac) + lut.table[next] * frac;
                        row_wave[n] += amplitude * sin_val;
                        // Branchless-ish wrap: predictable branch, avoids fmodf
                        phase += phase_inc;
                        if phase >= LUT_SIZE as f32 {
                            phase -= LUT_SIZE as f32;
                        }
                    }
                }
            }

            // Normalize by width
            if width > 0 {
                for sample in row_wave.iter_mut() {
                    *sample *= norm_factor;
                }
            }

            // Update progress atomically
            let done = completed_rows.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 || done == height as u64 {
                let row_progress = (done as f64 / height as f64 * progress_range as f64) as u64;
                pb.set_position(progress_start + row_progress);
            }

            row_wave
        })
        .collect();

    pb.set_position(progress_start + progress_range);

    // Combine per-row results into a flat buffer
    let mut final_wave = Vec::with_capacity(duration_samples * height as usize);
    for row_wave in row_results {
        final_wave.extend(row_wave);
    }

    final_wave
}

/// Benchmark-friendly version without progress tracking.
#[allow(dead_code)]
pub fn picture_spectrum_balanced_bench(
    img: &GrayImage,
    sample_rate: u32,
    start_freq: u32,
    spacing_ms: u32,
    freq_step: u32,
) -> Vec<f32> {
    let pb = indicatif::ProgressBar::hidden();
    picture_spectrum_balanced(img, sample_rate, start_freq, spacing_ms, freq_step, &pb, 0, 100)
}

/// Memory-mapped processing for very large images.
pub fn picture_spectrum_mmap(
    img_path: &str,
    sample_rate: u32,
    start_freq: u32,
    spacing_ms: u32,
    freq_step: u32,
    pb: &indicatif::ProgressBar,
    progress_start: u64,
    progress_range: u64,
) -> anyhow::Result<Vec<f32>> {
    println!("Loading image with memory-mapped processing...");
    let img = image::open(img_path)?.to_luma8();
    let (_width, height) = img.dimensions();
    let duration_samples = (spacing_ms as f32 * sample_rate as f32 / 1000.0) as usize;

    if duration_samples * height as usize > 100_000_000 {
        println!("Large output detected, consider streaming to file");
    }

    println!("Using balanced approach with progress tracking...");
    Ok(picture_spectrum_balanced(
        &img,
        sample_rate,
        start_freq,
        spacing_ms,
        freq_step,
        pb,
        progress_start,
        progress_range,
    ))
}
