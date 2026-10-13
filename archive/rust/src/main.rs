use std::f32::consts::PI;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

mod simd_optimized;
use simd_optimized::*;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input image path
    #[arg(short, long)]
    image: String,

    /// Sample rate in Hz
    #[arg(short = 'r', long, default_value_t = 44100)]
    sample_rate: u32,

    /// Starting frequency in Hz
    #[arg(short = 'f', long, default_value_t = 10000)]
    start_freq: u32,

    /// Spectrum row spacing in ms
    #[arg(short, long, default_value_t = 50)]
    spacing_ms: u32,

    /// Spectrum column spacing in Hz
    #[arg(short = 'z', long, default_value_t = 10)]
    freq_step: u32,

    /// Use SIMD optimizations (x86_64 only)
    #[arg(long)]
    simd: bool,

    /// Processing method: chunked, balanced, or mmap
    #[arg(short = 'm', long, default_value = "balanced")]
    method: String,

    /// Output file name (optional)
    #[arg(short, long)]
    output: Option<String>,
}

/// Optimized sine wave generation using pre-computed values
fn sin_wave_optimized(
    amplitude: f32,
    frequency: f32,
    sample_rate: f32,
    phase: f32,
    duration_ms: u32,
) -> Vec<f32> {
    let duration_samples = (duration_ms as f32 * sample_rate / 1000.0) as usize;
    let mut wave = Vec::with_capacity(duration_samples);

    let angular_freq = 2.0 * PI * frequency / sample_rate;
    let phase_rad = phase * PI / 180.0;

    // Use more efficient sine calculation
    for n in 0..duration_samples {
        let sample = amplitude * (angular_freq * n as f32 + phase_rad).sin();
        wave.push(sample);
    }

    wave
}

/// Alternative implementation using chunks for better memory efficiency
fn picture_spectrum_chunked(
    img_path: &str,
    sample_rate: u32,
    start_freq: u32,
    spacing_ms: u32,
    freq_step: u32,
    pb: &ProgressBar,
    progress_start: u64,
    progress_range: u64,
) -> Result<Vec<f32>> {
    let img = image::open(img_path)
        .context("Failed to open image")?
        .to_luma8();

    let (width, height) = img.dimensions();
    let sample_rate_f = sample_rate as f32;
    let duration_samples = (spacing_ms as f32 * sample_rate_f / 1000.0) as usize;

    println!("Processing image: {}x{} pixels", width, height);

    let mut final_wave = Vec::with_capacity(duration_samples * height as usize);

    // Process in chunks to manage memory usage
    const CHUNK_SIZE: u32 = 100;
    let total_chunks = (height + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (chunk_idx, chunk_start) in (0..height).step_by(CHUNK_SIZE as usize).enumerate() {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(height);

        let chunk_rows: Vec<_> = (chunk_start..chunk_end).collect();
        let chunk_data: Vec<Vec<f32>> = chunk_rows
            .par_iter()
            .map(|&row| {
                let mut row_wave = vec![0.0f32; duration_samples];

                for col in 0..width {
                    let pixel_value = img.get_pixel(col, row).0[0] as f32;
                    let amplitude = pixel_value / 256.0;
                    let frequency = (col * freq_step + start_freq) as f32;

                    if amplitude > 0.01 {
                        let wave = sin_wave_optimized(
                            amplitude,
                            frequency,
                            sample_rate_f,
                            amplitude,
                            spacing_ms,
                        );

                        for (i, &sample) in wave.iter().enumerate() {
                            if i < duration_samples {
                                row_wave[i] += sample;
                            }
                        }
                    }
                }

                if width > 0 {
                    let norm_factor = 1.0 / width as f32;
                    row_wave.iter_mut().for_each(|x| *x *= norm_factor);
                }

                row_wave
            })
            .collect();

        // Append chunk data to final wave
        for row_wave in chunk_data {
            final_wave.extend(row_wave);
        }

        // Update global progress bar
        let chunk_progress = ((chunk_idx + 1) as f64 / total_chunks as f64 * progress_range as f64) as u64;
        pb.set_position(progress_start + chunk_progress);
    }

    Ok(final_wave)
}

/// Write wave data to WAV file
fn write_wav_file(
    filename: &str, 
    wave_data: &[f32], 
    sample_rate: u32,
    pb: &ProgressBar,
    progress_start: u64,
    progress_range: u64,
) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(filename, spec).context("Failed to create WAV file")?;

    // Convert f32 samples to i16 with progress tracking
    for (i, &sample) in wave_data.iter().enumerate() {
        let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer
            .write_sample(sample_i16)
            .context("Failed to write sample")?;

        // Update progress every 10000 samples to avoid too frequent updates
        if i % 10000 == 0 {
            let write_progress = ((i as f64 / wave_data.len() as f64) * progress_range as f64) as u64;
            pb.set_position(progress_start + write_progress);
        }
    }

    pb.set_position(progress_start + progress_range);
    writer.finalize().context("Failed to finalize WAV file")?;

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let start_time = Instant::now();

    println!("Starting picture spectrum generation...");
    println!("Image: {}", args.image);
    println!("Sample rate: {} Hz", args.sample_rate);
    println!("Start frequency: {} Hz", args.start_freq);
    println!("Row spacing: {} ms", args.spacing_ms);
    println!("Frequency step: {} Hz", args.freq_step);

    // Create unified progress bar for the entire process
    let total_progress = 100u64;
    let pb = ProgressBar::new(total_progress);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent}% {msg} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Stage 1: Image loading (5%)
    pb.set_message("Loading image...");
    pb.set_position(0);

    // Generate spectrum using the selected method
    let wave_data = match args.method.as_str() {
        "chunked" => {
            pb.set_position(5); // Image loaded
            pb.set_message("Processing image (chunked)...");
            picture_spectrum_chunked(
                &args.image,
                args.sample_rate,
                args.start_freq,
                args.spacing_ms,
                args.freq_step,
                &pb,
                5,  // progress starts at 5%
                90, // spectrum generation takes 90%
            )?
        }
        "balanced" => {
            let img = image::open(&args.image)?.to_luma8();
            pb.set_position(5); // Image loaded
            pb.set_message("Processing image (balanced)...");
            picture_spectrum_balanced(
                &img,
                args.sample_rate,
                args.start_freq,
                args.spacing_ms,
                args.freq_step,
                &pb,
                5,  // progress starts at 5%
                90, // spectrum generation takes 90%
            )
        }
        "mmap" => {
            pb.set_position(5); // Image will be loaded in mmap function
            pb.set_message("Processing image (mmap)...");
            picture_spectrum_mmap(
                &args.image,
                args.sample_rate,
                args.start_freq,
                args.spacing_ms,
                args.freq_step,
                &pb,
                5,  // progress starts at 5%
                90, // spectrum generation takes 90%
            )?
        }
        _ => {
            eprintln!("Unknown method: {}. Using 'balanced'", args.method);
            let img = image::open(&args.image)?.to_luma8();
            pb.set_position(5); // Image loaded
            pb.set_message("Processing image (balanced)...");
            picture_spectrum_balanced(
                &img,
                args.sample_rate,
                args.start_freq,
                args.spacing_ms,
                args.freq_step,
                &pb,
                5,  // progress starts at 5%
                90, // spectrum generation takes 90%
            )
        }
    };

    // Stage 3: Writing WAV file (5%)
    pb.set_position(95);
    pb.set_message("Writing WAV file...");

    // Generate output filename
    let output_filename = args.output.unwrap_or_else(|| {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        format!(
            "{}ms_{}hz_{}.wav",
            args.spacing_ms, args.freq_step, timestamp
        )
    });

    // Write WAV file
    write_wav_file(
        &output_filename, 
        &wave_data, 
        args.sample_rate,
        &pb,
        95, // progress starts at 95%
        5,  // writing takes 5%
    )?;

    pb.set_position(100);
    pb.finish_with_message("Conversion completed successfully!");

    let elapsed = start_time.elapsed();
    println!("Conversion completed in {:.2}s", elapsed.as_secs_f64());
    println!("Output file: {}", output_filename);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_wave_generation() {
        let wave = sin_wave_optimized(1.0, 440.0, 44100.0, 0.0, 1000);
        assert_eq!(wave.len(), 44100); // 1 second at 44.1kHz

        // Check that the wave is not all zeros
        assert!(wave.iter().any(|&x| x.abs() > 0.1));
    }

    #[test]
    fn test_sin_wave_optimized() {
        let wave1 = sin_wave_optimized(1.0, 440.0, 44100.0, 0.0, 100);
        let wave2 = sin_wave_optimized(1.0, 440.0, 44100.0, 0.0, 100);

        assert_eq!(wave1.len(), wave2.len());

        // Waves should be approximately equal
        for (a, b) in wave1.iter().zip(wave2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
