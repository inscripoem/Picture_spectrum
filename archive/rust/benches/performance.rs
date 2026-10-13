use criterion::{black_box, criterion_group, criterion_main, Criterion};
use picture_spectrum::*;
use std::time::Duration;

fn benchmark_sin_wave(c: &mut Criterion) {
    let mut group = c.benchmark_group("sin_wave");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("optimized", |b| {
        b.iter(|| {
            let wave = sin_wave_optimized(
                black_box(0.5),
                black_box(440.0),
                black_box(44100.0),
                black_box(0.0),
                black_box(100),
            );
            black_box(wave)
        })
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("simd", |b| {
        b.iter(|| {
            let wave = sin_wave_simd(
                black_box(0.5),
                black_box(440.0),
                black_box(44100.0),
                black_box(0.0),
                black_box(100),
            );
            black_box(wave)
        })
    });

    group.finish();
}

fn create_test_image() -> image::GrayImage {
    let width = 100;
    let height = 100;
    let mut img = image::GrayImage::new(width, height);

    // Create a simple gradient pattern
    for y in 0..height {
        for x in 0..width {
            let value = ((x + y) * 255 / (width + height)) as u8;
            img.put_pixel(x, y, image::Luma([value]));
        }
    }

    img
}

fn benchmark_spectrum_generation(c: &mut Criterion) {
    let img = create_test_image();

    let mut group = c.benchmark_group("spectrum_generation");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    group.bench_function("balanced", |b| {
        b.iter(|| {
            let result = picture_spectrum_balanced_bench(
                &img,
                black_box(44100),
                black_box(10000),
                black_box(50),
                black_box(10),
            );
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_sin_wave, benchmark_spectrum_generation);
criterion_main!(benches);
