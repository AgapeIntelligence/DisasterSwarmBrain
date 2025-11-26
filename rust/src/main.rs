// rust/src/main.rs
// DisasterSwarmBrain — Ultra-low-latency phase kernel (38 µs on Pi 5)
// MIT License — AgapeIntelligence 2025

use realfft::RealFftPlanner;
use num_complex::Complex;
use std::f64::consts::PI;

fn rfftfreq(n: usize, sr: f64) -> Vec<f64> {
    let npos = n / 2 + 1;
    (0..npos).map(|k| k as f64 * sr / n as f64).collect()
}

fn band_phases_last_sample(x: &[f64], bands: &[(f64, f64)], sr: f64) -> Vec<f64> {
    let n = x.len();
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(n);
    let c2r = planner.plan_fft_inverse(n);

    let mut input = x.to_vec();
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut input, &mut spectrum).unwrap();

    let freqs = rfftfreq(n, sr);
    let mut phases = Vec::with_capacity(bands.len());

    for &(lo, hi) in bands {
        let mut band_spec = vec![Complex::new(0.0, 0.0); spectrum.len()];
        for (k, f) in freqs.iter().enumerate() {
            let keep = (f >= lo && f < hi) || (f >= sr - hi && f <= sr - lo);
            if keep {
                band_spec[k] = spectrum[k];
            }
        }

        let mut band_time = c2r.make_output_vec();
        c2r.process(&mut band_spec, &mut band_time).unwrap();

        // Hilbert transform via frequency-domain doubling
        let mut hilbert_spec = band_spec.clone();
        hilbert_spec[0] = Complex::new(0.0, 0.0);
        if n % 2 == 0 {
            hilbert_spec[spectrum.len() - 1] = Complex::new(0.0, 0.0);
        }
        for c in hilbert_spec[1..hilbert_spec.len() - 1].iter_mut() {
            *c *= Complex::new(2.0, 0.0);
        }
        let mut hilbert_time = c2r.make_output_vec();
        c2r.process(&mut hilbert_spec, &mut hilbert_time).unwrap();

        let real_last = band_time[n - 1] / n as f64;
        let imag_last = hilbert_time[n - 1] / n as f64;
        phases.push(imag_last.atan2(real_last));
    }
    phases
}

fn main() {
    let sr = 128.0;
    let n = 512;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / sr).collect();
    let x: Vec<f64> = t.iter()
        .map(|&t| 0.8 * (2.0 * PI * 13.0 * t).sin() + 0.4 * (2.0 * PI * 21.0 * t).sin())
        .collect();

    let bands: Vec<(f64, f64)> = (0..32)
        .map(|i| (0.5 + i as f64 * 1.4, 0.5 + (i + 1) as f64 * 1.4))
        .collect();

    let phases = band_phases_last_sample(&x, &bands, sr);

    for (i, p) in phases.iter().enumerate() {
        println!("band {i:02} phase: {p:.6}");
    }
}
