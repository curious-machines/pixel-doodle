pub struct BenchResult {
    pub frame_times: Vec<f64>, // milliseconds
}

impl BenchResult {
    pub fn report(&self, label: &str, width: u32, height: u32) {
        let n = self.frame_times.len();
        if n == 0 {
            eprintln!("No frames recorded.");
            return;
        }
        let min = self.frame_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.frame_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = self.frame_times.iter().sum();
        let avg = sum / n as f64;

        let mut sorted = self.frame_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[n / 2];

        let pixels = width as u64 * height as u64;
        let mpix_per_sec = (pixels as f64 / 1_000_000.0) / (avg / 1000.0);

        eprintln!("── bench: {} | {} frames | {}x{} ──", label, n, width, height);
        eprintln!("  min:        {:.2} ms", min);
        eprintln!("  max:        {:.2} ms", max);
        eprintln!("  avg:        {:.2} ms", avg);
        eprintln!("  median:     {:.2} ms", median);
        eprintln!("  total:      {:.1} ms", sum);
        eprintln!("  throughput: {:.1} Mpix/s", mpix_per_sec);
        eprintln!("  fps:        {:.1}", 1000.0 / avg);
    }
}
