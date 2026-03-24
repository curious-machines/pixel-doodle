use std::io::Write;

/// Write a u32 ARGB pixel buffer as a PPM image file.
pub fn write_ppm(path: &str, buffer: &[u32], width: u32, height: u32) {
    let mut file = std::fs::File::create(path).unwrap_or_else(|e| {
        eprintln!("Failed to create '{}': {}", path, e);
        std::process::exit(1);
    });
    write!(file, "P6\n{} {}\n255\n", width, height).unwrap();
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for &pixel in buffer {
        let r = ((pixel >> 16) & 0xFF) as u8;
        let g = ((pixel >> 8) & 0xFF) as u8;
        let b = (pixel & 0xFF) as u8;
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    file.write_all(&rgb).unwrap();
    eprintln!("[output] wrote {}x{} PPM to '{}'", width, height, path);
}
