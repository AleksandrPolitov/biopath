//! The build's log: one `info` line when the image is finished ([`log_summary`]), which benchmark
//! scripts read (`image total`). Everything goes through the `log` crate, so the library is silent
//! unless the caller installs a logger.

use crate::image::Image;

/// Human-readable byte size (binary units).
pub(crate) fn fmt_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

/// The finished image: its record blocks, its small blocks (no record) and its size.
pub(crate) fn log_summary(image: &Image, record_blocks: usize, small_blocks: usize) {
    log::info!(
        "[build] {record_blocks} record blocks, {small_blocks} small blocks; image total {} ({} \
         sections)",
        fmt_bytes(image.used()),
        image.header().toc_len,
    );
}
