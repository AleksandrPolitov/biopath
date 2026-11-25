use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

use anyhow::{Context, bail};

/// The first two bytes of a gzip (and BGZF) stream.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Opens the file and pipes it through a decompressor if it is gzip-compressed (plain gzip or
/// BGZF), recognised by its magic bytes whatever the file extension.
pub fn read_optionally_compressed_file<T>(
    path: impl AsRef<Path>,
    reader: impl FnOnce(&mut BufReader<dyn Read>) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let mut buf_reader = BufReader::new(open_optionally_compressed_file(path)?);
    reader(&mut buf_reader)
}

/// The file at `path`, decompressed if it starts with the gzip magic bytes.
pub fn open_optionally_compressed_file(path: impl AsRef<Path>) -> anyhow::Result<Box<dyn Read>> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("Failed to open file {:?}", path))?;
    let mut reader = BufReader::new(file);
    let head = reader
        .fill_buf()
        .with_context(|| format!("Failed to read file {:?}", path))?;
    if head.starts_with(&GZIP_MAGIC) {
        Ok(Box::new(flate2::bufread::MultiGzDecoder::new(reader)))
    } else {
        Ok(Box::new(reader))
    }
}

/// Whether an output file at `path` is gzip-compressed (a `.gz` / `.gzip` extension); an error for
/// the BGZF extensions, which the writer does not support.
fn compressed_output(path: &Path) -> anyhow::Result<bool> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("gz" | "gzip") => Ok(true),
        Some("bgzf" | "bgz" | "bgzip") => {
            bail!("Blocked gzip compression is not supported for output files (use .gz)")
        }
        _ => Ok(false),
    }
}

/// Create the file at `path`, gzip-compressed if its extension is `.gz` or `.gzip`, and write it
/// with `writer`.
pub fn write_optionally_compressed_file<T>(
    path: impl AsRef<Path>,
    writer: impl FnOnce(&mut BufWriter<dyn Write>) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let path = path.as_ref();
    let compressed = compressed_output(path).with_context(|| format!("Cannot write {path:?}"))?;
    let file = File::create(path).with_context(|| format!("Failed to create file {:?}", path))?;

    if compressed {
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut buf_writer = BufWriter::new(encoder);
        writer(&mut buf_writer)
    } else {
        writer(&mut BufWriter::new(file))
    }
}

/// Check, before any work, that an output file can be written at `path`, so a bad path fails at
/// once instead of after a long build or query run: not a directory, its directory exists, and the
/// file (if it exists) or its directory (if not) is writable. An absent file is created and removed
/// again. `what` names the output in the error (e.g. `"--query-out"`).
pub fn check_output_file(path: &Path, what: &str) -> anyhow::Result<()> {
    let directory = output_directory(path, what)?;
    if path.exists() {
        OpenOptions::new()
            .append(true)
            .open(path)
            .with_context(|| format!("{what} {path:?} is not writable"))?;
    } else {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("{what} {path:?} cannot be created in {directory:?}"))?;
        std::fs::remove_file(path)
            .with_context(|| format!("Failed to remove the probe file {path:?}"))?;
    }
    Ok(())
}

/// [`check_output_file`] for a file written by [`write_optionally_compressed_file`]: also refuses
/// the unsupported BGZF extensions.
pub fn check_compressed_output_file(path: &Path, what: &str) -> anyhow::Result<()> {
    compressed_output(path).with_context(|| format!("{what} {path:?}"))?;
    check_output_file(path, what)
}

/// Check, before any work, that a file can be created in `path`'s directory and moved over `path`:
/// `path` is not a directory, and its directory exists and is writable (probed with a hidden file
/// that is created and removed again). `what` names the output in the error.
pub fn check_output_directory(path: &Path, what: &str) -> anyhow::Result<()> {
    let directory = output_directory(path, what)?;
    let name = path
        .file_name()
        .map_or_else(Default::default, |name| name.to_string_lossy());
    let probe = directory.join(format!(".{name}.write-probe-{}", std::process::id()));
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .with_context(|| format!("{what} {path:?}: cannot create files in {directory:?}"))?;
    std::fs::remove_file(&probe)
        .with_context(|| format!("Failed to remove the probe file {probe:?}"))?;
    Ok(())
}

/// The directory an output file goes to (`.` for a bare file name); an error when `path` is a
/// directory or its directory does not exist.
fn output_directory<'path>(path: &'path Path, what: &str) -> anyhow::Result<&'path Path> {
    if path.is_dir() {
        bail!("{what} {path:?} is a directory, not a file");
    }
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    if !directory.is_dir() {
        bail!("{what} {path:?}: the directory {directory:?} does not exist");
    }
    Ok(directory)
}

pub fn write_human_readable_time(seconds: f64, mut writer: impl Write) -> std::io::Result<()> {
    if seconds < 1e-8 {
        write!(writer, "{:.2}ns", seconds * 1e9)
    } else if seconds < 1e-7 {
        write!(writer, "{:.1}ns", seconds * 1e9)
    } else if seconds < 1e-6 {
        write!(writer, "{:.0}ns", seconds * 1e9)
    } else if seconds < 1e-5 {
        write!(writer, "{:.2}μs", seconds * 1e6)
    } else if seconds < 1e-4 {
        write!(writer, "{:.1}μs", seconds * 1e6)
    } else if seconds < 1e-3 {
        write!(writer, "{:.0}μs", seconds * 1e6)
    } else if seconds < 1e-2 {
        write!(writer, "{:.2}ms", seconds * 1e3)
    } else if seconds < 1e-1 {
        write!(writer, "{:.1}ms", seconds * 1e3)
    } else if seconds < 1e0 {
        write!(writer, "{:.0}ms", seconds * 1e3)
    } else if seconds < 1e1 {
        write!(writer, "{:.2}s", seconds)
    } else if seconds < 60.0 {
        write!(writer, "{:.1}s", seconds)
    } else if seconds < 3600.0 {
        write!(
            writer,
            "{:.0}m {:.0}s",
            (seconds / 60.0).floor(),
            seconds % 60.0
        )
    } else if seconds < 86400.0 {
        write!(
            writer,
            "{:.0}h {:.0}m",
            (seconds / 3600.0).floor(),
            (seconds / 60.0).round() % 60.0,
        )
    } else if seconds < 86400.0 * 100.0 {
        write!(
            writer,
            "{:.0}d {:.0}h",
            (seconds / 86400.0).floor(),
            (seconds / 3600.0).round() % 24.0,
        )
    } else {
        write!(writer, "{:.0}d", seconds / 86400.0)
    }
}

pub fn write_human_readable_time_to_string(seconds: f64) -> String {
    let mut buf = Vec::new();
    write_human_readable_time(seconds, &mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}
