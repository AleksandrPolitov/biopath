//! The on-disk / in-memory **index image**: one flat byte arena holding every retained structure of
//! an engine as plain-data arrays, addressed through a table of contents.
//!
//! The same image serves both ways of running an engine:
//!
//! * **Build.** The engine build appends its finished arrays into an [`Image`] arena (a single
//!   contiguous anonymous reservation that never moves, so views stay valid while later parts are
//!   still being appended), then [`Image::finish`]es the header and TOC. Writing the index to disk is
//!   one `write_all` of the used prefix.
//! * **Load.** [`Image::open`] memory-maps the file and prefaults all of it, and the very same view
//!   types read it back.
//!
//! Every stored element is plain data (`Copy`, no pointers, no `bool`/enum fields — `u8` instead) so
//! that `&[u8]` → `&[T]` is a reinterpretation, and the format is native-endian with the word size and
//! element sizes recorded, so a mismatched binary refuses the file instead of misreading it. Sections
//! are addressed by stable [`SectionId`]s (never by position) and unknown sections are ignored, which
//! is what lets a derived structure be replaced without touching the rest of the file.

pub mod arena;
pub mod hash;
pub mod layout;
mod record;

pub use arena::{FORMAT_VERSION, Header, Image, Sec, SectionId, TocEntry};
pub(crate) use record::image_record;
pub use record::{ImageRecord, Padded};

/// Return freed heap pages to the OS (glibc `malloc_trim(0)`; a no-op without glibc).
///
/// After a block's owned build structures are copied into the image and dropped, glibc keeps the
/// freed small chunks resident in its heap while the image pages they were copied to are new, so
/// the process peak would be the block twice over. Trimming between the copies of a large block
/// returns those pages, bounding the transient to one piece. Cheap (milliseconds) and only called
/// for blocks large enough to matter.
pub fn trim_heap() {
    // SAFETY: `malloc_trim` only returns free heap memory to the OS.
    #[cfg(target_env = "gnu")]
    unsafe {
        libc::malloc_trim(0);
    }
}
