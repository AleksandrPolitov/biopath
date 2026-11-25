//! The image arena: one contiguous mapping holding header, table of contents and all sections.
//!
//! **Building:** [`Image::new_arena`] reserves a large anonymous mapping (`MAP_NORESERVE`, so only
//! touched pages cost memory) and hands out bump-allocated, zero-initialised typed slices. Because the
//! reservation never moves, a slice handed out early stays valid while later sections are appended —
//! which is what lets the build read back finished blocks (through the same views a query uses) while
//! it is still producing the rest. [`Image::finish`] writes the header and TOC into the reserved
//! prelude; [`Image::write_to`] writes the used prefix to a temporary file and renames it over the
//! target, so a process that has the old file memory-mapped keeps reading the old file.
//!
//! **Loading:** [`Image::open`] validates the header and the TOC, then memory-maps the file
//! read-only and prefaults all of it (like vg's `preload`), so queries never wait on a page fault.
//!
//! Format: native-endian, fixed-layout `#[repr(C)]` records; every TOC entry records its element size
//! and alignment, and a typed read checks them, so an index written by a binary with a different
//! layout is refused rather than misread.
//!
//! **Trust.** A `.bpi` file is trusted input, like vg's `.dist`. Loading refuses a truncated file, a
//! foreign one, another format version, byte order or word size, and a TOC whose sections do not
//! lie inside the file with their element layout; it does not validate the offsets stored inside
//! sections (the block directory, record-relative layouts, hash tables), which queries follow
//! unchecked for speed. A crafted file can therefore make queries read out of bounds: load only
//! indexes you built or trust.

use std::{
    io,
    marker::PhantomData,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use super::{ImageRecord, layout::align_up, record::image_record};
use crate::build_report::fmt_bytes;

/// Bumped whenever the image layout changes incompatibly; [`Image::open`] refuses any other version.
pub const FORMAT_VERSION: u32 = 19;
const MAGIC: [u8; 8] = *b"BPINDEX\0";
const ENDIAN_CHECK: u32 = 0x0A0B_0C0D;
/// The smallest build arena [`Image::new_arena`] settles for when the kernel refuses a larger one.
pub const MIN_RESERVATION: usize = 64 << 20;
/// Bytes reserved at offset 0 for the [`Header`].
const HEADER_REGION: usize = 4096;
/// Maximum number of sections (TOC entries); the TOC occupies `TOC_CAPACITY * 32` bytes after the
/// header.
pub const TOC_CAPACITY: usize = 512;
const TOC_REGION: usize = TOC_CAPACITY * std::mem::size_of::<TocEntry>();
/// Header + TOC; sections start here (a page multiple).
pub const PRELUDE_BYTES: usize = HEADER_REGION + TOC_REGION;
pub const PAGE: usize = 4096;

const _: () = assert!(PRELUDE_BYTES.is_multiple_of(PAGE));
const _: () = assert!(std::mem::size_of::<Header>() <= HEADER_REGION);
const _: () = assert!(std::mem::size_of::<TocEntry>() == 32);
const _: () = <Header as ImageRecord>::NO_PADDING;
const _: () = <TocEntry as ImageRecord>::NO_PADDING;

/// The fixed header at offset 0.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub magic: [u8; 8],
    pub format_version: u32,
    pub endian_check: u32,
    /// `size_of::<IndexType>()` in bytes (4 or 8).
    pub word_size: u32,
    pub toc_len: u32,
    /// Bytes of the image that are meaningful (`== file size`).
    pub used_bytes: u64,
    pub graph_node_count: u64,
    pub graph_edge_count: u64,
    /// Hash of node lengths + adjacency, so a GFA supplied at query time can be checked.
    pub graph_fingerprint: u64,
    /// Build-affecting knobs, engine-defined slots (see [`crate::sections::knob`]).
    pub build_knobs: [u64; 16],
}

image_record!(Header = [[u8; 8], u32, u32, u32, u32, u64, u64, u64, u64, [u64; 16]]);

/// One TOC entry: a typed byte range.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TocEntry {
    pub id: u32,
    pub elem_size: u32,
    pub elem_align: u32,
    pub _pad: u32,
    pub off: u64,
    /// Element count.
    pub len: u64,
}

image_record!(TocEntry = [u32, u32, u32, u32, u64, u64]);

/// A stable section identifier. Never reuse a number once it has shipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SectionId(pub u32);

/// A typed handle to a byte range of an image: element offset + count. Plain data, `Copy`, no
/// lifetime — the crate resolves it against an [`Image`] (`Image::get`).
pub struct Sec<T> {
    pub(crate) off: u64,
    pub(crate) len: u64,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for Sec<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Sec<T> {}
impl<T> std::fmt::Debug for Sec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sec({}+{})", self.off, self.len)
    }
}
impl<T> Default for Sec<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> Sec<T> {
    pub const fn empty() -> Self {
        Self {
            off: 0,
            len: 0,
            _t: PhantomData,
        }
    }

    /// A handle to `len` elements at byte offset `off`. Crate-private: a handle resolves unchecked
    /// (in release builds), so outside code gets handles only from a validated TOC.
    pub(crate) const fn new(off: u64, len: u64) -> Self {
        Self {
            off,
            len,
            _t: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len as usize * std::mem::size_of::<T>()
    }
}

/// See the module docs. A loaded image is trusted input (see **Trust** there): [`Image::open`]
/// checks the header and the TOC, not the records.
pub struct Image {
    /// The anonymous build reservation, or the file mapping of a loaded image; unmapped on drop.
    base: *mut u8,
    /// Mapped bytes.
    cap: usize,
    /// Bump pointer (build) / meaningful bytes (loaded).
    used: AtomicUsize,
    toc: Mutex<Vec<TocEntry>>,
    /// Process-unique id, so query-side memos can key on "which image" without pointers.
    instance_id: u64,
    frozen: std::sync::atomic::AtomicBool,
}

// SAFETY: the raw pointer is to a mapping we own; `alloc` hands out disjoint ranges via an atomic
// bump, and reads are plain shared borrows.
unsafe impl Send for Image {}
unsafe impl Sync for Image {}

static INSTANCE_IDS: AtomicU64 = AtomicU64::new(1);

// The mapping calls, with the target's own constants (Linux only, see the crate root).
use libc as sys;

/// Reserve `len` bytes of zeroed anonymous memory (`MAP_NORESERVE`: only touched pages cost memory).
fn map_anonymous(len: usize) -> io::Result<*mut u8> {
    let flags = sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_NORESERVE;
    let p = unsafe {
        sys::mmap(
            std::ptr::null_mut(),
            len,
            sys::PROT_READ | sys::PROT_WRITE,
            flags,
            -1,
            0,
        )
    };
    if p == sys::MAP_FAILED {
        Err(io::Error::last_os_error())
    } else {
        Ok(p as *mut u8)
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if self.cap > 0 {
            unsafe {
                sys::munmap(self.base as *mut _, self.cap);
            }
        }
    }
}

impl Image {
    /// Reserve a build arena. `reserve_hint` is the virtual size to reserve; it only needs to be an
    /// upper bound (untouched pages cost nothing). Tries progressively smaller reservations if the
    /// kernel refuses (an address-space limit, `ulimit -v`, or strict overcommit), down to
    /// [`MIN_RESERVATION`]; the image can then grow only up to the reservation granted.
    pub fn new_arena(reserve_hint: usize) -> io::Result<Self> {
        let mut cap = align_up(reserve_hint.max(MIN_RESERVATION), PAGE);
        loop {
            match map_anonymous(cap) {
                Ok(base) => {
                    // Transparent huge pages for the build arena: 512x fewer first-touch faults while
                    // the image fills, and fewer TLB misses for in-process queries. Advisory only.
                    // `SPQR_ARENA_HUGEPAGE=0` skips the advice: under THP `defrag=madvise` every
                    // first touch of an advised 2 MiB page compacts memory synchronously, which on a
                    // big, fragmented server can make filling the image ~1000x slower than memcpy.
                    if crate::env::arena_hugepages() {
                        unsafe {
                            sys::madvise(base as *mut _, cap, sys::MADV_HUGEPAGE);
                        }
                    }
                    let image = Self {
                        base,
                        cap,
                        used: AtomicUsize::new(PRELUDE_BYTES),
                        toc: Mutex::new(Vec::new()),
                        instance_id: INSTANCE_IDS.fetch_add(1, Ordering::Relaxed),
                        frozen: std::sync::atomic::AtomicBool::new(false),
                    };
                    return Ok(image);
                }
                Err(_) if cap > MIN_RESERVATION => {
                    cap = align_up(cap / 2, PAGE).max(MIN_RESERVATION);
                }
                Err(e) => {
                    return Err(io::Error::new(
                        e.kind(),
                        format!(
                            "cannot reserve {} of address space for the index image ({e}): the \
                             address-space limit (`ulimit -v`) is too low for even the smallest \
                             image; raise it",
                            fmt_bytes(cap)
                        ),
                    ));
                }
            }
        }
    }

    /// The default build reservation: generous virtual space (1 TiB on 64-bit), since only touched
    /// pages are ever committed.
    pub fn new_default_arena() -> io::Result<Self> {
        Self::new_arena(1usize << 40)
    }

    /// Process-unique id of this image instance.
    pub fn instance_id(&self) -> u64 {
        self.instance_id
    }

    /// Meaningful bytes so far.
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    #[inline]
    fn ptr_at(&self, off: usize) -> *mut u8 {
        debug_assert!(off <= self.cap);
        unsafe { self.base.add(off) }
    }

    /// Bump-allocate `len` zeroed elements aligned to `align` (at least `align_of::<T>()`), returning
    /// the handle and an exclusive slice to fill. Ranges never overlap, so handing out `&mut` from
    /// `&self` is sound; callers must not keep the slice past filling it (read it with `Self::get`).
    // The slice comes from the raw mapping, not from a shared borrow, and each range is fresh.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_aligned<T: ImageRecord>(&self, len: usize, align: usize) -> (Sec<T>, &mut [T]) {
        let () = T::NO_PADDING;
        assert!(
            !self.is_frozen(),
            "cannot allocate in a finished/loaded image"
        );
        let align = align.max(std::mem::align_of::<T>()).max(8);
        let bytes = len * std::mem::size_of::<T>();
        let mut cur = self.used.load(Ordering::Relaxed);
        loop {
            let start = align_up(cur, align);
            let end = start + bytes;
            assert!(
                end <= self.cap,
                "the index image needs more than the {} of address space reserved for it (the \
                 reservation shrinks to fit the address-space limit, `ulimit -v`, or strict \
                 overcommit): raise the limit and rebuild",
                fmt_bytes(self.cap)
            );
            match self
                .used
                .compare_exchange_weak(cur, end, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(self.ptr_at(start) as *mut T, len)
                    };
                    return (Sec::new(start as u64, len as u64), slice);
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// [`Self::alloc_aligned`] with the default (8-byte) alignment.
    pub fn alloc<T: ImageRecord>(&self, len: usize) -> (Sec<T>, &mut [T]) {
        self.alloc_aligned(len, 8)
    }

    /// Copy `data` into the arena (default alignment).
    #[cfg(test)]
    pub fn push<T: ImageRecord>(&self, data: &[T]) -> Sec<T> {
        let (sec, slice) = self.alloc::<T>(data.len());
        slice.copy_from_slice(data);
        sec
    }

    /// Copy `data` into a page-aligned range and register it under `id`.
    pub fn push_section<T: ImageRecord>(&self, id: SectionId, data: &[T]) -> Sec<T> {
        let (sec, slice) = self.alloc_aligned::<T>(data.len(), PAGE);
        slice.copy_from_slice(data);
        self.register(id, sec);
        sec
    }

    /// Allocate a page-aligned range, register it under `id`, and return it for in-place filling.
    pub fn alloc_section<T: ImageRecord>(&self, id: SectionId, len: usize) -> (Sec<T>, &mut [T]) {
        let (sec, slice) = self.alloc_aligned::<T>(len, PAGE);
        self.register(id, sec);
        (sec, slice)
    }

    /// Record `sec` in the TOC under `id`.
    pub fn register<T: ImageRecord>(&self, id: SectionId, sec: Sec<T>) {
        let mut toc = self.toc.lock().unwrap();
        assert!(
            toc.iter().all(|e| e.id != id.0),
            "section id {} registered twice",
            id.0
        );
        assert!(toc.len() < TOC_CAPACITY, "too many sections");
        toc.push(TocEntry {
            id: id.0,
            elem_size: std::mem::size_of::<T>() as u32,
            elem_align: std::mem::align_of::<T>() as u32,
            _pad: 0,
            off: sec.off,
            len: sec.len,
        });
    }

    /// The TOC (a copy).
    pub fn toc(&self) -> Vec<TocEntry> {
        self.toc.lock().unwrap().clone()
    }

    /// Whether a section is present.
    #[cfg(test)]
    pub fn has_section(&self, id: SectionId) -> bool {
        self.toc.lock().unwrap().iter().any(|e| e.id == id.0)
    }

    /// Look a typed section up by id: `None` if absent, an error if present with another element
    /// layout (a format mismatch the header check did not catch).
    pub(crate) fn section<T: ImageRecord>(&self, id: SectionId) -> io::Result<Option<Sec<T>>> {
        let toc = self.toc.lock().unwrap();
        let Some(entry) = toc.iter().find(|e| e.id == id.0) else {
            return Ok(None);
        };
        let layout = (std::mem::size_of::<T>(), std::mem::align_of::<T>());
        if (entry.elem_size as usize, entry.elem_align as usize) != layout {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "index image section {} has element layout {}B/{}B on disk but {}B/{}B in \
                     this binary: the index was written by an incompatible build; rebuild it",
                    id.0, entry.elem_size, entry.elem_align, layout.0, layout.1,
                ),
            ));
        }
        Ok(Some(Sec::new(entry.off, entry.len)))
    }

    /// Like [`Self::section`] but an absent section is an error naming it.
    pub(crate) fn require<T: ImageRecord>(&self, id: SectionId, name: &str) -> io::Result<Sec<T>> {
        self.section(id)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "index image lacks the required section {name} (id {})",
                    id.0
                ),
            )
        })
    }

    /// Resolve a handle to a slice. Unchecked in release builds: a handle comes from this image's
    /// validated TOC or from offsets inside its records (see **Trust** in the module docs).
    #[inline]
    pub(crate) fn get<T: ImageRecord>(&self, sec: Sec<T>) -> &[T] {
        let () = T::NO_PADDING;
        let off = sec.off as usize;
        let bytes = sec.byte_len();
        debug_assert!(
            off + bytes <= self.used.load(Ordering::Relaxed),
            "section {sec:?} out of range (used {})",
            self.used()
        );
        debug_assert_eq!(
            (self.base as usize + off) % std::mem::align_of::<T>(),
            0,
            "misaligned section {sec:?}"
        );
        if sec.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr_at(off) as *const T, sec.len as usize) }
    }

    /// A typed slice at an absolute byte offset (for sub-layouts inside a record).
    #[inline]
    pub(crate) fn get_at<T: ImageRecord>(&self, byte_off: usize, len: usize) -> &[T] {
        self.get(Sec::<T>::new(byte_off as u64, len as u64))
    }

    /// One element at an absolute byte offset.
    #[inline]
    pub(crate) fn get_one<T: ImageRecord>(&self, byte_off: usize) -> &T {
        &self.get_at::<T>(byte_off, 1)[0]
    }

    /// A mutable typed slice at an absolute byte offset — build time only, for filling a record that
    /// was reserved with [`Self::alloc`] as raw bytes.
    ///
    /// # Safety
    /// The range must lie within a range this thread obtained from `alloc`/`alloc_aligned` and not
    /// be read through any other reference while the returned slice lives.
    #[inline]
    #[allow(clippy::mut_from_ref)] // the caller guarantees exclusivity (see Safety)
    pub(crate) unsafe fn get_at_mut<T: ImageRecord>(
        &self,
        byte_off: usize,
        len: usize,
    ) -> &mut [T] {
        let () = T::NO_PADDING;
        debug_assert!(!self.is_frozen());
        debug_assert!(byte_off + len * std::mem::size_of::<T>() <= self.used());
        debug_assert_eq!(
            (self.base as usize + byte_off) % std::mem::align_of::<T>(),
            0
        );
        unsafe { std::slice::from_raw_parts_mut(self.ptr_at(byte_off) as *mut T, len) }
    }

    /// Write the header and TOC into the prelude and freeze the arena.
    pub fn finish(&self, mut header: Header) {
        let toc = self.toc.lock().unwrap();
        header.magic = MAGIC;
        header.format_version = FORMAT_VERSION;
        header.endian_check = ENDIAN_CHECK;
        header.toc_len = toc.len() as u32;
        // Round the image up to a page so the file is page-granular (and `used` is final).
        let used = align_up(self.used.load(Ordering::Acquire), PAGE);
        self.used.store(used, Ordering::Release);
        header.used_bytes = used as u64;
        unsafe {
            std::ptr::write(self.base as *mut Header, header);
            let entries = std::slice::from_raw_parts_mut(
                self.base.add(HEADER_REGION) as *mut TocEntry,
                toc.len(),
            );
            entries.copy_from_slice(&toc);
        }
        drop(toc);
        self.frozen.store(true, Ordering::Release);
    }

    /// Whether the image is finished/loaded (no more allocation).
    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::Acquire)
    }

    /// The header (valid after [`Self::finish`] or on a loaded image).
    pub fn header(&self) -> Header {
        unsafe { std::ptr::read(self.base as *const Header) }
    }

    /// A blank header for [`Self::finish`].
    pub fn header_template(word_size: usize) -> Header {
        Header {
            magic: MAGIC,
            format_version: FORMAT_VERSION,
            endian_check: ENDIAN_CHECK,
            word_size: word_size as u32,
            toc_len: 0,
            used_bytes: 0,
            graph_node_count: 0,
            graph_edge_count: 0,
            graph_fingerprint: 0,
            build_knobs: [0; 16],
        }
    }

    /// Write the used prefix to `path`. Requires [`Self::finish`]. The bytes go to a temporary
    /// file in the same directory, which is then renamed over `path`: an existing index at `path` is
    /// replaced whole, never truncated in place, so a process that has it memory-mapped keeps
    /// reading the old file (truncating it would kill that process with `SIGBUS`). On error
    /// `path` is untouched and the temporary file removed.
    pub fn write_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        assert!(self.is_frozen(), "finish() the image before writing it");
        use std::io::Write;
        let path = path.as_ref();
        let used = self.used();
        let bytes = unsafe { std::slice::from_raw_parts(self.base as *const u8, used) };
        let mut temp = TempFile::create_next_to(path)?;
        // Large writes in bounded chunks (write_all loops, but keep each syscall moderate).
        for chunk in bytes.chunks(256 << 20) {
            temp.file.write_all(chunk)?;
        }
        temp.file.flush()?;
        temp.persist(path)
    }

    /// Check, cheaply and before a long build, that [`Self::write_to`] can write `path`: that it is
    /// not a directory and that a temporary file can be created next to it (the directory exists
    /// and is writable). Leaves nothing behind.
    pub fn check_writable(path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("{} is a directory", path.display()),
            ));
        }
        TempFile::create_next_to(path).map(drop)
    }

    /// The whole used image as bytes (tests / hashing).
    #[cfg(test)]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.base as *const u8, self.used()) }
    }

    fn parse_header(bytes: &[u8]) -> io::Result<Header> {
        if bytes.len() < std::mem::size_of::<Header>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too small for an index header",
            ));
        }
        let header: Header = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const Header) };
        if header.magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a biopath index image (bad magic)",
            ));
        }
        if header.endian_check != ENDIAN_CHECK {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "index image written on a machine with another byte order",
            ));
        }
        if header.format_version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "index image format version {} but this binary reads version {FORMAT_VERSION}",
                    header.format_version
                ),
            ));
        }
        if header.toc_len as usize > TOC_CAPACITY {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "index image TOC too large",
            ));
        }
        Ok(header)
    }

    /// Open an index image file, checking its header and TOC (a `.bpi` is trusted input: see
    /// **Trust** in the module docs): memory-map it read-only and prefault all of it, so the load
    /// pays the whole read once and queries run warm.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        use std::os::unix::fs::FileExt;
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(&path)?;
        let file_len = file.metadata()?.len() as usize;
        let mut head = vec![0u8; HEADER_REGION.min(file_len)];
        file.read_exact_at(&mut head, 0)?;
        let header = Self::parse_header(&head)?;
        if header.used_bytes as usize != file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "index image header says {} bytes but the file has {file_len} (truncated or \
                     still being written?)",
                    header.used_bytes
                ),
            ));
        }
        if file_len < PRELUDE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "index image of {file_len} bytes is smaller than its header and table of \
                     contents ({PRELUDE_BYTES} bytes)"
                ),
            ));
        }
        let cap = align_up(file_len, PAGE);
        let p = unsafe {
            sys::mmap(
                std::ptr::null_mut(),
                cap,
                sys::PROT_READ,
                sys::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if p == sys::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // Prefer the blocking prefault (Linux >= 5.14); otherwise advise + touch.
        let rc = unsafe { sys::madvise(p, cap, sys::MADV_POPULATE_READ) };
        if rc != 0 {
            unsafe {
                sys::madvise(p, cap, sys::MADV_WILLNEED);
            }
            let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, file_len) };
            let mut acc = 0u64;
            for page in bytes.chunks(PAGE) {
                acc = acc.wrapping_add(page[0] as u64);
            }
            std::hint::black_box(acc);
        }
        let image = Self {
            base: p as *mut u8,
            cap,
            used: AtomicUsize::new(file_len),
            toc: Mutex::new(Vec::new()),
            instance_id: INSTANCE_IDS.fetch_add(1, Ordering::Relaxed),
            frozen: std::sync::atomic::AtomicBool::new(true),
        };
        let toc: Vec<TocEntry> = image
            .get_at::<TocEntry>(HEADER_REGION, header.toc_len as usize)
            .to_vec();
        for entry in &toc {
            check_toc_entry(entry, file_len)?;
        }
        *image.toc.lock().unwrap() = toc;
        Ok(image)
    }
}

/// Refuse a TOC entry whose section does not lie inside a file of `file_len` bytes, aligned for its
/// elements: every handle the image hands out comes from such an entry.
fn check_toc_entry(entry: &TocEntry, file_len: usize) -> io::Result<()> {
    let bad = |what: &str| {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index image section {} {what}", entry.id),
        ))
    };
    if !entry.elem_align.is_power_of_two() || !entry.elem_size.is_multiple_of(entry.elem_align) {
        return bad("has an invalid element layout");
    }
    if !entry.off.is_multiple_of(u64::from(entry.elem_align)) {
        return bad("is misaligned for its elements");
    }
    let end = entry
        .len
        .checked_mul(u64::from(entry.elem_size))
        .and_then(|bytes| entry.off.checked_add(bytes));
    if end.is_none_or(|end| end > file_len as u64) {
        return bad("extends past the file");
    }
    Ok(())
}

/// A temporary file next to a target path, removed on drop unless [`Self::persist`]ed over it.
struct TempFile {
    file: std::fs::File,
    path: std::path::PathBuf,
    persisted: bool,
}

impl TempFile {
    /// Create `.<name>.<pid>.<n>.tmp` in `target`'s directory (a fresh name, never an existing file).
    fn create_next_to(target: &Path) -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = target.file_name().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} names no file", target.display()),
            )
        })?;
        let mut temp_name = std::ffi::OsString::from(".");
        temp_name.push(name);
        temp_name.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let path = target.with_file_name(temp_name);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("cannot create a file next to {}: {e}", target.display()),
                )
            })?;
        Ok(Self {
            file,
            path,
            persisted: false,
        })
    }

    /// Rename the file over `target`.
    fn persist(mut self, target: &Path) -> io::Result<()> {
        std::fs::rename(&self.path, target)?;
        self.persisted = true;
        Ok(())
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.persisted {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(C)]
    struct Rec {
        a: u32,
        _pad: u32,
        b: u64,
    }
    image_record!(Rec = [u32, u32, u64]);

    #[test]
    fn alloc_push_get_round_trip() {
        let image = Image::new_arena(64 << 20).unwrap();
        let s1 = image.push(&[1u32, 2, 3]);
        let (s2, slice) = image.alloc::<Rec>(2);
        slice[0] = Rec {
            a: 1,
            _pad: 0,
            b: 2,
        };
        slice[1] = Rec {
            a: 3,
            _pad: 0,
            b: 4,
        };
        let s3 = image.push_section(SectionId(7), &[9u8, 8, 7]);
        assert_eq!(image.get(s1), &[1, 2, 3]);
        assert_eq!(
            image.get(s2)[1],
            Rec {
                a: 3,
                _pad: 0,
                b: 4
            }
        );
        assert_eq!(image.get(s3), &[9, 8, 7]);
        assert_eq!(s3.off as usize % PAGE, 0);
        assert!(image.has_section(SectionId(7)));
        assert_eq!(image.section::<u8>(SectionId(7)).unwrap().unwrap().len(), 3);
        assert!(image.section::<u8>(SectionId(8)).unwrap().is_none());
    }

    #[test]
    fn file_round_trip() {
        let image = Image::new_arena(64 << 20).unwrap();
        let data: Vec<u64> = (0..10_000).collect();
        image.push_section(SectionId(1), &data);
        image.push_section(
            SectionId(2),
            &[Rec {
                a: 5,
                _pad: 0,
                b: 6,
            }],
        );
        let mut header = Image::header_template(4);
        header.graph_node_count = 42;
        header.build_knobs[3] = 99;
        image.finish(header);
        let dir = std::env::temp_dir().join(format!("bpi-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("round.bpi");
        image.write_to(&path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len() as usize,
            image.used()
        );
        let loaded = Image::open(&path).unwrap();
        let h = loaded.header();
        assert_eq!(h.graph_node_count, 42);
        assert_eq!(h.build_knobs[3], 99);
        assert_eq!(h.word_size, 4);
        let s = loaded.section::<u64>(SectionId(1)).unwrap().unwrap();
        assert_eq!(loaded.get(s), &data[..]);
        let r = loaded.section::<Rec>(SectionId(2)).unwrap().unwrap();
        assert_eq!(
            loaded.get(r)[0],
            Rec {
                a: 5,
                _pad: 0,
                b: 6
            }
        );
        assert_eq!(loaded.as_bytes(), image.as_bytes());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_other_format_versions() {
        let image = Image::new_arena(64 << 20).unwrap();
        image.push_section(SectionId(1), &[1u64]);
        image.finish(Image::header_template(4));
        let dir = std::env::temp_dir().join(format!("bpi-test-version-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.bpi");
        let mut bytes = image.as_bytes().to_vec();
        let at = std::mem::offset_of!(Header, format_version);
        bytes[at..at + 4].copy_from_slice(&(FORMAT_VERSION - 1).to_ne_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = Image::open(&path).err().unwrap();
        assert!(
            err.to_string()
                .contains(&format!("format version {}", FORMAT_VERSION - 1)),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A finished one-section image (`u64` elements, id 1) as bytes.
    fn image_bytes() -> Vec<u8> {
        let image = Image::new_arena(MIN_RESERVATION).unwrap();
        image.push_section(SectionId(1), &[1u64, 2, 3]);
        image.finish(Image::header_template(4));
        image.as_bytes().to_vec()
    }

    /// Open `bytes` written to a file; the error message, if any.
    fn open_error(name: &str, bytes: &[u8]) -> Option<String> {
        let dir = std::env::temp_dir().join(format!("bpi-test-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        let result = Image::open(&path);
        std::fs::remove_file(&path).unwrap();
        result.err().map(|e| e.to_string())
    }

    /// Overwrite field `at` of the first TOC entry.
    fn set_toc(bytes: &mut [u8], at: usize, value: &[u8]) {
        let at = HEADER_REGION + at;
        bytes[at..at + value.len()].copy_from_slice(value);
    }

    #[test]
    fn rejects_sections_outside_the_file() {
        assert_eq!(open_error("ok.bpi", &image_bytes()), None);
        // `len * elem_size` wraps to 0 in u64.
        let mut bytes = image_bytes();
        set_toc(
            &mut bytes,
            std::mem::offset_of!(TocEntry, len),
            &(1u64 << 61).to_ne_bytes(),
        );
        let err = open_error("wrap.bpi", &bytes).unwrap();
        assert!(err.contains("extends past the file"), "{err}");
        let mut bytes = image_bytes();
        set_toc(
            &mut bytes,
            std::mem::offset_of!(TocEntry, off),
            &u64::MAX.to_ne_bytes(),
        );
        assert!(open_error("off.bpi", &bytes).is_some());
        let mut bytes = image_bytes();
        let off = PRELUDE_BYTES as u64 + 4;
        set_toc(
            &mut bytes,
            std::mem::offset_of!(TocEntry, off),
            &off.to_ne_bytes(),
        );
        let err = open_error("misaligned.bpi", &bytes).unwrap();
        assert!(err.contains("misaligned"), "{err}");
        let mut bytes = image_bytes();
        set_toc(
            &mut bytes,
            std::mem::offset_of!(TocEntry, elem_align),
            &0u32.to_ne_bytes(),
        );
        let err = open_error("align0.bpi", &bytes).unwrap();
        assert!(err.contains("invalid element layout"), "{err}");
    }

    #[test]
    fn rejects_a_file_shorter_than_its_toc() {
        let mut bytes = image_bytes();
        bytes.truncate(HEADER_REGION);
        let used = std::mem::offset_of!(Header, used_bytes);
        bytes[used..used + 8].copy_from_slice(&(HEADER_REGION as u64).to_ne_bytes());
        let toc_len = std::mem::offset_of!(Header, toc_len);
        bytes[toc_len..toc_len + 4].copy_from_slice(&(TOC_CAPACITY as u32).to_ne_bytes());
        let err = open_error("short.bpi", &bytes).unwrap();
        assert!(err.contains("smaller than its header"), "{err}");
    }

    #[test]
    fn a_section_with_another_layout_is_an_error() {
        let image = Image::new_arena(MIN_RESERVATION).unwrap();
        image.push_section(SectionId(1), &[1u64]);
        let err = image.section::<u32>(SectionId(1)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(image.require::<u64>(SectionId(1), "one").is_ok());
    }

    #[test]
    fn writing_replaces_a_mapped_file_without_truncating_it() {
        let dir = std::env::temp_dir().join(format!("bpi-test-replace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.bpi");
        let first = Image::new_arena(MIN_RESERVATION).unwrap();
        first.push_section(SectionId(1), &vec![7u64; 4096]);
        first.finish(Image::header_template(4));
        Image::check_writable(&path).unwrap();
        first.write_to(&path).unwrap();
        let mapped = Image::open(&path).unwrap();
        // A smaller image over the mapped file: the mapping keeps the old, complete file.
        let second = Image::new_arena(MIN_RESERVATION).unwrap();
        second.push_section(SectionId(1), &[9u64]);
        second.finish(Image::header_template(4));
        second.write_to(&path).unwrap();
        let old = mapped.require::<u64>(SectionId(1), "data").unwrap();
        assert!(mapped.get(old).iter().all(|&v| v == 7));
        let new = Image::open(&path).unwrap();
        assert_eq!(
            new.get(new.require::<u64>(SectionId(1), "data").unwrap()),
            &[9]
        );
        // Only the index is left in the directory (no temporary files).
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        assert!(Image::check_writable(&dir).is_err());
        assert!(Image::check_writable(dir.join("missing/index.bpi")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[should_panic(expected = "of address space reserved for it")]
    fn an_image_beyond_its_reservation_names_the_limit() {
        let image = Image::new_arena(MIN_RESERVATION).unwrap();
        image.alloc::<u8>(MIN_RESERVATION + 1);
    }

    #[test]
    fn rejects_garbage() {
        let dir = std::env::temp_dir().join(format!("bpi-test-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.bpi");
        std::fs::write(&path, vec![0u8; 8192]).unwrap();
        assert!(Image::open(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
