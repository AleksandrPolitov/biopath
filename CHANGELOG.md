# Changelog

## 0.4.0

The first public release of the SPQR-tree distance index. (biopath 0.1–0.3 on crates.io are an
earlier command-line tool with a different index.)

- `biopath index` builds an exact shortest-path distance index of a GFA or GBZ graph and writes it
  as a memory-mappable `.bpi` file (format version 19); `biopath query --index-in` maps it,
  prefaults all of it and answers distance queries from it without the graph.
- Strict GFA 1 input: lengths from `LN:i:` tags for `*` sequences; overlapping links, duplicate
  segment names, `J` lines and GFA 2 are refused with the line number.
- Word size 32 (default) or 64. A graph whose distances may not fit is refused before the build,
  with the word size to use. Query offsets beyond a node's end are refused when the query file is
  read.
- The same graph and options give a byte-identical index. Corrupt, truncated or foreign `.bpi`
  files are refused at load; a `.bpi` is still trusted input.
- Subcommands: `index`, `query`, `generate-random-queries`, `index-inspect`, `statistics` and
  `export-shrunk-gfa`. The log goes to stderr, a short summary at `info`.
- 64-bit Linux only.
