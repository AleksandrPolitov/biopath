#!/usr/bin/env bash
# Golden regression check for `biopath index` / `biopath query --index-in`.
#
# For every corpus graph under every config (tests/golden/configs.tsv) it builds a .bpi, answers
# the stored queries from it, and compares the SHA-256 of the image and of the answers with
# tests/golden/manifest.tsv.
#
#   scripts/golden.sh record   write the manifest with the current binary; also checks every
#                              answer against whole-graph Dijkstra
#   scripts/golden.sh check    compare the current binary with the manifest
#
# It also runs, per graph, the paths that build no index — `query --engine dijkstra`,
# the in-memory `query --engine dp` build, and `--compare-whole-graph-dijkstra` on the prod index —
# and requires their answers to equal the prod config's (exact distances do not depend on config).
#
# BIN overrides the binary (default target/release/biopath), WORK the scratch directory.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
GOLDEN=$ROOT/tests/golden
BIN=${BIN:-$ROOT/target/release/biopath}
WORK=${WORK:-$ROOT/target/golden}

mode=${1:?usage: golden.sh record|check}
[[ $mode == record || $mode == check ]] || { echo "unknown mode $mode" >&2; exit 2; }
[[ $# == 1 ]] || { echo "usage: golden.sh record|check (no other arguments)" >&2; exit 2; }

declare -A CONFIG_ENV CONFIG_FLAGS
while IFS=$'\t' read -r name env flags; do
    [[ -z $name || $name == \#* ]] && continue
    CONFIG_ENV[$name]=$env
    CONFIG_FLAGS[$name]=$flags
done < "$GOLDEN/configs.tsv"

# Run the binary with a clean environment plus the config's knobs.
run() {
    local config=$1; shift
    local env=${CONFIG_ENV[$config]}
    [[ $env == - ]] && env=
    # shellcheck disable=SC2086
    env -i PATH="$PATH" HOME="$HOME" RAYON_NUM_THREADS=1 $env "$BIN" "$@"
}

# Set Q to the query file of a graph, generated on first use: reachable walk queries of two
# lengths plus independent random pairs (mostly unreachable, which covers the "no path" answers).
# The stored files of graphs with zero-length nodes have no random pairs: they were recorded when
# `generate-random-queries` could not place an offset on such a node.
queries_for() {
    local name=$1 gfa=$2 amount=$3
    Q=$GOLDEN/queries/$name.tsv
    [[ -f $Q ]] && return
    [[ $mode == record ]] || { echo "missing $Q (run record first)" >&2; exit 1; }
    local tmp=$WORK/queries/$name
    mkdir -p "$(dirname "$tmp")"
    "$BIN" generate-random-queries --log-level warn --graph-gfa-in "$gfa" --query-out "$tmp.w8" --amount "$amount" --random-seed 1 --walk-length 8 > /dev/null
    "$BIN" generate-random-queries --log-level warn --graph-gfa-in "$gfa" --query-out "$tmp.w64" --amount "$amount" --random-seed 2 --walk-length 64 > /dev/null
    if ! "$BIN" generate-random-queries --graph-gfa-in "$gfa" --query-out "$tmp.rnd" --amount "$((amount / 4))" --random-seed 3 > /dev/null 2>&1; then
        echo "note: $name: no independent random pairs (generate-random-queries failed)" >&2
        : > "$tmp.rnd"
    fi
    cat "$tmp.w8" "$tmp.w64" "$tmp.rnd" > "$Q"
}

# The index-free paths on one graph: their answers must equal `$expected_out` (the prod answers).
other_paths() {
    local name=$1 gfa=$2 q=$3 expected_out=$4
    local out=$WORK/other/$name
    mkdir -p "$(dirname "$out")"
    run noshrink query --engine dijkstra --graph-gfa-in "$gfa" --query-in "$q" \
        --query-out "$out.dijkstra.out.tsv" > "$out.dijkstra.log" 2>&1 \
        || { echo "FAILED: $name --engine dijkstra (see $out.dijkstra.log)" >&2; return 1; }
    run noshrink query --engine dp --word-size 32 --graph-gfa-in "$gfa" --query-in "$q" \
        --query-out "$out.inmemory.out.tsv" > "$out.inmemory.log" 2>&1 \
        || { echo "FAILED: $name in-memory --engine dp (see $out.inmemory.log)" >&2; return 1; }
    run prod query --engine dp --word-size 32 --index-in "$WORK/prod/$name.bpi" --graph-gfa-in "$gfa" \
        --compare-whole-graph-dijkstra --query-in "$q" --query-out "$out.compare.out.tsv" > "$out.compare.log" 2>&1 \
        || { echo "FAILED: $name --compare-whole-graph-dijkstra (see $out.compare.log)" >&2; return 1; }
    grep -q "All query results match whole-graph Dijkstra" "$out.compare.log" \
        || { echo "FAILED: $name: --compare-whole-graph-dijkstra reports a mismatch (see $out.compare.log)" >&2; return 1; }
    local f
    for f in dijkstra inmemory compare; do
        cmp -s "$expected_out" "$out.$f.out.tsv" \
            || { echo "FAILED: $name: the $f answers differ from the prod index" >&2; return 1; }
    done
}

# Build + query one (graph, config); print the manifest line.
one() {
    local name=$1 gfa=$2 config=$3 q=$4 compare=$5
    local out=$WORK/$config/$name
    mkdir -p "$(dirname "$out")"
    local flags=${CONFIG_FLAGS[$config]}
    [[ $flags == - ]] && flags=
    # shellcheck disable=SC2086
    if ! run "$config" index --engine dp --word-size 32 $flags --graph-gfa-in "$gfa" --index-out "$out.bpi" > "$out.index.log" 2>&1; then
        echo "FAILED: index $name/$config (see $out.index.log)" >&2
        return 1
    fi
    if ! run "$config" query --engine dp --word-size 32 --index-in "$out.bpi" \
        --query-in "$q" --query-out "$out.out.tsv" > "$out.query.log" 2>&1; then
        echo "FAILED: query $name/$config (see $out.query.log)" >&2
        return 1
    fi
    if [[ $compare == 1 ]]; then
        run "$config" query --engine dp --word-size 32 --index-in "$out.bpi" --graph-gfa-in "$gfa" \
            --compare-whole-graph-dijkstra --query-in "$q" --query-out "$out.dijkstra.out.tsv" > "$out.dijkstra.log" 2>&1
        if grep -q "differ from whole-graph Dijkstra" "$out.dijkstra.log"; then
            echo "FAILED: $name/$config disagrees with whole-graph Dijkstra (see $out.dijkstra.log)" >&2
            return 1
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$config" \
        "$(sha256sum < "$out.bpi" | cut -c1-16)" "$(sha256sum < "$out.out.tsv" | cut -c1-16)" "$(stat -c %s "$out.bpi")"
}

[[ -x $BIN ]] || { echo "no binary at $BIN (cargo build --release)" >&2; exit 1; }
mkdir -p "$WORK"
result=$WORK/manifest.$mode.tsv
: > "$result"
failed=0

manifest=$GOLDEN/manifest.tsv
configs=$(grep -v '^#' "$GOLDEN/configs.tsv" | cut -f1)
for gfa in "$GOLDEN"/graphs/*.gfa; do
    name=$(basename "$gfa" .gfa)
    queries_for "$name" "$gfa" 400
    for config in $configs; do
        one "$name" "$gfa" "$config" "$Q" "$([[ $mode == record ]] && echo 1 || echo 0)" >> "$result" || failed=1
    done
    other_paths "$name" "$gfa" "$Q" "$WORK/prod/$name.out.tsv" || failed=1
done

if [[ $mode == record ]]; then
    [[ $failed == 0 ]] || { echo "record FAILED; manifest not written" >&2; exit 1; }
    {
        echo "# graph	config	bpi sha256 (16)	answers sha256 (16)	bpi bytes"
        echo "# recorded by $(basename "$BIN") from biopath $(git -C "$ROOT" rev-parse --short HEAD)"
        cat "$result"
    } > "$manifest"
    echo "recorded $(wc -l < "$result") entries in $manifest"
    exit 0
fi

expected=$(grep -v '^#' "$manifest" | sort)
actual=$(sort "$result")
if [[ $failed == 0 && $expected == "$actual" ]]; then
    echo "golden OK: $(wc -l < "$result") entries match $manifest"
    exit 0
fi
echo "golden MISMATCH against $manifest (< expected, > actual):" >&2
diff <(echo "$expected") <(echo "$actual") >&2 || true
exit 1
