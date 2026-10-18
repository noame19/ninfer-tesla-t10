#!/usr/bin/env python3
"""make_corpus.py -- build the long-context probe corpora from what is already in the tree.

The KVMem probes need two long texts: `filler.txt` (~10 MB of dense technical English) and
`novel.txt` (~390 KB of narrative Chinese). Committing them is wrong -- the filler is derivable
from the repository's own documentation, and the narrative one is a published novel, i.e.
third-party text that must not be redistributed.

So this script regenerates the filler from tracked Markdown, and points at an external novel if
one is configured. Byte equality with the historic `filler.txt` is *not* a goal: every consumer
sizes the corpus from the token count it measures at runtime (see the comments in
`kvmem_recall.py`), so a different but equally long body of prose is a valid substitute.

  ./make_corpus.py                        # write filler.txt into $OUT_DIR
  ./make_corpus.py --target-mib 10        # match the historic size
  NINFER_T10_RECALL_NOVEL=/path/novel.txt ./make_corpus.py --check
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # tools/t10-perf/make_corpus.py -> repository root


def markdown_sources() -> list[Path]:
    """Tracked prose, longest first so the corpus is not dominated by tiny files."""
    seen: dict[Path, int] = {}
    for pattern in ("docs/**/*.md", "*.md", "bench/**/*.md", "tools/**/*.md"):
        for path in ROOT.glob(pattern):
            if path.is_file():
                seen[path] = path.stat().st_size
    return [p for p, _ in sorted(seen.items(), key=lambda kv: -kv[1])]


def build_filler(target_bytes: int) -> str:
    """Concatenate the tree's own documentation until it is long enough.

    Technical English, plenty of identifiers and numbers, no repeated sentence -- which is what
    makes it a fair long-context filler: a model cannot complete it from a memorised n-gram.
    """
    chunks: list[str] = []
    total = 0
    sources = markdown_sources()
    if not sources:
        raise SystemExit("no Markdown found; run from a checkout of the repository")
    while total < target_bytes:
        for path in sources:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            chunks.append(text)
            total += len(text)
            if total >= target_bytes:
                break
    return "".join(chunks)[:target_bytes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Same default as env.sh's OUT_DIR, so the probes find the corpus without any export.
    parser.add_argument("--out-dir",
                        default=os.environ.get("OUT_DIR",
                                               str(Path(tempfile.gettempdir()) /
                                                   "ninfer-t10-perf")),
                        help="where to write filler.txt")
    parser.add_argument("--target-mib", type=float, default=10.0,
                        help="filler size; the historic file was 10,251,609 bytes (default 10)")
    parser.add_argument("--check", action="store_true",
                        help="only report what is present, write nothing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    filler_path = out_dir / "filler.txt"
    novel_path = Path(os.environ.get("NINFER_T10_RECALL_NOVEL",
                                     str(out_dir / "novel.txt")))

    if args.check:
        for label, path in (("filler", filler_path), ("novel", novel_path)):
            if path.is_file():
                print(f"{label:7s} {path} {path.stat().st_size} bytes")
            else:
                print(f"{label:7s} {path} MISSING")
        if not novel_path.is_file():
            print("\nThe narrative corpus is external on purpose: it is a published novel, so it is "
                  "never committed.\nPoint NINFER_T10_RECALL_NOVEL at any long narrative text "
                  "(several hundred KB) to run the probes.")
        return 0

    target = int(args.target_mib * 1024 * 1024)
    text = build_filler(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    filler_path.write_text(text, encoding="utf-8")
    print(f"wrote {filler_path} ({len(text)} bytes from {len(markdown_sources())} Markdown files)")
    if not novel_path.is_file():
        print(f"note: {novel_path} is absent; the narrative probes need it. "
              f"Set NINFER_T10_RECALL_NOVEL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
