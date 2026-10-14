"""Prove a block-imported artifact still holds the source GGUF's own bytes.

The GGUF block recipe's whole claim is that a row of the artifact is a row of the GGUF, byte for
byte, so that serving it cannot be worse than llama.cpp's own arithmetic on the same file. This
tool checks that claim from the outside:

* it reads the *artifact's* stored object bytes through the container reader, and the *GGUF's*
  tensor bytes straight off the mapped file, with row extents computed from the file's own tensor
  sizes rather than from any quantizer's block table;
* it derives the tower's geometry from the artifact's recorded config, and the row map of every
  imported parameter (identity, the head-interleaved ``attn_q`` split, or the exporter's tiled
  value-head order) from that geometry, independently of the recipe that wrote the artifact;
* it compares every row of every ``gguf_*`` object and reports exactly how many bytes match.

A mismatch means either the import re-quantized (the row's bytes differ everywhere) or a row map
moved the wrong rows; both are silent quality losses otherwise, since a wrong permutation still
loads and still produces fluent text.

Usage::

    python3 -m tools.convert.gguf_fidelity --gguf MODEL.gguf --artifact MODEL.ninfer

Exit status is 0 when every compared byte matches and the geometry agrees with the GGUF header.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE
from tools.artifact.reader import Artifact

from .gguf_geometry import Geometry, check_header, geometry_from_config
from .sources.gguf import GGUFFile

# A conversion method's source string: "blk.3.attn_q.weight[gguf_q4_k][4096, 2560]".
_SOURCE = re.compile(r"^(?P<tensor>.+)\[(?P<format>[a-z0-9_]+)\]\[(?P<rows>\d+), (?P<columns>\d+)\]$")


def _geometry_from_artifact(directory: Mapping) -> Geometry:
    """The tower the artifact records, so the check needs no side files beyond the config."""

    text = directory["components"]["text"]["config"]
    return geometry_from_config(text, mtp="mtp" in directory["components"])


def _tiled_to_grouped(geometry: Geometry) -> np.ndarray:
    """For grouped value head `h`, the exporter's tiled position, re-derived from the convention.

    llama.cpp's Qwen3.5 exporter tiles the GDN value heads so that repeating the key heads
    broadcasts them: grouped head `k * per_key + r` is stored at tiled position `r * key_heads + k`.
    """

    per_key = geometry.gdn_value_heads // geometry.gdn_key_heads
    heads = np.arange(geometry.gdn_value_heads)
    return (heads % per_key) * geometry.gdn_key_heads + heads // per_key


def _source_rows(tensor: str, parameter: str, geometry: Geometry, rows: int) -> np.ndarray:
    """Which stored row of `tensor` each logical row of `parameter` comes from."""

    index = np.arange(rows, dtype=np.int64)
    head_dim = geometry.gdn_head_dim
    key_dim = geometry.gdn_key_dim
    value_dim = geometry.gdn_value_dim

    def untiled(base: int) -> np.ndarray:
        permutation = _tiled_to_grouped(geometry)
        local = index // head_dim
        if local.max(initial=0) >= permutation.size:
            raise SystemExit(f"{parameter}: rows exceed the value heads")
        return base + permutation[local] * head_dim + index % head_dim

    if tensor.endswith("attn_qkv.weight"):
        role = parameter.rsplit("/", 1)[-1]
        if role == "query":
            return index
        if role == "key":
            return key_dim + index
        if role == "value":
            return untiled(2 * key_dim)
        raise SystemExit(f"{parameter}: unexpected role in a fused q/k/v tensor")
    if tensor.endswith("attn_gate.weight"):
        if rows != value_dim:
            raise SystemExit(f"{parameter}: z rows {rows} differ from the value width {value_dim}")
        return untiled(0)
    if tensor.endswith("attn_q.weight"):
        # One head's query and its interleaved output gate, head by head.
        d = geometry.attention_head_dim
        gate = parameter.endswith("/gate")
        if rows != geometry.attention_query_rows:
            raise SystemExit(f"{parameter}: query rows {rows} differ from the head width")
        return (index // d) * 2 * d + (d if gate else 0) + index % d
    return index


def _raw_rows(
    mapping: np.ndarray, tensor: str, info, rows: int, first: int, last: int
) -> bytes:
    """The GGUF's raw bytes for rows `[first, last)` of `tensor`, in logical row order."""

    stride = info.nbytes // info.shape[0]
    if stride * info.shape[0] != info.nbytes:
        raise SystemExit(f"{tensor}: tensor bytes are not a whole number of rows")
    begin = info.offset + first * stride
    end = info.offset + last * stride
    return bytes(mapping[begin:end])


def verify(gguf_path: Path, artifact_path: Path, directory_path: Path) -> dict:
    directory = json.loads(directory_path.read_text())
    geometry = _geometry_from_artifact(directory)
    report = {"compared_bytes": 0, "matched_bytes": 0, "objects": 0, "rows": 0, "failures": []}

    with GGUFFile(gguf_path) as gguf, Artifact(artifact_path) as artifact:
        # The artifact's own config must describe the same tower the GGUF holds.
        check_header(gguf.kv, geometry.with_mtp(geometry.mtp), source=str(gguf_path))
        raw = np.memmap(gguf_path, dtype=np.uint8, mode="r")
        base = gguf.data_offset
        for entry in directory["methods"]:
            if not entry["format"].startswith("gguf_"):
                continue
            objects = bytes(artifact.read_object(entry["object"]))
            total_rows = entry["shape"][0]
            if total_rows <= 0 or len(objects) % total_rows:
                raise SystemExit(f"{entry['object']}: payload is not a whole number of rows")
            row_bytes = len(objects) // total_rows
            offset = 0
            report["objects"] += 1
            for parameter, source in zip(
                entry["parameters"], entry["sources"], strict=True
            ):
                match = _SOURCE.match(source)
                if match is None:
                    raise SystemExit(f"{parameter}: cannot read its source reference {source!r}")
                tensor = match["tensor"]
                rows = int(match["rows"])
                info = gguf.info(tensor)
                # The source's own ggml type must be the format the artifact stores: a different
                # one would mean the import re-encoded the rows.
                stored = GGUF_FORMATS_BY_TYPE.get(info.type_id)
                if stored is None or stored.name != match["format"]:
                    raise SystemExit(
                        f"{parameter}: source is {info.type_name}, stored as {match['format']}"
                    )
                if info.shape[0] < rows or info.shape[-1] != int(match["columns"]):
                    raise SystemExit(f"{parameter}: source tensor is smaller than the import")
                index = _source_rows(tensor, parameter, geometry, rows)
                stride = info.nbytes // info.shape[0]
                if stride != row_bytes:
                    raise SystemExit(
                        f"{parameter}: source row is {stride} bytes, the artifact row is {row_bytes}"
                    )
                want = bytearray()
                for row in index:
                    begin = base + info.offset + int(row) * stride
                    want += bytes(raw[begin : begin + stride])
                got = objects[offset + 0 : offset + rows * row_bytes]
                offset += rows * row_bytes
                compared = min(len(want), len(got))
                report["compared_bytes"] += compared
                report["rows"] += rows
                if bytes(want) == got:
                    report["matched_bytes"] += compared
                    continue
                differing = sum(1 for a, b in zip(want, got, strict=True) if a != b)
                report["matched_bytes"] += compared - differing
                report["failures"].append(
                    f"{parameter} <- {tensor}: {differing} of {compared} bytes differ"
                )
            if offset != len(objects):
                raise SystemExit(f"{entry['object']}: sources do not cover the object")
        del raw
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prove a block-imported artifact holds the source GGUF's own bytes")
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--conversion", type=Path)
    arguments = parser.parse_args(argv)
    conversion = arguments.conversion or Path(str(arguments.artifact) + ".conversion.json")
    report = verify(arguments.gguf, arguments.artifact, conversion)
    compared, matched = report["compared_bytes"], report["matched_bytes"]
    share = 0.0 if compared == 0 else 100.0 * matched / compared
    print(
        f"{report['objects']} objects / {report['rows']} rows: "
        f"{matched} of {compared} bytes identical to the source GGUF ({share:.4f}%)"
    )
    for failure in report["failures"][:20]:
        print(f"  MISMATCH {failure}", file=sys.stderr)
    if report["failures"]:
        print(f"  {len(report['failures'])} objects differ", file=sys.stderr)
        return 1
    print("every imported row is byte-identical to the source GGUF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
