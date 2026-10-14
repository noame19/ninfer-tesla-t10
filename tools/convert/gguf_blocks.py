"""A dense Qwen3.5/3.8 GGUF kept in its own ggml block formats (GSQ-RCO mixed-precision releases).

The block-quantized main text projections, token table, output head and MTP head of an ``-mtp``
build are stored as the exporter's own blocks, without re-quantizing these matrices: a row
of the artifact is byte for byte a row of the GGUF. llama.cpp's Qwen3.5 exporter conventions are
undone where they touch rows or small tensors: value heads return to the grouped order,
zero-centred norms drop their stored ``1 + w`` and ``ssm_a`` becomes ``A_log`` again.
The small GDN alpha/beta gates use the runtime's BF16 format; gates stored as F32/F16 words or as
quantized blocks are decoded to the values they represent before that cast, while gates already
stored in BF16 retain their exact words.

The geometry is not fixed here. It is read from the base checkpoint's config and checked against
the GGUF's own header and tensor set (`gguf_geometry.py`), so the same recipe converts every dense
Qwen3.5/3.8 release -- the 27B widths, the 9B ones, and anything else a config describes -- while
a GGUF that disagrees with its config is refused instead of guessed at.

``ssm_out`` is the exception. Its tiled value heads are its input columns, which a row copy cannot
move without re-quantising, so its Use carries an ``input_columns`` auxiliary: column ``c`` of the
stored matrix multiplies element ``input_columns[c]`` of the grouped activation.

The GDN controls ``ssm_alpha`` and ``ssm_beta`` are BF16 operands of the Engine. Exporters store
them as BF16 words or as Q8_0 blocks (Underdog-Saluki and similar releases), whose values are
rounded to BF16: the one place a stored value changes.

Vision comes from the release's ``mmproj`` GGUF (``--source vision=mmproj.gguf``) with the official
Vision formats, DFlash2 from ``--source dflash2`` and the frontend resources from ``--model``, which
needs only the base checkpoint's config and tokenizer files.
"""

from __future__ import annotations

from math import prod

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .gguf_geometry import (
    BLOCKS,
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Geometry,
    Kind,
    check_header,
    expected_tensors,
    geometry_from_config,
    header_expectations,
)
from .methods import AuxiliaryValue, cast_direct, grouped_absmax, import_encoded
from .official_recipes import Q4, Q5, Q6, Q8
from .sources.gguf import TYPE_BF16, TYPE_F16, TYPE_F32, GGUFFile
from .sources.logical import EncodedRows, LogicalSource, array_source
from .ternary import (
    ATTENTION_HEADS,
    ATTENTION_HEAD_DIM,
    GDN_HEAD_DIM,
    GDN_KEY_HEADS,
    GDN_TAPS,
    GDN_VALUE_HEADS,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    VOCABULARY,
    RowMap,
    _direct,
    _flat,
    _norm,
    full_attention,
    rows,
)

# The geometry every default argument below uses, and the one the Base 27B artifacts were built
# with: one place to read, and the ternary and Flash-Next recipes keep their signatures.
QWEN38_27B_DENSE = Geometry(
    hidden=HIDDEN,
    intermediate=INTERMEDIATE,
    vocabulary=VOCABULARY,
    layers=LAYERS,
    attention_heads=ATTENTION_HEADS,
    attention_head_dim=ATTENTION_HEAD_DIM,
    kv_heads=4,
    gdn_key_heads=GDN_KEY_HEADS,
    gdn_value_heads=GDN_VALUE_HEADS,
    gdn_head_dim=GDN_HEAD_DIM,
    taps=GDN_TAPS,
    layer_types=tuple(
        FULL_ATTENTION if full_attention(layer) else LINEAR_ATTENTION
        for layer in range(LAYERS)
    ),
)

EXPECTED_HEADER = header_expectations(QWEN38_27B_DENSE)


def tiled_to_grouped_permutation(geometry: Geometry) -> np.ndarray:
    """For grouped value head `h = k * per_key + r`, the exporter's tiled position `r * K + k`.

    llama.cpp's exporter stores value heads tiled so that a plain repeat broadcasts the key
    heads; NInfer pairs value head `h` with key head `h // per_key` and wants the grouped order.
    """

    heads = np.arange(geometry.gdn_value_heads)
    per_key = geometry.gdn_value_heads_per_key
    return (heads % per_key) * geometry.gdn_key_heads + heads // per_key


def untile(array: np.ndarray, head_dim: int, geometry: Geometry) -> np.ndarray:
    """Reorder axis 0 from the exporter's tiled value-head order back to grouped."""

    expected = geometry.gdn_value_heads * head_dim
    if array.shape[0] != expected:
        raise ValueError(f"axis 0 has {array.shape[0]} entries, expected {expected}")
    view = array.reshape(geometry.gdn_value_heads, head_dim, *array.shape[1:])
    return np.ascontiguousarray(
        view[tiled_to_grouped_permutation(geometry)].reshape(array.shape)
    )


def attention_rows(gate: bool, geometry: Geometry) -> RowMap:
    """The query (or output gate) rows of the head-interleaved `attn_q`."""

    d = geometry.attention_head_dim

    def select(begin: int, end: int) -> np.ndarray:
        r = np.arange(begin, end, dtype=np.int64)
        return (r // d) * 2 * d + (d if gate else 0) + r % d

    return select


def untiled_rows(base: int, geometry: Geometry) -> RowMap:
    """Grouped value-head rows of a tiled family starting at source row `base`."""

    permutation = tiled_to_grouped_permutation(geometry)
    head_dim = geometry.gdn_head_dim

    def select(begin: int, end: int) -> np.ndarray:
        r = np.arange(begin, end, dtype=np.int64)
        return base + permutation[r // head_dim] * head_dim + r % head_dim

    return select


def _accepts(info, kind: Kind) -> bool:
    """Whether a stored tensor's ggml type satisfies one expected kind."""

    for entry in (kind,) if isinstance(kind, str) else kind:
        if entry == BLOCKS:
            if info.type_id in GGUF_FORMATS_BY_TYPE:
                return True
        elif info.type_name == entry:
            return True
    return False


def has_mtp(gguf: GGUFFile, geometry: Geometry = QWEN38_27B_DENSE) -> bool:
    return f"blk.{geometry.mtp_block}.nextn.eh_proj.weight" in gguf.tensors


def validate(gguf: GGUFFile, geometry: Geometry = QWEN38_27B_DENSE) -> Geometry:
    """Refuse any GGUF that is not this dense Qwen3.5/3.8 text tower of stored block types.

    `geometry` is the tower the base checkpoint describes. Whether the file carries the exporter's
    next-token-prediction block is the file's own answer, so it is read here rather than assumed;
    the returned Geometry is the file as it actually is, MTP block included.
    """

    full = geometry.with_mtp(has_mtp(gguf, geometry))
    check_header(gguf.kv, full, source=str(gguf.path))
    expected = expected_tensors(full)
    if set(gguf.tensors) != set(expected):
        missing = sorted(set(expected) - set(gguf.tensors))[:5]
        extra = sorted(set(gguf.tensors) - set(expected))[:5]
        raise ValueError(
            f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})"
        )
    for name, (shape, kind) in expected.items():
        info = gguf.tensors[name]
        if info.shape != shape or not _accepts(info, kind):
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")
    return full


def block_format(gguf: GGUFFile, tensor: str) -> str:
    return GGUF_FORMATS_BY_TYPE[gguf.info(tensor).type_id].name


GGML_Q8_0 = 8
GGML_Q2_0 = 42


def dequantize_q2_0(blocks: np.ndarray) -> np.ndarray:
    """ggml Q2_0 rows: 64 values per 18-byte block, a binary16 d then 16 code bytes; value j is
    d * (c - 1) with c the bit pair 2 (j % 4) of byte j / 4. gguf-py predates the type."""

    rows = blocks.shape[0]
    b = np.ascontiguousarray(blocks).reshape(rows, -1, 18)
    d = b[..., :2].copy().view("<f2").astype(np.float32)
    q = b[..., 2:]
    codes = np.stack([(q >> (2 * i)) & 3 for i in range(4)], axis=-1).reshape(rows, b.shape[1], 64)
    return ((codes.astype(np.float32) - 1.0) * d).reshape(rows, -1)


def dequantize_q8_0(blocks: np.ndarray) -> np.ndarray:
    """ggml Q8_0 rows: 32 values per 34-byte block, a binary16 d then 32 signed code bytes q;
    value j is d * q[j], exact in FP32."""

    rows = blocks.shape[0]
    b = np.ascontiguousarray(blocks).reshape(rows, -1, 34)
    d = b[..., :2].copy().view("<f2").astype(np.float32)
    q = b[..., 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(rows, -1)


def quantize_q8_0(values: np.ndarray) -> np.ndarray:
    """ggml quantize_row_q8_0_ref over rows of whole blocks, in FP32: per 32 values a binary16
    d = amax / 127 and codes round(x / d), halves away from zero. Returns [rows, row bytes]."""

    rows = values.shape[0]
    blocks = np.ascontiguousarray(values, dtype=np.float32).reshape(rows, -1, 32)
    d = np.abs(blocks).max(axis=-1) / np.float32(127.0)
    inverse = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d != 0)
    scaled = blocks * inverse[..., None]
    codes = (np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5))).astype(np.int8)
    out = np.empty((rows, blocks.shape[1], 34), dtype=np.uint8)
    out[..., :2] = d.astype("<f2")[..., None].view(np.uint8)
    out[..., 2:] = codes.view(np.uint8)
    return out.reshape(rows, -1)


def _dequantize(gguf: GGUFFile, tensor: str, first: int, last: int) -> torch.Tensor:
    blocks = gguf.read_blocks(tensor, first, last)
    local = {GGML_Q2_0: dequantize_q2_0, GGML_Q8_0: dequantize_q8_0}.get(
        gguf.info(tensor).type_id
    )
    if local is not None:
        values = local(blocks)
        return torch.from_numpy(np.ascontiguousarray(values)).reshape(last - first, -1)
    try:
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
    except ImportError as error:
        raise ValueError(
            "reading the values of a GGUF block matrix needs the `gguf` package; "
            "storing its blocks does not"
        ) from error
    kind = GGMLQuantizationType(gguf.info(tensor).type_id)
    values = dequantize(np.ascontiguousarray(blocks), kind)
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).reshape(
        last - first, -1
    )


def _values(gguf: GGUFFile, tensor: str) -> np.ndarray:
    """Every value of a stored tensor as float32 words, direct formats and blocks alike.

    The GDN gates are the only tensors this serves: a quantized gate decodes to the values it
    represents, which is what the Engine's BF16 operand needs.
    """

    info = gguf.info(tensor)
    if info.type_id in (TYPE_F32, TYPE_F16, TYPE_BF16):
        return gguf.read_direct(tensor)
    rows = prod(info.shape[:-1])
    decoded = _dequantize(gguf, tensor, 0, rows).numpy()
    return np.ascontiguousarray(decoded.reshape(info.shape))


def block_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Rows of a stored block matrix as encoded ggml rows, and their values on request."""

    format = block_format(gguf, tensor)
    empty = torch.empty(0, dtype=torch.float16)

    def encoded(begin: int, end: int) -> EncodedRows:
        index = select(begin, end)
        low, high = int(index.min()), int(index.max()) + 1
        blocks = gguf.read_blocks(tensor, low, high)
        return EncodedRows(
            format, torch.from_numpy(np.ascontiguousarray(blocks[index - low])), empty
        )

    def values(first: int, last: int) -> torch.Tensor:
        index = select(first, last)
        low, high = int(index.min()), int(index.max()) + 1
        return _dequantize(gguf, tensor, low, high)[torch.from_numpy(index - low)]

    return LogicalSource(
        shape, f"{tensor}[{format}]{list(shape)}", _flat(values, shape[1]), encoded
    )


def gdn_control_source(
    gguf: GGUFFile, tensor: str, geometry: Geometry = QWEN38_27B_DENSE
) -> LogicalSource:
    """A GDN A/B control as its BF16 operand, its value heads in the grouped order.

    BF16 words are stored exactly as they are. Every other storage an exporter may choose --
    F32/F16 words, or Q8_0 and the other block types -- decodes to the values it represents
    before the BF16 cast, so the operand does not acquire a second quantization.
    """

    if gguf.info(tensor).type_id == TYPE_BF16:
        words = untile(gguf.read_bf16_words(tensor), 1, geometry)
        return array_source(
            torch.from_numpy(np.ascontiguousarray(words.view(np.int16))).view(torch.bfloat16),
            tensor,
        )
    values = untile(_values(gguf, tensor), 1, geometry)
    return array_source(
        torch.from_numpy(np.ascontiguousarray(values)).to(torch.bfloat16), tensor
    )


def q2_native_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Import GSQ Q2_0 as native offset codes and the exact signed binary16 scale words.

    The row map selects rows within a tensor or flattened expert bank. The physical K padding
    holds logical integer zero (stored code 1); wholly padding groups have positive-zero scales.
    No floating-point dequantization or quantizer participates in the encoded import.
    """
    if gguf.info(tensor).type_id != GGML_Q2_0:
        raise ValueError(f"{tensor}: native Q2 import requires stored Q2_0 blocks")
    if len(shape) != 2 or shape[1] != gguf.info(tensor).shape[-1] or shape[1] % 64:
        raise ValueError(f"{tensor}: native Q2 import requires complete 64-value blocks")
    raw_source = block_source(gguf, tensor, shape, select)
    k = shape[1]
    groups, padded_groups = k // 64, ((k + 127) // 128) * 2

    def encoded(begin: int, end: int) -> EncodedRows:
        index = select(begin, end)
        low, high = int(index.min()), int(index.max()) + 1
        blocks = gguf.read_blocks(tensor, low, high)[index - low].reshape(end - begin, groups, 18)
        scale_words = np.zeros((end - begin, padded_groups), dtype="<u2")
        scale_words[:, :groups] = np.ascontiguousarray(blocks[..., :2]).view("<u2").squeeze(-1)
        scales = scale_words.view("<f2")
        if not np.isfinite(scales).all():
            raise ValueError(f"{tensor}: Q2 scales must be finite binary16 words")
        codes = np.full((end - begin, padded_groups, 16), 0x55, dtype=np.uint8)
        codes[:, :groups] = blocks[..., 2:]
        return EncodedRows("q2_g64_fp16", torch.from_numpy(codes.reshape(end - begin, -1)),
                           torch.from_numpy(scales), packed=True)

    return LogicalSource(shape, f"{tensor}[q2_g64_fp16]{list(shape)}",
                         raw_source.read_values, encoded)


def q8_native_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Preserve compatible Q8_0 integer codes and binary16 scales as native Q8 planes.

    Native Q8 excludes -128 and negative-zero scales, and requires zero codes for zero-scale
    groups. Refuse an incompatible source instead of silently changing its representation.
    """
    if gguf.info(tensor).type_id != 8:
        raise ValueError(f"{tensor}: native Q8 import requires stored Q8_0 blocks")
    if len(shape) != 2 or shape[1] != gguf.info(tensor).shape[-1] or shape[1] % 32:
        raise ValueError(f"{tensor}: native Q8 import requires complete 32-value blocks")
    raw_source = block_source(gguf, tensor, shape, select)
    groups, padded_groups = shape[1] // 32, ((shape[1] + 127) // 128) * 4

    def encoded(begin: int, end: int) -> EncodedRows:
        index = select(begin, end)
        low, high = int(index.min()), int(index.max()) + 1
        blocks = gguf.read_blocks(tensor, low, high)[index - low].reshape(end - begin, groups, 34)
        raw_words = np.ascontiguousarray(blocks[..., :2]).view("<u2").squeeze(-1)
        raw_scales = raw_words.view("<f2")
        raw_codes = blocks[..., 2:]
        if (not np.isfinite(raw_scales).all() or (raw_words == 0x8000).any() or
                (raw_codes == 0x80).any() or
                ((raw_scales == 0)[..., None] & (raw_codes != 0)).any()):
            raise ValueError(f"{tensor}: Q8_0 source is outside the native Q8 code/scale contract")
        scales = np.zeros((end - begin, padded_groups), dtype="<f2")
        scales[:, :groups] = raw_scales
        codes = np.zeros((end - begin, padded_groups, 32), dtype=np.uint8)
        codes[:, :groups] = raw_codes
        return EncodedRows("q8_g32_fp16", torch.from_numpy(codes.reshape(end - begin, -1)),
                           torch.from_numpy(scales), packed=True)

    return LogicalSource(shape, f"{tensor}[q8_g32_fp16]{list(shape)}",
                         raw_source.read_values, encoded)


def tiled_input_columns(geometry: Geometry = QWEN38_27B_DENSE) -> np.ndarray:
    """For each stored ``ssm_out`` column (tiled value heads), its grouped activation element."""

    head_dim = geometry.gdn_head_dim
    columns = np.arange(geometry.gdn_value_dim, dtype=np.int64)
    tiled_head, lane = columns // head_dim, columns % head_dim
    per_key = geometry.gdn_value_heads_per_key
    grouped_head = (
        (tiled_head % geometry.gdn_key_heads) * per_key
        + tiled_head // geometry.gdn_key_heads
    )
    return (grouped_head * head_dim + lane).astype(np.int32)


def _attention(
    gguf: GGUFFile, g: str, a: str, geometry: Geometry
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    hidden = geometry.hidden
    query_rows = geometry.attention_query_rows
    q_shape, kv_shape = (query_rows, hidden), (geometry.attention_kv_rows, hidden)
    encoded = {
        a + "query": block_source(
            gguf, g + "attn_q.weight", q_shape, attention_rows(False, geometry)
        ),
        a + "gate": block_source(
            gguf, g + "attn_q.weight", q_shape, attention_rows(True, geometry)
        ),
        a + "key": block_source(gguf, g + "attn_k.weight", kv_shape, rows()),
        a + "value": block_source(gguf, g + "attn_v.weight", kv_shape, rows()),
        a + "output": block_source(
            gguf, g + "attn_output.weight", (hidden, query_rows), rows()
        ),
    }
    direct = {
        a + "query_norm": _norm(gguf, g + "attn_q_norm.weight", True),
        a + "key_norm": _norm(gguf, g + "attn_k_norm.weight", True),
    }
    return encoded, direct


def _layer(
    gguf: GGUFFile, g: str, p: str, geometry: Geometry
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    hidden, intermediate = geometry.hidden, geometry.intermediate
    encoded = {
        p + "mlp/gate": block_source(
            gguf, g + "ffn_gate.weight", (intermediate, hidden), rows()
        ),
        p + "mlp/up": block_source(gguf, g + "ffn_up.weight", (intermediate, hidden), rows()),
        p + "mlp/down": block_source(
            gguf, g + "ffn_down.weight", (hidden, intermediate), rows()
        ),
    }
    direct = {
        p + "input_norm": _norm(gguf, g + "attn_norm.weight", True),
        p + "post_attention_norm": _norm(gguf, g + "post_attention_norm.weight", True),
    }
    return encoded, direct


def text_sources(
    gguf: GGUFFile, geometry: Geometry = QWEN38_27B_DENSE
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """Encoded block sources and direct sources of every GGUF-backed text (and MTP) parameter."""

    hidden = geometry.hidden
    key_dim, value_dim = geometry.gdn_key_dim, geometry.gdn_value_dim
    # A tied tower has one table for both roles: the exporter stores no `output.weight`, so the
    # head reads the token table. The recipe shares the two parameters, so the blocks are stored
    # once.
    head = "token_embd.weight" if geometry.tied else "output.weight"
    encoded = {
        "text/token_embedding": block_source(
            gguf, "token_embd.weight", (geometry.vocabulary, hidden), rows()
        ),
        "text/output_head": block_source(
            gguf, head, (geometry.vocabulary, hidden), rows()
        ),
    }
    direct = {"text/final_norm": _norm(gguf, "output_norm.weight", True)}
    for layer in range(geometry.layers):
        g, p = f"blk.{layer}.", f"text/layers/{layer}/"
        e, d = _layer(gguf, g, p, geometry)
        encoded |= e
        direct |= d
        if geometry.full_attention(layer):
            e, d = _attention(gguf, g, p + "attention/", geometry)
            encoded |= e
            direct |= d
            continue
        n = p + "gdn/"
        qkv = g + "attn_qkv.weight"
        encoded[n + "query"] = block_source(gguf, qkv, (key_dim, hidden), rows())
        encoded[n + "key"] = block_source(gguf, qkv, (key_dim, hidden), rows(key_dim))
        encoded[n + "value"] = block_source(
            gguf, qkv, (value_dim, hidden), untiled_rows(2 * key_dim, geometry)
        )
        encoded[n + "z"] = block_source(
            gguf, g + "attn_gate.weight", (value_dim, hidden), untiled_rows(0, geometry)
        )
        # Tiled input columns: the Use gathers the grouped activation (see tiled_input_columns).
        encoded[n + "output"] = block_source(
            gguf, g + "ssm_out.weight", (hidden, value_dim), rows()
        )
        for role, tensor in (("a_projection", "ssm_alpha.weight"), ("b_projection", "ssm_beta.weight")):
            direct[n + role] = gdn_control_source(gguf, g + tensor, geometry)
        ssm_a = untile(gguf.read_direct(g + "ssm_a"), 1, geometry).astype(np.float64)
        if not np.all(ssm_a < 0):
            raise ValueError(f"{g}ssm_a must be strictly negative (-exp(A_log))")
        direct[n + "a_log"] = _direct(
            np.log(-ssm_a).astype(np.float32), torch.float32, g + "ssm_a"
        )
        direct[n + "dt_bias"] = _direct(
            untile(gguf.read_direct(g + "ssm_dt.bias"), 1, geometry),
            torch.float32,
            g + "ssm_dt.bias",
        )
        taps = gguf.read_direct(g + "ssm_conv1d.weight")
        channels = np.concatenate(
            [
                taps[: 2 * key_dim],
                untile(taps[2 * key_dim :], geometry.gdn_head_dim, geometry),
            ]
        )
        direct[n + "convolution"] = _direct(
            channels.T, torch.bfloat16, g + "ssm_conv1d.weight"
        )
        direct[n + "norm"] = _norm(gguf, g + "ssm_norm.weight", False)
    if geometry.mtp:
        e, d = mtp_sources(gguf, geometry)
        encoded.update(e)
        direct.update(d)
    return encoded, direct


def mtp_sources(
    gguf: GGUFFile, geometry: Geometry = QWEN38_27B_DENSE
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """The exporter's MTP block, with its zero-centred (+1 stored) norms.

    This is the same MTP schema used by the dense GGUF recipe. A ternary target
    may use it independently; its main weights need not be GGUF block formats.
    """

    geometry = geometry.with_mtp(True)
    required = {
        name: value
        for name, value in expected_tensors(geometry).items()
        if name.startswith(f"blk.{geometry.mtp_block}.")
    }
    for name, (shape, kind) in required.items():
        info = gguf.info(name)
        if info.shape != shape:
            raise ValueError(f"{name}: MTP shape {info.shape} differs from {shape}")
        if kind == BLOCKS:
            block_format(gguf, name)
        elif not _accepts(info, ("F32", "F16", "BF16")):
            raise ValueError(f"{name}: MTP norm must be a direct format")
    g, p = f"blk.{geometry.mtp_block}.", "mtp/layers/0/"
    encoded, direct = _layer(gguf, g, p, geometry)
    e, d = _attention(gguf, g, p + "attention/", geometry)
    encoded.update(e)
    direct.update(d)
    encoded["mtp/input_projection"] = block_source(
        gguf, g + "nextn.eh_proj.weight", (geometry.hidden, 2 * geometry.hidden), rows()
    )
    for role, tensor in (("embedding_norm", "nextn.enorm.weight"),
                         ("hidden_norm", "nextn.hnorm.weight"),
                         ("final_norm", "nextn.shared_head_norm.weight")):
        direct["mtp/" + role] = _norm(gguf, g + tensor, True)
    return encoded, direct


def _vision_format(name: str) -> str:
    if name == "vision/patch_embedding":
        return Q6
    if name.startswith("vision/merger/"):
        return Q8
    if name.endswith(("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")):
        return Q4
    return Q5


def _vision_source(model, sources):
    from .sources.gguf_source import GGUFSource, HFAliasSource, qwen35_vision_name_map

    mmproj = sources["vision"]
    if not isinstance(mmproj, GGUFSource):
        return mmproj
    config = model.components["vision"]["config"]
    names, transforms, shapes, composites = qwen35_vision_name_map(
        config["depth"],
        hidden_size=config["hidden_size"],
        in_channels=config.get("in_channels", 3),
        temporal_patch_size=config["temporal_patch_size"],
        patch_size=config["patch_size"],
    )
    return HFAliasSource(
        mmproj, names, transforms=transforms, shapes=shapes, composites=composites
    )


def _companions(model, recipe, sources) -> None:
    """Vision from the mmproj GGUF in the official formats; DFlash2 as the official recipe."""

    vision = _vision_source(model, sources) if "vision" in model.components else None
    for name, parameter in model.parameters.items():
        if name.startswith("vision/"):
            source = model.source(name, vision)
            if parameter.projection:
                recipe.assign(
                    name, format=_vision_format(name), method=grouped_absmax, source=source
                )
            else:
                recipe.assign(
                    name,
                    format=parameter.direct_format,
                    method=cast_direct,
                    source=source,
                )
        elif name.startswith(("dflash/", "dflash2/")) and parameter.projection:
            if name.endswith(
                (
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            recipe.assign(name, format=Q8, method=grouped_absmax)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _group_same_format(recipe, names: list[str], formats: dict[str, str]) -> None:
    if len({formats[name] for name in names}) == 1:
        recipe.group(names)


def qwen35_dense_gguf(model, recipe, sources):
    """A dense Qwen3.5/3.8 GGUF in its own block formats from `--source gguf=MODEL.gguf`.

    The geometry comes from the base checkpoint's config, so one recipe serves every dense
    release: the 27B widths, the 9B ones, and whatever else a config describes. The GGUF's own
    header and tensor set must agree with that geometry. A mixture-of-experts release is refused
    here; its expert banks are `qwen3_8_flash_next_gguf`'s subject.
    """

    config = model.config
    if "num_experts" in config:
        raise ValueError("the dense GGUF block recipe requires Qwen3.5/3.8 Dense mathematics")
    gguf = sources["gguf"]
    if not isinstance(gguf, GGUFFile):
        raise ValueError("--source gguf must name the model's .gguf file")
    # The config gives the widths and the head the checkpoint records; the file's own header and
    # tensor set then decide whether the tower it holds is that one, MTP block included.
    geometry = validate(gguf, geometry_from_config(config, mtp=False))
    if "mtp" in model.components and not geometry.mtp:
        raise ValueError(
            f"{gguf.path}: the MTP component needs an -mtp GGUF "
            f"(blk.{geometry.mtp_block}.nextn.*)"
        )
    _companions(model, recipe, sources)
    encoded, direct = text_sources(gguf, geometry)
    formats = {}
    for name, source in encoded.items():
        if source.read_encoded is None:
            raise ValueError(f"{name}: the block import needs an encoded reader")
        format = source.read_encoded(0, 1).format
        formats[name] = format
        recipe.assign(
            name, format=format, method=import_encoded, source=source, activation_policy="AllowA8"
        )
    for name, source in direct.items():
        recipe.assign(
            name,
            format=model.parameters[name].direct_format,
            method=cast_direct,
            source=source,
        )
    if geometry.tied:
        # One table, two parameters. The alias has to be declared after both are assigned, because
        # a physical assignment clears it, and it makes the artifact store the blocks once while
        # the head keeps its own Use.
        recipe.share("text/output_head", "text/token_embedding")
    prefixes = [f"text/layers/{layer}/" for layer in range(geometry.layers)]
    if geometry.mtp:
        prefixes.append("mtp/layers/0/")
    for p in prefixes:
        if p.startswith("text/") and not geometry.full_attention(int(p.split("/")[2])):
            n = p + "gdn/"
            qkv = [n + "query", n + "key", n + "value"]
            recipe.group(qkv + [n + "z"] if formats[n + "z"] == formats[n + "query"] else qkv)
            recipe.separate([n + "a_projection", n + "b_projection"])
        else:
            a = p + "attention/"
            _group_same_format(
                recipe, [a + "query", a + "key", a + "gate", a + "value"], formats
            )
        _group_same_format(recipe, [p + "mlp/gate", p + "mlp/up"], formats)
    columns = AuxiliaryValue(
        "int32",
        (geometry.gdn_value_dim,),
        tiled_input_columns(geometry).astype("<i4").tobytes(),
    )
    for layer in range(geometry.layers):
        if geometry.full_attention(layer):
            continue
        name = f"text/layers/{layer}/gdn/output"
        for input_name in model.parameters[name].inputs:
            recipe.use(name, input_name, auxiliaries={"input_columns": columns})


RECIPES = {
    "qwen35_dense_gguf": qwen35_dense_gguf,
    # The name every published 27B GGUF artifact was converted with: the same recipe.
    "qwen3_8_27b_gguf": qwen35_dense_gguf,
}

# Kept importable under its original name; the recipe is geometry-general now.
qwen3_8_27b_gguf = qwen35_dense_gguf

__all__ = [
    "EXPECTED_HEADER",
    "QWEN38_27B_DENSE",
    "RECIPES",
    "block_format",
    "block_source",
    "expected_tensors",
    "gdn_control_source",
    "has_mtp",
    "qwen35_dense_gguf",
    "qwen3_8_27b_gguf",
    "q2_native_source",
    "q8_native_source",
    "text_sources",
    "tiled_input_columns",
    "validate",
]
