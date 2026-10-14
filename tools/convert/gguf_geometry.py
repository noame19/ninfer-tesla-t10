"""The geometry of a dense Qwen3.5/3.8 GGUF, and the tensor set a block import expects.

Pure data and header arithmetic: no torch and no gguf-py, so a saved GGUF header can be checked
without the weights beside it. The dense GGUF block recipe reads one `Geometry` from the base
checkpoint's config and validates the GGUF against it, which is what lets one recipe serve every
dense Qwen3.5/3.8 release instead of a single hard-coded width.

The GGUF header is the exporter's own description of the same tower, so the two must agree: a
mismatch means the GGUF is not the model the config describes, and the import must be refused
rather than guessed at.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TypeAlias

ARCHITECTURE = "qwen35"

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

# Stored tensor kinds, as `expected_tensors` reports them.
BLOCKS = "blocks"  # any ggml block type the gguf_blocks_v1 layout registers
DIRECT = ("F32", "F16", "BF16")
# The GDN gates are BF16 operands of the Engine. An export may store them as BF16 words or as
# quantized blocks; every stored form decodes to the same BF16 values.
CONTROL = (BLOCKS, "F32", "F16", "BF16")

DEFAULT_FULL_ATTENTION_INTERVAL = 4

# What `expected_tensors` reports for one tensor: one kind, or any of several.
Kind: TypeAlias = str | tuple[str, ...]
TensorSpec: TypeAlias = tuple[tuple[int, ...], Kind]


def _count(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _flag(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Geometry:
    """Row-major geometry of one dense Qwen3.5/3.8 text tower.

    `layer_types` names every block, in order, as `full_attention` or `linear_attention`; the
    artifact numbers its runtime blocks the same way. `mtp` adds the exporter's next-token
    prediction block after the last real layer (`blk.<layers>.*`). `tied` says the output head is
    the token table, which is what decides whether the GGUF stores an `output.weight` at all.
    """

    hidden: int
    intermediate: int
    vocabulary: int
    layers: int
    attention_heads: int
    attention_head_dim: int
    kv_heads: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_head_dim: int
    taps: int
    layer_types: tuple[str, ...]
    mtp: bool = False
    tied: bool = False

    def __post_init__(self) -> None:
        for label, value in (
            ("hidden", self.hidden),
            ("intermediate", self.intermediate),
            ("vocabulary", self.vocabulary),
            ("layers", self.layers),
            ("attention_heads", self.attention_heads),
            ("attention_head_dim", self.attention_head_dim),
            ("kv_heads", self.kv_heads),
            ("gdn_key_heads", self.gdn_key_heads),
            ("gdn_value_heads", self.gdn_value_heads),
            ("gdn_head_dim", self.gdn_head_dim),
            ("taps", self.taps),
        ):
            _count(value, label)
        if len(self.layer_types) != self.layers:
            raise ValueError(
                f"layer_types describes {len(self.layer_types)} blocks, expected {self.layers}"
            )
        unknown = sorted(set(self.layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION})
        if unknown:
            raise ValueError(f"unsupported layer types {unknown}")
        if self.gdn_value_heads % self.gdn_key_heads:
            raise ValueError("the GDN value heads must be divisible by the key heads")

    # --- derived widths -------------------------------------------------------------------
    @property
    def attention_query_rows(self) -> int:
        """Rows of `attn_q`: one query head and its interleaved output gate, per head."""
        return self.attention_heads * self.attention_head_dim

    @property
    def attention_kv_rows(self) -> int:
        return self.kv_heads * self.attention_head_dim

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_key_heads * self.gdn_head_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_value_heads * self.gdn_head_dim

    @property
    def gdn_channels(self) -> int:
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    @property
    def mtp_block(self) -> int:
        return self.layers

    @property
    def blocks(self) -> int:
        """`qwen35.block_count`: the real layers, plus the MTP block when there is one."""
        return self.layers + (1 if self.mtp else 0)

    @property
    def gdn_value_heads_per_key(self) -> int:
        return self.gdn_value_heads // self.gdn_key_heads

    # --- structure ------------------------------------------------------------------------
    def full_attention(self, layer: int) -> bool:
        return self.layer_types[layer] == FULL_ATTENTION

    def with_mtp(self, mtp: bool) -> Geometry:
        return replace(self, mtp=mtp)

    def with_tied(self, tied: bool) -> Geometry:
        return replace(self, tied=tied)

    def regular_interval(self) -> int | None:
        """The full-attention period when the layer types repeat exactly, else None.

        The exporter records it as `qwen35.full_attention_interval`; a pattern that does not
        repeat has no such key and is not checked against one.
        """
        for interval in range(1, self.layers + 1):
            if all(
                self.full_attention(index) == (index % interval == interval - 1)
                for index in range(self.layers)
            ):
                return interval
        return None


def geometry_from_config(config: Mapping, *, mtp: bool, tied: bool | None = None) -> Geometry:
    """The geometry of the tower a base checkpoint's resolved text config describes.

    `config` is the text config the model builder produced, so the names are that config's own:
    `head_dim` rather than a per-axis `key_length`, and the `linear_*` GDN fields. `tied`
    overrides the head the config records, for a caller that knows the checkpoint's own answer.
    """
    layers = _count(config.get("num_hidden_layers"), "text.num_hidden_layers")
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, Sequence) or isinstance(layer_types, str):
        raise ValueError("text.layer_types must describe every block")
    key_head_dim = _count(config.get("linear_key_head_dim"), "text.linear_key_head_dim")
    value_head_dim = _count(config.get("linear_value_head_dim"), "text.linear_value_head_dim")
    if key_head_dim != value_head_dim:
        raise ValueError(
            f"text: the GDN key ({key_head_dim}) and value ({value_head_dim}) head widths must agree"
        )
    return Geometry(
        hidden=_count(config.get("hidden_size"), "text.hidden_size"),
        intermediate=_count(config.get("intermediate_size"), "text.intermediate_size"),
        vocabulary=_count(config.get("vocab_size"), "text.vocab_size"),
        layers=layers,
        attention_heads=_count(config.get("num_attention_heads"), "text.num_attention_heads"),
        attention_head_dim=_count(config.get("head_dim"), "text.head_dim"),
        kv_heads=_count(config.get("num_key_value_heads"), "text.num_key_value_heads"),
        gdn_key_heads=_count(config.get("linear_num_key_heads"), "text.linear_num_key_heads"),
        gdn_value_heads=_count(
            config.get("linear_num_value_heads"), "text.linear_num_value_heads"
        ),
        gdn_head_dim=key_head_dim,
        taps=_count(config.get("linear_conv_kernel_dim"), "text.linear_conv_kernel_dim"),
        layer_types=tuple(str(kind) for kind in layer_types),
        mtp=mtp,
        tied=(
            _flag(config.get("tie_word_embeddings", False), "text.tie_word_embeddings")
            if tied is None
            else tied
        ),
    )


# --- the GGUF header ------------------------------------------------------------------------
def header_expectations(geometry: Geometry) -> dict[str, object]:
    """The header keys the llama.cpp Qwen3.5 exporter writes for this geometry."""

    expectations: dict[str, object] = {
        "general.architecture": ARCHITECTURE,
        "qwen35.block_count": geometry.blocks,
        "qwen35.embedding_length": geometry.hidden,
        "qwen35.feed_forward_length": geometry.intermediate,
        "qwen35.attention.head_count": geometry.attention_heads,
        "qwen35.attention.head_count_kv": geometry.kv_heads,
        "qwen35.attention.key_length": geometry.attention_head_dim,
        "qwen35.attention.value_length": geometry.attention_head_dim,
        "qwen35.ssm.conv_kernel": geometry.taps,
        "qwen35.ssm.state_size": geometry.gdn_head_dim,
        "qwen35.ssm.group_count": geometry.gdn_key_heads,
        "qwen35.ssm.time_step_rank": geometry.gdn_value_heads,
        "qwen35.ssm.inner_size": geometry.gdn_value_dim,
    }
    interval = geometry.regular_interval()
    if interval is not None:
        expectations["qwen35.full_attention_interval"] = interval
    return expectations


def check_header(header: Mapping, geometry: Geometry, *, source: str = "the GGUF") -> None:
    """Refuse a header that does not describe `geometry`."""

    for key, expected in header_expectations(geometry).items():
        found = header.get(key)
        if found != expected:
            raise ValueError(f"{source}: {key} = {found!r}, expected {expected!r}")


# --- the tensor set -------------------------------------------------------------------------
def _attention_tensors(prefix: str, geometry: Geometry) -> dict[str, TensorSpec]:
    hidden = geometry.hidden
    head_dim = geometry.attention_head_dim
    return {
        prefix + "attn_q.weight": (
            (2 * geometry.attention_query_rows, hidden),
            BLOCKS,
        ),
        prefix + "attn_k.weight": ((geometry.attention_kv_rows, hidden), BLOCKS),
        prefix + "attn_v.weight": ((geometry.attention_kv_rows, hidden), BLOCKS),
        prefix + "attn_output.weight": ((hidden, geometry.attention_query_rows), BLOCKS),
        prefix + "attn_q_norm.weight": ((head_dim,), "F32"),
        prefix + "attn_k_norm.weight": ((head_dim,), "F32"),
    }


def _layer_tensors(prefix: str, geometry: Geometry) -> dict[str, TensorSpec]:
    hidden, intermediate = geometry.hidden, geometry.intermediate
    return {
        prefix + "attn_norm.weight": ((hidden,), "F32"),
        prefix + "post_attention_norm.weight": ((hidden,), "F32"),
        prefix + "ffn_gate.weight": ((intermediate, hidden), BLOCKS),
        prefix + "ffn_up.weight": ((intermediate, hidden), BLOCKS),
        prefix + "ffn_down.weight": ((hidden, intermediate), BLOCKS),
    }


def _linear_attention_tensors(prefix: str, geometry: Geometry) -> dict[str, TensorSpec]:
    hidden = geometry.hidden
    return {
        prefix + "attn_qkv.weight": ((geometry.gdn_channels, hidden), BLOCKS),
        prefix + "attn_gate.weight": ((geometry.gdn_value_dim, hidden), BLOCKS),
        prefix + "ssm_out.weight": ((hidden, geometry.gdn_value_dim), BLOCKS),
        prefix + "ssm_alpha.weight": ((geometry.gdn_value_heads, hidden), CONTROL),
        prefix + "ssm_beta.weight": ((geometry.gdn_value_heads, hidden), CONTROL),
        prefix + "ssm_a": ((geometry.gdn_value_heads,), "F32"),
        prefix + "ssm_dt.bias": ((geometry.gdn_value_heads,), "F32"),
        prefix + "ssm_conv1d.weight": ((geometry.gdn_channels, geometry.taps), "F32"),
        prefix + "ssm_norm.weight": ((geometry.gdn_head_dim,), "F32"),
    }


def mtp_tensors(geometry: Geometry) -> dict[str, TensorSpec]:
    """The exporter's next-token-prediction block, stored after the last real layer."""

    hidden = geometry.hidden
    prefix = f"blk.{geometry.mtp_block}."
    return (
        _layer_tensors(prefix, geometry)
        | _attention_tensors(prefix, geometry)
        | {
            prefix + "nextn.eh_proj.weight": ((hidden, 2 * hidden), BLOCKS),
            prefix + "nextn.enorm.weight": ((hidden,), "F32"),
            prefix + "nextn.hnorm.weight": ((hidden,), "F32"),
            prefix + "nextn.shared_head_norm.weight": ((hidden,), "F32"),
        }
    )


def expected_tensors(geometry: Geometry) -> dict[str, TensorSpec]:
    """Row-major shape and kind of every tensor: `BLOCKS`, a ggml type name, or several kinds."""

    hidden, vocabulary = geometry.hidden, geometry.vocabulary
    out: dict[str, TensorSpec] = {
        "token_embd.weight": ((vocabulary, hidden), BLOCKS),
        "output_norm.weight": ((hidden,), "F32"),
    }
    # A tied tower stores no head of its own: llama.cpp's converters drop `output.weight` and the
    # runtime reads the token table. An untied one must store it.
    if not geometry.tied:
        out["output.weight"] = ((vocabulary, hidden), BLOCKS)
    for layer in range(geometry.layers):
        prefix = f"blk.{layer}."
        out |= _layer_tensors(prefix, geometry)
        if geometry.full_attention(layer):
            out |= _attention_tensors(prefix, geometry)
        else:
            out |= _linear_attention_tensors(prefix, geometry)
    if geometry.mtp:
        out |= mtp_tensors(geometry)
    return out


__all__ = [
    "ARCHITECTURE",
    "BLOCKS",
    "CONTROL",
    "DEFAULT_FULL_ATTENTION_INTERVAL",
    "DIRECT",
    "FULL_ATTENTION",
    "Geometry",
    "Kind",
    "LINEAR_ATTENTION",
    "TensorSpec",
    "check_header",
    "expected_tensors",
    "geometry_from_config",
    "header_expectations",
    "mtp_tensors",
]
