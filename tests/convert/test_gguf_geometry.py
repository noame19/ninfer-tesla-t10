from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.convert import gguf_blocks, gguf_geometry
from tools.convert.gguf_geometry import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Geometry,
    geometry_from_config,
)
from tools.convert.sources.gguf import GGUFFile, write_gguf

TYPE_F32 = 0
TYPE_Q8_0 = 8

# A second, deliberately different geometry: every width the 27B recipe used to hard-code is
# smaller or wider here, and the attention/GDN head counts do not match either.
SMALL = Geometry(
    hidden=64,
    intermediate=128,
    vocabulary=128,
    layers=2,
    attention_heads=2,
    attention_head_dim=32,
    kv_heads=1,
    gdn_key_heads=2,
    gdn_value_heads=4,
    gdn_head_dim=128,
    taps=4,
    layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
)


def _q8_0(rows: int, columns: int, seed: int) -> bytes:
    generator = np.random.default_rng(seed)
    blocks = generator.integers(0, 256, size=(rows, columns // 32, 34), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 32)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 32, 2)
    return blocks.tobytes()


def _f32(count: int, seed: int) -> bytes:
    generator = np.random.default_rng(seed)
    return generator.uniform(-1.0, 1.0, size=count).astype(np.float32).tobytes()


def _entries(geometry: Geometry) -> list[tuple[str, tuple[int, ...], int, bytes]]:
    """Every tensor the geometry expects, as a stored Q8_0 matrix or an F32 vector."""

    entries = []
    for index, (name, (shape, kind)) in enumerate(
        sorted(gguf_geometry.expected_tensors(geometry).items())
    ):
        kinds = (kind,) if isinstance(kind, str) else kind
        if gguf_geometry.BLOCKS in kinds and len(shape) == 2:
            entries.append((name, shape, TYPE_Q8_0, _q8_0(shape[0], shape[1], index)))
        elif name.endswith(".ssm_a"):
            # The exporter stores -exp(A_log): strictly negative words.
            generator = np.random.default_rng(index)
            words = -np.exp(generator.uniform(-3.0, -1.0, size=shape)).astype(np.float32)
            entries.append((name, shape, TYPE_F32, words.tobytes()))
        else:
            entries.append((name, shape, TYPE_F32, _f32(int(np.prod(shape)), index)))
    return entries


def _write(path, geometry: Geometry, entries) -> None:
    kv = gguf_geometry.header_expectations(geometry)
    write_gguf(path, kv, entries)


def test_the_default_geometry_reproduces_the_27b_widths():
    # The published Base 27B artifacts were converted with exactly these numbers, and the recipe
    # must keep converting them unchanged.
    geometry = gguf_blocks.QWEN38_27B_DENSE
    assert geometry.hidden == 5120
    assert geometry.intermediate == 17408
    assert geometry.layers == 64
    assert geometry.attention_query_rows == 6144
    assert geometry.gdn_channels == 10240
    assert geometry.gdn_value_dim == 6144
    assert geometry.regular_interval() == 4
    assert gguf_blocks.EXPECTED_HEADER["qwen35.ssm.inner_size"] == 6144
    assert gguf_blocks.EXPECTED_HEADER["qwen35.full_attention_interval"] == 4
    # The published recipe name still resolves to the recipe.
    assert gguf_blocks.RECIPES["qwen3_8_27b_gguf"] is gguf_blocks.RECIPES["qwen35_dense_gguf"]


def test_geometry_follows_the_config_not_a_constant():
    config = {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "layer_types": [
            FULL_ATTENTION if index % 4 == 3 else LINEAR_ATTENTION for index in range(32)
        ],
    }
    geometry = geometry_from_config(config, mtp=False)
    assert geometry.hidden == 4096
    assert geometry.layers == 32
    assert geometry.attention_query_rows == 4096
    assert geometry.gdn_key_dim == 2048
    assert geometry.gdn_value_dim == 4096
    assert geometry.gdn_channels == 8192
    assert geometry.regular_interval() == 4
    assert geometry.blocks == 32
    assert geometry.with_mtp(True).blocks == 33
    assert gguf_geometry.header_expectations(geometry)["qwen35.ssm.inner_size"] == 4096


def test_a_config_that_does_not_describe_a_tower_is_refused():
    base = {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "layer_types": [LINEAR_ATTENTION] * 32,
    }
    with pytest.raises(ValueError, match="layer_types"):
        geometry_from_config({**base, "layer_types": [LINEAR_ATTENTION] * 8}, mtp=False)
    with pytest.raises(ValueError, match="head widths must agree"):
        geometry_from_config({**base, "linear_value_head_dim": 64}, mtp=False)
    with pytest.raises(ValueError, match="must be a positive integer"):
        geometry_from_config({**base, "hidden_size": 0}, mtp=False)
    with pytest.raises(ValueError, match="divisible by the key heads"):
        Geometry(
            hidden=64,
            intermediate=128,
            vocabulary=128,
            layers=1,
            attention_heads=1,
            attention_head_dim=32,
            kv_heads=1,
            gdn_key_heads=3,
            gdn_value_heads=4,
            gdn_head_dim=128,
            taps=4,
            layer_types=(LINEAR_ATTENTION,),
        )


def test_a_smaller_tower_validates_and_imports(tmp_path):
    path = tmp_path / "small.gguf"
    _write(path, SMALL, _entries(SMALL))
    with GGUFFile(path) as gguf:
        assert gguf_blocks.has_mtp(gguf, SMALL) is False
        gguf_blocks.validate(gguf, SMALL)
        encoded, direct = gguf_blocks.text_sources(gguf, SMALL)
    # One GDN output carries the tiled-column auxiliary, so it is present as a source.
    assert "text/layers/0/gdn/output" in encoded
    assert "text/layers/1/attention/query" in encoded
    assert "text/layers/1/gdn/output" not in encoded
    assert direct["text/layers/0/gdn/norm"].shape == (SMALL.gdn_head_dim,)
    # The convolution operand is the exporter's (taps, channels) matrix, transposed.
    assert direct["text/layers/0/gdn/convolution"].shape == (SMALL.taps, SMALL.gdn_channels)


def test_a_gguf_of_another_geometry_is_refused(tmp_path):
    path = tmp_path / "small.gguf"
    _write(path, SMALL, _entries(SMALL))
    with GGUFFile(path) as gguf, pytest.raises(
        ValueError, match="qwen35.block_count = 2, expected 64"
    ):
        gguf_blocks.validate(gguf, gguf_blocks.QWEN38_27B_DENSE)


def test_a_header_that_disagrees_with_the_config_is_refused(tmp_path):
    path = tmp_path / "small.gguf"
    kv = {**gguf_geometry.header_expectations(SMALL), "qwen35.embedding_length": 96}
    write_gguf(path, kv, _entries(SMALL))
    with GGUFFile(path) as gguf, pytest.raises(
        ValueError, match="qwen35.embedding_length = 96, expected 64"
    ):
        gguf_blocks.validate(gguf, SMALL)


def test_a_missing_tensor_is_refused(tmp_path):
    path = tmp_path / "small.gguf"
    entries = [e for e in _entries(SMALL) if e[0] != "blk.0.ssm_out.weight"]
    _write(path, SMALL, entries)
    with GGUFFile(path) as gguf, pytest.raises(
        ValueError, match="missing \\['blk.0.ssm_out.weight'\\]"
    ):
        gguf_blocks.validate(gguf, SMALL)


def test_the_tiled_column_map_follows_the_geometry():
    columns = gguf_blocks.tiled_input_columns(SMALL)
    assert sorted(columns.tolist()) == list(range(SMALL.gdn_value_dim))
    per_key = SMALL.gdn_value_heads_per_key
    for tiled_head in range(SMALL.gdn_value_heads):
        key_head, repeat = tiled_head % SMALL.gdn_key_heads, tiled_head // SMALL.gdn_key_heads
        assert (
            columns[tiled_head * SMALL.gdn_head_dim + 5]
            == (key_head * per_key + repeat) * SMALL.gdn_head_dim + 5
        )


@pytest.mark.parametrize("type_id", (TYPE_F32, TYPE_Q8_0))
def test_gdn_controls_accept_every_storage_an_exporter_chooses(tmp_path, type_id):
    # Coder390 stores the gates as BF16, Swift and 0.8B as Q8_0, 4B as F32 and the 9B releases
    # as Q4_K: all four must reach the Engine's BF16 operand in the grouped head order.
    name = "blk.0.ssm_alpha.weight"
    shape = (SMALL.gdn_value_heads, SMALL.hidden)
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) / 100.0
    raw = _q8_0(shape[0], shape[1], 3) if type_id == TYPE_Q8_0 else values.tobytes()
    path = tmp_path / "gate.gguf"
    write_gguf(path, {"general.architecture": "qwen35"}, [(name, shape, type_id, raw)])
    with GGUFFile(path) as gguf:
        source = gguf_blocks.gdn_control_source(gguf, name, SMALL)
        got = source.rows(0, SMALL.gdn_value_heads)
    # Grouped head `k * per_key + r` is the exporter's tiled head `r * key_heads + k`.
    tiled = [repeat * SMALL.gdn_key_heads + key for key in range(SMALL.gdn_key_heads)
             for repeat in range(SMALL.gdn_value_heads_per_key)]
    assert got.dtype == torch.bfloat16
    if type_id == TYPE_F32:
        want = torch.from_numpy(values[tiled]).to(torch.bfloat16)
    else:
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(shape[0], -1, 34)
        scales = blocks[..., :2].copy().view(np.float16).astype(np.float32)
        codes = blocks[..., 2:].copy().view(np.int8).astype(np.float32)
        want = torch.from_numpy((codes * scales).reshape(shape)[tiled]).to(torch.bfloat16)
    assert torch.equal(got, want)
