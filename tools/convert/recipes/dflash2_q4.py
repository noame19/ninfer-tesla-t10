"""DFlash2 draft weights compressed, so the 27B artifact still fits a 16 GiB Turing card.

The official ``qwen35_dense_gguf`` recipe stores every draft projection at ``q8_g32_fp16``
(``tools/convert/gguf_blocks.py`` ``_companions``). On a Tesla T10 (16,384 MiB) the text tower plus
the KV reservation already reach 15,148 MiB, so a 2004 MiB draft cannot load.

The draft has four format constraints that no single choice satisfies on its own:

* the fused Q/K/V projection is Q8-only (``attn_input_proj``'s three-output overload calls
  ``require_q8_rowsplit``), 157.3 M parameters;
* ``context_key``/``context_value`` are **aliases** of key/value -- they must keep pointing at the
  same Q8 rows, because ``context_kv_materialize`` requires exactly
  ``Q8_G32_FP16 + RowSplit + (1024, 5120)``;
* the two dynamic-convolution ``kernel_projection`` matrices are BF16-only;
* the two selector codebooks are BF16 by contract.

``Recipe.assign`` drops an alias when it changes anything physical about that name
(``tools/convert/recipe.py`` ``self.aliases.pop(name, None)``), so this override never names the
constrained tensors at all. It works from an **allowlist** of the matrices that are free -- the
attention output projection and the MLP with the feature projection, 1572.9 M parameters -- which
is Q4-safe by ``src/ops/linear/q4/q4_dispatch.cpp`` ``kShapes``: (5120,4096) attention output,
(34816,5120) joined gate/up, (5120,17408) down, (5120,25600) feature.

Pass as ``--override tools/convert/recipes/dflash2_q4.py:dflash2_q4``.
"""
from __future__ import annotations

Q4 = "q4_g64_fp16"

# Every draft projection this override is allowed to compress. Anything not listed keeps whatever
# the base recipe chose, which is Q8 for the Q/K/V parent and BF16 for the conv and selector
# matrices. Listed by suffix so one entry covers all five layers.
_Q4 = (
    "/attention/output",
    "/mlp/gate",
    "/mlp/up",
    "/mlp/down",
)
_Q4_EXACT = ("dflash2/feature_projection",)


def dflash2_q4(model, recipe, sources) -> None:
    """Compress the free draft matrices to Q4 and leave every constrained tensor untouched."""
    if "dflash2" not in model.components:
        raise ValueError("this override needs --components dflash2")

    from tools.convert.methods import grouped_search

    compressed = []
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if not (name in _Q4_EXACT or (name.startswith("dflash2/") and name.endswith(_Q4))):
            continue
        compressed.append(name)
        recipe.assign(name, format=Q4, method=grouped_search)

    if not compressed:
        raise ValueError("dflash2_q4 matched no draft projection")
    print(f"dflash2_q4: {len(compressed)} matrices at {Q4}", flush=True)
