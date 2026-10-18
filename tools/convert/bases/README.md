# Base checkpoint configs

`tools.convert --model DIR` wants a directory carrying the model's `config.json` and its frontend
resources. When the weights come from a GGUF (`--source gguf=…`), the checkpoint's own weight files
are never read — only these configs are — so this directory holds just those, one subdirectory per
tower, instead of the multi-gigabyte downloads.

Copy a whole Hugging Face snapshot here and the weights come along; that also works and is what the
first conversion of each model used. Nothing in the build or the tests reads this directory; it is
an input record for the commands in `docs/weight-conversion.md`.

| Subdirectory | Registry key | Converted to |
|---|---|---|
| `qwen3.8-27b/` | `qwen35_dense_gguf` | Qwen3.8-27B Q8MTP, Swift-1.5 Q3_K_XL, DFlash2-Q4 |
| `qwen3.6-27b/` | `qwen35_dense_gguf` | Qwen3.6-27B UD-IQ3_XXS |
| `ornith-1.0-9b/` | `qwen35_dense_gguf` | Ornith-1.0-9B MTP Q6_K |
| `ornith-1.5-9b-mtp/` | `qwen35_dense_gguf` | Ornith-1.5-9B MTP Q6_K |
| `qwenpaw-9b-q6/` | `qwen35_dense_gguf` | QwenPaw-Flash-9B heretic MTP Q6_K |

The 9B towers share a tokenizer but not a `config.json`, so each keeps its own.
