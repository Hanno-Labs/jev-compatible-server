# Models and backends

The bundled registry is defined in
[`configs/public-models.json`](../configs/public-models.json). Supported entries
are enabled; catalogued models whose native readouts are not implemented remain
explicitly disabled.

## Supported models

| Model | Backend | Readout | Question types |
| --- | --- | --- | --- |
| `jaredpalmer/kev-0.5b` | Transformers | pointer head | `choice`, `score`, `noul` |
| `jaredpalmer/kev-0.6b` | Transformers | pointer head | `choice`, `score`, `noul` |
| `jaredpalmer/kev-4b` | Transformers | pointer head | `choice`, `score`, `noul` |
| `jaredpalmer/kev-8b` | Transformers | pointer head | `choice`, `score`, `noul` |

## Catalogued models awaiting readouts

| Model | Required integration | Status |
| --- | --- | --- |
| `C-Tianyu/NanoJev` | candidate-set attention head | pending |
| `com-kotobalabs/open-jev-deberta-v3-large` | encoder option-scoring head | pending |
| `pngwn/system-one-qwen3.5-4b-scorer` | scalar scoring head | pending |
| `aac6fef/laya-mlx` | MLX marker-scalar readout | pending |

Pending entries are not silently routed through an incompatible loader and do
not count as supported models.

## Execution backends and readouts

| Execution backend | Readout | Required artifacts |
| --- | --- | --- |
| llama.cpp/GGUF | token logits | GGUF plus prompt and decision-token metadata |
| Transformers | token logits | causal LM plus prompt and decision-token metadata |
| Transformers | pointer head | backbone, optional LoRA, pointer tensors, and packing metadata |

An execution backend runs the underlying neural network. A decision readout
defines how the model represents candidates and turns its outputs into scores.
Keeping these concerns separate lets model families reuse the same runtime.

Token-logit models expose one vocabulary token for each candidate. The runtime
formats the prompt, reads the next-token logits for valid candidate tokens, and
normalizes them into the requested answer type.

Pointer-head models encode the state, question, and candidates together. The
built-in pointer readout applies the declared backbone and optional LoRA weights,
then scores candidate representations with the model's query/key head. The
runtime owns this implementation and does not import an author's serving code.
