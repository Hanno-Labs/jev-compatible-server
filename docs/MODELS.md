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
| `KaLM-Embedding/KaLM-Reranker-V1-Nano-R2` | Transformers | encoder-decoder margin | `choice`, `score`, `noul` |
| `KaLM-Embedding/KaLM-Reranker-V1-Small-R2` | Transformers | encoder-decoder margin | `choice`, `score`, `noul` |
| `KaLM-Embedding/KaLM-Reranker-V1-Large-R2` | Transformers | encoder-decoder margin | `choice`, `score`, `noul` |
| `pngwn/system-one-qwen3.5-4b-scorer` | Transformers | scalar sequence classifier | `choice`, `score`, `noul` |
| `FINAL-Bench/ZTC-Judge-4B` | Transformers | candidate verifier plus last-hidden-state RBF probe | `choice`, `score`, `noul` |
| `FINAL-Bench/ZTC-Judge-9B` | Transformers | candidate verifier plus last-hidden-state RBF probe | `choice`, `score`, `noul` |
| `FINAL-Bench/ZTC-Judge-27B` | Transformers | candidate verifier plus last-hidden-state RBF probe | `choice`, `score`, `noul` |
| `FINAL-Bench/Darwin-397B-ZTC` | Transformers | calibrated last-hidden-state linear probe | `noul` |

## Catalogued models awaiting readouts

| Model | Required integration | Status |
| --- | --- | --- |
| `C-Tianyu/NanoJev` | candidate-set attention head | pending |
| `com-kotobalabs/open-jev-deberta-v3-large` | encoder option-scoring head | pending |
| `aac6fef/laya-mlx` | MLX marker-scalar readout | pending |
| `PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct` | generative structured PASS/FAIL judge | pending |
| `convaiinnovations/laya-multilingual` | option-marker set head | pending |
| `vectara/hallucination_evaluation_model` | pairwise consistency classifier mapping | pending |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | generative decision recipe | pending |

Pending entries are not silently routed through an incompatible loader and do
not count as supported models.

## Execution backends and readouts

| Execution backend | Readout | Required artifacts |
| --- | --- | --- |
| llama.cpp/GGUF | token logits | GGUF plus prompt and decision-token metadata |
| Transformers | token logits | causal LM plus prompt and decision-token metadata |
| Transformers | pointer head | backbone, optional LoRA, pointer tensors, and packing metadata |
| Transformers | encoder-decoder margin | conditional-generation model plus declarative prompt, label, pooling, and aggregation metadata |
| Transformers | scalar sequence classifier | backbone, optional PEFT adapter, per-candidate input templates, and calibration metadata |
| Transformers | hidden-state probe | base checkpoint, input template, probe tensors, token selector, and output transform |

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

Encoder-decoder margin models encode candidate documents, run a configured
decoder prompt, and score the positive-label logit minus the negative-label
logit. The same readout supports every model whose registry recipe declares the
required templates, labels, pooling, limits, and aggregation behavior.

Scalar sequence-classifier models render one sequence per candidate, preserve
the question and candidate tail when truncating, score each sequence with a
single-logit classification head, and normalize the configured candidate set.
The loader supports a full checkpoint or a PEFT adapter over a declared
backbone.

Hidden-state probe models can render either one input per shared state or one
input per candidate. The runtime extracts the configured hidden-state position
and applies a linear or RBF probe. Input modes, templates, probe filenames,
model revisions, aggregation, and output transforms all come from the registry;
the implementation contains no ZTC model-name branches.

The ZTC Judge checkpoints are external answer verifiers. For every API
candidate, the runtime serializes the shared state and question as the problem,
uses that candidate as the proposed answer, and applies the published RBF
probe. It normalizes those unbounded verifier scores across candidates for
`choice` and `score`, or compares the true and false candidates for `noul`.
Those normalized values are decision distributions, not the model-card API's
unpublished absolute calibration.

Darwin has a different contract: its probe predicts correctness of Darwin's
own answer before generation. Its public probe includes `cal_A`, `cal_B`,
`s_mean`, and `s_std`, so the runtime exposes that calibrated value as `noul`.
It does not pretend that Darwin can rank externally supplied candidates.

Darwin-397B-ZTC is enabled because the readout is implemented, but loading its
397B checkpoint requires a multi-GPU or offloaded Transformers deployment. A
single 80 GB GPU is not sufficient.
