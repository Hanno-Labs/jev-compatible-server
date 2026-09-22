# Models and backends

The bundled registry is defined in
[`configs/public-models.json`](../configs/public-models.json). Supported entries
are enabled; catalogued models whose native readouts are not implemented remain
explicitly disabled.

## Model aliases

Pass a value from the **Alias for `--model`** column to the server. These are
names in the bundled registry, not Hugging Face repository IDs. For example,
`--model bosun-v3.1-0.6b` selects `Hanno-Labs/bosun-v3.1-0.6b`. The alias is
resolved and loaded at startup; an unknown or disabled alias fails before the
server accepts requests. Without `--model`, requests can select an alias in
their `model` field, and the default is `kev-4b`.

The table lists every enabled bundled alias. A custom `DECISION_REGISTRY` file
defines its own aliases. The backend column reflects the registry entry; model
size and hardware requirements still apply.

| Alias for `--model` | Model repository | Registry backend |
| --- | --- | --- |
| `bge-reranker-v2-m3` | `BAAI/bge-reranker-v2-m3` | Transformers |
| `bosun-v3.1-0.6b` | `Hanno-Labs/bosun-v3.1-0.6b` | Transformers |
| `bosun-v3.1-1.7b` | `Hanno-Labs/bosun-v3.1-1.7b` | Transformers |
| `certo` | `altslate/certo-decision-model` | Transformers |
| `darwin-397b-ztc` | `FINAL-Bench/Darwin-397B-ZTC` | Transformers |
| `decider-2b` | `Mapika/decider-2b` | Transformers |
| `decider-35b-a3b` | `Mapika/decider-35b-a3b` | Transformers |
| `djev` | `google/diffusiongemma-26B-A4B-it` | Transformers |
| `djev-thinking` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | Transformers |
| `gliner2-large-v1` | `fastino/gliner2-large-v1` | Transformers |
| `gliner2.5-base` | `fastino/gliner2.5-base-v1` | Transformers |
| `gliner2.5-multi-v1` | `fastino/gliner2.5-multi-v1` | Transformers |
| `gliner2.5-small-v1` | `fastino/gliner2.5-small-v1` | Transformers |
| `gte-reranker-modernbert-base` | `Alibaba-NLP/gte-reranker-modernbert-base` | Transformers |
| `jeff` | `knowledgator/gliformer-large-v1` | Transformers |
| `jev-local` | `Qwen/Qwen3.5-9B` | Transformers |
| `jqv` | `Qwen/Qwen3-32B` | Transformers |
| `kalm-jev-large` | `KaLM-Embedding/KaLM-Reranker-V1-Large-R2` | Transformers |
| `kalm-jev-nano` | `KaLM-Embedding/KaLM-Reranker-V1-Nano-R2` | Transformers |
| `kalm-jev-small` | `KaLM-Embedding/KaLM-Reranker-V1-Small-R2` | Transformers |
| `kev-0.5b` | `jaredpalmer/kev-0.5b` | Transformers |
| `kev-0.6b` | `jaredpalmer/kev-0.6b` | Transformers |
| `kev-4b` | `jaredpalmer/kev-4b` | Transformers |
| `kev-8b` | `jaredpalmer/kev-8b` | Transformers |
| `laya` | `convaiinnovations/laya` | Transformers |
| `litjev` | `Qwen/Qwen3.8-27B` | Transformers |
| `mxbai-rerank-base-v2` | `mixedbread-ai/mxbai-rerank-base-v2` | Transformers |
| `open-alternative-jev` | `IkerMoel/open-alternative-jev` | Transformers |
| `open-jev-zefan-2b` | `ZefanCai/Open-Jev-2B` | Transformers |
| `open-jev-zefan-9b` | `ZefanCai/Open-Jev-9B` | Transformers |
| `opendecision` | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | Transformers |
| `openjev-thinking` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | Transformers |
| `openjev-verdict` | `heman10x/rlcd-modernbert-151m` | Transformers |
| `qwen3-reranker-4b` | `Qwen/Qwen3-Reranker-4B` | Transformers |
| `reflex-27b` | `Qwen/Qwen3.8-27B` | Transformers |
| `reflex-4b` | `Qwen/Qwen3.5-4B` | Transformers |
| `semif-openjev-qwen3.5-4b` | `TheoLeeCJ/SemIf` | Transformers |
| `simplejev-qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | Transformers |
| `simplejev-qwen3.8-27b` | `Qwen/Qwen3.8-27B` | Transformers |
| `smalljev` | `isHeSatoshi/smalljev-semantic-v9` | Transformers |
| `system-one-qwen3.5-4b-scorer` | `pngwn/system-one-qwen3.5-4b-scorer` | Transformers |
| `winnow-12b` | `EldanRing/Winnow-12B` | llama |
| `zerank-2` | `zeroentropy/zerank-2-reranker` | Transformers |
| `ztc-judge-27b` | `FINAL-Bench/ZTC-Judge-27B` | Transformers |
| `ztc-judge-4b` | `FINAL-Bench/ZTC-Judge-4B` | Transformers |
| `ztc-judge-9b` | `FINAL-Bench/ZTC-Judge-9B` | Transformers |

The bundled registry explicitly limits `darwin-397b-ztc` to `noul` and
`smalljev` to `choice`. See the registry JSON for each entry's readout and
question-type metadata.

## Catalogued models awaiting readouts

| Disabled alias | Model repository | Required integration |
| --- | --- | --- |
| `nanojev` | `C-Tianyu/NanoJev` | candidate-set attention head |
| `open-jev-deberta-v3-large` | `com-kotobalabs/open-jev-deberta-v3-large` | encoder option-scoring head |
| `laya-mlx` | `aac6fef/laya-mlx` | MLX marker-scalar readout |
| `patronus-lynx-8b-instruct` | `PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct` | generative structured PASS/FAIL judge |
| `laya-multilingual` | `convaiinnovations/laya-multilingual` | option-marker set head |
| `vectara-hhem-2.1` | `vectara/hallucination_evaluation_model` | pairwise consistency classifier mapping |
| `qwen3-next-80b-a3b-instruct` | `Qwen/Qwen3-Next-80B-A3B-Instruct` | generative decision recipe |

Pending entries are not silently routed through an incompatible loader and do
not count as supported models.

## Execution backends and readouts

| Execution backend | Readout | Required artifacts |
| --- | --- | --- |
| llama.cpp/GGUF | token logits | GGUF plus prompt and decision-token metadata |
| Transformers | token logits | causal LM plus prompt and decision-token metadata |
| Transformers | native Bosun decision tokens | pinned base, PEFT adapter, tokenizer, decision-token rows, and remote model class |
| Transformers | pointer head | backbone, optional LoRA, pointer tensors, and packing metadata |
| Transformers | encoder-decoder margin | conditional-generation model plus declarative prompt, label, pooling, and aggregation metadata |
| Transformers | scalar sequence classifier | backbone, optional PEFT adapter, per-candidate input templates, and calibration metadata |
| Transformers | hidden-state probe | base checkpoint, input template, probe tensors, token selector, and output transform |

An execution backend runs the underlying neural network. A decision readout
defines how the model represents candidates and turns its outputs into scores.
Keeping these concerns separate lets model families reuse the same runtime.

Bosun publishes a Transformers model class that reconstructs its pinned Qwen
base, PEFT adapter, tokenizer, and learned decision-token rows. The runtime opts
into that reviewed remote code, calls its candidate-aligned `predict` method,
and translates the returned distribution into Jev `choice`, `score`, and
`noul` answers. Registry entries pin the exact model revisions.

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
