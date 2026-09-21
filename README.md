# jev-compatible-server

Open inference runtime for decision models.

Run supported decision models on your own infrastructure and expose them through
one typed API. The runtime loads model-specific decision recipes, executes them
through llama.cpp/GGUF or Hugging Face Transformers, and returns normalized
`choice`, `score`, and `noul` answers.

The included HTTP server implements Jev's `POST /v1/systemone` contract so an
existing Jev client can cut over without changing its request and response
shapes. Jev compatibility is an interface to the runtime, not the model
abstraction itself.

## Quick start

Prerequisites: [install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
and Git. Then clone this repository:

The bundled registry includes the public Kev checkpoints. This example starts
the smallest one through the Transformers pointer-head readout:

```bash
git clone https://github.com/Hanno-Labs/jev-compatible-server.git
cd jev-compatible-server

DECISION_REGISTRY="$PWD/configs/public-models.json" \
uv run --extra transformers jev-compatible-server
```

The first request downloads `jaredpalmer/kev-0.5b` and its Qwen backbone from
Hugging Face:

```bash
curl http://localhost:8000/v1/systemone \
  --header 'content-type: application/json' \
  --data '{
    "model": "kev-0.5b",
    "state": "A customer says they were charged twice and need help today.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should own this ticket?",
        "criteria": {
          "billing": "Payment, invoice, or refund problems",
          "technical": "Product bugs and technical failures"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does this ticket need immediate attention?"
      }
    }
  }'
```

The response preserves the question keys and returns each model-derived
distribution. Values depend on the model; the response shape is:

```json
{
  "model": "jaredpalmer/kev-0.5b",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.65,
        "technical": 0.35
      },
      "confidence": 0.65
    },
    "urgent": {
      "type": "noul",
      "noul": 0.72
    }
  },
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

The numeric values above illustrate the schema; they are not claimed predictions
for the example request.

## What the runtime provides

- A Jev-compatible HTTP interface at `/v1/systemone` and `/systemone`.
- Keyed, multi-question requests over one shared state.
- `choice`, ordinal `score`, and binary-probability `noul` questions.
- A common answer schema across different model architectures.
- GGUF token-logit execution through `llama-cpp-python`.
- Hugging Face token-logit and pointer-head execution through Transformers.
- Declarative model recipes from model metadata, standalone config, or a
  service-owned registry.
- Lazy model loading, model selection per request, and bounded dynamic
  microbatching for concurrent requests.

The runtime is organized around two independent extension points:

```text
Jev-compatible request
        |
        v
model registry and microbatch scheduler
        |
        +-- execution backend: llama.cpp/GGUF | Transformers
        |
        +-- decision readout: token logits | pointer head | future readouts
        |
        v
normalized choice / score / noul answers
```

An execution backend runs the underlying neural network. A decision readout
defines how that model represents candidates and turns its outputs into scores.
Keeping these concerns separate lets multiple model families reuse the same
runtime implementation.

## Supported models

These are the enabled entries in
[`configs/public-models.json`](configs/public-models.json):

| Model | Execution backend | Readout | Question types |
| --- | --- | --- | --- |
| `jaredpalmer/kev-0.5b` | Transformers | pointer head | `choice`, `score`, `noul` |
| `jaredpalmer/kev-0.6b` | Transformers | pointer head | `choice`, `score`, `noul` |
| `jaredpalmer/kev-4b` | Transformers | pointer head | `choice`, `score`, `noul` |
| `jaredpalmer/kev-8b` | Transformers | pointer head | `choice`, `score`, `noul` |

The following public models are catalogued but disabled because their native
readouts are not implemented yet. They are not silently routed through an
incompatible loader and do not count as supported models.

| Model | Required integration | Status |
| --- | --- | --- |
| `C-Tianyu/NanoJev` | candidate-set attention head | pending |
| `com-kotobalabs/open-jev-deberta-v3-large` | encoder option-scoring head | pending |
| `pngwn/system-one-qwen3.5-4b-scorer` | scalar scoring head | pending |
| `aac6fef/laya-mlx` | MLX marker-scalar readout | pending |

## Backends and readouts

| Execution backend | Readout | Model artifacts |
| --- | --- | --- |
| llama.cpp/GGUF | token logits | GGUF plus prompt and decision-token metadata |
| Transformers | token logits | causal LM plus prompt and decision-token metadata |
| Transformers | pointer head | backbone, optional LoRA, pointer tensors, and packing metadata |

Token-logit models expose one vocabulary token for each candidate. The runtime
formats the model's prompt, reads the next-token logits for only those candidate
tokens, and normalizes them into the requested answer type.

Pointer-head models encode the state, question, and candidates together. The
built-in pointer readout applies the published backbone and LoRA weights, then
scores candidate representations with the model's query/key head. It is an
owned implementation and does not import a model author's serving repository.

## Jev-compatible API

A request contains one arbitrary `state` and one or more named questions:

- `choice` selects among 2–255 keyed criteria and returns the full probability
  distribution.
- `score` scores an ordered list of 2–255 criteria and returns its expected
  ordinal value, distribution, confidence, and legend.
- `noul` returns a probability between zero and one. Custom `true` and `false`
  criteria are optional.

Every answer retains the corresponding question key. A request can mix all
three question types, subject to the capabilities of the selected model.

The optional top-level `model` field selects a registry entry. When it is
omitted, the registry's `default` entry is used.

## Adding a model

If a new model uses an existing execution backend and readout, adding it is a
data change: publish the decision metadata with the model or add a registry
entry. No model-name branch is required in the service.

For a token-logit model, the recipe contains a prompt template and the token ID
for each possible answer label:

```json
{
  "model": "my-decision-model",
  "decision.prompt_template": "State:\n{state}\n\nQuestion:\n{instructions}\nCriteria:\n{criteria}\nAnswer:",
  "decision.tokens": {
    "true": 1001,
    "false": 1002,
    "0": 1010,
    "1": 1011,
    "billing": 1020,
    "technical": 1021
  }
}
```

If the model introduces new readout math—rather than a new configuration of an
existing readout—the runtime needs one reusable readout implementation. Future
models with the same architecture can then select it declaratively.

## Model registry

The service-owned registry is useful when published model metadata is missing,
stale, or cannot carry all required artifacts. Registry configuration overrides
model metadata and file defaults.

```json
{
  "default": "decision-qwen",
  "models": {
    "decision-qwen": {
      "backend": "transformers",
      "model": "org/decision-qwen",
      "config": {
        "decision.prompt_template": "State: {state}\nQuestion: {instructions}\nCriteria: {criteria}\nAnswer:",
        "decision.tokens": {
          "true": 1001,
          "false": 1002
        }
      }
    },
    "decision-gguf": {
      "backend": "llama",
      "model": "/models/decision.gguf",
      "config_path": "/etc/decision/decision-gguf.json"
    }
  }
}
```

Start the server with `DECISION_REGISTRY=/path/to/registry.json`. Entries are
loaded on first use and cached for later requests.

Without a registry, select one model directly:

```bash
DECISION_BACKEND=transformers \
DECISION_MODEL_ID=your-org/your-model \
DECISION_CONFIG=/path/to/decision.json \
uv run --extra transformers jev-compatible-server
```

For GGUF, set `DECISION_BACKEND=llama`, `DECISION_MODEL_PATH`, and install the
`llama` extra instead.

## Batching and performance

Concurrent HTTP calls are collected for a short, bounded window. A registry
batch is then grouped by model so each loaded runtime receives compatible
requests together. The Transformers implementations combine the questions in a
group into shared forward-pass batches.

- `DECISION_MAX_BATCH_SIZE` defaults to `16`.
- `DECISION_BATCH_WAIT_MS` defaults to `5`.

This is dynamic request microbatching, not continuous token-level batching.
Throughput, latency, and memory depend on the model, backend, hardware, request
shape, and these settings; the project does not yet publish a general
performance claim.

## Project status

The runtime is currently `0.1.0`. The protocol, registry, token-logit readout,
pointer-head readout, and dynamic microbatcher are implemented. Broader readout
coverage, per-model conformance results, reproducible performance benchmarks,
and a stable publisher-facing recipe specification remain active work.

The immediate compatibility target is Jev's typed decision contract. The larger
goal is an open runtime where model authors can publish a decision model once
and users can run it locally or in their own cloud without adopting the
author's serving stack.
