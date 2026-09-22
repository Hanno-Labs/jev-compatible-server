<h1 align="center">jev-compatible-server</h1>

<p align="center">
  Run open decision models behind a Jev-compatible API.
</p>

<p align="center">
  <a href="docs/README.md">Documentation</a> ·
  <a href="docs/API.md">API</a> ·
  <a href="docs/MODELS.md">Models</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then start
the server with the Transformers backend:

```bash
uvx --from 'jev-compatible-server[transformers] @ git+https://github.com/Hanno-Labs/jev-compatible-server.git' jev-compatible-server
```

Send a decision request to `POST /v1/systemone`:

```bash
curl http://localhost:8000/v1/systemone \
  --header 'content-type: application/json' \
  --data '{
    "model": "kev-0.5b",
    "state": "A customer says they were charged twice.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should own this ticket?",
        "criteria": {
          "billing": "Payment, invoice, or refund problems",
          "technical": "Product bugs and technical failures"
        }
      }
    }
  }'
```

The first request downloads the selected model and its backbone from Hugging
Face. See the [API reference](docs/API.md) for all question and response types.

## Installation

`uvx` installs the server into an isolated environment and runs it directly.
Select the extra for the inference backend you need:

```bash
# Hugging Face Transformers models
uvx --from 'jev-compatible-server[transformers] @ git+https://github.com/Hanno-Labs/jev-compatible-server.git' jev-compatible-server

# llama.cpp/GGUF models
DECISION_BACKEND=llama \
DECISION_MODEL_PATH=/path/to/model.gguf \
uvx --from 'jev-compatible-server[llama] @ git+https://github.com/Hanno-Labs/jev-compatible-server.git' jev-compatible-server
```

The bundled public-model registry is used by default. Custom GGUF models and
registries require the environment described in the
[model recipe guide](docs/MODEL_RECIPES.md).

## Description

`jev-compatible-server` is an open inference runtime for decision models. It
accepts one shared state with one or more typed questions, runs the selected
model through llama.cpp or Hugging Face Transformers, and returns normalized
`choice`, `score`, and `noul` answers. Models that implement only part of that
contract return an explicit `unsupported` result for each incompatible
question without discarding compatible answers in the same request.

The HTTP interface implements Jev's `POST /v1/systemone` request and response
shape so applications can move between hosted Jev and self-hosted models
without replacing their decision API.

## Goals

- Provide a common runtime for open decision models, as llama.cpp does for
  language models.
- Preserve the Jev API contract for straightforward application cutover.
- Keep model behavior declarative when an existing backend and readout can run
  it.
- Support multiple execution engines without coupling applications to model
  architecture.
- Make batching, model coverage, and compatibility behavior explicit and
  testable.

## Supported backends

| Backend | Model format | Built-in readouts |
| --- | --- | --- |
| llama.cpp | GGUF | token logits |
| Hugging Face Transformers | Transformers checkpoints | token logits, pointer head, encoder-decoder margin, scalar sequence classifier, hidden-state probe |

Backends execute the neural network; readouts convert model outputs into typed
decision probabilities. See [models and backends](docs/MODELS.md) for supported
checkpoints and the exact distinction.

## Documentation

- [API reference](docs/API.md)
- [Supported models and backends](docs/MODELS.md)
- [Model recipes and registries](docs/MODEL_RECIPES.md)
- [Batching and performance](docs/PERFORMANCE.md)

## Contributing

Contributions for new backends, reusable readouts, model recipes, tests, and
documentation are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) before making
a change.

## Acknowledgements

`jev-compatible-server` builds on
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python),
[Transformers](https://github.com/huggingface/transformers),
[FastAPI](https://github.com/fastapi/fastapi), and
[uv](https://github.com/astral-sh/uv). It also depends on the authors who
publish open decision-model checkpoints and document their readout contracts.
