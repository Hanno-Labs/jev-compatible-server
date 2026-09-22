# Model recipes and registries

If a model uses an existing execution backend and readout, adding it is a data
change. Publish the decision metadata with the model or add a registry entry;
the service does not need a model-name branch.

If a model introduces new readout math, the runtime needs one reusable readout
implementation. Future models with the same architecture can then select it
declaratively.

## Shared recipes

Top-level `recipes` hold reusable decision metadata. Models reference a recipe
by name and can override only the fields that differ. Recipe names describe the
readout contract, not a model or vendor:

```json
{
  "recipes": {
    "encoder-decoder-yes-no-margin-v1": {
      "decision": {
        "readout": "encoder_decoder_margin",
        "labels": {"positive": "yes", "negative": "no"},
        "encoder": {
          "document_template": "<Document>: {document}",
          "pooling": "mean_chunks",
          "chunk_size": 4
        },
        "decoder": {
          "template": "...{instruction}...{query}...",
          "pad_to_multiple_of": 8
        },
        "instruction_adapters": {
          "choice": "...",
          "score": "...",
          "noul": "..."
        },
        "candidates": {
          "choice": {
            "criterion_template": "{key}: {criterion}",
            "empty_criterion_template": "{key}"
          },
          "score": {"criterion_template": "{criterion}"},
          "noul": {
            "no_criteria_template": "{state}",
            "criterion_template": "{criterion}"
          }
        },
        "aggregation": {
          "choice_temperature": 1.0,
          "score_temperature": 1.0,
          "noul_a": 1.0,
          "noul_b": 0.0,
          "confidence": "normalized_entropy"
        }
      }
    }
  },
  "models": {
    "example-reranker": {
      "backend": "transformers",
      "model": "org/example-reranker",
      "recipe": "encoder-decoder-yes-no-margin-v1"
    }
  }
}
```

The runtime never branches on `example-reranker`. A future compatible
checkpoint can reuse the recipe without a code change. A new readout algorithm,
rather than a new model name, is what requires runtime code.

The bundled registry also includes a `sequence_classifier_margin` recipe. It
declares a base model, optional PEFT adapter, state/question/candidate templates,
tail-preserving truncation, batch size, temperature, and confidence method. A
compatible scalar-head checkpoint needs only a model entry referencing that
recipe.

The generic `hidden_state_probe` readout is selected the same way. It supports
`linear` and `rbf` probe math, shared-state and per-candidate input modes, a
configurable last-token selector, pinned model/probe revisions, candidate
aggregation, and calibration values loaded from the probe artifact.

A shared-state confidence probe can declare only `noul`:

```json
{
  "decision": {
    "readout": "hidden_state_probe",
    "question_types": ["noul"],
    "input": {
      "mode": "shared_state",
      "template": "Question: {question}\nAnswer: {answer}",
      "fields": {"question": "question", "answer": "answer"},
      "max_length": 2048
    },
    "hidden": {"token": "last_non_pad"},
    "probe": {
      "kind": "rbf",
      "repo": "auto",
      "file": "probe.npz"
    },
    "output": {"transform": "sigmoid"}
  }
}
```

An answer-verifier probe can instead set `input.mode` to `candidate_tasks`, use
`{query}`, `{instructions}`, and `{candidate}` in its input template, and reuse
the same declarative `instruction_adapters`, `candidates`, and `aggregation`
objects as the margin readouts. That mode produces all three question types by
normalizing the probe's raw candidate scores; it does not require a new runtime
class or a model-name branch.

`decision.question_types` is backend-independent. Any existing or future
readout can declare a subset of `choice`, `score`, and `noul`; the runtime then
returns per-question `unsupported` results for the rest.

## Token-logit recipe

A token-logit recipe supplies the prompt and token ID representing each answer
label:

```json
{
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

## Service-owned registry

The registry fills gaps when published model metadata is missing, stale, or
cannot describe every required artifact. Registry configuration overrides model
metadata and file defaults.

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

Start with a custom registry:

```bash
DECISION_REGISTRY=/path/to/registry.json jev-compatible-server
```

Registry entries are loaded on first use and cached for later requests.
Pass `--model ALIAS` to load one registry entry at startup and reject requests
for any other alias.

## Single-model configuration

Run one Transformers model without a registry:

```bash
DECISION_BACKEND=transformers \
DECISION_MODEL_ID=your-org/your-model \
DECISION_CONFIG=/path/to/decision.json \
jev-compatible-server
```

For GGUF, set `DECISION_BACKEND=llama`, `DECISION_MODEL_PATH`, and
`DECISION_CONFIG`, and install the `llama` extra.
