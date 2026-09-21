# Model recipes and registries

If a model uses an existing execution backend and readout, adding it is a data
change. Publish the decision metadata with the model or add a registry entry;
the service does not need a model-name branch.

If a model introduces new readout math, the runtime needs one reusable readout
implementation. Future models with the same architecture can then select it
declaratively.

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
