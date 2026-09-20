# Jev-compatible decision inference server

This package exposes the Jev `POST /v1/systemone` contract while keeping model
execution behind a runtime adapter. Requests contain one `state` and a keyed
`questions` map. The supported question primitives are `choice`, `score`, and
`noul`; multiple questions are evaluated together and answers retain the same
keys.

The service includes two runtime adapters:

- `DECISION_BACKEND=llama` for GGUF token-logit models through
  `llama-cpp-python`.
- `DECISION_BACKEND=transformers` for Hugging Face causal token-logit models
  through Transformers.

Both adapters read the same model-side decision configuration. A minimal config
looks like this:

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

For models whose published metadata is stale or incomplete, use a service-owned
registry. Registry values override model metadata, so no model or service code
change is needed:

```json
{
  "default": "decision-qwen",
  "models": {
    "decision-qwen": {
      "backend": "transformers",
      "model": "org/decision-qwen",
      "config": {
        "decision.prompt_template": "State: {state}\\nQuestion: {instructions}\\nCriteria: {criteria}\\nAnswer:",
        "decision.tokens": {"true": 1001, "false": 1002}
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

Start with `DECISION_REGISTRY=/etc/decision/registry.json`. Clients can select
an entry using Jev's existing `model` field; requests without one use the
registry default. Registry entries are lazy-loaded and requests for the same
entry are batched together.

`configs/public-models.json` contains the public catalog. The Kev entries are
enabled and use the owned pointer-head adapter. NanoJev, open-jev-DeBERTa, and
System One are recorded as disabled pending their distinct attention/scalar
readout adapters; they are not silently routed through an incompatible loader.

Run the server with:

```bash
DECISION_BACKEND=transformers \
DECISION_MODEL_ID=your-org/your-model \
DECISION_CONFIG=decision.json \
uv run decision-serve
```

Incoming requests are collected for a short, bounded microbatch window. This
preserves Jev's wire contract while allowing concurrent calls to share one
forward pass. `DECISION_MAX_BATCH_SIZE` and `DECISION_BATCH_WAIT_MS` control the
tradeoff between throughput and tail latency.

Native scalar/pointer heads need a runtime-specific readout adapter; the
pointer-head path is now included for models that publish the reusable
backbone + LoRA + pointer-head recipe. Its registry metadata must provide
`decision.readout: "pointer_head"`, `decision.head_path`, and
`decision.packing` with the five delimiter-token names. This is an owned
implementation and does not import a model author's serving repository.
