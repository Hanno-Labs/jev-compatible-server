# API reference

The server accepts `POST` requests at both `/v1/systemone` and `/systemone`.
`/v1/systemone` is the stable Jev-compatible route.

## Request

A request contains an arbitrary shared `state`, an optional registry `model`,
and one or more named `questions`:

```json
{
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
    "severity": {
      "type": "score",
      "instructions": "How severe is this ticket?",
      "criteria": ["low", "medium", "high"]
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does this ticket need immediate attention?"
    }
  }
}
```

Question types:

- `choice` selects among 2–255 keyed criteria and returns the complete
  probability distribution.
- `score` evaluates an ordered list of 2–255 criteria and returns the expected
  ordinal value, distribution, confidence, and legend.
- `noul` returns a probability between zero and one. Custom `true` and `false`
  criteria are optional.

The question names are application-defined and preserved in the response. One
request can mix all three question types when the selected model supports them.

## Partial model support

A model can declare only the question types its native readout implements. The
server still returns HTTP `200` for a valid mixed request: supported questions
contain normal answers, while each unsupported question has an explicit result:

```json
{
  "type": "unsupported",
  "question_type": "choice",
  "reason": "question_type_not_supported",
  "supported_types": ["noul"]
}
```

If every question is unsupported, the backend is not invoked and usage remains
zero. Unknown models, disabled models, malformed inputs, and inference failures
remain request-level errors.

## Response

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
    "severity": {
      "type": "score",
      "score": 1.4,
      "probabilities": {
        "0": 0.1,
        "1": 0.4,
        "2": 0.5
      },
      "confidence": 0.5,
      "legend": ["low", "medium", "high"]
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

These numeric values illustrate the schema; they are not predictions for the
example request. Usage fields remain zero when the backend does not report
token counts.

## ZTC candidate verification

The bundled ZTC Judge entries support all three question types. The runtime
serializes the arbitrary shared `state` and question instructions as the
problem, then sends each criterion through the checkpoint as a proposed answer:

```json
{
  "model": "ztc-judge-4b",
  "state": {"ticket": "I was charged twice."},
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
}
```

Darwin ZTC instead accepts its own complete pre-generation prompt as `state`
and supports only `noul`, because its published probe estimates confidence in
Darwin's own forthcoming answer rather than scoring an external candidate.

## Model selection

The optional top-level `model` selects an enabled
[registry alias](MODELS.md#model-aliases). When omitted,
the registry's `default` entry is used. Start the server with `--model ALIAS` to
load and pin one registry entry before serving; requests may then omit `model`
or name that same alias. Requests naming another model return HTTP `422`.
Unknown and disabled models also return HTTP `422`.

## Health

`GET /health` returns the server status and active runtime name. In registry
mode it does not load every registered model; with `--model`, it reports the
pinned alias after that model has loaded.
