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

## Model selection

The optional top-level `model` selects an enabled registry entry. When omitted,
the registry's `default` entry is used. Unknown and disabled models return an
HTTP `422` response.

## Health

`GET /health` returns the server status and active runtime name without loading
every registered model.
