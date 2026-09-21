# Documentation

`jev-compatible-server` exposes decision models through one Jev-compatible
HTTP API while keeping model execution and decision readout separate.

- [API reference](API.md) — request and response shapes for `choice`, `score`,
  and `noul` questions.
- [Supported models and backends](MODELS.md) — enabled checkpoints, execution
  engines, and readout coverage.
- [Model recipes and registries](MODEL_RECIPES.md) — configure a model without
  adding model-specific branches to the service.
- [Batching and performance](PERFORMANCE.md) — current batching behavior and
  tuning controls.
- [Contributing](../CONTRIBUTING.md) — development and contribution guidance.
