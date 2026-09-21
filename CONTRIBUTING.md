# Contributing

Contributions are welcome for reusable decision readouts, execution backends,
model recipes, compatibility tests, performance work, and documentation.

Before contributing:

1. Keep model behavior declarative when an existing backend and readout already
   cover it. New model IDs should normally require registry metadata, not a
   model-name conditional.
2. Add a reusable readout only when the model introduces genuinely new output
   semantics or scoring math.
3. Include tests for protocol compatibility and invalid configuration.
4. Keep the README concise and put detailed material in `docs/`.

Run the test and static-analysis suite before submitting a change:

```bash
uv run --with pytest pytest
uvx ruff check .
uv run --with mypy mypy src
```

Open an issue before a large architectural change so the backend/readout
boundary and compatibility contract can be agreed first.
