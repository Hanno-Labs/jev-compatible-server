# Batching and performance

Concurrent HTTP calls are collected for a short, bounded window. A registry
batch is grouped by model so each loaded runtime receives compatible requests
together. Transformers implementations combine questions in a group into shared
forward-pass batches.

Configuration:

- `DECISION_MAX_BATCH_SIZE` defaults to `16`.
- `DECISION_BATCH_WAIT_MS` defaults to `5`.

This is dynamic request microbatching, not continuous token-level batching.
Throughput, latency, and memory depend on the model, backend, hardware, request
shape, and these settings. The project does not yet publish a general
performance claim.
