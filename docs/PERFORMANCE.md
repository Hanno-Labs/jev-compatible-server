# Batching and performance

Concurrent HTTP calls are collected for a short, bounded window. A registry
batch is grouped by model so each loaded runtime receives compatible requests
together. Transformers implementations combine questions in a group into shared
forward-pass batches.

Encoder-decoder margin recipes additionally cache pooled document encoder
states in a bounded LRU. This avoids repeated encoder work when candidates are
reused across requests; the bundled recipe defaults to 256 MiB and four
documents per model batch.

Hidden-state probe recipes batch rendered states into forward passes and score
all supported questions for a state from the same hidden vector. They generate
no output tokens. Their registry `decision.batch_size` bounds the internal
model batch independently of the HTTP request microbatch.

Configuration:

- `DECISION_MAX_BATCH_SIZE` defaults to `16`.
- `DECISION_BATCH_WAIT_MS` defaults to `5`.

This is dynamic request microbatching, not continuous token-level batching.
Throughput, latency, and memory depend on the model, backend, hardware, request
shape, and these settings. The project does not yet publish a general
performance claim.
