"""FastAPI application exposing the Jev-compatible endpoint."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .backends import LlamaBackend, load_decision_config
from .batching import DecisionBatcher
from .protocol import DecisionRequest, DecisionResponse
from .registry import ModelRegistry, RegistryRuntime, build_transformers_runtime
from .runtime import DecisionRuntime, RuntimeErrorBase, apply_question_type_support


def build_runtime() -> DecisionRuntime:
    registry_path = os.environ.get("DECISION_REGISTRY")
    if registry_path:
        return RegistryRuntime(ModelRegistry.from_file(registry_path))
    explicit_model = os.environ.get("DECISION_MODEL_ID") or os.environ.get(
        "DECISION_MODEL_PATH"
    )
    if not explicit_model and "DECISION_BACKEND" not in os.environ:
        return RegistryRuntime(ModelRegistry.from_builtin())
    backend = os.environ.get("DECISION_BACKEND", "transformers").lower()
    config_path = os.environ.get("DECISION_CONFIG")
    config = load_decision_config(config_path) if config_path else {}
    if backend == "llama":
        model_path = os.environ.get("DECISION_MODEL_PATH")
        if not model_path:
            raise RuntimeErrorBase("DECISION_MODEL_PATH is required for the llama backend")
        return apply_question_type_support(
            LlamaBackend(model_path, config=config), config
        )
    if backend == "transformers":
        model_id = os.environ.get("DECISION_MODEL_ID")
        if not model_id:
            raise RuntimeErrorBase("DECISION_MODEL_ID is required for the transformers backend")
        return apply_question_type_support(
            build_transformers_runtime(model_id, config), config
        )
    raise RuntimeErrorBase(f"unknown DECISION_BACKEND: {backend}")


def create_app(runtime: DecisionRuntime | None = None) -> FastAPI:
    selected_runtime = runtime or build_runtime()
    batcher = DecisionBatcher(
        selected_runtime,
        max_batch_size=int(os.environ.get("DECISION_MAX_BATCH_SIZE", "16")),
        wait_ms=int(os.environ.get("DECISION_BATCH_WAIT_MS", "5")),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await batcher.start()
        yield
        await batcher.close()

    app = FastAPI(title="jev-compatible-server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": selected_runtime.model_name}

    @app.post("/v1/systemone", response_model=DecisionResponse)
    @app.post("/systemone", response_model=DecisionResponse)
    async def system_one(request: DecisionRequest) -> DecisionResponse:
        try:
            return await batcher.submit(request)
        except RuntimeErrorBase as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="decision inference failed") from exc

    return app


app: FastAPI | None = None


def main() -> None:
    import uvicorn

    uvicorn.run("jev_compatible_server.app:create_app", factory=True, host="0.0.0.0", port=8000)
