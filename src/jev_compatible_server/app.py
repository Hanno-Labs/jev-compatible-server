"""FastAPI application exposing the Jev-compatible endpoint."""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .backends import LlamaBackend, load_decision_config
from .batching import DecisionBatcher
from .protocol import DecisionRequest, DecisionResponse
from .registry import ModelRegistry, RegistryRuntime, build_transformers_runtime
from .runtime import DecisionRuntime, RuntimeErrorBase, apply_question_type_support


logger = logging.getLogger(__name__)


def _positive_batch_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError("must be a positive integer")
    return parsed


def _model_batch_size_override(explicit: int | None = None) -> int | None:
    if explicit is not None:
        if isinstance(explicit, bool) or explicit <= 0:
            raise RuntimeErrorBase("model batch size override must be a positive integer")
        return explicit
    configured = os.environ.get("DECISION_MODEL_BATCH_SIZE")
    if configured is None:
        return None
    try:
        return _positive_batch_size(configured)
    except ValueError as exc:
        raise RuntimeErrorBase(
            "DECISION_MODEL_BATCH_SIZE must be a positive integer"
        ) from exc


def build_runtime(
    *,
    model: str | None = None,
    model_batch_size: int | None = None,
) -> DecisionRuntime:
    model_batch_size = _model_batch_size_override(model_batch_size)
    registry_path = os.environ.get("DECISION_REGISTRY")
    if registry_path:
        return RegistryRuntime(
            ModelRegistry.from_file(registry_path),
            pinned_model=model,
            model_batch_size=model_batch_size,
        )
    explicit_model = os.environ.get("DECISION_MODEL_ID") or os.environ.get(
        "DECISION_MODEL_PATH"
    )
    if not explicit_model and "DECISION_BACKEND" not in os.environ:
        return RegistryRuntime(
            ModelRegistry.from_builtin(),
            pinned_model=model,
            model_batch_size=model_batch_size,
        )
    if model is not None:
        raise RuntimeErrorBase(
            "--model cannot be combined with DECISION_BACKEND, "
            "DECISION_MODEL_ID, or DECISION_MODEL_PATH"
        )
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
            build_transformers_runtime(
                model_id,
                config,
                batch_size_override=model_batch_size,
            ),
            config,
        )
    raise RuntimeErrorBase(f"unknown DECISION_BACKEND: {backend}")


def create_app(
    runtime: DecisionRuntime | None = None,
    *,
    model: str | None = None,
    model_batch_size: int | None = None,
) -> FastAPI:
    selected_runtime = runtime or build_runtime(
        model=model,
        model_batch_size=model_batch_size,
    )
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
            frames = traceback.extract_tb(exc.__traceback__)[-12:]
            frame_locations = " > ".join(
                f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.name}"
                for frame in frames
            )
            logger.error(
                "decision inference failed exception_type=%s frames=%s",
                type(exc).__name__,
                frame_locations,
            )
            raise HTTPException(status_code=500, detail="decision inference failed") from exc

    return app


app: FastAPI | None = None


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve a Jev-compatible decision API")
    parser.add_argument(
        "--model",
        help=(
            "pin and load one registry model at startup, "
            "for example bosun-v3.1-0.6b"
        ),
    )
    parser.add_argument(
        "--model-batch-size",
        type=_positive_batch_size,
        help=(
            "override decision.batch_size for Transformers model forward passes; "
            "defaults to the selected model recipe"
        ),
    )
    args = parser.parse_args(argv)
    uvicorn.run(
        create_app(model=args.model, model_batch_size=args.model_batch_size),
        host="0.0.0.0",
        port=8000,
    )
