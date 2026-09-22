import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import uvicorn

import jev_compatible_server.app as app_module
from jev_compatible_server import registry as registry_module
from jev_compatible_server.app import build_runtime
from jev_compatible_server.protocol import DecisionRequest, DecisionResponse, Usage
from jev_compatible_server.registry import (
    ModelRegistry,
    RegistryRuntime,
    build_transformers_runtime,
)
from jev_compatible_server.runtime import DecisionRuntime, RuntimeErrorBase


class FakeRuntime(DecisionRuntime):
    model_name = "org/model-a"

    def __init__(self) -> None:
        self.requests: list[DecisionRequest] = []

    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        self.requests.extend(requests)
        return [
            DecisionResponse(model=self.model_name, answers={}, usage=Usage())
            for _ in requests
        ]


def test_main_forwards_startup_model_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: dict[str, object] = {}
    fake_app = object()

    def create_app(
        *,
        model: str | None = None,
        model_batch_size: int | None = None,
    ) -> object:
        selected["model"] = model
        selected["model_batch_size"] = model_batch_size
        return fake_app

    def run(app: object, *, host: str, port: int) -> None:
        selected["app"] = app
        selected["host"] = host
        selected["port"] = port

    monkeypatch.setattr(app_module, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "run", run)

    app_module.main(["--model", "bosun-v3.1-0.6b", "--model-batch-size", "4"])

    assert selected == {
        "model": "bosun-v3.1-0.6b",
        "model_batch_size": 4,
        "app": fake_app,
        "host": "0.0.0.0",
        "port": 8000,
    }


def test_registry_resolves_default_and_overrides_config(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "default": "qwen",
                "models": {
                    "qwen": {
                        "backend": "transformers",
                        "model": "org/qwen",
                        "config": {"decision.readout": "token_logits"},
                    }
                },
            }
        )
    )
    registry = ModelRegistry.from_file(path)
    name, entry = registry.resolve(None)
    assert name == "qwen"
    assert entry.resolved_config()["decision.readout"] == "token_logits"


def test_builtin_registry_resolves_public_default() -> None:
    registry = ModelRegistry.from_builtin()
    name, entry = registry.resolve(None)
    assert name == "kev-4b"
    assert entry.model == "jaredpalmer/kev-4b"


def test_registry_runtime_eagerly_loads_and_pins_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelRegistry.from_json(
        json.dumps(
            {
                "default": "model-b",
                "models": {
                    "model-a": {
                        "backend": "transformers",
                        "model": "org/model-a",
                    },
                    "model-b": {
                        "backend": "transformers",
                        "model": "org/model-b",
                    },
                },
            }
        )
    )
    fake_runtime = FakeRuntime()
    loaded_models: list[str] = []

    def build_backend(
        model_id: str,
        config: dict[str, object],
        *,
        batch_size_override: int | None = None,
    ) -> DecisionRuntime:
        del config, batch_size_override
        loaded_models.append(model_id)
        return fake_runtime

    monkeypatch.setattr(registry_module, "build_transformers_runtime", build_backend)

    runtime = RegistryRuntime(registry, pinned_model="model-a")

    assert runtime.model_name == "model-a"
    assert runtime.pinned_model == "model-a"
    assert loaded_models == ["org/model-a"]

    request = DecisionRequest.model_validate(
        {
            "state": "state",
            "questions": {"q": {"type": "noul", "instructions": "Decide."}},
        }
    )
    response = runtime.decide_batch([request])

    assert len(response) == 1
    assert fake_runtime.requests == [request]
    assert loaded_models == ["org/model-a"]


def test_registry_runtime_rejects_request_for_different_pinned_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelRegistry.from_json(
        json.dumps(
            {
                "models": {
                    "model-a": {
                        "backend": "transformers",
                        "model": "org/model-a",
                    },
                    "model-b": {
                        "backend": "transformers",
                        "model": "org/model-b",
                    },
                }
            }
        )
    )
    monkeypatch.setattr(
        registry_module,
        "build_transformers_runtime",
        lambda model_id, config, *, batch_size_override=None: FakeRuntime(),
    )
    runtime = RegistryRuntime(registry, pinned_model="model-a")
    request = DecisionRequest.model_validate(
        {
            "model": "model-b",
            "state": "state",
            "questions": {"q": {"type": "noul", "instructions": "Decide."}},
        }
    )

    with pytest.raises(
        RuntimeErrorBase,
        match="server is pinned to model 'model-a'; request selected 'model-b'",
    ):
        runtime.decide_batch([request])


def test_registry_recipe_is_shared_and_model_config_wins() -> None:
    registry = ModelRegistry.from_json(
        json.dumps(
            {
                "recipes": {
                    "margin": {
                        "decision": {
                            "readout": "encoder_decoder_margin",
                            "limits": {"query": 512, "document": 1024},
                        }
                    }
                },
                "models": {
                    "model-a": {
                        "backend": "transformers",
                        "model": "org/model-a",
                        "recipe": "margin",
                        "config": {"decision": {"limits": {"query": 256}}},
                    }
                },
            }
        )
    )
    _, entry = registry.resolve("model-a")
    config = entry.resolved_config(registry.definition.recipes[entry.recipe or ""])
    assert config["decision"]["readout"] == "encoder_decoder_margin"
    assert config["decision"]["limits"] == {"query": 256, "document": 1024}


def test_registry_rejects_missing_recipe() -> None:
    with pytest.raises(RuntimeErrorBase, match="unregistered recipe"):
        ModelRegistry.from_json(
            json.dumps(
                {
                    "models": {
                        "model-a": {
                            "backend": "transformers",
                            "model": "org/model-a",
                            "recipe": "missing",
                        }
                    }
                }
            )
        )


def test_builtin_registry_supports_encoder_decoder_margin_models() -> None:
    registry = ModelRegistry.from_builtin()
    expected = {
        "kalm-jev-nano": "KaLM-Embedding/KaLM-Reranker-V1-Nano-R2",
        "kalm-jev-small": "KaLM-Embedding/KaLM-Reranker-V1-Small-R2",
        "kalm-jev-large": "KaLM-Embedding/KaLM-Reranker-V1-Large-R2",
    }
    for name, model_id in expected.items():
        _, entry = registry.resolve(name)
        config = entry.resolved_config(registry.definition.recipes[entry.recipe or ""])
        assert entry.model == model_id
        assert config["decision"]["readout"] == "encoder_decoder_margin"


def test_transformers_dispatch_uses_readout_not_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        registry_module,
        "EncoderDecoderMarginBackend",
        lambda model_id, config: sentinel,
    )
    runtime = build_transformers_runtime(
        "any-org/any-model",
        {"decision": {"readout": "encoder_decoder_margin"}},
    )
    assert runtime is sentinel


def test_transformers_dispatch_supports_scalar_sequence_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        registry_module,
        "SequenceClassifierMarginBackend",
        lambda model_id, config: sentinel,
    )
    runtime = build_transformers_runtime(
        "another-org/another-model",
        {"decision": {"readout": "sequence_classifier_margin"}},
    )
    assert runtime is sentinel


@pytest.mark.parametrize(
    ("readout", "backend_name"),
    [
        ("nli_entailment", "NLIEntailmentBackend"),
        ("gliclass_calibrated", "GLiClassCalibratedBackend"),
    ],
)
def test_transformers_dispatch_supports_classifier_adapters(
    monkeypatch: pytest.MonkeyPatch,
    readout: str,
    backend_name: str,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        registry_module,
        backend_name,
        lambda model_id, config: sentinel,
    )

    runtime = build_transformers_runtime(
        "classifier-org/classifier-model",
        {"decision": {"readout": readout}},
    )

    assert runtime is sentinel


@pytest.mark.parametrize(
    ("readout", "backend_name"),
    [
        ("openjev_scalar_head", "OpenJevScalarHeadBackend"),
        ("semantic_option_head", "SmallJevSemanticBackend"),
    ],
)
def test_transformers_dispatch_supports_custom_head_adapters(
    monkeypatch: pytest.MonkeyPatch,
    readout: str,
    backend_name: str,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        registry_module,
        backend_name,
        lambda model_id, config: sentinel,
    )

    runtime = build_transformers_runtime(
        "custom-org/custom-model",
        {"decision": {"readout": readout}},
    )

    assert runtime is sentinel


@pytest.mark.parametrize(
    ("readout", "backend_name"),
    [
        ("bosun_decision_tokens", "BosunDecisionBackend"),
        ("diffusion_structured_read", "DjevHTTPRuntime"),
        ("gliformer_native", "JeffHTTPRuntime"),
        ("gliner2_multilabel", "GLiNER2Runtime"),
    ],
)
def test_transformers_dispatch_supports_native_model_readouts(
    monkeypatch: pytest.MonkeyPatch,
    readout: str,
    backend_name: str,
) -> None:
    sentinel = object()
    received_config: dict[str, object] = {}

    def build_backend(model_id: str, config: dict[str, object]) -> object:
        assert model_id == "public-org/public-model"
        received_config.update(config)
        return sentinel

    monkeypatch.setattr(registry_module, backend_name, build_backend)

    runtime = build_transformers_runtime(
        "public-org/public-model",
        {"decision": {"readout": readout}},
        batch_size_override=16,
    )

    assert runtime is sentinel
    assert received_config["decision.batch_size"] == 16


@pytest.mark.parametrize(
    ("native_contract", "backend_name"),
    [
        ("djev", "DjevThinkingRuntime"),
        ("openjev", "OpenJevThinkingHTTPRuntime"),
    ],
)
def test_transformers_dispatches_diffusion_thinking_by_native_contract(
    monkeypatch: pytest.MonkeyPatch,
    native_contract: str,
    backend_name: str,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        registry_module,
        backend_name,
        lambda model_id, config: sentinel,
    )

    runtime = build_transformers_runtime(
        "public-org/diffusion-model",
        {
            "decision": {
                "readout": "diffusion_thinking_read",
                "native_contract": native_contract,
            }
        },
    )

    assert runtime is sentinel


def test_transformers_rejects_ambiguous_diffusion_thinking_contract() -> None:
    with pytest.raises(RuntimeErrorBase, match="native_contract"):
        build_transformers_runtime(
            "public-org/diffusion-model",
            {"decision": {"readout": "diffusion_thinking_read"}},
        )


def test_registry_dispatches_winnow_before_generic_llama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelRegistry.from_json(
        json.dumps(
            {
                "models": {
                    "winnow": {
                        "backend": "llama",
                        "model": "EldanRing/Winnow-12B-Q8_0.gguf",
                        "config": {
                            "decision": {
                                "readout": "winnow_shared_branch",
                                "endpoint": "http://winnow:8091/v1/systemone",
                            }
                        },
                    }
                }
            }
        )
    )
    sentinel = object()
    received: dict[str, object] = {}

    def build_winnow(model_id: str, config: dict[str, object]) -> object:
        received["model_id"] = model_id
        received["config"] = config
        return sentinel

    monkeypatch.setattr(registry_module, "WinnowHTTPRuntime", build_winnow)
    monkeypatch.setattr(
        registry_module,
        "LlamaBackend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generic llama backend must not replace Winnow's planner")
        ),
    )

    _, entry = registry.resolve("winnow")
    runtime = RegistryRuntime(registry)._runtime("winnow", entry)

    assert runtime is sentinel
    assert received["model_id"] == "EldanRing/Winnow-12B-Q8_0.gguf"
    assert received["config"] == {
        "decision": {
            "readout": "winnow_shared_branch",
            "endpoint": "http://winnow:8091/v1/systemone",
        }
    }


def test_transformers_dispatch_supports_hidden_state_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    received_config: dict[str, object] = {}

    def build_backend(model_id: str, config: dict[str, object]) -> object:
        assert model_id == "some-org/some-model"
        received_config.update(config)
        return sentinel

    monkeypatch.setattr(
        registry_module,
        "HiddenStateProbeBackend",
        build_backend,
    )
    config = {
        "decision": {"readout": "hidden_state_probe", "batch_size": 8}
    }
    runtime = build_transformers_runtime(
        "some-org/some-model",
        config,
        batch_size_override=64,
    )
    assert runtime is sentinel
    assert received_config["decision.batch_size"] == 64
    assert config["decision"]["batch_size"] == 8


def test_build_runtime_uses_builtin_registry_without_model_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DECISION_REGISTRY",
        "DECISION_BACKEND",
        "DECISION_MODEL_ID",
        "DECISION_MODEL_PATH",
        "DECISION_CONFIG",
        "DECISION_MODEL_BATCH_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
    runtime = build_runtime()
    assert isinstance(runtime, RegistryRuntime)
    assert runtime.registry.resolve(None)[0] == "kev-4b"
    assert runtime.model_batch_size is None


def test_build_runtime_pins_bundled_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DECISION_REGISTRY",
        "DECISION_BACKEND",
        "DECISION_MODEL_ID",
        "DECISION_MODEL_PATH",
        "DECISION_CONFIG",
        "DECISION_MODEL_BATCH_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
    fake_runtime = FakeRuntime()
    loaded_models: list[str] = []

    def build_backend(
        model_id: str,
        config: dict[str, object],
        *,
        batch_size_override: int | None = None,
    ) -> DecisionRuntime:
        del config, batch_size_override
        loaded_models.append(model_id)
        return fake_runtime

    monkeypatch.setattr(registry_module, "build_transformers_runtime", build_backend)

    runtime = build_runtime(model="bosun-v3.1-0.6b")

    assert isinstance(runtime, RegistryRuntime)
    assert runtime.model_name == "bosun-v3.1-0.6b"
    assert runtime.pinned_model == "bosun-v3.1-0.6b"
    assert loaded_models == ["Hanno-Labs/bosun-v3.1-0.6b"]


def test_build_runtime_rejects_model_pin_with_direct_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DECISION_REGISTRY", raising=False)
    monkeypatch.setenv("DECISION_BACKEND", "transformers")
    monkeypatch.setenv("DECISION_MODEL_ID", "org/direct-model")
    monkeypatch.delenv("DECISION_MODEL_PATH", raising=False)

    with pytest.raises(
        RuntimeErrorBase,
        match="--model cannot be combined with DECISION_BACKEND",
    ):
        build_runtime(model="bosun-v3.1-0.6b")


def test_build_runtime_accepts_model_batch_size_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DECISION_REGISTRY",
        "DECISION_BACKEND",
        "DECISION_MODEL_ID",
        "DECISION_MODEL_PATH",
        "DECISION_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DECISION_MODEL_BATCH_SIZE", "64")

    runtime = build_runtime()

    assert isinstance(runtime, RegistryRuntime)
    assert runtime.model_batch_size == 64


def test_build_runtime_explicit_model_batch_size_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_MODEL_BATCH_SIZE", "16")

    runtime = build_runtime(model_batch_size=64)

    assert isinstance(runtime, RegistryRuntime)
    assert runtime.model_batch_size == 64


def test_build_runtime_rejects_invalid_model_batch_size_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_MODEL_BATCH_SIZE", "0")

    with pytest.raises(
        RuntimeErrorBase,
        match="DECISION_MODEL_BATCH_SIZE must be a positive integer",
    ):
        build_runtime()


def test_registry_skips_model_load_when_every_question_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelRegistry.from_json(
        json.dumps(
            {
                "models": {
                    "noul-only": {
                        "backend": "transformers",
                        "model": "org/noul-only",
                        "config": {
                            "decision": {
                                "readout": "hidden_state_probe",
                                "question_types": ["noul"],
                            }
                        },
                    }
                }
            }
        )
    )
    runtime = RegistryRuntime(registry)
    monkeypatch.setattr(
        runtime,
        "_runtime",
        lambda name, entry: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    request = DecisionRequest.model_validate(
        {
            "model": "noul-only",
            "state": "state",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": {"a": None, "b": None},
                }
            },
        }
    )

    response = runtime.decide(request)

    assert response.answers["route"].type == "unsupported"
