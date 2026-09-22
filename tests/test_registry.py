import json
from pathlib import Path

import pytest

from jev_compatible_server import registry as registry_module
from jev_compatible_server.app import build_runtime
from jev_compatible_server.protocol import DecisionRequest
from jev_compatible_server.registry import (
    ModelRegistry,
    RegistryRuntime,
    build_transformers_runtime,
)
from jev_compatible_server.runtime import RuntimeErrorBase


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
