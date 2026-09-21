import json
from pathlib import Path

import pytest

from jev_compatible_server.app import build_runtime
from jev_compatible_server.registry import ModelRegistry, RegistryRuntime


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


def test_build_runtime_uses_builtin_registry_without_model_environment(
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
    runtime = build_runtime()
    assert isinstance(runtime, RegistryRuntime)
    assert runtime.registry.resolve(None)[0] == "kev-4b"
