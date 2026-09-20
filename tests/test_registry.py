import json

from model_decision_serve.registry import ModelRegistry


def test_registry_resolves_default_and_overrides_config(tmp_path) -> None:
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

