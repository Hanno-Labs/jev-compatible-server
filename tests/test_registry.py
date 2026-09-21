import json
from pathlib import Path

from jev_compatible_server.registry import ModelRegistry


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
