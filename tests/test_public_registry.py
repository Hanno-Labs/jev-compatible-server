import json
from pathlib import Path

from model_decision_serve.registry import ModelRegistry


def test_public_registry_excludes_bosun_and_lists_public_ids() -> None:
    path = Path(__file__).parents[1] / "configs" / "public-models.json"
    registry = ModelRegistry.from_file(path)
    assert "bosun" not in registry.definition.models
    assert registry.definition.models["kev-4b"].support_status == "supported"
    assert registry.definition.models["nanojev"].enabled is False
    assert json.loads(path.read_text())["models"]["kev-8b"]["model"] == "jaredpalmer/kev-8b"
    laya = registry.definition.models["laya-mlx"]
    assert laya.enabled is False
    assert laya.backend == "mlx"
    assert laya.resolved_config()["decision.readout"] == "marker_scalar_head"
