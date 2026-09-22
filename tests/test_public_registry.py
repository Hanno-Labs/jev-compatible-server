import json
from pathlib import Path

from jev_compatible_server.registry import ModelRegistry


def test_public_registry_lists_bosun_and_public_ids() -> None:
    path = Path(__file__).parents[1] / "configs" / "public-models.json"
    registry = ModelRegistry.from_file(path)
    expected_bosun = {
        "bosun-v3.1-0.6b": "Hanno-Labs/bosun-v3.1-0.6b",
        "bosun-v3.1-1.7b": "Hanno-Labs/bosun-v3.1-1.7b",
    }
    for name, model_id in expected_bosun.items():
        entry = registry.definition.models[name]
        config = entry.resolved_config(
            registry.definition.recipes[entry.recipe or ""]
        )
        assert entry.model == model_id
        assert entry.support_status == "supported"
        assert config["decision"]["readout"] == "bosun_decision_tokens"
        assert config["decision"]["question_types"] == [
            "choice",
            "score",
            "noul",
        ]
    assert registry.definition.models["kev-4b"].support_status == "supported"
    assert (
        registry.definition.models["system-one-qwen3.5-4b-scorer"].support_status
        == "supported"
    )
    assert registry.definition.models["nanojev"].enabled is False
    assert json.loads(path.read_text())["models"]["kev-8b"]["model"] == "jaredpalmer/kev-8b"
    laya = registry.definition.models["laya-mlx"]
    assert laya.enabled is False
    assert laya.backend == "mlx"
    assert laya.resolved_config()["decision.readout"] == "marker_scalar_head"
    for name in (
        "patronus-lynx-8b-instruct",
        "laya-multilingual",
        "vectara-hhem-2.1",
        "qwen3-next-80b-a3b-instruct",
    ):
        assert registry.definition.models[name].enabled is False
        assert registry.definition.models[name].support_status == "pending"


def test_public_registry_supports_all_ztc_hidden_state_probes() -> None:
    registry = ModelRegistry.from_builtin()
    expected = {
        "ztc-judge-4b": "FINAL-Bench/ZTC-Judge-4B",
        "ztc-judge-9b": "FINAL-Bench/ZTC-Judge-9B",
        "ztc-judge-27b": "FINAL-Bench/ZTC-Judge-27B",
        "darwin-397b-ztc": "FINAL-Bench/Darwin-397B-ZTC",
    }
    for name, model_id in expected.items():
        _, entry = registry.resolve(name)
        config = entry.resolved_config(registry.definition.recipes[entry.recipe or ""])
        assert entry.model == model_id
        assert entry.support_status == "supported"
        assert config["decision"]["readout"] == "hidden_state_probe"
        if name == "darwin-397b-ztc":
            assert config["decision"]["question_types"] == ["noul"]
            assert config["decision"]["input"]["mode"] == "shared_state"
        else:
            assert config["decision"]["question_types"] == [
                "choice",
                "score",
                "noul",
            ]
            assert config["decision"]["input"]["mode"] == "candidate_tasks"
