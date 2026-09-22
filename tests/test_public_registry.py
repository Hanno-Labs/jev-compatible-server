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
    system_one = registry.definition.models["system-one-qwen3.5-4b-scorer"]
    system_one_config = system_one.resolved_config(
        registry.definition.recipes[system_one.recipe or ""]
    )
    assert system_one_config["decision"]["loader"]["revision"] == (
        "1001bb4d826a52d1f399e183466143f4da7b741b"
    )
    assert system_one_config["decision"]["loader"]["adapter_revision"] == (
        "e6464dce15f013c2ef641593a85cc6afcdaea928"
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


def test_public_registry_supports_native_catalog_servers_with_pinned_weights() -> None:
    registry = ModelRegistry.from_builtin()
    expected = {
        "djev": {
            "readout": "diffusion_structured_read",
            "endpoint": "http://127.0.0.1:8000/v1/request",
            "weights": "f7f5b7f5fa82ffc52addd066915886d497f5517b",
        },
        "jeff": {
            "readout": "gliformer_native",
            "endpoint": "http://127.0.0.1:8000/v1/systemone",
            "weights": "d0a4e53d09cebe6bc963dd9be319d4279084bb2d",
        },
        "openjev-thinking": {
            "readout": "diffusion_thinking_read",
            "endpoint": "http://127.0.0.1:8080/v1/systemone",
            "weights": "ec4ff3df205028f4e81c954c2227f9312b3ec2ea",
        },
        "winnow-12b": {
            "readout": "winnow_shared_branch",
            "endpoint": "http://127.0.0.1:8091/v1/systemone",
            "weights": "b6ac22b0d51b69b18200acacb3fbdd98073fffe8",
        },
    }
    for name, contract in expected.items():
        _, entry = registry.resolve(name)
        config = entry.resolved_config()
        decision = config["decision"]
        assert entry.support_status == "supported"
        assert decision["readout"] == contract["readout"]
        assert decision["endpoint"] == contract["endpoint"]
        assert decision["weights"]["revision"] == contract["weights"]

    djev_thinking = registry.definition.models["djev-thinking"]
    assert djev_thinking.enabled is True
    assert djev_thinking.support_status == "supported"
    djev_thinking_decision = djev_thinking.resolved_config()["decision"]
    assert djev_thinking_decision["native_contract"] == "djev"
    assert djev_thinking_decision["endpoint"].endswith(":8011/v1/systemone")
    assert djev_thinking_decision["request_fields"] == {"think": 64}


def test_public_registry_supports_catalog_causal_profiles_with_pinned_weights() -> None:
    registry = ModelRegistry.from_builtin()
    expected = {
        "semif-openjev-qwen3.5-4b": (
            "semif",
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        ),
        "open-alternative-jev": (
            "open_alternative",
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        ),
        "decider-2b": (
            "decider",
            "Mapika/decider-2b",
            "fa996cea58e1c1d8d1ab4d7124154f303b017f95",
        ),
        "decider-35b-a3b": (
            "decider",
            "Mapika/decider-35b-a3b",
            "d9783c58ad4fbff3a6030c8332d29738ec49d22a",
        ),
    }
    for name, (profile, model_id, revision) in expected.items():
        _, entry = registry.resolve(name)
        decision = entry.resolved_config()["decision"]
        assert entry.enabled is True
        assert entry.support_status == "supported"
        assert decision["readout"] == "causal_options"
        assert decision["profile"] == profile
        assert decision["loader"]["model"] == model_id
        assert decision["loader"]["revision"] == revision


def test_public_registry_supports_pinned_fastino_gliner2_models() -> None:
    registry = ModelRegistry.from_builtin()
    expected = {
        "gliner2.5-base": (
            "fastino/gliner2.5-base-v1",
            "1a8bc24e00dc7300b9017c81d63e3dcdabb26596",
        ),
        "gliner2-large-v1": (
            "fastino/gliner2-large-v1",
            "f32ea6ef6e26d8264fdc72431b4b3b041eadc537",
        ),
        "gliner2.5-multi-v1": (
            "fastino/gliner2.5-multi-v1",
            "a221b77a8baf4a613b8f8652661d41fa10a5641e",
        ),
        "gliner2.5-small-v1": (
            "fastino/gliner2.5-small-v1",
            "7e6f537f10337497069276892a5ef435028252ce",
        ),
    }
    for name, (model_id, revision) in expected.items():
        _, entry = registry.resolve(name)
        decision = entry.resolved_config()["decision"]
        assert entry.model == model_id
        assert entry.enabled is True
        assert entry.support_status == "supported"
        assert decision["readout"] == "gliner2_multilabel"
        assert decision["gliner2_load_options"]["revision"] == revision


def test_public_registry_supports_pinned_jev_local_contract() -> None:
    registry = ModelRegistry.from_builtin()
    _, entry = registry.resolve("jev-local")
    decision = entry.resolved_config()["decision"]
    assert entry.model == "Qwen/Qwen3.5-9B"
    assert entry.enabled is True
    assert entry.support_status == "supported"
    assert decision["readout"] == "jev_local_options"
    assert decision["implementation"]["revision"] == (
        "56bfc2a96543f2fc6a4d4227460a2c17553c6e24"
    )
    assert decision["loader"]["revision"] == (
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    assert decision["temperatures"] == {
        "choice": 0.5,
        "noul": 0.25,
        "score": 0.25,
    }
