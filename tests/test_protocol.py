import pytest

from jev_compatible_server.encoder_decoder import (
    aggregate_margin_answers,
    compile_margin_tasks,
)
from jev_compatible_server.protocol import DecisionRequest
from jev_compatible_server.registry import ModelRegistry


def test_jev_request_accepts_mixed_questions() -> None:
    request = DecisionRequest.model_validate(
        {
            "model": "jev-latest",
            "state": "A support ticket",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "Payments", "technical": "Bugs"},
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["low", "high"],
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            },
        }
    )
    assert list(request.questions) == ["route", "severity", "urgent"]


def test_encoder_decoder_recipe_compiles_exact_margin_tasks() -> None:
    registry = ModelRegistry.from_builtin()
    _, entry = registry.resolve("kalm-jev-nano")
    config = entry.resolved_config(registry.definition.recipes[entry.recipe or ""])
    request = DecisionRequest.model_validate(
        {
            "state": {"ticket": "refund", "days": 30},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a queue.",
                    "criteria": {"billing": "Payment help", "other": None},
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate severity.",
                    "criteria": ["low", {"label": "high"}],
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Is this urgent?",
                    "criteria": {"true": "Needs action now", "false": "Can wait"},
                },
                "refund": {"type": "noul", "instructions": "Is this a refund?"},
            },
        }
    )
    metadata = config["decision"]
    tasks = compile_margin_tasks(request, metadata)
    assert [(task.question_id, task.option_id, task.document) for task in tasks] == [
        ("route", "billing", "billing: Payment help"),
        ("route", "other", "other"),
        ("severity", "0", "low"),
        ("severity", "1", '{"label":"high"}'),
        ("urgent", "true", "Needs action now"),
        ("urgent", "false", "Can wait"),
        ("refund", "", '{"days":30,"ticket":"refund"}'),
    ]
    assert tasks[0].query == '{"days":30,"ticket":"refund"}'
    assert tasks[0].instruction.startswith("Choose a queue.\n\nEvaluate whether")


def test_encoder_decoder_recipe_aggregates_all_question_types() -> None:
    registry = ModelRegistry.from_builtin()
    _, entry = registry.resolve("kalm-jev-nano")
    config = entry.resolved_config(registry.definition.recipes[entry.recipe or ""])
    request = DecisionRequest.model_validate(
        {
            "state": "A production outage",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": {"ops": "Operations", "sales": "Sales"},
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate.",
                    "criteria": ["low", "medium", "high"],
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Urgent?",
                    "criteria": {"true": "urgent", "false": "not urgent"},
                },
                "outage": {"type": "noul", "instructions": "Outage?"},
            },
        }
    )
    metadata = config["decision"]
    tasks = compile_margin_tasks(request, metadata)
    answers = aggregate_margin_answers(
        request,
        tasks,
        [3.0, 1.0, -1.0, 0.0, 2.0, 4.0, 1.0, -1.0],
        metadata,
    )
    assert answers["route"].choice == "ops"
    assert answers["route"].confidence == pytest.approx(0.4729346589968385)
    assert answers["severity"].score == pytest.approx(1.8017846683472736)
    assert answers["urgent"].noul == pytest.approx(0.9525741268224334)
    assert answers["outage"].noul == pytest.approx(0.2689414213699951)


def test_scalar_sequence_classifier_recipe_uses_yes_no_candidates() -> None:
    registry = ModelRegistry.from_builtin()
    _, entry = registry.resolve("system-one-qwen3.5-4b-scorer")
    config = entry.resolved_config(registry.definition.recipes[entry.recipe or ""])
    metadata = config["decision"]
    request = DecisionRequest.model_validate(
        {
            "state": "The payment service is down.",
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is this urgent?"}
            },
        }
    )
    tasks = compile_margin_tasks(request, metadata)
    assert [(task.option_id, task.instruction, task.document) for task in tasks] == [
        ("true", "Is this urgent?", "yes"),
        ("false", "Is this urgent?", "no"),
    ]
    answers = aggregate_margin_answers(request, tasks, [3.5, 0.0], metadata)
    assert answers["urgent"].noul == pytest.approx(0.8807970779778823)
