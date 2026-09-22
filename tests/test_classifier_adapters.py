from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_compatible_server.classifier_adapters import (
    VERDICT_ABSTENTION_ID,
    RLCDTemperatureCalibrator,
    aggregate_verdict_answers,
    compile_verdict_tasks,
    load_rlcd_calibrator,
    project_entailment_logits,
    render_gliclass_prompt,
)
from jev_compatible_server.encoder_decoder import MarginTask
from jev_compatible_server.protocol import DecisionRequest
from jev_compatible_server.runtime import RuntimeErrorBase


def _verdict_request() -> DecisionRequest:
    return DecisionRequest.model_validate(
        {
            "state": {"ticket": "My card was charged twice."},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose the customer-support route.",
                    "criteria": {
                        "billing": "dispute a duplicate charge",
                        "technical": "fix a login problem",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is this?",
                    "criteria": ["low impact", "high impact"],
                },
                "supported": {
                    "type": "noul",
                    "instructions": "The evidence supports a duplicate charge.",
                },
            },
        }
    )


class FakeTensor:
    ndim = 2
    shape = (2, 3)

    def __init__(self) -> None:
        self.selected: int | None = None

    def __getitem__(self, key: tuple[slice, int]) -> FakeTensor:
        assert key[0] == slice(None)
        self.selected = key[1]
        return self

    def isfinite(self) -> FakeTensor:
        return self

    def all(self) -> bool:
        return True


def test_nli_entailment_projection_selects_named_label_not_fixed_column() -> None:
    logits = FakeTensor()

    assert project_entailment_logits(logits, {"contradiction": 2, "ENTAILMENT": 1}) is logits
    assert logits.selected == 1


def test_nli_entailment_projection_rejects_missing_label() -> None:
    with pytest.raises(RuntimeErrorBase, match="entailment"):
        project_entailment_logits(FakeTensor(), {"neutral": 0})


def test_verdict_prompt_uses_all_labels_before_the_single_text_input() -> None:
    assert render_gliclass_prompt("Route?", "duplicate card charge", ["billing", "technical"]) == (
        "<<LABEL>>billing<<LABEL>>technical<<SEP>>Question: Route?\n\nContext:\nduplicate card charge"
    )


def test_verdict_compilation_reserves_the_last_slot_for_explicit_abstention() -> None:
    tasks = compile_verdict_tasks(_verdict_request())
    by_question: dict[str, list[MarginTask]] = {}
    for task in tasks:
        by_question.setdefault(task.question_id, []).append(task)

    route = by_question["route"]
    assert [task.document for task in route] == [
        "It is dispute a duplicate charge",
        "It is fix a login problem",
        "insufficient evidence",
    ]
    assert route[-1].option_id == VERDICT_ABSTENTION_ID
    assert [task.document for task in by_question["severity"]] == [
        "low impact (Value: 0)",
        "high impact (Value: 1)",
        "insufficient evidence",
    ]
    assert [task.document for task in by_question["supported"]] == [
        "true: The evidence supports a duplicate charge.",
        "false: not The evidence supports a duplicate charge.",
        "insufficient evidence",
    ]


def test_verdict_rejects_a_25th_substantive_candidate() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": {},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": {str(index): str(index) for index in range(25)},
                }
            },
        }
    )

    with pytest.raises(RuntimeErrorBase, match="24 substantive"):
        compile_verdict_tasks(request)


def test_verdict_maps_the_calibrated_full_distribution_including_abstention() -> None:
    request = _verdict_request()
    tasks = compile_verdict_tasks(request)
    margins = [
        1.0,
        0.0,
        4.0,  # route abstains
        0.0,
        1.0,
        2.0,  # score abstains
        2.0,
        0.0,
        1.0,  # noul: P(true | sufficient evidence)
    ]

    answers = aggregate_verdict_answers(request, tasks, margins)

    route = answers["route"]
    assert route.choice == VERDICT_ABSTENTION_ID
    assert VERDICT_ABSTENTION_ID in route.probabilities
    assert route.probabilities[VERDICT_ABSTENTION_ID] > route.probabilities["billing"]
    score = answers["severity"]
    assert VERDICT_ABSTENTION_ID in score.probabilities
    assert score.score > 0.5
    noul = answers["supported"]
    assert noul.noul > 0.8


def test_rlcd_calibrator_uses_per_cardinality_temperature() -> None:
    calibrator = RLCDTemperatureCalibrator(2.0, {"3": 4.0})

    assert calibrator.temperature_for(3) == 4.0
    assert calibrator.temperature_for(2) == 2.0


def test_rlcd_calibrator_loads_safe_public_json_contract(tmp_path: Path) -> None:
    path = tmp_path / "calibrator.json"
    path.write_text(json.dumps({"format_version": "rlcd-calibrator-v1", "temperature": 2.5, "per_k": {"3": 4.0}}))

    calibrator = load_rlcd_calibrator(path)

    assert calibrator.temperature_for(3) == 4.0


def test_rlcd_calibrator_rejects_opaque_or_unsupported_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "calibrator.json"
    path.write_text(json.dumps({"format_version": "pickle", "temperature": 2.5}))

    with pytest.raises(RuntimeErrorBase, match="unsupported Verdict calibrator format"):
        load_rlcd_calibrator(path)


def test_rlcd_calibrator_rejects_non_cardinality_per_k_keys(tmp_path: Path) -> None:
    path = tmp_path / "calibrator.json"
    path.write_text(
        json.dumps(
            {
                "format_version": "rlcd-calibrator-v1",
                "temperature": 2.5,
                "per_k": {"three": 4.0},
            }
        )
    )

    with pytest.raises(RuntimeErrorBase, match="positive integers"):
        load_rlcd_calibrator(path)
