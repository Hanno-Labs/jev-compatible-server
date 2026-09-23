from __future__ import annotations

import pytest
from jev_compatible_server.certo import certo_answer, certo_options
from jev_compatible_server.protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from jev_compatible_server.runtime import RuntimeErrorBase


def test_certo_choice_options_preserve_ids_and_descriptions() -> None:
    question = ChoiceQuestion(
        type="choice", instructions="Route the request.",
        criteria={"billing": "refunds and charges", "support": None},
    )
    assert certo_options(question) == [
        ("billing", "refunds and charges"), ("support", "support")
    ]
    answer = certo_answer(question, {"billing": 0.8, "support": 0.2})
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "billing"
    assert answer.confidence == 0.8


def test_certo_noul_uses_author_no_criteria_options() -> None:
    question = NoulQuestion(type="noul", instructions="Is the light on?")
    assert certo_options(question) == [
        ("true", "Is the light on? — true"),
        ("false", "Is the light on? — false"),
    ]
    answer = certo_answer(question, {"true": 0.7, "false": 0.3})
    assert isinstance(answer, NoulAnswer)
    assert answer.noul == 0.7


def test_certo_score_is_categorical_expected_level() -> None:
    question = ScoreQuestion(
        type="score", instructions="Rate severity.", criteria=["low", "medium", "high"]
    )
    assert certo_options(question) == [("0", "low"), ("1", "medium"), ("2", "high")]
    answer = certo_answer(question, {"0": 0.2, "1": 0.3, "2": 0.5})
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == pytest.approx(1.3)
    assert answer.legend == ["low", "medium", "high"]


def test_certo_rejects_misaligned_probability_keys() -> None:
    question = NoulQuestion(type="noul", instructions="Is the light on?")
    with pytest.raises(RuntimeErrorBase, match="match presented options"):
        certo_answer(question, {"true": 1.0})
