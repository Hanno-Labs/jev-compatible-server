import pytest

from jev_compatible_server.jev_local_options import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    UPSTREAM_SOURCE_REVISION,
    answer_from_mean_logprobabilities,
    render_prefix,
)
from jev_compatible_server.protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
)


def _request() -> DecisionRequest:
    return DecisionRequest.model_validate(
        {
            "state": {"message": "Payouts failed"},
            "questions": {
                "choice": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "Payments", "technical": None},
                },
                "score": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["low", "high"],
                },
                "noul": {
                    "type": "noul",
                    "instructions": "Is this urgent?",
                    "criteria": {"true": "urgent", "false": "not urgent"},
                },
            },
        }
    )


def test_pinned_upstream_source_and_model_revisions() -> None:
    assert UPSTREAM_SOURCE_REVISION == "56bfc2a96543f2fc6a4d4227460a2c17553c6e24"
    assert DEFAULT_MODEL == "Qwen/Qwen3.5-9B"
    assert DEFAULT_MODEL_REVISION == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def test_prompt_preserves_upstream_option_and_noul_verbalizers() -> None:
    request = _request()
    choice = request.questions["choice"]
    noul = request.questions["noul"]
    assert isinstance(choice, ChoiceQuestion)
    assert isinstance(noul, NoulQuestion)
    assert render_prefix(
        request.state,
        choice.instructions,
        list(choice.criteria),
        {"billing": "Payments"},
        None,
    ) == (
        "State: {'message': 'Payouts failed'}\n"
        "Question: Which team?\n"
        "billing means: Payments\n"
        "Answer with exactly one of: billing | technical\n"
        "Answer:"
    )
    assert render_prefix(
        request.state,
        noul.instructions,
        ["Yes", "No"],
        None,
        None,
        {"true": "urgent", "false": "not urgent"},
    ).endswith("Yes means: urgent\nNo means: not urgent\nAnswer:")


def test_choice_score_and_noul_keep_native_distributions() -> None:
    request = _request()
    choice = answer_from_mean_logprobabilities(
        request.questions["choice"], [0.0, -1.0], temperature=0.5
    )
    assert isinstance(choice, ChoiceAnswer)
    assert choice.choice == "billing"
    assert choice.probabilities["billing"] == pytest.approx(0.8807970779778823)
    assert choice.confidence == choice.probabilities[choice.choice]

    score = answer_from_mean_logprobabilities(
        request.questions["score"], [-1.0, 0.0], temperature=0.25
    )
    assert isinstance(score, ScoreAnswer)
    assert score.probabilities == {
        "0": pytest.approx(0.017986209962091555),
        "1": pytest.approx(0.9820137900379085),
    }
    assert score.score == score.probabilities["1"]
    assert score.legend == ["low", "high"]

    noul = answer_from_mean_logprobabilities(
        request.questions["noul"], [0.0, -1.0], temperature=0.25
    )
    assert isinstance(noul, NoulAnswer)
    assert noul.noul == pytest.approx(0.9820137900379085)
