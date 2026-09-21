import pytest

from jev_compatible_server.encoder_decoder import MarginTask
from jev_compatible_server.hidden_state_probe import (
    HiddenStateProbeBackend,
    render_probe_input,
    render_probe_task,
)
from jev_compatible_server.protocol import DecisionRequest
from jev_compatible_server.runtime import RuntimeErrorBase


def test_probe_input_renders_declared_state_fields() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": {"payload": {"question": "2 + 2?"}, "answer": "4"},
            "questions": {
                "correct": {
                    "type": "noul",
                    "instructions": "Is this correct?",
                }
            },
        }
    )
    metadata = {
        "input": {
            "template": "Question: {question}\nAnswer: {answer}",
            "fields": {"question": "payload.question", "answer": "answer"},
        }
    }

    assert render_probe_input(request, metadata) == "Question: 2 + 2?\nAnswer: 4"


def test_probe_input_rejects_missing_state_path() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": {"question": "2 + 2?"},
            "questions": {
                "correct": {
                    "type": "noul",
                    "instructions": "Is this correct?",
                }
            },
        }
    )
    metadata = {
        "input": {
            "template": "{answer}",
            "fields": {"answer": "answer"},
        }
    }

    with pytest.raises(RuntimeErrorBase, match="state path is missing: answer"):
        render_probe_input(request, metadata)


def test_probe_task_renders_candidate_verifier_fields() -> None:
    task = MarginTask(
        question_id="route",
        option_id="billing",
        instruction="Choose the best route.",
        query='{"ticket":"duplicate charge"}',
        document="billing: billing issue",
    )
    metadata = {
        "input": {
            "template": "Q={query}\nI={instructions}\nA={candidate}",
        }
    }

    assert render_probe_task(task, metadata) == (
        'Q={"ticket":"duplicate charge"}\n'
        "I=Choose the best route.\n"
        "A=billing: billing issue"
    )


def test_candidate_probe_scores_are_aggregated_into_choice_answer() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": {"ticket": "duplicate charge"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a route.",
                    "criteria": {
                        "billing": "billing issue",
                        "technical": "technical issue",
                    },
                }
            },
        }
    )
    backend = object.__new__(HiddenStateProbeBackend)
    backend.model_name = "answer-verifier"
    backend.metadata = {
        "input": {
            "template": "Q={query}\nI={instructions}\nA={candidate}",
        },
        "instruction_adapters": {"choice": "", "score": "", "noul": ""},
        "candidates": {
            "choice": {
                "criterion_template": "{key}: {criterion}",
                "empty_criterion_template": "{key}",
            },
            "score": {"criterion_template": "{criterion}"},
            "noul": {
                "no_criteria_options": {"true": "yes", "false": "no"},
                "criterion_template": "{criterion}",
            },
        },
        "aggregation": {
            "choice_temperature": 1.0,
            "score_temperature": 1.0,
            "noul_a": 1.0,
            "noul_b": 0.0,
            "noul_mode": "true_false_softmax",
            "confidence": "normalized_entropy",
        },
    }
    rendered: list[str] = []

    def score_texts(texts: list[str]) -> tuple[list[float], list[int]]:
        rendered.extend(texts)
        return [2.0, 0.0], [12, 13]

    backend._score_texts = score_texts  # type: ignore[method-assign]

    response = backend._decide_candidate_tasks([request])[0]

    answer = response.answers["route"]
    assert answer.type == "choice"
    assert answer.choice == "billing"
    assert answer.probabilities["billing"] > answer.probabilities["technical"]
    assert response.usage.input_tokens == 25
    assert rendered[0].endswith("A=billing: billing issue")
