from types import SimpleNamespace

import pytest

from jev_compatible_server.jev_local_options import (
    CANDIDATE_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    UPSTREAM_SOURCE_REVISION,
    JevLocalOptionsBackend,
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


def test_candidate_microbatch_matches_serial_teacher_forcing() -> None:
    torch = pytest.importorskip("torch")

    class ToyTokenizer:
        def __call__(self, text: str, *, return_tensors: str) -> SimpleNamespace:
            assert return_tensors == "pt"
            tokens = torch.tensor([[ord(char) % 128 for char in text]])
            return SimpleNamespace(input_ids=tokens, attention_mask=torch.ones_like(tokens))

    class ToyModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, tokens: object, *, attention_mask: object) -> SimpleNamespace:
            self.calls += 1
            assert (tokens[attention_mask == 0] == 0).all()
            logits = torch.arange(128, dtype=torch.float32).expand(
                tokens.shape[0], tokens.shape[1], -1
            )
            return SimpleNamespace(logits=logits)

    backend = object.__new__(JevLocalOptionsBackend)
    backend._torch = torch
    backend._tokenizer = ToyTokenizer()
    backend._model = ToyModel()
    prefix = "Question:"
    candidates = ["A" * (index % 5 + 1) for index in range(17)]
    actual = backend._candidate_means(prefix, candidates)
    assert backend._model.calls == 3
    assert CANDIDATE_BATCH_SIZE == 8

    expected = []
    for candidate in candidates:
        encoded = backend._tokenizer(prefix + " " + candidate, return_tensors="pt")
        full_ids = encoded.input_ids[0].tolist()
        common = len(prefix)
        output = backend._model(encoded.input_ids, attention_mask=encoded.attention_mask)
        log_probabilities = torch.log_softmax(output.logits[0], dim=-1)
        terms = [
            log_probabilities[index - 1, encoded.input_ids[0, index]].item()
            for index in range(common, len(full_ids))
        ]
        expected.append(sum(terms) / len(terms))
    assert actual == pytest.approx(expected)
