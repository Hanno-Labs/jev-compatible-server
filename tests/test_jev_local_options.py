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


def test_candidate_batch_preserves_native_token_logprobabilities() -> None:
    torch = pytest.importorskip("torch")

    class Tokenizer:
        pad_token_id = 63
        eos_token_id = 62

        def __call__(self, value: str, *, return_tensors: str) -> SimpleNamespace:
            assert return_tensors == "pt"
            ids = [1, *(ord(character) % 50 + 2 for character in value)]
            tokens = torch.tensor([ids], dtype=torch.long)
            return SimpleNamespace(input_ids=tokens, attention_mask=torch.ones_like(tokens))

    class Model:
        device = torch.device("cpu")
        calls = 0

        def __call__(self, tokens: object, *, attention_mask: object) -> SimpleNamespace:
            assert attention_mask is not None
            assert (tokens[attention_mask == 0] == Tokenizer.pad_token_id).all()
            self.calls += 1
            vocabulary = torch.arange(64, dtype=torch.float32)
            logits = ((tokens.unsqueeze(-1) + 1) * (vocabulary + 1)) / 100
            return SimpleNamespace(logits=logits)

    tokenizer = Tokenizer()
    model = Model()
    backend = JevLocalOptionsBackend.__new__(JevLocalOptionsBackend)
    backend._torch = torch
    backend._tokenizer = tokenizer
    backend._model = model
    candidates = ["A" * (index % 5 + 1) for index in range(17)]
    assert CANDIDATE_BATCH_SIZE == 8
    actual = backend._candidate_means("prefix:", candidates)
    assert model.calls == 3

    expected = []
    for candidate in candidates:
        encoded = tokenizer("prefix: " + candidate, return_tensors="pt")
        ids = encoded.input_ids
        logits = model(ids, attention_mask=encoded.attention_mask).logits[0]
        common = len(tokenizer("prefix:", return_tensors="pt").input_ids[0])
        log_probabilities = torch.log_softmax(logits, dim=-1)
        terms = [
            log_probabilities[index - 1, ids[0, index]].item()
            for index in range(common, ids.shape[1])
        ]
        expected.append(sum(terms) / len(terms))
    assert actual == pytest.approx(expected, abs=1e-6)
