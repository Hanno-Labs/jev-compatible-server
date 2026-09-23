import math
from types import SimpleNamespace

import pytest
from jev_compatible_server.backends import PointerTransformersBackend
from jev_compatible_server.custom_heads import (
    ConfiguredCustomHeadBackend,
    SmallJevSemanticBackend,
    build_smalljev_aux_head,
    build_smalljev_semantic_ids,
    calibration_temperature,
    compile_openjev_tasks,
    custom_head_metadata,
    format_openjev_answers,
    render_custom_head_task,
)
from jev_compatible_server.encoder_decoder import MarginTask
from jev_compatible_server.protocol import DecisionRequest
from jev_compatible_server.runtime import RuntimeErrorBase, softmax


def test_openjev_metadata_requires_declared_artifact_contract() -> None:
    metadata = custom_head_metadata(
        {
            "decision": {
                "readout": "openjev_scalar_head",
                "loader": {},
                "artifacts": {},
                "input": {},
            }
        }
    )

    assert metadata["readout"] == "openjev_scalar_head"


def test_custom_head_overlong_tokenization_keeps_candidate_tail() -> None:
    torch = pytest.importorskip("torch")

    class Tokenizer:
        truncation_side = "left"

        def apply_chat_template(self, messages: list[dict[str, str]], **_: object) -> str:
            return messages[0]["content"]

        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, object]:
            assert kwargs["truncation"] is True
            assert kwargs["max_length"] == 8
            assert self.truncation_side == "left"
            rows = [[ord(char) for char in text][-8:] for text in texts]
            return {
                "input_ids": torch.tensor(rows),
                "attention_mask": torch.ones((len(rows), 8), dtype=torch.long),
            }

    class Model:
        def __call__(self, **kwargs: object) -> object:
            ids = kwargs["input_ids"]
            return SimpleNamespace(last_hidden_state=ids.float().unsqueeze(-1))

    backend = object.__new__(ConfiguredCustomHeadBackend)
    backend._torch = torch
    backend._tokenizer = Tokenizer()
    backend._batch_size = 2
    backend._max_length = 8
    backend._truncate_overlong = True
    backend._add_generation_prompt = True
    backend._enable_thinking = False
    backend._device = "cpu"
    backend._model = Model()
    backend._apply_head = lambda hidden: hidden[:, 0]

    scores, counts = backend._score_texts(["old context ANSWER!"])

    assert scores == [float(ord("!"))]
    assert counts == [8]


def test_pointer_batches_bound_padding_and_dense_mask_without_losing_rows() -> None:
    lengths = [4000, 29000, 400, 3100, 28000, 600, 30000, 2800, 500]
    batches = PointerTransformersBackend._encoding_batches(
        [{"ids": range(length)} for length in lengths]
    )
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert len(batches[0]) == 3
    for batch in batches:
        batch_lengths = [lengths[index] for index in batch]
        assert len(batch) <= 8
        assert max(batch_lengths) <= 2 * min(batch_lengths)
        assert len(batch) * max(batch_lengths) ** 2 <= 1_000_000_000


@pytest.mark.parametrize("value", [0, -0.1, math.inf, math.nan, True, "1.0"])
def test_calibration_temperature_rejects_invalid_saved_values(value: object) -> None:
    with pytest.raises(RuntimeErrorBase, match="positive finite"):
        calibration_temperature({"temperature": value}, "temperature")


def test_calibration_temperature_requires_configured_artifact_field() -> None:
    assert calibration_temperature({"saved_temperature": 1.25}, "saved_temperature") == 1.25
    with pytest.raises(RuntimeErrorBase, match="missing field: temperature"):
        calibration_temperature({"saved_temperature": 1.25}, "temperature")


def test_custom_head_renderer_uses_declared_candidate_template() -> None:
    task = MarginTask(
        question_id="route",
        option_id="billing",
        instruction="Choose a route.",
        query='{"ticket":"duplicate charge"}',
        document="billing: payment issue",
    )
    metadata = {"input": {"template": "State={query}\nQuestion={instructions}\nOption={candidate}"}}

    assert render_custom_head_task(task, metadata) == (
        'State={"ticket":"duplicate charge"}\n'
        "Question=Choose a route.\n"
        "Option=billing: payment issue"
    )


def test_custom_head_metadata_rejects_unknown_readout() -> None:
    with pytest.raises(RuntimeErrorBase, match="custom-head backend requires"):
        custom_head_metadata({"decision": {"readout": "pointer_head"}})


def test_openjev_compiler_uses_candidate_verifier_and_special_noul_prompt() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": {"ticket": "duplicate charge"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a route.",
                    "criteria": {"billing": "payment issue", "other": None},
                },
                "confirmed": {
                    "type": "noul",
                    "instructions": "Was it confirmed?",
                    "criteria": {"true": "the bank confirmed it", "false": "no confirmation"},
                },
            },
        }
    )

    route, confirmed = compile_openjev_tasks(request)

    assert route.prompts[0] == (
        'Context:\n{"ticket": "duplicate charge"}\n\n'
        "Question: Choose a route.\n"
        "Proposed answer: billing: payment issue\n"
        "Is this proposed answer correct? Answer Yes or No."
    )
    assert route.prompts[1].endswith("Proposed answer: other\nIs this proposed answer correct? Answer Yes or No.")
    assert confirmed.prompts == (
        (
            'Context:\n{"ticket": "duplicate charge"}\n\n'
            "Question: Was it confirmed?\n"
            "Yes means: the bank confirmed it\nNo means: no confirmation\n"
            "Is the answer to this question yes? Answer Yes or No."
        ),
    )


def test_openjev_answers_apply_temperature_to_choice_and_noul() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": "state",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "route",
                    "criteria": {"a": "A", "b": "B"},
                },
                "yes": {"type": "noul", "instructions": "yes?"},
            },
        }
    )
    answers = format_openjev_answers(compile_openjev_tasks(request), [2.0, 0.0, 2.0], 2.0)

    assert answers["route"].probabilities["a"] == pytest.approx(0.7310585786)
    assert answers["yes"].noul == pytest.approx(0.7310585786)


class _FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        prefix = [999] if add_special_tokens else []
        return {"input_ids": prefix + [ord(char) for char in text]}


def test_smalljev_semantic_builder_keeps_only_option_text_in_spans() -> None:
    tokenizer = _FakeTokenizer()
    ids, spans = build_smalljev_semantic_ids(
        tokenizer, "state", "question", ["billing", "technical"]
    )

    assert "".join(chr(value) for value in ids[spans[0][0] : spans[0][1]]).strip() == "billing"
    assert "".join(chr(value) for value in ids[spans[1][0] : spans[1][1]]).strip() == "technical"
    assert chr(ord("A")) not in "".join(chr(value) for value in ids[spans[0][0] : spans[0][1]])


def test_smalljev_semantic_builder_uses_public_1024_limit_and_state_trim() -> None:
    tokenizer = _FakeTokenizer()
    ids, _ = build_smalljev_semantic_ids(tokenizer, "x" * 4000, "q", ["alpha"])

    assert len(ids) <= 1024


def test_smalljev_semantic_builder_rejects_more_than_letter_verbalizers() -> None:
    with pytest.raises(RuntimeErrorBase, match="at most 26"):
        build_smalljev_semantic_ids(_FakeTokenizer(), "s", "q", [str(i) for i in range(27)])


def test_smalljev_pinned_aux_head_requires_exact_tensor_shapes() -> None:
    torch = pytest.importorskip("torch")
    payload = {
        "noul": {
            "bit.weight": torch.zeros((1, 4)),
            "bit.bias": torch.zeros(1),
        }
    }
    head = build_smalljev_aux_head(
        torch, payload, name="noul", prefix="bit", hidden_size=4, outputs=1, device="cpu"
    )
    assert torch.sigmoid(head(torch.ones(4))).item() == 0.5
    payload["noul"]["bit.weight"] = torch.zeros((2, 4))
    with pytest.raises(RuntimeErrorBase, match="invalid shape"):
        build_smalljev_aux_head(
            torch, payload, name="noul", prefix="bit", hidden_size=4, outputs=1, device="cpu"
        )


def test_smalljev_semantic_backend_uses_all_three_published_heads() -> None:
    torch = pytest.importorskip("torch")

    class Model:
        def __call__(self, *, input_ids: object, **_: object) -> object:
            return SimpleNamespace(
                hidden_states=(torch.ones((1, input_ids.shape[1], 4)),)
            )

    backend = object.__new__(SmallJevSemanticBackend)
    backend.model_name = "smalljev"
    backend._torch = torch
    backend._tokenizer = _FakeTokenizer()
    backend._model = Model()
    backend._device = "cpu"
    backend._max_length = 1024
    backend._head = torch.nn.Linear(4, 1)
    backend._noul_head = torch.nn.Linear(4, 1)
    backend._score_head = torch.nn.Linear(4, 8)
    with torch.no_grad():
        backend._head.weight.zero_()
        backend._head.bias.zero_()
        backend._noul_head.weight.zero_()
        backend._noul_head.bias.zero_()
        backend._score_head.weight.zero_()
        backend._score_head.bias.copy_(torch.arange(8, dtype=torch.float32))
    request = DecisionRequest.model_validate(
        {
            "state": "state",
            "questions": {
                "choice": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": {f"c{i}": f"Option {i}" for i in range(27)},
                },
                "binary": {"type": "noul", "instructions": "Yes?"},
                "ordinal": {
                    "type": "score",
                    "instructions": "Rate it.",
                    "criteria": ["low", "medium", "high"],
                },
            },
        }
    )

    response = backend.decide(request)

    assert len(response.answers["choice"].probabilities) == 27
    assert response.answers["choice"].probabilities["c26"] == pytest.approx(1 / 27)
    assert response.answers["binary"].noul == 0.5
    assert list(response.answers["ordinal"].probabilities) == ["0", "1", "2"]
    assert response.answers["ordinal"].score == pytest.approx(
        sum(index * probability for index, probability in enumerate(softmax([0.0, 1.0, 2.0])))
    )
    assert response.usage.input_tokens > 0
