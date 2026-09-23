import pytest

from jev_compatible_server.causal_options import (
    CausalOptionsBackend,
    OptionPrompt,
    option_letter,
)
from jev_compatible_server.protocol import DecisionRequest
from jev_compatible_server.runtime import RuntimeErrorBase, softmax


def backend(profile: str, two: bool = False) -> CausalOptionsBackend:
    value = object.__new__(CausalOptionsBackend)
    value.model_name = "fake"
    value.metadata = {"profile": profile, "two_order_aggregation": two}
    return value


def request() -> DecisionRequest:
    return DecisionRequest.model_validate(
        {
            "state": {"x": 1},
            "questions": {
                "pick": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {"odd/key": "one", "even:key": "two"},
                },
                "score": {
                    "type": "score",
                    "instructions": "score",
                    "criteria": ["low", "high"],
                },
                "truth": {"type": "noul", "instructions": "true?"},
            },
        }
    )


def test_letters() -> None:
    assert [option_letter(i) for i in (0, 25, 26)] == ["A", "Z", "AA"]


def test_jqv_contract() -> None:
    compiled = backend("jqv")._compile(request(), "pick", request().questions["pick"])
    assert compiled.labels == {"A": "odd/key", "B": "even:key"}
    assert compiled.suffixes == {"A": " A", "B": " B"}
    assert "<think>\n\n</think>\n\nAnswer:" in compiled.prompt


def test_reflex_averages_probabilities_after_remapping() -> None:
    value = backend("reflex")
    req = request().model_copy(
        update={"questions": {"pick": request().questions["pick"]}}
    )
    value._score = lambda compiled: {"A": 4.0, "B": 0.0}  # type: ignore[method-assign]
    answer = value.decide(req).answers["pick"]
    assert answer.type == "choice"
    assert answer.probabilities["odd/key"] == pytest.approx(.5)
    assert answer.probabilities["even:key"] == pytest.approx(.5)


def test_litjev_and_simplejev_disable_thinking() -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: object, **kwargs: object
        ) -> str:
            del messages
            assert kwargs["enable_thinking"] is False
            return "CHAT"

    for profile in ("litjev", "simplejev_v1"):
        value = backend(profile)
        value._tokenizer = Tokenizer()
        compiled = value._compile(request(), "score", request().questions["score"])
        assert compiled.prompt.startswith("CHAT")
    assert compiled.prompt.endswith('{"answer": ')


def test_generic_profile_uses_configured_template_and_token_strings() -> None:
    value = backend("generic")
    value.metadata.update(
        {
            "prompt_template": "State={state}\nQ={instructions}\n{options}\nAnswer=",
            "option_tokens": {"A": " yes", "B": " no"},
        }
    )

    compiled = value._compile(request(), "pick", request().questions["pick"])

    assert compiled.labels == {"A": "odd/key", "B": "even:key"}
    assert compiled.suffixes == {"A": " yes", "B": " no"}
    assert "A. one" in compiled.prompt


def test_semif_and_open_alternative_use_native_no_thinking_turns() -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, str]], **kwargs: object
        ) -> str:
            assert kwargs["enable_thinking"] is False
            return "CHAT\n" + messages[-1]["content"]

    rendered: dict[str, str] = {}
    for profile in ("semif", "open_alternative"):
        value = backend(profile)
        value._tokenizer = Tokenizer()
        compiled = value._compile(request(), "pick", request().questions["pick"])
        rendered[profile] = compiled.prompt
        assert compiled.prompt.startswith("CHAT\n")
        assert compiled.suffixes == {"A": "A", "B": "B"}
    assert '"evidence": {"x": 1}' in rendered["semif"]
    assert "Context:" in rendered["open_alternative"]


def test_decider_and_system_one_open_preserve_answer_slot_contracts() -> None:
    decider = backend("decider")
    compiled = decider._compile(request(), "pick", request().questions["pick"])
    assert compiled.prompt.endswith("Answer: (")
    assert "(A) odd/key: one" in compiled.prompt
    assert compiled.add_special_tokens is False

    decider_truth = decider._compile(request(), "truth", request().questions["truth"])
    assert decider_truth.labels == {"A": "false", "B": "true"}
    assert "(A) no" in decider_truth.prompt
    assert "(B) yes" in decider_truth.prompt

    system_one = backend("system_one_open")
    truth = system_one._compile(request(), "truth", request().questions["truth"])
    assert truth.labels == {"A": "false", "B": "true"}
    assert "Question (yes/no): true?" in truth.prompt
    assert truth.prompt.endswith("Answer: (")
    assert truth.add_special_tokens is True


def test_system_one_open_rejects_unpacked_multi_question_requests() -> None:
    value = backend("system_one_open")
    with pytest.raises(RuntimeErrorBase, match="packed multi-question"):
        value.decide(request())


def test_system_one_sg_preserves_choice_index_prompt_and_yes_first_order() -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, str]], **kwargs: object
        ) -> str:
            assert kwargs["enable_thinking"] is False
            assert kwargs["add_generation_prompt"] is True
            return messages[-1]["content"]

    value = backend("system_one_sg")
    value._tokenizer = Tokenizer()
    choice = value._compile(request(), "pick", request().questions["pick"])
    assert choice.labels == {"0": "odd/key", "1": "even:key"}
    assert choice.suffixes == {"0": "0", "1": "1"}
    assert '"name": "odd/key", "criteria": "one"' in choice.prompt
    assert choice.prompt.endswith("choice_index:")
    score = value._compile(request(), "score", request().questions["score"])
    assert score.labels == {"0": "0", "1": "1"}
    truth = value._compile(request(), "truth", request().questions["truth"])
    assert truth.labels == {"0": "true", "1": "false"}
    assert '"name": "yes", "criteria": "Yes"' in truth.prompt
    assert '"name": "no", "criteria": "No"' in truth.prompt

    answer = value._answer(request().questions["pick"], {"odd/key": .75, "even:key": .25})
    assert answer.confidence == pytest.approx(.18872187554086717)


def test_system_one_sg_scores_multitoken_decimal_indices() -> None:
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def encode(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            assert text.startswith("p")
            return [99] + [int(digit) for digit in text[1:]]

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids: object, attention_mask: object, **kwargs: object) -> object:
            del attention_mask
            assert kwargs == {"logits_to_keep": 1, "use_cache": False}
            logits = torch.zeros((*input_ids.shape, 100))
            logits[:, -1, 1] = 2.0
            logits[:, -1, 0] = 1.0
            if input_ids.shape[1] > 1:
                logits[:, -1, 0] = 3.0
                logits[:, -1, 1] = 4.0
            return type("Output", (), {"logits": logits})()

    value = backend("system_one_sg")
    value._torch = torch
    value._tokenizer = Tokenizer()
    value._model = Model()
    compiled = OptionPrompt(
        "p", {str(i): str(i) for i in range(12)},
        {str(i): str(i) for i in range(12)},
    )
    scores = value._score(compiled)
    root_logits = torch.zeros(100)
    root_logits[0], root_logits[1] = 1.0, 2.0
    child_logits = torch.zeros(100)
    child_logits[0], child_logits[1] = 3.0, 4.0
    root = torch.log_softmax(root_logits, dim=-1)
    child = torch.log_softmax(child_logits, dim=-1)
    assert scores["1"] == pytest.approx(float(root[1]))
    assert scores["10"] == pytest.approx(float(root[1] + child[0]))
    assert scores["11"] == pytest.approx(float(root[1] + child[1]))
    value.metadata = {"profile": "litjev", "sequence_option_extension": True}
    assert value._score(compiled) == scores


def test_simplejev_noul_and_boundary_validation() -> None:
    value = backend("simplejev_v1")
    value._temperature = lambda: 1.0  # type: ignore[method-assign]
    scores = {str(i): float(i) for i in range(1, 10)}
    values = softmax([scores[str(i)] for i in range(1, 10)])
    assert sum(values) == pytest.approx(1.0)

    class Tokenizer:
        def encode(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return [1] if text == "p" else [2]

    value._tokenizer = Tokenizer()
    with pytest.raises(RuntimeErrorBase, match="one-token continuation"):
        value._token_ids(OptionPrompt("p", {"A": "x"}, {"A": " A"}))


def test_simplejev_noul_uses_its_rating_readout() -> None:
    value = backend("simplejev_v1")
    value._temperature = lambda: 1.0  # type: ignore[method-assign]
    value._score = (  # type: ignore[method-assign]
        lambda compiled: {str(i): float(i) for i in range(1, 10)}
    )

    class Tokenizer:
        def apply_chat_template(
            self, messages: object, **kwargs: object
        ) -> str:
            del messages, kwargs
            return "CHAT"

    value._tokenizer = Tokenizer()

    req = request().model_copy(
        update={"questions": {"truth": request().questions["truth"]}}
    )
    answer = value.decide(req).answers["truth"]

    assert answer.type == "noul"
    assert 0.5 < answer.noul < 0.99
