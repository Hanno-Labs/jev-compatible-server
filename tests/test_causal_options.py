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
