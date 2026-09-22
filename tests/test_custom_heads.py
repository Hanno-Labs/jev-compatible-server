import math

import pytest

from jev_compatible_server.custom_heads import (
    build_smalljev_semantic_ids,
    calibration_temperature,
    compile_openjev_tasks,
    custom_head_metadata,
    format_openjev_answers,
    render_custom_head_task,
)
from jev_compatible_server.encoder_decoder import MarginTask
from jev_compatible_server.protocol import DecisionRequest
from jev_compatible_server.runtime import RuntimeErrorBase


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
