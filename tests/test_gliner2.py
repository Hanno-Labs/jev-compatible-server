from __future__ import annotations

from jev_compatible_server.gliner2 import GLiNER2Runtime
from jev_compatible_server.protocol import DecisionRequest


def test_gliner2_returns_explicitly_unsupported_typed_questions() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": "The card was retained.",
            "questions": {
                "route": {"type": "choice", "instructions": "Select a team.", "criteria": {"cash": None, "card": None}},
                "valid": {"type": "noul", "instructions": "Is it valid?"},
                "severity": {"type": "score", "instructions": "Rate it.", "criteria": ["low", "high"]},
            },
        }
    )

    response = GLiNER2Runtime("fastino/gliner2-large-v1").decide(request)

    assert response.model == "fastino/gliner2-large-v1"
    answers = response.model_dump(mode="json")["answers"]
    assert {name: answer["type"] for name, answer in answers.items()} == {
        "route": "unsupported",
        "valid": "unsupported",
        "severity": "unsupported",
    }
    assert all(answer["supported_types"] == [] for answer in answers.values())
