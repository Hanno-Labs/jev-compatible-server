from model_decision_serve.protocol import DecisionRequest


def test_jev_request_accepts_mixed_questions() -> None:
    request = DecisionRequest.model_validate(
        {
            "model": "jev-latest",
            "state": "A support ticket",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "Payments", "technical": "Bugs"},
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["low", "high"],
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            },
        }
    )
    assert list(request.questions) == ["route", "severity", "urgent"]

