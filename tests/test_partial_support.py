from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from jev_compatible_server.app import create_app
from jev_compatible_server.protocol import (
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    Usage,
)
from jev_compatible_server.runtime import DecisionRuntime, QuestionTypeRuntime


class NoulOnlyFakeRuntime(DecisionRuntime):
    model_name = "fake/noul-only"

    def __init__(self) -> None:
        self.requests: list[DecisionRequest] = []

    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        self.requests.extend(requests)
        return [
            DecisionResponse(
                model=self.model_name,
                answers={
                    name: NoulAnswer(type="noul", noul=0.75)
                    for name in request.questions
                },
                usage=Usage(input_tokens=7),
            )
            for request in requests
        ]


def _mixed_request() -> DecisionRequest:
    return DecisionRequest.model_validate(
        {
            "state": {"question": "2 + 2?", "answer": "4"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a route.",
                    "criteria": {"accept": None, "review": None},
                },
                "correct": {
                    "type": "noul",
                    "instructions": "Is the answer correct?",
                },
            },
        }
    )


def test_partial_runtime_returns_supported_and_unsupported_answers() -> None:
    inner = NoulOnlyFakeRuntime()
    runtime = QuestionTypeRuntime(inner, ("noul",))

    response = runtime.decide(_mixed_request())

    assert list(inner.requests[0].questions) == ["correct"]
    assert response.answers["correct"].type == "noul"
    unsupported = response.answers["route"]
    assert unsupported.type == "unsupported"
    assert unsupported.question_type == "choice"
    assert unsupported.supported_types == ["noul"]
    assert response.usage.input_tokens == 7


def test_all_unsupported_questions_skip_backend_inference() -> None:
    inner = NoulOnlyFakeRuntime()
    runtime = QuestionTypeRuntime(inner, ("noul",))
    request = _mixed_request().model_copy(
        update={"questions": {"route": _mixed_request().questions["route"]}}
    )

    response = runtime.decide(request)

    assert inner.requests == []
    assert response.answers["route"].type == "unsupported"
    assert response.usage == Usage()


def test_partial_result_is_returned_over_systemone_http_route() -> None:
    runtime = QuestionTypeRuntime(NoulOnlyFakeRuntime(), ("noul",))

    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/v1/systemone",
            json=_mixed_request().model_dump(mode="json"),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answers"]["route"] == {
        "type": "unsupported",
        "question_type": "choice",
        "reason": "question_type_not_supported",
        "supported_types": ["noul"],
    }
    assert payload["answers"]["correct"] == {"type": "noul", "noul": 0.75}


def test_inference_error_logs_type_and_location_without_request_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingRuntime(DecisionRuntime):
        model_name = "fake/failing"

        def decide_batch(
            self, requests: Sequence[DecisionRequest]
        ) -> list[DecisionResponse]:
            raise MemoryError("private-input-marker")

    with TestClient(create_app(FailingRuntime())) as client:
        response = client.post(
            "/v1/systemone",
            json=_mixed_request().model_dump(mode="json"),
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "decision inference failed"
    assert "exception_type=MemoryError" in caplog.text
    assert "frames=" in caplog.text
    assert "private-input-marker" not in caplog.text
