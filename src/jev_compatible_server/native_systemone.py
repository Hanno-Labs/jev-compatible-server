"""Adapters for public servers that own their typed-decision readout.

These models publish more than a checkpoint: their probability readout is part
of a separately versioned native server.  Forwarding the typed request to that
server preserves its fixed canvas/branching, calibration, and thinking policy
instead of substituting a generic next-token approximation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .protocol import (
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulQuestion,
    ScoreQuestion,
)
from .runtime import DecisionRuntime, RuntimeErrorBase

JsonObject = dict[str, Any]
PostJSON = Callable[[str, JsonObject, float], JsonObject]


def _decision_value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    nested = config.get("decision")
    if isinstance(nested, Mapping) and key in nested:
        default = nested[key]
    return config.get(f"decision.{key}", default)


def _post_json(url: str, body: JsonObject, timeout_seconds: float) -> JsonObject:
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read(1_000).decode("utf-8", errors="replace")
        raise RuntimeErrorBase(
            f"native decision server returned HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeErrorBase("native decision server is unavailable") from exc
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeErrorBase("native decision server returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeErrorBase("native decision server returned a non-object response")
    return decoded


class NativeSystemOneHTTPRuntime(DecisionRuntime):
    """Forward the exact typed request to a model's published server.

    Required config: ``decision.endpoint``.  ``decision.native_model`` selects
    the server-side alias, ``decision.request_fields`` supplies documented
    request extensions (for example OpenJev's ``think``), and
    ``decision.timeout_seconds`` bounds each native read.
    """

    def __init__(
        self,
        model_id: str,
        *,
        config: Mapping[str, Any] | None = None,
        post_json: PostJSON | None = None,
    ) -> None:
        self.config = dict(config or {})
        endpoint = _decision_value(self.config, "endpoint")
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
            raise RuntimeErrorBase("native_systemone requires decision.endpoint as an HTTP URL")
        native_model = _decision_value(self.config, "native_model", model_id)
        if not isinstance(native_model, str) or not native_model:
            raise RuntimeErrorBase("decision.native_model must be a non-empty string")
        fields = _decision_value(self.config, "request_fields", {})
        if not isinstance(fields, Mapping):
            raise RuntimeErrorBase("decision.request_fields must be an object")
        timeout = _decision_value(self.config, "timeout_seconds", 120.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise RuntimeErrorBase("decision.timeout_seconds must be a positive number")
        self.model_name = str(_decision_value(self.config, "public_model_name", model_id))
        self.endpoint = endpoint
        self.native_model = native_model
        self.request_fields = dict(fields)
        self.timeout_seconds = float(timeout)
        self._post_json = post_json or _post_json

    def _body(self, request: DecisionRequest) -> JsonObject:
        body = request.model_dump(mode="json")
        body["model"] = self.native_model
        for key, value in self.request_fields.items():
            if key in {"state", "questions", "model"}:
                raise RuntimeErrorBase(
                    "decision.request_fields may not override state, questions, or model"
                )
            body[str(key)] = value
        return body

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        return [
            self._decode(
                request,
                self._post_json(self.endpoint, self._body(request), self.timeout_seconds),
            )
            for request in requests
        ]

    def _decode(self, request: DecisionRequest, raw: JsonObject) -> DecisionResponse:
        raw_answers = raw.get("answers")
        if not isinstance(raw_answers, Mapping) or set(raw_answers) != set(request.questions):
            raise RuntimeErrorBase(
                "native decision server did not return exactly one answer per question"
            )
        normalized: JsonObject = {
            "model": self.model_name,
            "answers": {},
            "usage": raw.get("usage", {}),
        }
        for name, question in request.questions.items():
            answer = raw_answers[name]
            if not isinstance(answer, Mapping) or answer.get("type") != question.type:
                raise RuntimeErrorBase(
                    f"native server returned an invalid {question.type} answer for {name!r}"
                )
            value = dict(answer)
            if isinstance(question, ScoreQuestion) and isinstance(value.get("legend"), Mapping):
                legend = value["legend"]
                if any(str(index) not in legend for index in range(len(question.criteria))):
                    raise RuntimeErrorBase(
                        f"native server returned an incomplete score legend for {name!r}"
                    )
                value["legend"] = [legend[str(index)] for index in range(len(question.criteria))]
            self._validate_answer(question, value, name)
            normalized["answers"][name] = value
        try:
            response = DecisionResponse.model_validate(normalized)
        except ValueError as exc:
            raise RuntimeErrorBase("native decision server returned an invalid typed response") from exc
        usage = response.usage
        if usage.input_tokens < 0 or usage.output_tokens < 0:
            raise RuntimeErrorBase("native decision server returned negative token usage")
        return response

    @staticmethod
    def _validate_answer(question: Any, answer: Mapping[str, Any], name: str) -> None:
        if isinstance(question, NoulQuestion):
            value = answer.get("noul")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise RuntimeErrorBase(f"native server returned invalid noul probability for {name!r}")
            return
        probabilities = answer.get("probabilities")
        expected = (
            list(question.criteria)
            if isinstance(question, ChoiceQuestion)
            else [str(index) for index in range(len(question.criteria))]
        )
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(expected):
            raise RuntimeErrorBase(f"native server returned a misaligned probability distribution for {name!r}")
        values = [probabilities[key] for key in expected]
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values
        ):
            raise RuntimeErrorBase(f"native server returned invalid probabilities for {name!r}")
        if not math.isclose(sum(float(value) for value in values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise RuntimeErrorBase(f"native server probabilities do not sum to one for {name!r}")


class SystemOneOpenHTTPRuntime(NativeSystemOneHTTPRuntime):
    """Adapt System One Open's list-shaped public ``/decide`` API."""

    def _body(self, request: DecisionRequest) -> JsonObject:
        questions: list[JsonObject] = []
        for name, question in request.questions.items():
            item: JsonObject = {
                "id": name,
                "type": question.type,
                "instructions": question.instructions,
            }
            criteria = question.model_dump(mode="json").get("criteria")
            if isinstance(question, ChoiceQuestion):
                item["options"] = criteria
            elif isinstance(question, ScoreQuestion):
                item["levels"] = criteria
            elif criteria is not None:
                item["criteria"] = criteria
            questions.append(item)
        return {"state": request.state, "questions": questions}

    def _decode(self, request: DecisionRequest, raw: JsonObject) -> DecisionResponse:
        items = raw.get("answers")
        if not isinstance(items, list) or len(items) != len(request.questions):
            raise RuntimeErrorBase("System One Open returned an invalid answer list")
        answers: JsonObject = {}
        for item in items:
            if not isinstance(item, Mapping):
                raise RuntimeErrorBase("System One Open returned a non-object answer")
            name = item.get("id")
            if not isinstance(name, str) or name not in request.questions or name in answers:
                raise RuntimeErrorBase("System One Open returned an unknown or duplicate answer id")
            question = request.questions[name]
            if item.get("type") != question.type:
                raise RuntimeErrorBase("System One Open returned a mismatched answer type")
            answer = dict(item)
            answer.pop("id")
            if isinstance(question, ScoreQuestion):
                answer["legend"] = question.criteria
            if isinstance(question, NoulQuestion):
                answer.pop("confidence", None)
            self._validate_answer(question, answer, name)
            answers[name] = answer
        try:
            return DecisionResponse.model_validate(
                {"model": self.model_name, "answers": answers, "usage": {}}
            )
        except ValueError as exc:
            raise RuntimeErrorBase("System One Open returned an invalid typed response") from exc


class DjevHTTPRuntime(NativeSystemOneHTTPRuntime):
    """DJeV's native ``/v1/request`` contract and structured distributions."""

    def __init__(
        self,
        model_id: str,
        *,
        config: Mapping[str, Any] | None = None,
        post_json: PostJSON | None = None,
    ) -> None:
        merged = dict(config or {})
        decision = dict(merged.get("decision", {})) if isinstance(merged.get("decision"), Mapping) else {}
        decision.setdefault("native_model", "djev")
        merged["decision"] = decision
        super().__init__(model_id, config=merged, post_json=post_json)

    def _body(self, request: DecisionRequest) -> JsonObject:
        body = super()._body(request)
        # DJeV rejects criterion descriptions longer than 500 Unicode characters.
        # Keep the frozen question and criterion keys intact; adapt only the
        # model-facing descriptions in this detached request body.
        for question in body["questions"].values():
            criteria = question.get("criteria")
            if isinstance(criteria, dict):
                question["criteria"] = {
                    key: value[:500] if isinstance(value, str) else value
                    for key, value in criteria.items()
                }
            elif isinstance(criteria, list):
                question["criteria"] = [
                    value[:500] if isinstance(value, str) else value for value in criteria
                ]
        options = _decision_value(self.config, "options", {})
        if not isinstance(options, Mapping):
            raise RuntimeErrorBase("decision.options must be an object")
        body["options"] = dict(options)
        return body


class DjevThinkingRuntime(NativeSystemOneHTTPRuntime):
    """DJeV Spark's ``think`` extension over its typed ``/v1/systemone`` API.

    The separate DJeV Spark server writes an optional bounded thought before a
    structured read, then returns only the normal typed response.  It is a
    distinct public contract from djev-dev's ``/v1/request`` service.
    """

    def __init__(
        self,
        model_id: str,
        *,
        config: Mapping[str, Any] | None = None,
        post_json: PostJSON | None = None,
    ) -> None:
        super().__init__(model_id, config=config, post_json=post_json)
        if not self.endpoint.rstrip("/").endswith("/v1/systemone"):
            raise RuntimeErrorBase(
                "djev_thinking requires decision.endpoint ending in /v1/systemone"
            )
        if "think" not in self.request_fields:
            raise RuntimeErrorBase(
                "djev_thinking requires decision.request_fields.think as a token budget"
            )
        think = self.request_fields["think"]
        if (
            isinstance(think, bool)
            or not isinstance(think, int)
            or not 0 <= think <= 4096
        ):
            raise RuntimeErrorBase("DJeV think must be an integer from 0 through 4096")


class OpenJevThinkingHTTPRuntime(NativeSystemOneHTTPRuntime):
    """OpenJev's documented ``think`` extension over native typed reads."""

    def __init__(
        self,
        model_id: str,
        *,
        config: Mapping[str, Any] | None = None,
        post_json: PostJSON | None = None,
    ) -> None:
        super().__init__(model_id, config=config, post_json=post_json)
        if "think" not in self.request_fields:
            raise RuntimeErrorBase(
                "openjev_thinking requires decision.request_fields.think as a token budget"
            )
        think = self.request_fields["think"]
        if (
            isinstance(think, bool)
            or not isinstance(think, int)
            or not 0 <= think <= 4096
        ):
            raise RuntimeErrorBase("OpenJev think must be an integer from 0 through 4096")


class JeffHTTPRuntime(NativeSystemOneHTTPRuntime):
    """Jeff's GLiFormer server, including its native grouping and temperature."""


class WinnowHTTPRuntime(NativeSystemOneHTTPRuntime):
    """Winnow's llama.cpp server, retaining its shared-prefix branch planner."""
