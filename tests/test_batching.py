import asyncio
from collections.abc import Sequence

from jev_compatible_server.batching import DecisionBatcher
from jev_compatible_server.protocol import DecisionRequest, DecisionResponse, Usage
from jev_compatible_server.runtime import DecisionRuntime


class FakeRuntime(DecisionRuntime):
    model_name = "fake"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        self.batch_sizes.append(len(requests))
        return [DecisionResponse(model=self.model_name, answers={}, usage=Usage()) for _ in requests]


def test_concurrent_requests_are_microbatched() -> None:
    async def run() -> None:
        runtime = FakeRuntime()
        batcher = DecisionBatcher(runtime, max_batch_size=8, wait_ms=20)
        request = DecisionRequest.model_validate({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}})
        await asyncio.gather(*(batcher.submit(request) for _ in range(3)))
        await batcher.close()
        assert runtime.batch_sizes == [3]

    asyncio.run(run())


class SelectivelyFailingRuntime(FakeRuntime):
    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        self.batch_sizes.append(len(requests))
        if any(request.state == "bad" for request in requests):
            raise RuntimeError("bad request")
        return [
            DecisionResponse(model=self.model_name, answers={}, usage=Usage())
            for _ in requests
        ]


def test_failed_microbatch_is_retried_per_request() -> None:
    async def run() -> None:
        runtime = SelectivelyFailingRuntime()
        batcher = DecisionBatcher(runtime, max_batch_size=8, wait_ms=20)

        def request(state: str) -> DecisionRequest:
            return DecisionRequest.model_validate(
                {
                    "state": state,
                    "questions": {
                        "q": {"type": "noul", "instructions": "?"}
                    },
                }
            )

        results = await asyncio.gather(
            batcher.submit(request("good-1")),
            batcher.submit(request("bad")),
            batcher.submit(request("good-2")),
            return_exceptions=True,
        )
        await batcher.close()

        assert runtime.batch_sizes == [3, 1, 1, 1]
        assert isinstance(results[0], DecisionResponse)
        assert isinstance(results[1], RuntimeError)
        assert str(results[1]) == "bad request"
        assert isinstance(results[2], DecisionResponse)

    asyncio.run(run())
