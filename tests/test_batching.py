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
