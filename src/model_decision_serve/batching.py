"""Small dynamic microbatcher for Jev-compatible requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .protocol import DecisionRequest, DecisionResponse
from .runtime import DecisionRuntime


@dataclass
class _Pending:
    request: DecisionRequest
    future: asyncio.Future[DecisionResponse]


class DecisionBatcher:
    """Collects concurrent requests for a bounded interval and runs one batch."""

    def __init__(self, runtime: DecisionRuntime, *, max_batch_size: int = 16, wait_ms: int = 5):
        self.runtime = runtime
        self.max_batch_size = max_batch_size
        self.wait_seconds = wait_ms / 1000
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None

    async def submit(self, request: DecisionRequest) -> DecisionResponse:
        if self._worker is None:
            await self.start()
        future: asyncio.Future[DecisionResponse] = asyncio.get_running_loop().create_future()
        await self._queue.put(_Pending(request, future))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.wait_seconds
            while len(batch) < self.max_batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            try:
                responses = await asyncio.to_thread(
                    self.runtime.decide_batch, [item.request for item in batch]
                )
                if len(responses) != len(batch):
                    raise RuntimeError("runtime returned the wrong batch length")
                for item, response in zip(batch, responses, strict=True):
                    item.future.set_result(response)
            except Exception as exc:
                for item in batch:
                    item.future.set_exception(exc)

