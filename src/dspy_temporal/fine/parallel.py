"""Workflow-safe parallel helpers for fine mode.

``dspy.Parallel`` is thread-based (``ParallelExecutor`` -> ``ThreadPoolExecutor``)
and drives *synchronous* module calls, so it can't run inside a Temporal workflow:
threads aren't allowed in the sandbox, and a sync LM call hits ``WorkflowLM.forward``
(which raises). The durable-mode way to fan out is ``asyncio.gather`` over the
*async* path -- each ``await predictor.acall(...)`` becomes its own
``dspy_lm_call`` activity, and Temporal's single, deterministic event loop runs
them concurrently (``asyncio.gather`` is sandbox-safe; ``as_completed``/``wait``,
which depend on completion *order*, are not).

These are thin wrappers so a user module's ``aforward`` can fan out without
importing ``asyncio`` directly, and so there's one documented entry point.
Loaded into the workflow via ``imports_passed_through`` (host code); keep it
replay-safe -- pure ``asyncio.gather`` only.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Iterable


async def gather(*coros: Awaitable[Any]) -> list[Any]:
    """Run awaitables concurrently and return their results in order.

    The fine-mode-safe fan-out primitive: ``await dspy_temporal.gather(
    p1.acall(**a), p2.acall(**b), ...)`` dispatches each leaf LM/tool call as its
    own concurrent Temporal activity.
    """
    return await asyncio.gather(*coros)


async def aparallel(pairs: Iterable[tuple[Any, dict[str, Any]]]) -> list[Any]:
    """Run ``(module, inputs)`` pairs concurrently via each module's async path.

    The fine-mode analog of ``dspy.Parallel``: ``await module.acall(**inputs)``
    for every pair, concurrently. Each pair's leaf LM/tool calls become their own
    activities.
    """
    return await asyncio.gather(*(module.acall(**inputs) for module, inputs in pairs))
