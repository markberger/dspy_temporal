"""Unit tests for the fine-mode async fan-out helpers."""

import pytest

from dspy_temporal.fine.parallel import aparallel, gather


@pytest.mark.asyncio
async def test_gather_returns_results_in_order():
    async def value(v):
        return v

    assert await gather(value(1), value("two"), value(3)) == [1, "two", 3]


@pytest.mark.asyncio
async def test_aparallel_awaits_each_module_acall_with_inputs():
    class _Mod:
        def __init__(self, tag):
            self.tag = tag

        async def acall(self, **kwargs):
            return (self.tag, kwargs)

    out = await aparallel([(_Mod("a"), {"x": 1}), (_Mod("b"), {"y": 2})])
    assert out == [("a", {"x": 1}), ("b", {"y": 2})]
