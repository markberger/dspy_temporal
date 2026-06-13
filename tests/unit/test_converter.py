"""Tests for the client connect helper (no server; Client.connect monkeypatched)."""

import pytest

from dspy_temporal import converter


@pytest.mark.asyncio
async def test_connect_forwards_pydantic_converter(monkeypatch):
    captured = {}

    async def fake_connect(target_host, **kwargs):
        captured["target_host"] = target_host
        captured.update(kwargs)
        return "CLIENT"

    monkeypatch.setattr(converter.Client, "connect", fake_connect)

    result = await converter.connect(
        "host:1234", namespace="ns", rpc_metadata={"k": "v"}
    )

    assert result == "CLIENT"
    assert captured["target_host"] == "host:1234"
    assert captured["namespace"] == "ns"
    assert captured["data_converter"] is converter.pydantic_data_converter
    assert captured["rpc_metadata"] == {"k": "v"}  # extra kwargs pass through
