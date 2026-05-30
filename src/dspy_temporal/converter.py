"""Temporal data-converter wiring.

We use Temporal's pydantic data converter so our pydantic input/output models
(and nested types) serialize cleanly on both the client and the worker.
"""

from __future__ import annotations

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

data_converter = pydantic_data_converter


async def connect(target_host: str = "localhost:7233", *, namespace: str = "default", **kwargs) -> Client:
    """Connect a Temporal ``Client`` configured with the pydantic data converter."""
    return await Client.connect(
        target_host,
        namespace=namespace,
        data_converter=data_converter,
        **kwargs,
    )
