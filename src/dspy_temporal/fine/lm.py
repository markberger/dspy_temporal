"""``WorkflowLM`` -- the seam that turns each DSPy LM call into a Temporal activity.

In fine mode the program's orchestration runs *in the workflow* (deterministic
Python), but the actual model HTTP call must run in an activity. DSPy's async
path makes this a clean override: the adapter calls ``await lm.acall(messages=...)``,
so a ``dspy.BaseLM`` subclass that dispatches ``acall`` to an activity reroutes
every LM call without touching the rest of DSPy.

We override ``acall`` (not ``aforward``) because the activity returns
*already-processed* outputs -- the same list ``dspy.LM.__call__`` produces -- so
the adapter's ``_call_postprocess`` parses them with no further work. We then
replicate the one side effect ``acall`` normally has that callers depend on:
feeding the usage tracker, so ``prediction.get_lm_usage()`` keeps working.

This module is loaded into the workflow via ``imports_passed_through`` (host
code), but it is defined-by-us, so it must stay replay-safe: no wall-clock, no
randomness -- only ``workflow.execute_activity`` and pure data shaping.
"""

from __future__ import annotations

from typing import Any

import dspy
from temporalio import workflow

from ..models import LMCallInput, LMCallOutput
from ..options import CallOptions
from ..serde import json_safe

LM_ACTIVITY_NAME = "dspy_lm_call"

# Placeholder model id. The *real* model lives in the worker process (applied in
# the activity); the workflow never sees it and must not serialize it.
_WORKER_MODEL_PLACEHOLDER = "dspy-temporal/worker"


class WorkflowLM(dspy.BaseLM):
    """A ``dspy.BaseLM`` whose every call is a ``dspy_lm_call`` activity."""

    def __init__(self, *, options: CallOptions | None = None):
        super().__init__(model=_WORKER_MODEL_PLACEHOLDER)
        self._options = options or CallOptions()

    async def acall(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        out = await workflow.execute_activity(
            LM_ACTIVITY_NAME,
            LMCallInput(prompt=prompt, messages=messages, lm_kwargs=json_safe(kwargs)),
            result_type=LMCallOutput,
            start_to_close_timeout=self._options.start_to_close_timeout(),
            retry_policy=self._options.retry_policy(),
        )

        # Replicate dspy.LM.forward's usage-tracker side effect so the
        # outermost Module.acall (which wraps the run in track_usage()) can read
        # tokens back via prediction.get_lm_usage(). Key by the worker's request
        # model so the lm_usage shape matches coarse mode exactly.
        tracker = dspy.settings.usage_tracker
        if tracker is not None and out.usage:
            key = out.model or out.response_model or self.model
            tracker.add_usage(key, dict(out.usage))

        # The adapter's _call_postprocess parses these into output fields.
        return out.outputs

    # `acall` is the only entry DSPy's async path uses; the sync `__call__`
    # would run a blocking HTTP request inside the workflow, which must never
    # happen. Guard it loudly so a misconfigured (sync) program fails fast
    # instead of stalling the workflow task.
    def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover - guard
        raise RuntimeError(
            "WorkflowLM only supports the async path (use program.acall in the "
            "fine workflow); a synchronous LM call cannot run inside a workflow."
        )
