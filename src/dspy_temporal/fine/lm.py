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

Each ``WorkflowLM`` carries an :class:`LMSpec` describing its predictor's
*effective* LM (model id, capability flags, sampling kwargs). DSPy decides
whether to attach a structured ``response_format`` (``JSONAdapter``) and what
temperature to use *in the workflow, before the call* -- reading exactly these
attributes -- so the spec makes those decisions match the real worker LM. The
spec is resolved up front by the ``dspy_describe_lms`` activity (the model itself
never crosses the boundary; only its description does).

This module is loaded into the workflow via ``imports_passed_through`` (host
code), but it is defined-by-us, so it must stay replay-safe: no wall-clock, no
randomness -- only ``workflow.execute_activity`` and pure data shaping.
"""

from __future__ import annotations

from typing import Any

import dspy
from temporalio import workflow

from ..models import LMCallInput, LMCallOutput, LMSpec
from ..options import CallOptions
from ..serde import encode_lm_kwargs

LM_ACTIVITY_NAME = "dspy_lm_call"


class WorkflowLM(dspy.BaseLM):
    """A ``dspy.BaseLM`` whose every call is a ``dspy_lm_call`` activity.

    Carries an :class:`LMSpec` so it stands in faithfully for the worker LM the
    activity will actually run: same model id, same capability flags (so
    ``JSONAdapter`` branches correctly workflow-side), same sampling ``kwargs``
    (so ``Predict``'s n>1 temperature handling matches coarse mode).
    """

    def __init__(
        self,
        *,
        spec: LMSpec,
        lm_ref: str | None = None,
        program: str | None = None,
        options: CallOptions | None = None,
    ):
        super().__init__(model=spec.model, model_type=spec.model_type)
        # Mirror the real LM's sampling kwargs so workflow-side config derivation
        # (predict.py reads lm.kwargs for temperature / n) matches the activity.
        self.kwargs = dict(spec.kwargs)
        self._spec = spec
        self._lm_ref = lm_ref
        self._program = program
        self._options = options or CallOptions()

    # Capability flags are setterless @property on BaseLM, so we override them as
    # properties (not instance attributes) reading from the spec. JSONAdapter
    # reads these in the workflow to decide whether to attach a response_format.
    @property
    def supported_params(self) -> set[str]:
        return set(self._spec.supported_params)

    @property
    def supports_response_schema(self) -> bool:
        return self._spec.supports_response_schema

    @property
    def supports_function_calling(self) -> bool:
        return self._spec.supports_function_calling

    async def acall(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        out = await workflow.execute_activity(
            LM_ACTIVITY_NAME,
            LMCallInput(
                prompt=prompt,
                messages=messages,
                lm_kwargs=encode_lm_kwargs(kwargs),
                lm_ref=self._lm_ref,
                program=self._program,
            ),
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
