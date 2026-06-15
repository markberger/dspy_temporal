"""Declare a program reference, then bind a ``dspy.Module`` to it on the worker.

The surface is split into a cheap *declaration* and an explicit *binding*:

- ``program(name, *, mode=..., options=..., activity_task_queue=..., result=...)``
  constructs an immutable :class:`TemporalProgram` reference. It is pure -- no
  registry mutation, no model load, no I/O -- so a workflow file and a thin client
  can import it normally (no ``imports_passed_through`` dance) and the workflow
  class stays cheap to import.
- ``ref.bind(impl)`` registers the heavy implementation (a live ``dspy.Module`` or
  a zero-arg builder) under the ref's name. This is the only side-effecting step,
  and it belongs on the worker.

The reference is the single source of truth for *how* the program runs:

- ``await ref.run(**inputs)`` -- inside a user-authored ``@workflow.defn`` it
  dispatches our activities inline (compose a program into your own workflow),
  carrying the ref's ``options`` (timeout/retry) and routing the activity to
  ``activity_task_queue`` when set; outside any workflow it degrades to a plain
  local DSPy call. When ``result`` is set, the returned ``dspy.Prediction`` is
  passed through that adapter so workflow code speaks the typed model, not dspy.
- ``await ref.start(client, *, task_queue=..., **inputs)`` -- start the program as
  a standalone workflow from a client (the by-name escape hatch ``run_program``).

Per-call tweaks use fluent, copy-returning helpers so ``run(**inputs)`` stays pure
inputs: ``ref.with_options(CallOptions(...))`` and ``ref.on_task_queue("gpu")``
each return a modified copy of the frozen ref.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from temporalio import workflow

from ..client import run_program
from ..config import RunMode, run_program_async_or_sync
from ..execute import execute_coarse, execute_fine
from ..options import CallOptions
from ..registry import ModuleSource, default_registry, register_program

if TYPE_CHECKING:
    import dspy


@dataclass(frozen=True, kw_only=True)
class TemporalProgram:
    """An immutable reference to a deployed program.

    Carries everything a workflow needs to *dispatch* the program -- its ``name``,
    its ``mode`` (coarse = whole-program activity, fine = per-call activities), the
    default ``options`` (timeout/retry) applied to the activity, an optional
    ``activity_task_queue`` to route the LM-heavy activity to a dedicated pool, and
    an optional ``result`` adapter that shapes the ``dspy.Prediction`` into a typed
    value. Constructed by :func:`program`; the implementation is attached
    separately on the worker via :meth:`bind`.
    """

    name: str
    mode: RunMode = RunMode.COARSE
    options: CallOptions | None = None
    # Route the program activity to a dedicated queue (the "cheap workflow workers
    # + dedicated activity pool" split). ``None`` co-locates it with the calling
    # workflow's queue. Coarse mode only for now -- ignored in fine mode, whose
    # per-call activities co-locate.
    activity_task_queue: str | None = None
    # Explicit ``dspy.Prediction -> T`` adapter. When set, :meth:`run` returns
    # ``result(prediction)`` (typically a pydantic model), keeping dspy out of
    # workflow code. When ``None``, ``run`` returns the raw ``Prediction``.
    result: Callable[[Any], Any] | None = None

    def bind(self, impl: ModuleSource) -> TemporalProgram:
        """Register ``impl`` under this ref's name and return ``self``.

        ``impl`` is a live ``dspy.Module`` (e.g. a compiled program with few-shot
        demos: its prototype stays in worker memory and each run gets a fresh,
        LM-stripped clone) or a zero-arg builder. This is the heavy, side-effecting
        step -- call it on the worker, never in a workflow file (the registry
        guards against an import-time bind inside the sandbox). The ref's ``mode``
        is recorded so a by-name ``run_program`` resolves the same mode.
        """
        register_program(self.name, impl, mode=self.mode)
        return self

    def with_options(self, options: CallOptions) -> TemporalProgram:
        """Return a copy of this ref with ``options`` overridden for the next run."""
        return replace(self, options=options)

    def on_task_queue(self, task_queue: str) -> TemporalProgram:
        """Return a copy routing the program activity to ``task_queue``."""
        return replace(self, activity_task_queue=task_queue)

    async def run(self, **inputs) -> Any:
        """Run the program, dispatching by execution context.

        Inside a workflow (a user's own ``@workflow.defn`` that awaits this) the
        call dispatches our activities inline via ``execute_coarse`` /
        ``execute_fine``, carrying this ref's ``options`` and (coarse only)
        ``activity_task_queue``. Outside any workflow it degrades to a plain
        in-process DSPy call against the locally configured LM (no worker-LM
        injection -- ``start`` is the path that uses the worker).

        When ``self.result`` is set, the ``dspy.Prediction`` is passed through it
        and the adapted value is returned; otherwise the raw ``Prediction``.
        """
        if workflow.in_workflow():
            if self.mode == RunMode.FINE:
                pred = await execute_fine(self.name, inputs, self.options)
            else:
                pred = await execute_coarse(
                    self.name,
                    inputs,
                    self.options,
                    task_queue=self.activity_task_queue,
                )
        else:
            # In-process degrade: build from the registry and run locally.
            program = default_registry().build(self.name)
            if self.mode == RunMode.FINE:
                pred = await program.acall(**inputs)
            else:
                # Coarse: prefer the async path (so concurrent sub-calls trace
                # correctly), falling back to the sync call for forward-only modules.
                pred = await run_program_async_or_sync(program, inputs)
        return self.result(pred) if self.result is not None else pred

    async def start(
        self,
        client,
        /,
        *,
        task_queue: str,
        workflow_id: str | None = None,
        options: CallOptions | None = None,
        **inputs,
    ) -> dspy.Prediction:
        """Start this program as a standalone workflow and await its result.

        Delegates to :func:`dspy_temporal.run_program` (the by-name escape hatch)
        using this ref's ``mode``. ``task_queue`` is required -- the reference is
        about composing the program *inside your own workflow*; starting it as its
        own workflow needs an explicit serving queue.

        Program inputs are passed as keywords (``ref.start(client, task_queue=q,
        question=...)``). ``client`` is positional-only so an input field may be
        named ``client``; ``task_queue`` / ``workflow_id`` / ``options`` are
        reserved control knobs, so a program needing inputs by those names must use
        :func:`dspy_temporal.run_program` (it takes inputs as an explicit dict).
        """
        return await run_program(
            client,
            self.name,
            inputs,
            task_queue=task_queue,
            workflow_id=workflow_id,
            options=options,
            mode=self.mode,
        )


def program(
    name: str,
    *,
    mode: RunMode = RunMode.COARSE,
    options: CallOptions | None = None,
    activity_task_queue: str | None = None,
    result: Callable[[Any], Any] | None = None,
) -> TemporalProgram:
    """Declare a program reference. Pure: no registry mutation, no model load, no I/O.

    Safe to import from a workflow file and a thin client. Attach the
    implementation on the worker with ``ref.bind(impl)``. ``mode`` selects coarse
    vs. fine; ``options`` sets the default activity timeout/retry; pass
    ``activity_task_queue`` to route the LM-heavy activity to a dedicated pool
    (coarse mode); pass ``result`` to adapt the ``dspy.Prediction`` into a typed
    value so workflow code never touches dspy.
    """
    return TemporalProgram(
        name=name,
        mode=mode,
        options=options,
        activity_task_queue=activity_task_queue,
        result=result,
    )
