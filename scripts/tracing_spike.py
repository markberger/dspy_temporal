"""De-risk spike: does the Temporal activity span parent the DSPy spans?

The coarse activity runs sync in a ThreadPoolExecutor. OTel context is a
ContextVar that does NOT auto-propagate into executor threads, so the activity
span (current on the event-loop thread) may not be current in the worker thread.

This script runs the REAL coarse path (Temporal TracingInterceptor on client +
worker, a minimal DSPy callback emitting spans inside the activity) under two
executors and reports the resulting span tree / parenting:

  1. default ThreadPoolExecutor      (what build_worker uses today)
  2. context-copying ThreadPoolExecutor (proposed fix)

Run:  uv run python scripts/tracing_spike.py
"""

import asyncio
import contextvars
import uuid
from concurrent.futures import ThreadPoolExecutor

import dspy
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from dspy.utils.dummies import DummyLM
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import set_span_in_context
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

import dspy_temporal as dt
from dspy_temporal.coarse.activities import run_program_activity
from dspy_temporal.coarse.workflow import DSPyProgramWorkflow
from dspy_temporal.config import RunConfig
from dspy_temporal.converter import data_converter
from dspy_temporal.sandbox import PASSTHROUGH_MODULES

# --- OTel setup: global provider exporting to memory ------------------------
exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("spike")


class SpikeCallback(BaseCallback):
    """Minimal span-emitting callback: parents via DSPy's ACTIVE_CALL_ID map."""

    def __init__(self):
        self._spans = {}

    def _start(self, call_id, name):
        parent_id = ACTIVE_CALL_ID.get()
        parent_span = self._spans.get(parent_id)
        ctx = set_span_in_context(parent_span) if parent_span else None  # None -> current
        self._spans[call_id] = tracer.start_span(name, context=ctx)

    def _end(self, call_id):
        span = self._spans.pop(call_id, None)
        if span:
            span.end()

    def on_module_start(self, call_id, instance, inputs):
        self._start(call_id, f"dspy.module {type(instance).__name__}")

    def on_module_end(self, call_id, outputs, exception=None):
        self._end(call_id)

    def on_lm_start(self, call_id, instance, inputs):
        self._start(call_id, f"chat {getattr(instance, 'model', '?')}")

    def on_lm_end(self, call_id, outputs, exception=None):
        self._end(call_id)


class ContextCopyingExecutor(ThreadPoolExecutor):
    """Carries the caller's contextvars (incl. OTel context) into the worker thread."""

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(lambda: ctx.run(fn, *args, **kwargs))


def _runner():
    restr = SandboxRestrictions.default.with_passthrough_modules(
        *PASSTHROUGH_MODULES, "opentelemetry"
    )
    return SandboxedWorkflowRunner(restrictions=restr)


def _report(label):
    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    print(f"\n===== {label} =====")
    print(f"{'name':40} {'span_id':>18} {'parent_id':>18}  trace")
    for s in spans:
        pid = s.parent.span_id if s.parent else None
        print(f"{s.name:40} {s.context.span_id:>18x} {(pid or 0):>18x}  {s.context.trace_id:x}")

    # Worker-side activity span id(s). The root dspy span is the dspy.module
    # whose parent is one of them (not an inner module).
    activity_ids = {s.context.span_id for s in spans if "RunActivity" in s.name}
    dspy_modules = [s for s in spans if s.name.startswith("dspy.module")]
    lm_span = next((s for s in spans if s.name.startswith("chat ")), None)
    dspy_root = next(
        (s for s in dspy_modules if s.parent and s.parent.span_id in activity_ids), None
    )

    print("\n  analysis:")
    if not dspy_modules:
        print("  - NO dspy span emitted (callback didn't fire?)"); return
    if not activity_ids:
        print("  - NO Temporal activity span found"); return

    parented = dspy_root is not None
    lm_nested = lm_span and lm_span.parent and any(
        lm_span.parent.span_id == m.context.span_id for m in dspy_modules
    )
    unique = {s.name for s in spans}
    dup = len(spans) - len(unique)
    print(f"  - dspy root parented to activity span : {bool(parented)}")
    print(f"  - lm span nested under a dspy module  : {bool(lm_nested)}")
    print(f"  - duplicate-named spans (replay?)     : {dup}")
    verdict = "PASS (unified trace)" if (parented and lm_nested) else "FAIL"
    print(f"  => {verdict}")


async def run_variant(env, label, executor, register_on_worker):
    exporter.clear()
    tq = f"spike-{uuid.uuid4().hex[:8]}"
    # Per Temporal docs: register the interceptor on the CLIENT only; the worker
    # inherits it. Registering on both client AND worker double-emits spans.
    worker_interceptors = [TracingInterceptor()] if register_on_worker else []
    worker = Worker(
        env.client,
        task_queue=tq,
        workflows=[DSPyProgramWorkflow],
        activities=[run_program_activity],
        activity_executor=executor,
        workflow_runner=_runner(),
        interceptors=worker_interceptors,
    )
    async with worker:
        # Wrap the client call in a root span so the interceptor creates the
        # workflow/activity spans (always_create_workflow_spans=False by default).
        with tracer.start_as_current_span("client.run"):
            await dt.run_program(
                env.client, "qa", {"question": "sky?"}, task_queue=tq
            )
    _report(label)


async def main():
    dspy.configure(callbacks=[SpikeCallback()])
    dt.register_program("qa", lambda: dspy.ChainOfThought("question -> answer"))
    dt.set_worker_lm(DummyLM([{"reasoning": "r", "answer": "blue"}] * 10))

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter,
        interceptors=[TracingInterceptor()],
    ) as env:
        await run_variant(
            env, "interceptor on BOTH client+worker (anti-pattern)",
            ThreadPoolExecutor(max_workers=4), register_on_worker=True,
        )
        await run_variant(
            env, "interceptor on CLIENT only (per Temporal docs)",
            ThreadPoolExecutor(max_workers=4), register_on_worker=False,
        )


if __name__ == "__main__":
    asyncio.run(main())
