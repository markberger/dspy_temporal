# LLM Tracing Design (OpenTelemetry, dual-emit)

## Goal & decision

Capture LLM traces for DSPy programs running on Temporal using the open standard,
**OpenTelemetry**. Emit **dual-convention** attributes on every span:

- **OTel GenAI semantic conventions** (`gen_ai.*`) — vendor-neutral; renders in
  Langfuse, Grafana/Tempo, Honeycomb, Jaeger, Datadog.
- **OpenInference** (`openinference.span.kind`, `llm.*`) — first-class in Arize
  **Phoenix**.

Both namespaces coexist on one span (no conflict). We own the mapping in one
place so the cost is bounded. Tracing is an **optional extra**, off by default,
**zero-overhead** when unconfigured.

## Non-negotiable design principles (staff review)

1. **No monkeypatching.** We use only first-class extension points:
   DSPy's `dspy.settings.callbacks` (`BaseCallback`) and Temporal's
   `interceptors=` API. We explicitly reject `openinference-instrumentation-dspy`
   — it `wrapt`-patches `dspy.LM.__call__`/`Predict.forward`, which our fine-mode
   `WorkflowLM` replaces (→ no-op or double-count), and it records **no token
   usage**. We emit OpenInference attributes ourselves, better.
2. **Tracing can never break a run.** DSPy swallows callback exceptions
   (`callback.py:271,283`); span emission lives only in callbacks/activities.
3. **Replay-safe.** No span emission in workflow code (it is replayed). All
   emission is in activities (not replayed). Temporal's interceptor uses a
   replay-safe tracer.
4. **Lean core.** OTel is imported lazily, only when tracing is enabled; the core
   install and the workflow sandbox never import it.

## Instrumentation seam: a DSPy `BaseCallback`

Register one `DSPyOTelCallback` via `dspy.settings.callbacks` at worker startup.
DSPy fires paired hooks around every module / LM / tool / adapter call
(`callback.py`): `on_module_start/end`, `on_lm_start/end`, `on_tool_start/end`,
`on_adapter_*`. Confirmed facts that shape the implementation:

- `on_*_start(call_id, instance, inputs)` gets the **instance** and **inputs**
  (`inputs` = call kwargs, e.g. the LM's `messages`/`prompt`).
- `on_*_end(call_id, outputs, exception)` gets **outputs** + any **exception**,
  but **not** the instance.
- `ACTIVE_CALL_ID` (a `ContextVar`) holds the **parent** call_id at the moment
  `on_*_start` runs (it's reassigned to the current call_id only afterward,
  `callback.py:300‑304`).

### Span nesting (explicit, not ambient)

Maintain `self._spans: dict[call_id, Span]` (thread-safe). Parent each span via
the **explicit** parent call_id, not OTel's ambient context:

```
on_X_start(call_id, instance, inputs):
    parent_id   = ACTIVE_CALL_ID.get()
    parent_ctx  = set_span_in_context(self._spans[parent_id]) if parent_id in self._spans
                  else self._root_ctx              # top module → activity span
    span = tracer.start_span(name, context=parent_ctx, kind=INTERNAL)
    self._spans[call_id] = span
    set start attributes (request model, params, span kind, content if opted-in)

on_X_end(call_id, outputs, exception):
    span = self._spans.pop(call_id)
    set end attributes (response model, tokens, finish reasons, cost, content)
    if exception: span.record_exception(exception); span.set_status(ERROR)
    span.end()
```

Why explicit over ambient: it keys parenting off DSPy's own `ACTIVE_CALL_ID`
rather than OTel's ambient context, so a span parents to the DSPy call that
spawned it even when the two contexts diverge.

#### Concurrency & nesting (the `dspy.Parallel` limitation)

`ACTIVE_CALL_ID` is a **ContextVar**. It propagates correctly on the synchronous
path and across `asyncio.gather` — asyncio copies the contextvar context into each
Task (PEP 567), so a parent module's call_id reaches its concurrent children and
the spans nest. It does **not** propagate across `dspy.Parallel`, which runs items
on a `ThreadPoolExecutor`: the worker re-applies only DSPy's `thread_local_overrides`
and never `copy_context()`s (`dspy/utils/parallelizer.py`), so `ACTIVE_CALL_ID`
reverts to its `None` default in the worker thread. The shared `call_id → span`
dict still crosses the thread boundary fine — but its **key** is what's lost, so
each parallel item's sub-calls orphan into **new trace roots**.

Consequences and guidance:

- **Use the async interface for concurrency.** Concurrency expressed as
  `asyncio.gather` over `.acall()` nests correctly with no patching. The coarse
  activity runs programs via `program.acall` by default (falling back to the sync
  call for `forward`-only modules), so async-capable programs get a correct trace
  tree automatically. This is verified by a unit test (async fan-out → one trace)
  and the `dspy.Parallel` orphaning is locked by a characterization test.
- **`dspy.Parallel` (sync threadpool) orphans spans** — a documented limitation,
  not a regression. `dspy.context` carries DSPy *settings* into a `Parallel` block
  but cannot carry `ACTIVE_CALL_ID`, so it is not a workaround.
- **We do not monkeypatch** DSPy's parallel executor to fix this (it would patch
  internals, be brittle across versions, and is unnecessary given the async path).
  The permanent root-cause fix belongs upstream in DSPy: propagate the context into
  `ParallelExecutor` workers (`copy_context()` / re-set `ACTIVE_CALL_ID`).

## Attribute mapping (dual-emit)

Source of truth differs by datum:

| Datum | gen_ai | OpenInference | DSPy source |
|---|---|---|---|
| span kind | span name `chat {model}` + `gen_ai.operation.name` | `openinference.span.kind` = LLM/CHAIN/TOOL | hook type |
| request model | `gen_ai.request.model` | `llm.model_name` | `instance.model` (on_lm_start) |
| response model | `gen_ai.response.model` | (same key) | `instance.history[-1]["response_model"]` |
| input tokens | `gen_ai.usage.input_tokens` | `llm.token_count.prompt` | `history[-1]["usage"]` |
| output tokens | `gen_ai.usage.output_tokens` | `llm.token_count.completion` | `history[-1]["usage"]` |
| temp/top_p/max | `gen_ai.request.*` | `llm.invocation_parameters` | `instance.kwargs` |
| finish reasons | `gen_ai.response.finish_reasons` | — | `history[-1]["response"]` |
| cost (custom) | `gen_ai.usage.cost` | — | `history[-1]["cost"]` |
| content (opt-in) | events `gen_ai.{input,output}.messages` | `llm.input_messages.*`, `input.value` | on_lm_start `inputs`, outputs |
| error | `record_exception` + status | (same) | `exception` arg |

### The token-source wrinkle (honest limitation)

`on_lm_end` does **not** carry usage/model. We read them from the LM history
entry added during the call (capture `len(instance.history)` in `on_lm_start`,
read the new entries in `on_lm_end` via the stored instance). Consequences:

- Couples token/cost/model attributes to `disable_history=False`. Coarse mode
  keeps history on, so this is fine; the callback degrades gracefully to
  metadata-without-tokens if history is disabled.
- The history list lives on the LM **instance**, and the coarse worker shares one
  `worker_lm` across every activity running concurrently in the
  `ThreadPoolExecutor` (up to `max_concurrent_activities`). So under concurrency —
  whether from `dspy.Parallel` *within* a program or from independent jobs racing
  in separate activities — appends from different calls interleave in that shared
  list, and `history[start:]` can pick up another call's entry. We therefore only
  attribute usage/cost/model when **exactly one** entry was appended during the
  call (`len(new_entries) == 1`); otherwise we omit the response attributes rather
  than risk reporting the wrong call's tokens. The result is "missing under
  contention" instead of "wrong under contention" — the safer failure for a
  billing/observability signal. **Fine mode (shipped) fixes this entirely**: each
  LM call is its own activity running on an isolated `worker_lm.copy()`, so exactly
  one entry is appended to *that copy's* history and the response attributes are
  always attributed correctly.

## Temporal boundary: context propagation

Use Temporal's official `temporalio.contrib.opentelemetry.TracingInterceptor` on
**client + worker** (first-class, no patching). It creates the
StartWorkflow → Workflow → Activity spans and propagates trace context through
Temporal headers.

```
[client] StartWorkflow:DSPyProgram
└─ RunWorkflow:DSPyProgram
   └─ RunActivity:dspy_run_program          ← Temporal interceptor (activity span)
      └─ dspy.program <Name>  (CHAIN)        ← our callback (root dspy span)
         └─ chat <model>      (LLM)          ← our callback (gen_ai + OpenInference)
```

### ThreadPoolExecutor parenting — RESOLVED by spike (not a risk)

> Update: the coarse activity drives the program via DSPy's async path
> (`program.acall`) so in-program concurrency traces correctly — see "Concurrency &
> nesting". It does so with `asyncio.run(...)` on its own worker-pool thread while
> staying a **synchronous** activity (so the heartbeat watchdog — which beats from a
> daemon thread — keeps working, and a sync-only program can't block the worker's
> shared event loop). The throwaway loop's contextvar context still carries
> `ACTIVE_CALL_ID` into `asyncio.gather` children. So the spike below (the
> sync-activity `ThreadPoolExecutor` case) applies directly, and the integration
> test still guards the parenting.

Original concern: the coarse activity runs sync in a `ThreadPoolExecutor`, and OTel
context is a `ContextVar` that doesn't auto-propagate into executor threads — so
the activity span might not be current in the worker thread, orphaning the dspy
spans.

**Spike result (a throwaway script, since removed; see commit `ebd91d5` and
`tests/integration/test_tracing_workflow.py`): the default executor already works.**
Temporal's SDK propagates the contextvar context into the sync-activity worker
thread, so a span created inside the activity (root dspy span, `context=None`
→ current) parents to the Temporal `RunActivity` span and yields one unified
trace:

```
client.run → StartWorkflow → RunWorkflow → StartActivity
  → RunActivity:dspy_run_program
     → dspy.module ChainOfThought → dspy.module Predict → chat <model>
```

A `ContextCopyingExecutor` (override `submit` with `copy_context().run`) was
tested too and behaves identically — so **we do NOT add it**; the existing
`build_worker` executor is sufficient. We keep an integration test asserting the
activity span parents the dspy spans, as a guard against future Temporal changes.

**Duplicate spans — root-caused, not a Temporal bug.** The doubled Temporal spans
in the first spike were caused by registering the interceptor on **both** the
client and the worker — the documented anti-pattern (package README,
Troubleshooting #3 + Best Practice #4: *"Never register the same
plugin/interceptor on both client and worker"*). Re-running with the interceptor
on the **client only** → **0 duplicates**, parenting still PASS. **Rule: attach
the interceptor/plugin at the client; the worker inherits it. `build_worker` must
NOT re-add it.**

### Interceptor vs Plugin (Temporal ships both)

| | `TracingInterceptor` (legacy) | `OpenTelemetryPlugin` (new, "recommended") |
|---|---|---|
| Workflow span duration | zero-duration | accurate |
| TracerProvider | any | must use `create_tracer_provider()` (replay-safe) |
| OTel APIs inside workflow | no | yes |
| API stability | stable | experimental |
| Register | client only | client only |

**Recommendation: `TracingInterceptor` for v1.** We emit all DSPy spans in
**activities**, not workflow code, so the plugin's headline advantage (accurate
workflow-span durations + OTel APIs in workflows) doesn't benefit us, while the
interceptor is stable and works with the user's **existing** TracerProvider (no
forced `create_tracer_provider()`). Revisit the plugin if we ever want
workflow-internal spans. Either way, register on the client only.

## Privacy

- Default **metadata only** (model, tokens, finish reasons, timing) — **no
  message text**.
- Content is **opt-in**: `setup_tracing(capture_content=...)` overriding the OTel
  env var `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
- When on, gen_ai content is emitted as **span events** (collector-droppable);
  OpenInference content as span attributes. Content dual-emit is Phase B (the two
  models diverge most here); metadata dual-emit ships in Phase A.

## Packaging, config, wiring

New subpackage `src/dspy_temporal/tracing/`:
- `__init__.py` — `setup_tracing()` (builds `TracerProvider`+OTLP exporter, returns
  the `TracingInterceptor`, registers the callback). Lazy OTel imports.
- `callback.py` — `DSPyOTelCallback(BaseCallback)`.
- `semconv.py` — the single dual-emit attribute mapping (tracks the evolving
  gen_ai spec in one file).
- `config.py` — content-capture / env-var resolution.

Optional dependency group:
```toml
[project.optional-dependencies]
tracing = ["opentelemetry-sdk>=1.27", "opentelemetry-exporter-otlp-proto-grpc>=1.27"]
```
Touchpoints (registration model matters — see duplicate-spans finding):
- **`connect()`** attaches the `TracingInterceptor` to the **client** (the single
  registration point). Workers built from that client inherit it.
- **`build_worker`** registers only the **DSPy callback** (`dspy.settings.callbacks`)
  at worker startup. It must **not** add the interceptor (double-registration →
  duplicate spans).
- **`sandbox.py`** adds `opentelemetry` to passthrough (used only in
  activities/interceptor). Guard all OTel imports so a core install never loads
  them.

`setup_tracing(...)` ties it together: builds/uses a `TracerProvider` + OTLP
exporter, returns the interceptor for `connect()`, and registers the callback.

## Testing

- **Unit (no network):** OTel `InMemorySpanExporter` + `DummyLM`; assert span
  names, hierarchy (program→lm), `gen_ai.*` **and** `llm.*` attributes incl. token
  counts; assert no content by default, content present when opted in.
- **Retry/error:** force an activity failure; assert span status ERROR + recorded
  exception; assert a retried attempt is a distinct span tree tagged with
  `temporal.activity.attempt`.
- **Integration:** `WorkflowEnvironment` + in-memory exporter + interceptor on
  client & worker; assert the activity span **parents** the DSPy spans (validates
  the executor bridge).
- Keep tracing in its own subpackage; run its tests with the `tracing` extra so
  the existing 90% gate isn't muddied by optional code.

## Phases

- **A (SHIPPED):** callback + semconv metadata mapping (dual-emit) +
  `setup_tracing` + interceptor wiring (client-only) + sandbox passthrough +
  unit/integration tests. Coarse mode. Includes a basic opt-in content path.
  No context-copying executor needed (spike showed the default works).
- **B:** content capture (dual-emit, opt-in), tool/adapter spans, cost, metrics
  (`gen_ai.client.token.usage` histograms).
- **C (SHIPPED):** fine mode — per-LM-call (and per-tool-call) activity spans,
  workflow-as-parent; no span emission in workflow code (the workflow runs the
  module with `callbacks=[]`; spans are emitted only inside the activities). Each
  LM call runs on an isolated `worker_lm.copy()`, so its `history[-1]` is
  unambiguous and the usage/cost/model attributes are always attributed — the
  concurrency wrinkle above does not apply. See `src/dspy_temporal/fine/` and
  `tests/integration/test_fine_tracing_workflow.py`.

## Open question for later

- gen_ai semconv is still "Development" (semconv 1.40, 2026); attribute names
  churn. We pin our mapping in `semconv.py` and opt into
  `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` — accept periodic
  rename maintenance.
