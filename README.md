# dspy-temporal

[![lint](https://github.com/markberger/dspy_temporal/actions/workflows/lint.yml/badge.svg)](https://github.com/markberger/dspy_temporal/actions/workflows/lint.yml)
[![test](https://github.com/markberger/dspy_temporal/actions/workflows/test.yml/badge.svg)](https://github.com/markberger/dspy_temporal/actions/workflows/test.yml)

**Run [DSPy](https://dspy.ai) programs on [Temporal](https://temporal.io) as durable
workflows — with retries, timeouts, and tracing — without writing Temporal code.**

You write a normal `dspy.Module`. `dspy-temporal` wraps it in a Temporal workflow so a
crash mid-run resumes instead of starting over, transient LM errors retry on a policy you
control, and every run emits an OpenTelemetry trace — all behind a small API.

```python
import dspy
import dspy_temporal as dt

qa = dt.program("qa")                                          # a lightweight reference
qa.bind(lambda: dspy.ChainOfThought("question -> answer"))     # attach the impl (on the worker)

pred = await qa.start(client, task_queue="dspy-temporal", question="Why is the sky blue?")
print(pred.answer)
```

## Why

- **No Temporal boilerplate.** No workflow/activity classes to author — `program()` declares a
  reference, `bind()` attaches the implementation on the worker, and you `start()` or compose it.
- **Durable by default.** Runs survive worker crashes and restarts. In fine mode, completed
  LM and tool calls are recorded in history and never re-charged on replay.
- **Retries & timeouts you control.** Per-run `CallOptions` set attempt counts, backoff, and
  timeouts; non-retryable errors (e.g. context-window overflow) are excluded by default.
- **Secrets never cross the wire.** Only the program *name* and call *inputs* enter Temporal
  history — never a live LM or API key. The LM is configured on the worker.
- **First-class observability.** One OpenTelemetry trace per run, dual-emitted as gen_ai
  semantic conventions *and* OpenInference (Arize Phoenix), with token usage and cost.
- **Bring your own workflow.** The `program()` reference is dspy-free and side-effect-free, so
  your `@workflow.defn` imports it with a plain `import` (no passthrough dance), composes it with
  `await ref.run(...)`, and gets back a typed result — dspy never enters your workflow code.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Concepts](#concepts)
- [Execution modes: coarse vs. fine](#execution-modes-coarse-vs-fine)
- [Configuring the LM](#configuring-the-lm)
- [Tuning retries & timeouts](#tuning-retries--timeouts)
- [Recipes](#recipes)
  - [Bind a compiled / live module](#bind-a-compiled--live-module)
  - [Compose a program inside your own workflow](#compose-a-program-inside-your-own-workflow)
  - [Fine-grained mode (per-call activities)](#fine-grained-mode-per-call-activities)
  - [Wire your own worker with the plugin](#wire-your-own-worker-with-the-plugin)
- [Tracing](#tracing)
- [Run locally with Docker Compose](#run-locally-with-docker-compose)
- [API reference](#api-reference)
- [Development](#development)

## Install

`dspy-temporal` isn't on PyPI yet — install it from Git:

```bash
pip install "git+https://github.com/markberger/dspy_temporal"
```

For LLM tracing (OpenTelemetry), add the `tracing` extra:

```bash
pip install "dspy-temporal[tracing] @ git+https://github.com/markberger/dspy_temporal"
```

> Once published, `pip install dspy-temporal` (and `dspy-temporal[tracing]`) will be the
> install line; the Git URL is the interim path.

You'll also need a running Temporal server. For local development, the
[Docker Compose stack](#run-locally-with-docker-compose) brings one up for you; otherwise
the [Temporal CLI](https://docs.temporal.io/cli) (`temporal server start-dev`) is the
quickest path.

## Quickstart

Running a program has three pieces: a **program** (your `dspy.Module`), a **worker** (the process
that runs it), and a **starter** (kicks off a run). The snippets below are plain Python —
run them with `python your_script.py`.

**1. Declare a program reference** (`program.py`):

```python
import dspy_temporal as dt

qa = dt.program("qa")
```

`program()` returns a lightweight, immutable **reference** — just a name (plus optional
`mode`, `options`, and a typed `result` adapter). It registers nothing and loads no model, so
this module is safe to import from anywhere (your workflow, a thin client). The
implementation is attached separately, on the worker.

**2. Run a worker** (`worker.py` — binds the implementation, then serves):

```python
import asyncio
import dspy
import dspy_temporal as dt
from program import qa

async def main():
    dt.configure_lm_from_env()                       # DSPY_LM_MODEL + provider keys from env
    qa.bind(lambda: dspy.ChainOfThought("question -> answer"))   # attach the impl here
    client = await dt.connect("localhost:7233")
    worker = dt.build_worker(client, task_queue="dspy-temporal")
    await worker.run()

asyncio.run(main())
```

`bind()` takes a **zero-arg builder** that returns a fresh `dspy.Module` (or a live instance —
see [below](#bind-a-compiled--live-module)). It's the heavy, side-effecting step that
populates the worker's registry, so it belongs on the worker — never in a workflow file.

Point the worker at its LM via environment variables:

```bash
export DSPY_LM_MODEL=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

**3. Start a run** (`run.py`):

```python
import asyncio
import dspy_temporal as dt
from program import qa     # the reference from program()

async def main():
    client = await dt.connect("localhost:7233")
    pred = await qa.start(client, task_queue="dspy-temporal", question="Why is the sky blue?")
    print(pred.answer)

asyncio.run(main())
```

A runnable version of this flow lives in `examples/` (`qa_program.py`, `worker.py`,
`run.py`).

## Concepts

- **Declaration vs. implementation.** `program(name, ...)` is the *declaration* — a pure,
  immutable reference the workflow and a thin client import. `ref.bind(impl)` is the
  *implementation* — the heavy, side-effecting registration the worker does. Splitting them is
  what lets a workflow file import the reference cheaply and safely.
- **Builder, not instance.** You bind a zero-arg callable that returns a fresh `dspy.Module`.
  Each run builds its own program, so there's no shared mutable state and nothing live to
  serialize. (You can also [bind a compiled instance](#bind-a-compiled--live-module).)
- **The boundary is name + inputs.** Only the program name and call inputs cross into
  Temporal. The LM — and its API key — is configured on the worker from its environment and
  never enters workflow history.
- **The reference carries the run knobs.** A `TemporalProgram` knows its `mode`, default
  `options`, optional `activity_task_queue`, and `result` adapter. `ref.run(**inputs)` composes
  it inside your own workflow; `ref.start(client, task_queue=..., **inputs)` runs it as a
  standalone workflow. Per-call tweaks return a modified copy:
  `ref.with_options(...)` / `ref.on_task_queue(...)`.
- **Predictions round-trip (or your typed result).** Without `result`, `run`/`start` return a
  `dspy.Prediction` with `get_lm_usage()` intact; with a `result` adapter, `run`/`start` return
  your own type so caller code speaks pydantic, not dspy.

## Execution modes: coarse vs. fine

The same worker serves both modes; you pick per program with `mode=` on `program()`.

| | **Coarse** (default) | **Fine** |
|---|---|---|
| What runs in an activity | the whole `dspy.Module` | each LM call and each tool call |
| Where orchestration runs | inside the activity | in the workflow (deterministic replay) |
| Durability granularity | job-level: a crash re-runs the program | step-level: completed LM/tool calls survive a crash and aren't re-charged |
| Tracing | `Workflow → Activity → dspy.module → chat <model>` | per-call spans, each LM call on an isolated copy with exact token usage |
| Best for | "just run it" — full DSPy fidelity, lowest friction | long/agentic runs (e.g. ReAct) where resuming mid-run matters |
| Opt in | `mode=dt.RunMode.COARSE` (default) | `mode=dt.RunMode.FINE` |

**Coarse mode** keeps DSPy fully intact — adapters, caching, in-program concurrency — and is
the low-friction path. **Fine mode** turns each step into its own activity so long runs
resume from the last completed step instead of re-calling the model; see
[Fine-grained mode](#fine-grained-mode-per-call-activities) for usage and its limits.

## Configuring the LM

The LM lives only on the worker. The standard path reads it from the environment:

```python
dt.configure_lm_from_env()                 # reads DSPY_LM_MODEL, provider keys via litellm
dt.configure_lm_from_env("openai/gpt-4o")  # or pass the model id explicitly
```

To supply a pre-built `dspy.LM` (custom client, headers, etc.), use `set_worker_lm`:

```python
dt.set_worker_lm(dspy.LM("openai/gpt-4o-mini", temperature=0.0))
```

Programs that bind their own per-predictor `.lm` in the builder keep it; the worker LM is
the default applied only to predictors that don't carry one.

## Tuning retries & timeouts

Set defaults on the reference (`program(..., options=...)`), or override per run. For
`start()`, pass `options=`; inside a workflow, use the fluent copy `ref.with_options(...)`:

```python
# default for every run of this program:
qa = dt.program("qa", options=dt.CallOptions(maximum_attempts=5))

# one-off override at start:
pred = await qa.start(
    client,
    task_queue="dspy-temporal",
    question="Why is the sky blue?",
    options=dt.CallOptions(maximum_attempts=5, start_to_close_timeout_seconds=120),
)

# one-off override when composing in a workflow:
pred = await qa.with_options(dt.CallOptions(start_to_close_timeout_seconds=900)).run(
    question="Why is the sky blue?"
)
```

| Field | Default | Meaning |
|---|---|---|
| `start_to_close_timeout_seconds` | `300.0` | per-attempt activity timeout |
| `maximum_attempts` | `3` | total attempts before failing the run |
| `initial_interval_seconds` | `1.0` | first retry backoff |
| `backoff_coefficient` | `2.0` | backoff multiplier |
| `maximum_interval_seconds` | `60.0` | backoff ceiling |
| `heartbeat_timeout_seconds` | `None` | when set, the activity self-heartbeats at ~⅓ this interval |
| `non_retryable_error_types` | `["ContextWindowExceededError"]` | errors that fail fast instead of retrying |

## Recipes

### Bind a compiled / live module

`bind()` also accepts a **live `dspy.Module` instance** — handy for a program optimized with a
DSPy teleprompter, whose predictors carry few-shot demos. The instance stays in worker memory
as a prototype; each run gets a fresh, LM-stripped `deepcopy` (demos preserved, no bound LM or
API key ever serialized):

```python
compiled = dspy.ChainOfThought("question -> answer")   # or a compiled program with demos
qa = dt.program("qa")
qa.bind(compiled)                                      # on the worker
```

Runnable example: `examples/instance_program.py`.

### Compose a program inside your own workflow

This is the path the declaration/implementation split is built for. Because `program()` is a
pure reference, three roles each import exactly what they need:

**`refs.py` — the shared, side-effect-free reference** (safe to import from the workflow
*and* a thin client):

```python
import dspy_temporal as dt

agent = dt.program("compose_qa")   # no registry mutation, no model load, no dspy import
```

**`workflow.py` — your workflow** imports the reference with a *plain* `import` (no
`imports_passed_through()` dance) and calls `await agent.run(...)`:

```python
from temporalio import workflow
from refs import agent

@workflow.defn
class ResearchWorkflow:
    @workflow.run
    async def run(self, question: str) -> str:
        first = await agent.run(question=question)          # one durable activity step
        followup = await agent.run(question=f"Summarize in one word: {first.answer}")
        return followup.answer
```

**`worker.py` — binds the implementation** and serves both the program and your workflow:

```python
agent.bind(lambda: dspy.ChainOfThought("question -> answer"))
worker = dt.build_worker(client, task_queue="dspy-temporal",
                         extra_workflows=[ResearchWorkflow])
```

`agent.run()` is **context-aware**: inside a `@workflow.defn` it dispatches our activity inline
(interleave it with timers, activities, child workflows as one replayable execution); outside
any workflow it degrades to a plain local DSPy call. Because the workflow class is now cheap to
import, a thin client starts it **type-safely** —
`await client.start_workflow(ResearchWorkflow.run, question, id=..., task_queue=...)`.

**Typed results.** Give the reference a `result` adapter so `run()` (composing in your workflow)
and `start()` (standalone) both return your own type and dspy never leaks into caller code
(validation, e.g. a confidence clamp, lives on the model):

```python
agent = dt.program("compose_qa", result=lambda p: Answer(text=str(p.answer)))
```

**Start now, poll later.** `ref.start()` awaits the result. When a caller can't hold the
connection open for the length of an LM run (a web request, a dashboard), use
`ref.start_nowait(...)` instead: it returns a Temporal `WorkflowHandle` immediately. Drive
`handle.describe()` / `await handle.result()` on your own schedule, and decode the result with
`ref.result_of(handle)` — which re-applies the same `result` adapter, so you still get your
typed value (just deferred). No hand-authored `@workflow.defn` needed; the program *is* the
workflow.

```python
handle = await agent.start_nowait(client, task_queue="dspy-temporal", question="Why is the sky blue?")
# ... return to the caller now; later (even another request — re-obtain the handle by id):
handle = client.get_workflow_handle(handle.id)
answer = await agent.result_of(handle)        # -> your Answer type (or a dspy.Prediction)
```

A thin client that never imported the program reference has the by-name pair
`start_program_nowait(client, name, inputs, *, task_queue, mode=...)` (parallel to
`run_program`) and `dspy_temporal.client.prediction_of(handle)`.

**Dedicated activity pool.** `program(..., activity_task_queue="gpu-pool")` (or
`agent.on_task_queue("gpu-pool")`) routes the LM-heavy activity to its own worker pool while
your workflow stays on the cheap queue — in coarse mode the single program activity, in fine
mode every per-call `dspy_lm_call` / `dspy_tool_call`. Runnable example (also demonstrating the
typed `result` adapter above — it returns a pydantic `Answer`): `examples/compose_refs.py`
+ `examples/compose_program.py` + `examples/run_compose.py`.

### Fine-grained mode (per-call activities)

Opt in per program with `mode=dt.RunMode.FINE`. Tools are ordinary Python functions you hand
to `dspy.ReAct` in the builder — there's no fine-mode-specific tool API:

```python
import dspy
import dspy_temporal as dt

def get_weather(city: str) -> str:
    """Return a weather report for a city."""   # body runs in an activity → real I/O is fine
    return f"The weather in {city} is sunny."

def build_agent() -> dspy.Module:
    # The builder runs in the WORKFLOW: construct dspy objects only — no network/file/DB here.
    return dspy.ReAct("question -> answer", tools=[get_weather])

weather_agent = dt.program("weather_agent", mode=dt.RunMode.FINE)
weather_agent.bind(build_agent)               # on the worker
```

Run it the usual way — `await weather_agent.start(client, task_queue="dspy-temporal",
question=...)`. In the Temporal UI you'll see distinct `dspy_lm_call` / `dspy_tool_call`
activities per run. Runnable example: `examples/react_program.py`, `examples/run_react.py`.

**Where each piece runs:** tool *bodies* and LM HTTP calls run in activities (real I/O
allowed); the builder, the ReAct loop, and adapter format/parse run in the workflow as
deterministic Python. Tool args arrive JSON-native and are coerced to the annotated types;
return values are JSON-ified, so tools should return JSON-native data or pydantic models (not
live handles). Both sync and async tools work.

**Multiple LMs & structured outputs are supported.** Bind a predictor's own `.lm` in the
builder (`self.summarize.lm = dspy.LM("openai/gpt-4o")`); a one-shot `dspy_describe_lms`
activity carries each LM's model + capabilities to the workflow up front, while the
credentials stay on the worker. For `JSONAdapter`, configure it on the worker once
(`dspy.configure(adapter=dspy.JSONAdapter())`) — its structured `response_format` crosses the
boundary as a JSON schema.

**Limitations** (use coarse mode if you need these):

1. **Sequential async only** — programs that fan out internally (`dspy.Parallel`, threads,
   `asyncio.gather`) aren't supported in the workflow.
2. **No ReAct context-window-truncation fallback** — a `ContextWindowExceededError` becomes a
   Temporal `ActivityError` across the boundary, so ReAct's truncate-and-retry won't trigger.
3. **Tools resolved via `program.tools`** — covers ReAct and any module exposing a `.tools`
   dict; a custom tool-calling module without `.tools` isn't supported yet.

### Wire your own worker with the plugin

If you already construct your own `temporalio.worker.Worker` (custom interceptors, tuning,
your own workflows/activities), add DSPy support with `DSPyPlugin` instead of `build_worker`:

```python
from temporalio.worker import Worker
import dspy_temporal as dt

worker = Worker(client, task_queue="dspy-temporal", plugins=[dt.DSPyPlugin()])
```

`DSPyPlugin` is a **combined client + worker plugin**. Passed to the **client** it installs
the pydantic data converter *and* propagates to any `Worker` built from that client — so over
a vanilla `Client.connect()` you'd pass it there to keep pydantic models round-tripping:

```python
client = await Client.connect("localhost:7233", plugins=[dt.DSPyPlugin()])
worker = Worker(client, task_queue="dspy-temporal")   # DSPy set added automatically
```

(`dt.connect()` already sets the converter, so over it the plugin is worker-only.) Apply it on
the client **or** a `Worker`/`Replayer` — not both. The plugin **extends** (never overwrites)
anything you pass explicitly; add your own workflows with
`DSPyPlugin(extra_workflows=[ResearchWorkflow])` and extra sandbox-passthrough prefixes with
`DSPyPlugin(extra_passthrough_modules=("my_pkg",))`. Runnable example:
`examples/worker_plugin.py`.

## Tracing

Capture LLM traces with OpenTelemetry — dual-emitted as both **gen_ai** semantic conventions
(Langfuse, Grafana/Tempo, Honeycomb, Datadog) and **OpenInference** (Arize Phoenix). Install
the extra, call `setup_tracing` once, and pass the returned interceptor to the **client** (the
worker inherits it):

```python
import dspy_temporal as dt
from dspy_temporal.tracing import setup_tracing

interceptor = setup_tracing(service_name="qa-worker")   # OTLP by default; reads OTEL_EXPORTER_OTLP_*
client = await dt.connect("localhost:7233", interceptors=[interceptor])
worker = dt.build_worker(client, task_queue="dspy-temporal")
await worker.run()
```

You get one trace per run. In **coarse** mode: `Workflow → Activity → dspy.module → chat
<model>`. In **fine** mode the tree follows the per-call activities (`dspy_lm_call`,
`dspy_tool_call`). LM spans carry token usage, model, finish reasons, and cost. Prompt and
completion **content is off by default**; enable it with `setup_tracing(capture_content=True)`
or `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.

Buffered spans are force-flushed when the worker stops gracefully, so the last activities'
spans aren't lost (a `BatchSpanProcessor` otherwise only flushes on its timer / `atexit`).
Pass `setup_tracing(flush_on_worker_stop=False)` if you manage flushing for a provider you
reuse elsewhere.

> **Register the interceptor on the client only** — adding it to the worker too double-emits
> spans.

> **`dspy.Parallel` is not traced.** Span nesting follows DSPy's `ACTIVE_CALL_ID` contextvar,
> which propagates on the sync path and across `asyncio.gather` but **not** across
> `dspy.Parallel` (its `ThreadPoolExecutor` doesn't copy contextvars), so parallel sub-calls
> orphan into separate trace roots. For a correct trace tree under concurrency, fan out with
> `asyncio.gather` over `module.acall` instead of `dspy.Parallel`; the coarse activity runs
> programs via `acall` by default, so this works out of the box. (The permanent fix is an
> upstream DSPy `copy_context()` in its parallel executor.)

## Run locally with Docker Compose

`docker-compose.yml` brings up a Temporal dev server, an
[Arize Phoenix](https://phoenix.arize.com) UI, and a traced worker for the example `qa`
program — so you can fire a DSPy program at `gpt-5-nano` and watch the trace.

```bash
cp .env.example .env            # then put a real OPENROUTER_API_KEY in .env
docker compose up --build       # temporal (7233/8233), phoenix (6006/4317), worker
```

With the stack up, start a run from the host. Point the starter at Phoenix so it emits the
root `StartWorkflow` span and propagates context — that's what ties the worker spans and the
DSPy spans into a **single** trace:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
    uv run --extra tracing python examples/run.py "Why is the sky blue?"
```

- **Phoenix UI:** http://localhost:6006 — `Workflow → Activity → dspy.module → chat
  openrouter/openai/gpt-5-nano`, with token usage, model, and (content capture is on in
  compose) the prompt/completion text.
- **Temporal UI:** http://localhost:8233 — the workflow execution.

The Temporal dev server uses an in-memory store, so workflow history resets when the stack
restarts.

## API reference

Everything below is importable as `dt.X` (`import dspy_temporal as dt`).

| Symbol | Summary |
|---|---|
| `program(name, *, mode=RunMode.COARSE, options=None, activity_task_queue=None, result=None)` | Declare an immutable program reference (pure: no registration, no I/O). |
| `TemporalProgram` | The reference `program()` returns. `.bind(impl)` registers a builder/live module (worker-side); `.run(**inputs)` composes it in your workflow; `.start(client, *, task_queue, **inputs)` runs it standalone and awaits; `.start_nowait(...)` runs it standalone and returns a `WorkflowHandle` (decode later with `.result_of(handle)`); `.with_options(...)` / `.on_task_queue(...)` return modified copies. |
| `run_program(client, name, inputs, *, task_queue, workflow_id=None, options=None, mode=None)` | Low-level by-name start, awaited (a thin client must pass `mode`). |
| `start_program_nowait(client, name, inputs, *, task_queue, workflow_id=None, options=None, mode=None)` | Low-level by-name non-blocking start → `WorkflowHandle`; decode with `dspy_temporal.client.prediction_of(handle)`. |
| `build_worker(client, *, task_queue, max_concurrent_activities=100, extra_workflows=(), extra_passthrough_modules=(), **worker_kwargs)` | Build a `Worker` serving all bound programs. |
| `DSPyPlugin(...)` | Client+worker plugin to wire DSPy into a `Worker` you build yourself. |
| `connect(address, **kwargs)` | Connect a Temporal `Client` with the pydantic data converter installed. |
| `configure_lm_from_env(model=None, **lm_kwargs)` | Build a `dspy.LM` from env and set it as the worker LM. |
| `set_worker_lm(lm)` | Set a pre-built `dspy.LM` as the worker LM. |
| `RunMode.COARSE` / `RunMode.FINE` | Execution mode (see the table above). |
| `CallOptions(...)` | Per-run retry/timeout tuning. |
| `setup_tracing(...)` | (from `dspy_temporal.tracing`) Returns the OTel interceptor for the client. |

## Development

Contributing? Clone and install everything (dev + tracing — the full suite needs both):

```bash
git clone https://github.com/markberger/dspy_temporal
cd dspy_temporal
uv sync --all-extras
```

The repo uses [`uv`](https://docs.astral.sh/uv/), so contributor commands run through
`uv run …`:

```bash
uv run pytest                                                              # run the suite
uv run pytest --cov=dspy_temporal --cov-branch --cov-report=term-missing   # with coverage
uv run ruff format .                                                       # format
uv run ruff check --select I --fix .                                       # sort imports
uv run pre-commit install                                                  # enable git hooks (once)
```

Tests use DSPy's `DummyLM` and a time-skipping `WorkflowEnvironment` (an ephemeral in-process
Temporal server), so they need no network or API keys. Coverage is 100% line and ~99.8% branch
— the only gap is two partial branches in `tracing/callback.py` — enforced by a 90% floor
(`fail_under`).

### Parallel work (git worktrees)

To run several sessions at once — each on its own branch — spin up a worktree:

```bash
scripts/wt new my-feature      # creates .worktrees/my-feature with its own venv + .env
cd .worktrees/my-feature
```

`scripts/wt list` / `scripts/wt rm <name>` manage them. Editing, `pytest`, and `ruff` are
collision-free across worktrees; the live `docker compose` stack is the one thing only one
worktree may run at a time. See [CLAUDE.md](CLAUDE.md) for the full rules.
