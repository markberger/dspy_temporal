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

# client = await dt.connect("localhost:7233")
pred = await qa.start(client, task_queue="dspy-temporal", question="Why is the sky blue?")
print(pred.answer)
```

## Why

- **No activity boilerplate.** `DSPyPlugin` contributes the DSPy activities and workflows, so
  there are none to author — declare a `program()` reference, `bind()` the impl on the worker,
  then compose it in your own `@workflow.defn` with `await ref.run(...)` or `start()` it as its
  own workflow.
- **Durable by default.** Runs survive worker crashes and restarts. In fine mode, completed
  LM and tool calls are recorded in history and never re-charged on replay.
- **Retries & timeouts you control.** Per-run `CallOptions` set attempt counts, backoff, and
  timeouts; non-retryable errors (e.g. context-window overflow) are excluded by default.
- **Secrets never cross the wire.** Only the program *name* and call *inputs* enter Temporal
  history — never a live LM or API key. The LM is configured on the worker.
- **First-class observability.** One OpenTelemetry trace per run, dual-emitted as gen_ai
  semantic conventions *and* OpenInference (Arize Phoenix), with token usage and cost.
- **Bring your own workflow.** The dspy-free `program()` reference imports into any
  `@workflow.defn` with a plain `import` and composes via `await ref.run(...)` — typed result
  and all, dspy never entering your workflow code.

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

**Requirements:** Python 3.10–3.14 and DSPy 3.2.x (pulled in automatically), plus a running
Temporal server. For local development, the
[Docker Compose stack](#run-locally-with-docker-compose) brings one up for you; otherwise
the [Temporal CLI](https://docs.temporal.io/cli) (`temporal server start-dev`) is the
quickest path.

## Quickstart

Most teams already run a Temporal worker. `dspy-temporal` slots into it as a **plugin**: you
declare a program **reference**, compose it inside your own `@workflow.defn`, and add DSPy
support to your worker with `DSPyPlugin`. The LM and its API key stay on the worker — only the
program name and inputs cross into Temporal. (Just want to run a program with no Temporal code?
Jump to the [shortcut](#shortcut-run-a-program-with-no-temporal-code).)

**1. Declare the program** (`program.py` — pure and dspy-free, safe to import anywhere):

```python
from pydantic import BaseModel
import dspy_temporal as dt

class Answer(BaseModel):
    text: str

# `result` shapes the dspy.Prediction into a typed Answer, so workflow and caller
# code speak pydantic, never dspy.
qa = dt.program("qa", result=lambda p: Answer(text=str(p.answer)))
```

`program()` returns a lightweight, immutable **reference** — just a name plus optional `mode`,
`options`, and (here) a typed `result` adapter. It registers nothing and loads no model, so
your workflow and a thin client both import it cheaply. The implementation is attached
separately, on the worker.

**2. Compose it in your workflow** (`workflow.py`):

```python
from temporalio import workflow
from program import Answer, qa

@workflow.defn
class ResearchWorkflow:
    @workflow.run
    async def run(self, question: str) -> Answer:
        first = await qa.run(question=question)             # a durable, retried activity step → Answer
        followup = await qa.run(question=f"Summarize in one word: {first.text}")
        return followup
```

`qa.run()` is **context-aware**: inside a `@workflow.defn` it dispatches a durable activity you
interleave with timers, signals, and child workflows; the DSPy/LM work runs in that activity,
never in the workflow sandbox. (Outside any workflow it degrades to a plain local DSPy call.)

**3. Add DSPy to your worker** with `DSPyPlugin` (`worker.py`). The plugin contributes the DSPy
activities and generic workflows and **extends** — never overwrites — your own; pass your
composed workflow via `extra_workflows=`:

```python
import asyncio
import dspy
from temporalio.worker import Worker
import dspy_temporal as dt
from program import qa
from workflow import ResearchWorkflow

class AnswerQuestion(dspy.Signature):
    """Answer the user's question concisely and factually."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

def build_qa() -> dspy.Module:
    # Zero-arg builder: a fresh module per run; the worker supplies the LM.
    return dspy.ChainOfThought(AnswerQuestion)

async def main():
    dt.configure_lm_from_env()          # DSPY_LM_MODEL + provider keys from env
    qa.bind(build_qa)                    # attach the impl here, on the worker

    client = await dt.connect("localhost:7233")   # installs the pydantic data converter
    worker = Worker(                              # your own Worker — add interceptors, activities, tuning
        client,
        task_queue="dspy-temporal",
        plugins=[dt.DSPyPlugin(extra_workflows=[ResearchWorkflow])],
    )
    await worker.run()

asyncio.run(main())
```

`bind()` takes a **zero-arg builder** returning a fresh `dspy.Module` (or a live, compiled
instance — see [below](#bind-a-compiled--live-module)); it's the heavy, side-effecting step, so
it belongs on the worker, never in a workflow file. Point the worker at its LM:

```bash
export DSPY_LM_MODEL=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

> Apply the plugin on the client **or** the worker, not both. Over `dt.connect()` (which
> already installs the converter) it's worker-only, as above; over a vanilla
> `temporalio.client.Client`, pass it to the client instead —
> `Client.connect(..., plugins=[dt.DSPyPlugin()])` — and it propagates to the worker.

**4. Start it** the normal Temporal way. The reference keeps `ResearchWorkflow` cheap to
import, so the call stays type-safe:

```python
client = await dt.connect("localhost:7233")
answer = await client.execute_workflow(           # answer is your typed Answer
    ResearchWorkflow.run, "Why is the sky blue?",
    id="research-1", task_queue="dspy-temporal",
)
print(answer.text)
```

Runnable: `examples/worker_plugin.py` with the `examples/compose_*.py` trio.

### Shortcut: run a program with no Temporal code

No workflow of your own — you just want a DSPy program to run durably? `dt.build_worker` wires
the plugin for you, and `ref.start()` runs the program *as* its own workflow (no `@workflow.defn`
to write):

```python
# worker.py — same program.py + build_qa as above
qa.bind(build_qa)
worker = dt.build_worker(client, task_queue="dspy-temporal")
await worker.run()
```

```python
# run.py
answer = await qa.start(client, task_queue="dspy-temporal", question="Why is the sky blue?")
print(answer.text)                      # start() awaits the run; the result adapter returns your Answer
```

Runnable: `examples/qa_program.py` + `examples/worker.py` + `examples/run.py`.

## Concepts

The Quickstart showed the split in practice — `program()` declares a pure reference; the
worker does the heavy `bind()`. The rest of the model:

- **Builder, not instance.** You bind a zero-arg callable that returns a fresh `dspy.Module`.
  Each run builds its own program, so there's no shared mutable state and nothing live to
  serialize. (You can also [bind a compiled instance](#bind-a-compiled--live-module).)
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

Coarse keeps DSPy fully intact — adapters, caching, in-program concurrency — at job-level
durability; fine trades some of that for step-level durability. See
[Fine-grained mode](#fine-grained-mode-per-call-activities) for its usage and limits.

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

The [Quickstart](#quickstart) covered the shape — declare a reference, compose it with
`await ref.run(...)` in your `@workflow.defn`, wire the worker with
`DSPyPlugin(extra_workflows=[...])`, and start it with native `client.execute_workflow(...)`.
Three knobs that path unlocks:

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

The [Quickstart](#quickstart) already wired `DSPyPlugin` into a `Worker`. A few more notes for
production setups:

- **`build_worker` is this plugin wired for you.** `dt.build_worker(client, ...)` just builds a
  `Worker` through `DSPyPlugin`; reach for the plugin directly when you construct your own
  `Worker` (custom interceptors, activities, tuning).
- **Client or worker, never both.** `DSPyPlugin` is a combined client + worker plugin: on a
  client it installs the pydantic converter and propagates to any `Worker` built from it; on a
  worker it contributes the DSPy set. Apply it in exactly one place — the same plugin also
  works on a `Replayer` for replay tests.
- **Widen the sandbox when needed.** `DSPyPlugin(extra_passthrough_modules=("my_pkg",))` shares
  extra module prefixes with the fine-mode sandbox, for the rare builder whose imports would
  otherwise trip it.

Runnable example: `examples/worker_plugin.py`.

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

> **`dspy.Parallel` is not traced.** Fan out with `asyncio.gather` over `module.acall`, not
> `dspy.Parallel` — its `ThreadPoolExecutor` doesn't copy DSPy's `ACTIVE_CALL_ID` contextvar,
> so parallel sub-calls orphan into separate trace roots. The coarse activity runs programs via
> `acall`, so this works out of the box. (Permanent fix: an upstream `copy_context()` in DSPy's
> parallel executor.)

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
