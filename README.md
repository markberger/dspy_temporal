# dspy-temporal

[![lint](https://github.com/markberger/dspy_temporal/actions/workflows/lint.yml/badge.svg)](https://github.com/markberger/dspy_temporal/actions/workflows/lint.yml)
[![test](https://github.com/markberger/dspy_temporal/actions/workflows/test.yml/badge.svg)](https://github.com/markberger/dspy_temporal/actions/workflows/test.yml)

**Deploy [DSPy](https://dspy.ai) programs on [Temporal](https://temporal.io) as durable
workflows — with retries, timeouts, and tracing — without writing Temporal code.**

You write a normal `dspy.Module`. `dspy-temporal` wraps it in a Temporal workflow so a
crash mid-run resumes instead of starting over, transient LM errors retry on a policy you
control, and every run emits an OpenTelemetry trace — all behind a three-line API.

```python
import dspy
import dspy_temporal as dt

qa = dt.deploy(lambda: dspy.ChainOfThought("question -> answer"),
               name="qa", task_queue="dspy-temporal")

pred = await qa.start(client, question="Why is the sky blue?")
print(pred.answer)
```

## Why

- **No Temporal boilerplate.** No workflow/activity classes to author — `deploy()` registers
  your program and hands back a handle you can `start()`.
- **Durable by default.** Runs survive worker crashes and restarts. In fine mode, completed
  LM and tool calls are recorded in history and never re-charged on replay.
- **Retries & timeouts you control.** Per-run `CallOptions` set attempt counts, backoff, and
  timeouts; non-retryable errors (e.g. context-window overflow) are excluded by default.
- **Secrets never cross the wire.** Only the program *name* and call *inputs* enter Temporal
  history — never a live LM or API key. The LM is configured on the worker.
- **First-class observability.** One OpenTelemetry trace per run, dual-emitted as gen_ai
  semantic conventions *and* OpenInference (Arize Phoenix), with token usage and cost.
- **Composable.** Drop a deployed program into your own `@workflow.defn` and interleave it
  with timers, activities, and child workflows as one replayable execution.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Concepts](#concepts)
- [Execution modes: coarse vs. fine](#execution-modes-coarse-vs-fine)
- [Configuring the LM](#configuring-the-lm)
- [Tuning retries & timeouts](#tuning-retries--timeouts)
- [Recipes](#recipes)
  - [Deploy a compiled / live module](#deploy-a-compiled--live-module)
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

A deployment has three pieces: a **program** (your `dspy.Module`), a **worker** (the process
that runs it), and a **starter** (kicks off a run). The snippets below are plain Python —
run them with `python your_script.py`.

**1. Define and deploy a program** (`program.py`):

```python
import dspy
import dspy_temporal as dt

qa = dt.deploy(
    lambda: dspy.ChainOfThought("question -> answer"),
    name="qa",
    task_queue="dspy-temporal",
)
```

`deploy()` takes a **zero-arg builder** that returns a fresh `dspy.Module`. The returned
`qa` handle carries the `task_queue` and `mode`, so you never repeat them when starting a
run. `task_queue` is what keeps the worker and the run on the same queue.

**2. Run a worker** (`worker.py` — importing `program.py` registers the builder):

```python
import asyncio
import dspy_temporal as dt
import program  # registers "qa"

async def main():
    dt.configure_lm_from_env()                       # DSPY_LM_MODEL + provider keys from env
    client = await dt.connect("localhost:7233")
    worker = dt.build_worker(client, task_queue="dspy-temporal")
    await worker.run()

asyncio.run(main())
```

Point the worker at its LM via environment variables:

```bash
export DSPY_LM_MODEL=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

**3. Start a run** (`run.py`):

```python
import asyncio
import dspy_temporal as dt
from program import qa     # the handle returned by deploy

async def main():
    client = await dt.connect("localhost:7233")
    pred = await qa.start(client, question="Why is the sky blue?")
    print(pred.answer)

asyncio.run(main())
```

A runnable version of this flow lives in `examples/` (`qa_program.py`, `worker.py`,
`run.py`).

## Concepts

- **Builder, not instance.** You register a zero-arg callable that returns a fresh
  `dspy.Module`. Each run builds its own program, so there's no shared mutable state and
  nothing live to serialize. (You can also [deploy a compiled instance](#deploy-a-compiled--live-module).)
- **The boundary is name + inputs.** Only the program name and call inputs cross into
  Temporal. The LM — and its API key — is configured on the worker from its environment and
  never enters workflow history.
- **The handle is the source of truth.** `deploy()` returns a `TemporalProgram` that knows
  its own `task_queue` and `mode`. `handle.start(client, **inputs)` runs it as a standalone
  workflow; `handle.run(**inputs)` composes it inside your own workflow. You never re-pass
  queue or mode.
- **Predictions round-trip.** `start()` returns a `dspy.Prediction` with `get_lm_usage()`
  intact, just as a local DSPy call would.

## Execution modes: coarse vs. fine

The same worker serves both modes; you pick per program with `mode=` on `deploy()`.

| | **Coarse** (default) | **Fine** |
|---|---|---|
| What runs in an activity | the whole `dspy.Module` | each LM call and each tool call |
| Where orchestration runs | inside the activity | in the workflow (deterministic replay) |
| Durability granularity | job-level: a crash re-runs the program | step-level: completed LM/tool calls survive a crash and aren't re-charged |
| Tracing | `Workflow → Activity → dspy.module → chat <model>` | per-call spans, each LM call on an isolated copy with exact token usage |
| Best for | "just deploy it" — full DSPy fidelity, lowest friction | long/agentic runs (e.g. ReAct) where resuming mid-run matters |
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

Pass `CallOptions` per run to override the defaults:

```python
pred = await qa.start(
    client,
    question="Why is the sky blue?",
    options=dt.CallOptions(maximum_attempts=5, start_to_close_timeout_seconds=120),
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

### Deploy a compiled / live module

`deploy()` also accepts a **live `dspy.Module` instance** — handy for a program optimized
with a DSPy teleprompter, whose predictors carry few-shot demos. The instance stays in
worker memory as a prototype; each run gets a fresh, LM-stripped `deepcopy` (demos
preserved, no bound LM or API key ever serialized):

```python
program = dspy.ChainOfThought("question -> answer")   # or a compiled program with demos
qa = dt.deploy(program, name="qa", task_queue="dspy-temporal")
```

Runnable example: `examples/deploy_instance.py`.

### Compose a program inside your own workflow

The handle's `await agent.run(**inputs)` is **context-aware**:

- **inside a user-authored `@workflow.defn`** it dispatches our activities inline, so you can
  interleave DSPy calls with your own workflow logic (timers, activities, child workflows) as
  one durable, replayable execution;
- **outside any workflow** it degrades to a plain local DSPy call against your configured
  `dspy.settings` LM.

```python
from datetime import timedelta
import dspy
from temporalio import workflow
import dspy_temporal as dt

agent = dt.deploy(lambda: dspy.ChainOfThought("question -> answer"),
                  name="compose_qa", task_queue="dspy-temporal")

@workflow.defn
class ResearchWorkflow:
    @workflow.run
    async def run(self, question: str) -> str:
        first = await agent.run(question=question)          # one durable activity step
        await workflow.sleep(timedelta(seconds=1))          # ...interleave your own logic...
        followup = await agent.run(question=f"Summarize in one word: {first.answer}")
        return followup.answer
```

Register the user workflow on the worker with
`build_worker(..., extra_workflows=[ResearchWorkflow])` (or the [plugin](#wire-your-own-worker-with-the-plugin)).
`agent.run()` is the compose verb; `agent.start(client, ...)` is how you start the deployed
program *itself* as a standalone workflow. Runnable example: `examples/compose_program.py` +
`examples/run_compose.py`.

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

agent = dt.deploy(build_agent, name="weather_agent",
                  task_queue="dspy-temporal", mode=dt.RunMode.FINE)
```

Run it the usual way — `await agent.start(client, question=...)`. In the Temporal UI you'll
see distinct `dspy_lm_call` / `dspy_tool_call` activities per run. Runnable example:
`examples/react_program.py`, `examples/run_react.py`.

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
| `deploy(source, *, name, task_queue, mode=RunMode.COARSE)` | Register a builder or live module; returns a `TemporalProgram`. |
| `TemporalProgram` | Handle from `deploy`. `.start(client, **inputs)` runs it standalone; `.run(**inputs)` composes it in your workflow. |
| `run_program(client, name, inputs, *, task_queue, workflow_id=None, options=None, mode=None)` | Low-level by-name start (a thin client must pass `mode`). |
| `build_worker(client, *, task_queue, max_concurrent_activities=100, extra_workflows=(), extra_passthrough_modules=(), **worker_kwargs)` | Build a `Worker` serving all deployed programs. |
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
