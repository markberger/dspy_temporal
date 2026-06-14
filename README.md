# dspy-temporal

[![lint](https://github.com/markberger/dspy_temporal/actions/workflows/lint.yml/badge.svg)](https://github.com/markberger/dspy_temporal/actions/workflows/lint.yml)
[![test](https://github.com/markberger/dspy_temporal/actions/workflows/test.yml/badge.svg)](https://github.com/markberger/dspy_temporal/actions/workflows/test.yml)

Deploy [DSPy](https://dspy.ai) programs on [Temporal](https://temporal.io) as durable
workflows — with retries, timeouts, and observability — without writing Temporal code.

## Status

- **Coarse mode (shipped):** a whole `dspy.Module` runs inside one Temporal activity. DSPy
  is fully intact (adapters, caching, retries). Durability is job-level: a crash re-runs the
  program. This is the low-friction "just deploy it" path.
- **Fine-grained mode (shipped):** each LM call (and ReAct tool call) becomes its own
  activity, with the program's orchestration running in the workflow. Completed LM/tool
  calls are recorded in Temporal history, so long/agentic runs resume from the last
  completed step instead of re-calling the model, and each LM call gets an isolated span
  (no token-attribution ambiguity under concurrency). Opt in with `RunConfig(mode=RunMode.FINE)`.
  See [Fine-grained mode](#fine-grained-mode) for usage and limits.

## How it works

- You register a **zero-arg builder** that returns a fresh `dspy.Module`. Only the program
  *name* + call *inputs* cross the Temporal boundary — never a live LM or API key.
- The worker configures the LM from its environment and runs the program inside an activity.
- A thin workflow invokes that activity with your retry policy and timeouts.

## Install

```bash
uv sync --all-extras   # dev + tracing; the full test suite needs both
```

## Parallel work (git worktrees)

To run several Claude/dev sessions at once — each on its own branch — spin up a
worktree:

```bash
scripts/wt new my-feature      # creates .worktrees/my-feature with its own venv + .env
cd .worktrees/my-feature
```

`scripts/wt list` / `scripts/wt rm <name>` manage them. Editing, `pytest`, and
`ruff` are collision-free across worktrees; the live `docker compose` stack is
the one thing only one worktree may run at a time. See [CLAUDE.md](CLAUDE.md)
for the rules.

## Usage

**1. Define + register a program** (`program.py`):

```python
import dspy
import dspy_temporal as dt

qa = dt.deploy(
    lambda: dspy.ChainOfThought("question -> answer"),
    name="qa",
    config=dt.RunConfig(task_queue="dspy-temporal"),
)
```

**2. Run a worker** (imports `program.py` so the builder is registered):

```python
import asyncio, dspy_temporal as dt
import program  # registers "qa"

async def main():
    dt.configure_lm_from_env()           # reads DSPY_LM_MODEL + provider keys from env
    client = await dt.connect("localhost:7233")
    worker = dt.build_worker(client, config=dt.RunConfig(task_queue="dspy-temporal"))
    await worker.run()

asyncio.run(main())
```

**3. Start a run from anywhere:**

```python
client = await dt.connect("localhost:7233")
pred = await dt.run_program(client, "qa", {"question": "Why is the sky blue?"},
                            task_queue="dspy-temporal")
print(pred.answer)
```

Set the LM via env on the worker:

```bash
export DSPY_LM_MODEL=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

A runnable example lives in `examples/` (`qa_program.py`, `worker.py`, `run.py`).

## More ways to deploy, run, and wire

The `deploy` / `run_program` / `build_worker` trio above is the stable baseline. A few
conveniences layer on top — each is optional, and nothing about the baseline changes.

### Deploy a live (or compiled) module instance

The headline `deploy` above takes a zero-arg builder; **it also accepts a live `dspy.Module`
instance** — handy for a program you've optimized with a DSPy teleprompter, whose predictors
carry few-shot demos. The instance stays in worker memory as a prototype; each run gets a
fresh, LM-stripped `deepcopy` (demos preserved, no bound LM or API key ever serialized into
Temporal history):

```python
import dspy
import dspy_temporal as dt

program = dspy.ChainOfThought("question -> answer")   # or a compiled program carrying demos
agent = dt.deploy(program, name="qa", mode=dt.RunMode.COARSE, task_queue="dspy-temporal")
```

`deploy` also takes a builder (`dt.deploy(lambda: ..., name=...)`); pass an explicit
`config=dt.RunConfig(...)` to override the `mode`/`task_queue` it otherwise assembles from the
keywords. Runnable example: `examples/deploy_instance.py`.

### Compose a deployed program inside your OWN workflow

The handle returned by `deploy` has a **context-aware** `await
agent.run(**inputs)`:

- **inside a user-authored `@workflow.defn`** it dispatches our activities inline, so you can
  interleave DSPy calls with your own workflow logic (timers, other activities, child
  workflows) as one durable, replayable execution;
- **outside any workflow** it degrades to a plain local DSPy call (against your configured
  `dspy.settings` LM).

```python
from datetime import timedelta
import dspy
from temporalio import workflow
import dspy_temporal as dt

agent = dt.deploy(lambda: dspy.ChainOfThought("question -> answer"),
                  name="compose_qa", mode=dt.RunMode.COARSE, task_queue="dspy-temporal")

@workflow.defn
class ResearchWorkflow:
    @workflow.run
    async def run(self, question: str) -> str:
        first = await agent.run(question=question)          # one durable activity step
        await workflow.sleep(timedelta(seconds=1))          # ...interleave your own logic...
        followup = await agent.run(question=f"Summarize in one word: {first.answer}")
        return followup.answer
```

Register the user workflow on the worker with `build_worker(..., extra_workflows=[ResearchWorkflow])`
(or the plugin below). To start a deployed program as a standalone workflow from a client, use
`await dt.run_program(client, "compose_qa", {...})`. If you'd rather compose without a handle,
`dt.execute_coarse` / `dt.execute_fine` are exported too. Runnable example:
`examples/compose_program.py` + `examples/run_compose.py`.

### Wire with a plugin (client + worker)

If you already construct your own `temporalio.worker.Worker` (custom interceptors, tuning, your
own workflows/activities), add DSPy support with `DSPyPlugin` instead of `build_worker`:

```python
from temporalio.worker import Worker
import dspy_temporal as dt

worker = Worker(client, task_queue="dspy-temporal", plugins=[dt.DSPyPlugin()])
```

`DSPyPlugin` is a **combined client + worker plugin**: passed to the **client** it installs the
pydantic data converter *and*, because it's also a worker plugin, propagates to any `Worker`
built from that client — so over a vanilla `Client.connect()` you'd pass it there to keep
pydantic models round-tripping:

```python
client = await Client.connect("localhost:7233", plugins=[dt.DSPyPlugin()])
worker = Worker(client, task_queue="dspy-temporal")   # DSPy set added automatically
```

(`dt.connect()` already sets the converter, so over it the plugin is worker-only.) Apply it on
the client **or** a `Worker`/`Replayer` directly — not both. The plugin contributes the same
four activities, both generic workflows, and the DSPy sandbox runner — **extending** (never
overwriting) anything you pass explicitly. Add your own composed workflows via
`DSPyPlugin(extra_workflows=[ResearchWorkflow])` and extra sandbox-passthrough prefixes via
`DSPyPlugin(extra_passthrough_modules=("my_pkg",))`. `build_worker` stays the one-call path (it
builds its worker through this same plugin) and shares the exact same activity/workflow set
(`dt.DSPY_ACTIVITIES` / `dt.DSPY_WORKFLOWS`). Runnable example: `examples/worker_plugin.py`.

## Fine-grained mode

Coarse mode runs the whole program in one activity. **Fine mode** instead runs the
program's *orchestration* in the workflow and turns each LM call and each tool call into
its own activity. The payoff:

- **Durable resume:** a completed LM/tool call is in Temporal history, so a crash + replay
  resumes from the last finished step — no duplicate model spend, and long agentic runs
  survive restarts.
- **Per-call tracing:** each LM call runs on an isolated LM copy and emits its own span
  with correct `gen_ai.usage.*` tokens (the coarse shared-history attribution caveat is gone).

Opt in per program with `RunConfig(mode=RunMode.FINE)`. Tools are ordinary Python functions you
hand to `dspy.ReAct` in the builder — there is no fine-mode-specific tool API:

```python
import dspy
import dspy_temporal as dt

def get_weather(city: str) -> str:
    """Return a weather report for a city."""   # body runs in an activity → real I/O is fine
    return f"The weather in {city} is sunny."

def build_agent() -> dspy.Module:
    # The builder runs in the WORKFLOW: only construct dspy objects here — no network/file/DB.
    return dspy.ReAct("question -> answer", tools=[get_weather])

agent = dt.deploy(build_agent, name="weather_agent",
                  config=dt.RunConfig(task_queue="dspy-temporal", mode=dt.RunMode.FINE))
```

The same worker serves both modes (it registers both workflows and all activities), so no
worker change is needed. Run it the usual way — `run_program(..., mode=RunMode.FINE)`. In the
Temporal UI you'll see distinct `dspy_lm_call` / `dspy_tool_call` activities per run. A
runnable example is in `examples/` (`react_program.py`, `run_react.py`).

**Where each piece runs:** the tool *bodies* and the LM HTTP calls run in activities (real
I/O allowed); the builder, the ReAct loop, and adapter format/parse run in the workflow as
deterministic Python. Tool args arrive JSON-native and are coerced to the annotated types;
tool return values are JSON-ified, so tools should return JSON-native data or pydantic
models (not live handles). Both sync and async tool functions work.

**Multiple LMs and structured outputs (supported):**

- **Per-predictor LMs** — bind a predictor's own `.lm` in the builder
  (`self.summarize.lm = dspy.LM("openai/gpt-4o")`); the worker resolves each predictor's LM
  by name. A one-shot `dspy_describe_lms` activity carries each LM's model + capabilities to
  the workflow up front (so JSONAdapter branches correctly), and the LM/credentials stay on
  the worker — only a description crosses the wire.
- **JSONAdapter / structured outputs** — a structured `response_format` (the pydantic class
  `JSONAdapter` builds from the signature) now crosses the boundary as its JSON schema.
  Configure the adapter on the worker once: `dspy.configure(adapter=dspy.JSONAdapter())`.

**Limitations (use coarse mode if you need these):**

1. **Sequential async only** — programs that fan out internally (`dspy.Parallel`, threads,
   `asyncio.gather`) aren't supported in the workflow.
2. **No ReAct context-window-truncation fallback** — a `ContextWindowExceededError` becomes
   a Temporal `ActivityError` across the boundary, so ReAct's truncate-and-retry won't trigger.
3. **Tools resolved via `program.tools`** — covers ReAct and any module exposing a `.tools`
   dict; a custom tool-calling module without `.tools` isn't supported yet.

## Tracing (optional)

Capture LLM traces with OpenTelemetry — dual-emitted as both **gen_ai** semantic
conventions (Langfuse, Grafana/Tempo, Honeycomb, Datadog) and **OpenInference**
(Arize Phoenix). Install the extra and call `setup_tracing` once, then pass the
returned interceptor to the **client** (the worker inherits it):

```bash
uv sync --extra tracing
```

```python
import dspy_temporal as dt
from dspy_temporal.tracing import setup_tracing

interceptor = setup_tracing(service_name="qa-worker")   # OTLP by default; reads OTEL_EXPORTER_OTLP_*
client = await dt.connect("localhost:7233", interceptors=[interceptor])
worker = dt.build_worker(client, config=dt.RunConfig(task_queue="dspy-temporal"))
await worker.run()
```

You get one trace per run. In **coarse** mode: `Workflow → Activity → dspy.module →
chat <model>`. In **fine** mode the span tree follows the per-call activities:
`Workflow → dspy_lm_call activity → chat <model>` for each LM call and
`Workflow → dspy_tool_call activity → execute_tool <name>` for each tool call (no
`dspy.module` span — the module orchestrates in the workflow). Either way LM spans carry
token usage, model, finish reasons, and cost. Prompt/completion **content is off by
default**; enable it with `setup_tracing(capture_content=True)` or
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. Register the interceptor on the
**client only** — adding it to the worker too double-emits spans.

> **`dspy.Parallel` is not traced.** Span nesting follows DSPy's `ACTIVE_CALL_ID`
> contextvar, which propagates on the sync path and across `asyncio.gather` (asyncio
> copies the context into each task) but **not** across `dspy.Parallel`, which runs
> items on a `ThreadPoolExecutor` that doesn't copy contextvars — so parallel
> sub-calls orphan into separate trace roots. For a correct trace tree under
> concurrency, fan out with the async interface (`asyncio.gather` over `module.acall`)
> instead of `dspy.Parallel`; the coarse activity runs programs via `acall` by default
> so this works out of the box. (The permanent fix is an upstream DSPy
> `copy_context()` in its parallel executor.)

## Run locally with Docker Compose (Phoenix tracing)

`docker-compose.yml` brings up a [Temporal](https://temporal.io) dev server, an
[Arize Phoenix](https://phoenix.arize.com) UI, and a traced worker for the example
`qa` program — so you can fire a DSPy program at `gpt-5-nano` and watch the trace.

```bash
cp .env.example .env            # then put a real OPENROUTER_API_KEY in .env
docker compose up --build       # temporal (7233/8233), phoenix (6006/4317), worker
```

With the stack up, start a run from the host and view the trace. Point the starter at
Phoenix so it emits the root `StartWorkflow` span and propagates context — that's what
ties the worker's `RunWorkflow`/`StartActivity`/`RunActivity` and the DSPy spans into a
**single** trace (standard distributed tracing; without it they fragment into separate
traces):

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
    uv run --extra tracing python examples/run.py "Why is the sky blue?"
```

- **Phoenix UI:** http://localhost:6006 — the trace `Workflow → Activity → dspy.module
  → chat openrouter/openai/gpt-5-nano` with token usage, model, and (content capture is
  on in compose) the prompt/completion text.
- **Temporal UI:** http://localhost:8233 — the workflow execution.

The worker reads `TEMPORAL_ADDRESS`, `DSPY_LM_MODEL`, and `OTEL_EXPORTER_OTLP_ENDPOINT`
from its environment; it enables tracing automatically when an OTLP endpoint is set.
The Temporal dev server uses an in-memory store, so workflow history resets when the
stack restarts.

## Formatting

Code is formatted and imports sorted with [Ruff](https://docs.astral.sh/ruff/),
configured in `pyproject.toml` (`[tool.ruff]`). After `uv sync --all-extras`, enable the
git hook once so formatting runs automatically on every commit:

```bash
uv run pre-commit install
```

Run it by hand anytime:

```bash
uv run ruff format .                     # format
uv run ruff check --select I --fix .     # sort imports
uv run pre-commit run --all-files        # both hooks over the whole repo
```

In editors, the [Ruff extension](https://docs.astral.sh/ruff/editors/) gives
format-on-save with the same config.

## Tests

```bash
uv run pytest                                   # run the suite
uv run pytest --cov=dspy_temporal --cov-branch --cov-report=term-missing   # with coverage
```

Unit tests use `ActivityEnvironment`; integration tests use a time-skipping
`WorkflowEnvironment`. All tests use DSPy's `DummyLM`, so they need no network or
API keys. Coverage is 100% line+branch with a 90% floor (`fail_under`).

CI-style gate — enforced in CI on every PR (see the **test** badge above):

```bash
uv sync --all-extras
uv run pytest --cov=dspy_temporal --cov-branch \
  --cov-report=term-missing --cov-report=xml --cov-fail-under=90
```
