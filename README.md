# dspy-temporal

Deploy [DSPy](https://dspy.ai) programs on [Temporal](https://temporal.io) as durable
workflows — with retries, timeouts, and observability — without writing Temporal code.

## Status

- **Coarse mode (shipped):** a whole `dspy.Module` runs inside one Temporal activity. DSPy
  is fully intact (adapters, caching, retries). Durability is job-level: a crash re-runs the
  program. This is the low-friction "just deploy it" path.
- **Fine-grained mode (planned):** each LM call (and ReAct tool call) becomes its own
  activity with orchestration in the workflow, so long/agentic runs resume from the last
  completed step. See the design plan.

## How it works

- You register a **zero-arg builder** that returns a fresh `dspy.Module`. Only the program
  *name* + call *inputs* cross the Temporal boundary — never a live LM or API key.
- The worker configures the LM from its environment and runs the program inside an activity.
- A thin workflow invokes that activity with your retry policy and timeouts.

## Install

```bash
uv sync --extra dev
```

## Usage

**1. Define + register a program** (`program.py`):

```python
import dspy
import dspy_temporal as dt

qa = dt.deploy_module(
    "qa",
    lambda: dspy.ChainOfThought("question -> answer"),
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

You get one trace per run: `Workflow → Activity → dspy.module → chat <model>`, with
token usage, model, finish reasons, and cost on the LM spans. Prompt/completion
**content is off by default**; enable it with `setup_tracing(capture_content=True)`
or `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. Register the interceptor on
the **client only** — adding it to the worker too double-emits spans.

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
    uv run python examples/run.py "Why is the sky blue?"
```

- **Phoenix UI:** http://localhost:6006 — the trace `Workflow → Activity → dspy.module
  → chat openrouter/openai/gpt-5-nano` with token usage, model, and (content capture is
  on in compose) the prompt/completion text.
- **Temporal UI:** http://localhost:8233 — the workflow execution.

The worker reads `TEMPORAL_ADDRESS`, `DSPY_LM_MODEL`, and `OTEL_EXPORTER_OTLP_ENDPOINT`
from its environment; it enables tracing automatically when an OTLP endpoint is set.
The Temporal dev server uses an in-memory store, so workflow history resets when the
stack restarts.

## Tests

```bash
uv run pytest                                   # run the suite
uv run pytest --cov=dspy_temporal --cov-branch --cov-report=term-missing   # with coverage
```

Unit tests use `ActivityEnvironment`; integration tests use a time-skipping
`WorkflowEnvironment`. All tests use DSPy's `DummyLM`, so they need no network or
API keys. Coverage is 100% line+branch with a 90% floor (`fail_under`).

CI-style gate:

```bash
uv sync --extra dev
uv run pytest --cov=dspy_temporal --cov-branch \
  --cov-report=term-missing --cov-report=xml --cov-fail-under=90
```
