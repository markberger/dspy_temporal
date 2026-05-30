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
