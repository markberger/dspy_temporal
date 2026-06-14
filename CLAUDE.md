# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

Today, this project is unreleased so do not worry about maintaining backwards
compatibility. Strive to build the best library possible despite the existing
implementation details.

## Parallel sessions via git worktrees

**Worktrees are the default for any branch work — including a single session.**
Start with `scripts/wt new <name>` and work inside `.worktrees/<name>`; don't
`git checkout -b` in the main checkout. Switching the main checkout's branch
moves it off whatever branch you (or another session) left it on and drags
untracked files along — the main checkout should stay a stable, untouched
anchor.

Run several sessions at once, each on its own branch, with `scripts/wt`:

```bash
scripts/wt new <name> [base]    # create .worktrees/<name>, build its venv, link .env
scripts/wt list                 # show all worktrees
scripts/wt rm  <name> [--delete-branch] [--force]
```

Each worktree is a full, independent checkout under `.worktrees/<name>` (nested,
gitignored) that shares this repo's single `.git`. `wt new` gives it its own
`.venv` (`uv sync --all-extras`, so the full suite incl. tracing tests runs) on
the repo's pinned Python (`.python-version`) and a symlinked `.env`, so it's
ready to use:

```bash
scripts/wt new my-feature
cd .worktrees/my-feature
```

`base` defaults to `origin/main`. Worktrees don't show up in the main checkout's
`git status` (the `.worktrees/` dir is ignored).

## What's safe to run in parallel

Editing, `uv run pytest`, and `ruff` are **fully isolated** across worktrees.
Tests use DSPy's `DummyLM` and a time-skipping `WorkflowEnvironment` (an
ephemeral in-process Temporal server on a random port) — no shared server, no
fixed ports, no API keys. Run them in any number of worktrees simultaneously.

## The one collision rule: the live Docker stack

`docker compose up` binds fixed host ports (`7233`/`8233` Temporal,
`6006`/`4317` Phoenix). Only **one** worktree may run the live stack at a time.
To run two, set `COMPOSE_PROJECT_NAME` and remap the published ports in an
override. Likewise, if you point a bare `examples/worker.py` at a shared
`localhost:7233`, give each worktree a **distinct `task_queue`** so workers
don't split each other's tasks.

## Pre-commit is shared

Hooks live in the common `.git/hooks` (shared by all worktrees). Install **once
from the main checkout** and never from an ephemeral worktree — the hook
hardcodes the installing venv's interpreter, and `wt rm` would delete it:

```bash
uv run pre-commit install        # run this in the MAIN checkout only
```

## Standard dev commands

```bash
uv sync --all-extras                                  # install deps incl. tracing (per worktree)
uv run pytest                                         # run the suite
uv run pytest --cov=dspy_temporal --cov-branch --cov-report=term-missing
uv run ruff format .                                  # format
uv run ruff check --select I --fix .                  # sort imports
uv run pre-commit run --all-files                     # both hooks over the repo
```

Coverage floor is 90% (`fail_under`); the suite currently sits at 100%
line+branch. See `README.md` for project usage and the Docker/tracing stack.
