# Worker image: runs the example DSPy-on-Temporal worker with tracing enabled.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Install dependencies first (cached unless the lock/manifest changes).
# README.md + src are needed because hatchling builds the project during sync.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra tracing

# The example program + worker entrypoint.
COPY examples ./examples

CMD ["uv", "run", "--no-sync", "python", "examples/worker.py"]
