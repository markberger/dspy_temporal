"""A fine-mode program with two predictors, each bound to its own LM.

This exercises the per-predictor multi-LM path: the ``draft`` predictor runs on
one cheap OpenRouter model and the ``refine`` predictor on a *different* one. In
fine mode the worker resolves each predictor's bound ``.lm`` by name and runs it
in its own ``dspy_lm_call`` activity, so a single run shows up as two distinct
model keys in ``prediction.get_lm_usage()`` (and two LM spans when tracing is on).

No API key is ever placed on a ``dspy.LM`` here -- litellm reads
``OPENROUTER_API_KEY`` from the worker's environment at call time, and the model
spec that crosses to the workflow has secret kwargs stripped.

Imported by ``examples/worker.py`` so the builder registers at worker startup.
"""

import dspy

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"

# Two cheap OpenRouter models from different providers, so the per-predictor
# split is obvious in lm_usage / traces. Swap freely -- only the worker env needs
# OPENROUTER_API_KEY.
DRAFT_MODEL = "openrouter/openai/gpt-4o-mini"
REFINE_MODEL = "openrouter/qwen/qwen3.5-9b"


class TwoLMQA(dspy.Module):
    """Draft an answer with one LM, then refine it with another."""

    def __init__(self):
        super().__init__()
        self.draft = dspy.ChainOfThought("question -> draft_answer")
        self.refine = dspy.ChainOfThought("question, draft_answer -> answer")
        # Bind a distinct LM per predictor. set_lm pushes the LM onto the
        # submodule's inner Predict, which is what named_predictors() (and thus
        # the worker's per-predictor resolution) reads.
        self.draft.set_lm(dspy.LM(DRAFT_MODEL))
        self.refine.set_lm(dspy.LM(REFINE_MODEL))

    # Fine mode runs the program via program.acall -> aforward in the workflow,
    # so the orchestration MUST be async (a sync forward would attempt a blocking
    # LM call inside the workflow, which WorkflowLM guards against).
    async def aforward(self, question: str) -> dspy.Prediction:
        drafted = await self.draft.acall(question=question)
        return await self.refine.acall(question=question, draft_answer=drafted.draft_answer)


two_lm_qa = dt.deploy_module(
    "two_lm_qa",
    TwoLMQA,
    config=dt.RunConfig(task_queue=TASK_QUEUE, mode=dt.RunMode.FINE),
)
