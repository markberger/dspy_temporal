"""A fine-mode program that fans out concurrently with ``dspy_temporal.gather``.

In fine mode the program's ``aforward`` runs *in the workflow*, so the way to run
sub-calls concurrently is the async path: ``await dspy_temporal.gather(...)`` over
each predictor's ``.acall(...)``. Each leaf LM call becomes its own
``dspy_lm_call`` activity, and Temporal's deterministic event loop runs them
concurrently -- so a 3-way fan-out issues three activities at once (visible in the
Temporal UI), and a crash resumes from whichever branches already finished.

(Thread-based ``dspy.Parallel`` can't run in the workflow -- threads aren't
allowed in the sandbox and it drives *sync* calls; use ``gather`` here, or run the
program in coarse mode.)

A predictor may also bind its own LM (``self.summarize.lm = dspy.LM(...)``) to use
a different model per step; the worker resolves each predictor's LM by name.

Imported by ``examples/worker.py`` so the builder registers at worker startup.
"""

import dspy

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"


class FanOutQA(dspy.Module):
    """Ask the same question three ways at once, then combine the answers."""

    def __init__(self):
        super().__init__()
        self.factual = dspy.Predict("question -> answer")
        self.creative = dspy.Predict("question -> answer")
        self.concise = dspy.Predict("question -> answer")
        self.combine = dspy.Predict("question, drafts -> answer")

    async def aforward(self, question):
        # Three concurrent LM-call activities (one per predictor).
        factual, creative, concise = await dt.gather(
            self.factual.acall(question=question),
            self.creative.acall(question=question),
            self.concise.acall(question=question),
        )
        drafts = " | ".join([factual.answer, creative.answer, concise.answer])
        # A final activity that synthesizes the drafts.
        return await self.combine.acall(question=question, drafts=drafts)


fan_out_qa = dt.deploy_module(
    "fan_out_qa",
    FanOutQA,
    config=dt.RunConfig(task_queue=TASK_QUEUE, mode=dt.RunMode.FINE),
)
