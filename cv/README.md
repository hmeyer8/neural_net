# Computer Vision — paused

**Status as of 2026-09-04: the drills are live, the projects are not built.**

This track was planned as tensors through to a deployed inference service — classification,
detection, segmentation, serving. I got through the drill infrastructure and the curriculum design,
and then moved to the `agent/` track before any of the four projects produced a number.

I'm writing that down instead of leaving empty directories behind, because a directory named
`detect/` with nothing in it makes a claim, and the claim would be false. The four project
directories that used to sit here were removed in the same commit as this README.

## What is actually here

`reps.py` — a retrieval drill for PyTorch/tensor API fluency. Every function is a spec with an
empty body; you fill it from memory and the file scores you. Closed book, shapes predicted before
running. This still runs daily and is the reason the tensor API is not something I have to look up.

`call_box.md` — the bounded curriculum behind the drill. About 75 calls across three tiers, plus an
explicit list of what I am deliberately *not* learning cold. The sorting principle is frequency,
not difficulty: anything in the inner loop of a training run goes cold, anything that appears once
at the top of a script gets looked up forever.

`reps.md` — the score log. The trend is the signal, not any single day.

## Why it stopped where it did

Two reasons, and the second is the real one.

The stated reason is in the journal (8/13): I wanted to take a model from raw data all the way to
something a person could call, and the marginal return on more fundamentals had dropped.

The honest reason is that I then went a step further and changed problem domains entirely. The work
I want to be doing is applied LLM systems — agents, retrieval, evaluation, the operations around
models rather than the models themselves. Continuing to build a fourth CNN would have been
comfortable and off-target. `ROADMAP.md` has the full reasoning.

## What carried over

The method, which was always the point:

- **Retrieval drills over re-reading.** `agent/reps.py` is the same file for a different API.
- **A bounded call box.** `agent/call_box.md` is the same idea for LLM and RAG primitives.
- **A project is done when there is a number, a baseline, and a written account of how it fails.**
  This survives intact and is the standard the agent track is held to.
- **The experiment ledger.** Same file, `../experiments/experiments.csv`, now carrying eval runs
  instead of training runs.

## If this restarts

The project I'd build is still the one worth building, and it's the reason the abstention idea runs
through the agent track too:

**Equipment Listing Image QA.** Given the images attached to a listing, return structured,
actionable output rather than a bare class label — asset class, view angle, whether the image is
usable, and a `manual_review` flag. The abstention flag is the interesting part: a model that knows
when to abstain is worth more to a business than a model with two more points of accuracy, because
it lets you set the precision you actually need and route the rest to a human.

That argument turned out to transfer completely. In the agent track it's the same design: an answer
the system isn't confident in should become an escalation, not a guess.

## Data policy

Open-licensed only, license recorded in a dataset card. No employer data goes in this repo.
