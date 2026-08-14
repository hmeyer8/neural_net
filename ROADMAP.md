# Roadmap — Computer Vision, 8 Weeks

The plan I'm running from August 2026. Day-level detail lives in my own notes; this is the version that matters to anyone reading the repo — what I'm building, in what order, and how I know when a week is actually finished.

## What I'm optimizing for

Not "complete a curriculum." The target is a specific capability:

> Hand me a blank repo and an ambiguous image problem. I ask what the classes are, what the label quality looks like, how imbalanced it is, what a wrong answer costs. I build a baseline before anything clever. I measure it honestly, find where it fails, fix the data before the architecture, and ship it behind an endpoint in a container.

Everything below is in service of that. If a week's work doesn't move me toward it, the week was wrong.

## Structure

**3 focused hours/day, 6 days/week, 8 weeks.** The seventh day is flex — catch-up, a write-up, or nothing. It's in the plan on purpose, not as slack I'm hoping not to use.

Each day runs the same shape:

| Block | Time | What |
|---|---|---|
| Recall | 15 min | Closed book. Three things from yesterday, one shape predicted from memory. |
| Learn | 35 min | One resource. Notes in a fixed format, not transcription. |
| Ugly pass | 20 min | Deliberately bad first version. Hardcoded, no functions. Just make it run. |
| Build | 65 min | The real implementation, typed by hand. |
| Closed-book rebuild | 30 min | Tutorial closed. Rebuild the core idea, change one thing, break something on purpose. |
| Journal + commit | 15 min | What I built, the biggest bug and why it happened, one thing I can now explain cold. |

The ugly pass is there for a reason. My failure mode isn't quitting halfway, it's not starting because I want the first version to be good. Twenty minutes of deliberately bad code removes that as an option.

## Ground rules

- **Type it, don't paste it.**
- **Predict before you run.** Every shape, written down first. A wrong prediction is the day's real lesson.
- **Help ladder, in order:** memory → docs → AI for a conceptual hint → AI for pseudocode → the actual solution, and then I close it and rewrite from memory.
- **Open-licensed data only.** No employer data, no proprietary imagery, license recorded in every dataset card. A project I can't show anyone is worth nothing.
- **Every run gets logged.** Seed, change, metric, notes. If I can't reproduce it from a clean clone, it isn't a result.

## The eight weeks

### Week 1 — Calibrate and close the API gap
The math isn't the gap; I built micrograd. The gap is library fluency — `axis` vs `dim`, `view` vs `reshape`, when `keepdim` is load-bearing. Day 1 is a cold API diagnostic that decides how much of Days 2–4 I actually need. Then datasets, dataloaders, splits, and a deliberately dumb baseline so every later number has something to beat.

**Gate:** training loop from a blank file; splits with an assertion, not a glance; a baseline with a number attached.

### Week 2 — CNNs, and debugging them
Manual convolution in NumPy before touching `Conv2d`. Then CIFAR-10, checkpointing, and the day that matters most: breaking my own working code six ways and diagnosing each from symptoms alone — missing `zero_grad`, LR off by 100× in both directions, wrong normalization stats, shuffled labels, missing `model.eval()`. Output is a written playbook I'll use for years.

**Gate:** build and train a CNN cold; overfit a single batch to ~100% as a pipeline sanity check; `docs/debugging_playbook.md` exists and is mine.

### Week 3 — Transfer learning and my own dataset
Pretrained backbones, residual blocks, freeze vs fine-tune and why fine-tuning wants a lower LR. Then I build a dataset from open sources with **written labeling rules including the ambiguous cases** — which I suspect is the actual skill here, not the modeling.

**Gate:** documented dataset card; head swapped from memory; three runs compared on identical splits, and I know whether the gap between them exceeds seed noise.

### Week 4 — Evaluation
The week I expect to be least fun and most valuable. Precision and recall by hand before any library. Threshold sweeps and an abstain path that returns `manual_review: true` instead of guessing. Every validation error pulled and categorized by hand — including how many "errors" are actually my labels being wrong. Then a data fix, retrained with nothing else changed, to see whether it beats a day of architecture work. It usually does.

**Gate:** categorized, counted failure analysis; a reproducibility audit from a clean clone in under 30 minutes.

### Week 5 — Object detection
IoU and NMS implemented myself before using `torchvision.ops`. Detector output fields typed out one by one. Detection metrics reasoned through by hand at two IoU thresholds. Then fine-tuning a pretrained detector on a small hand-annotated set — partly to learn how slow annotation actually is, which is a business fact as much as a technical one.

**Gate:** my own IoU and NMS; I can explain a detector's output field by field; I can argue when detection is the *wrong* tool.

### Week 6 — Capstone: define it before coding it
**Equipment Listing Image QA.** Given the images from an equipment listing, return structured output — asset class, view angle, image usable, confidence, and a manual-review flag.

Two full days before any model code: who reads the output, what decision they make with it, exactly what JSON leaves the system, what a false positive costs versus a false negative, and the one number that decides whether V1 shipped. Then baseline → failure analysis → data fix → one controlled experiment.

**Gate:** the problem stated in business terms with an error-cost analysis, and I can express the model's value in minutes saved per listing.

### Week 7 — Turn it into software
Repo restructured, config extracted, robust inference that survives corrupt files and grayscale inputs, FastAPI with real status codes, model loaded once at startup, tests for model / inference / API separately, and a working Dockerfile.

**Gate:** container runs, endpoint returns the real schema, `pytest` green.

### Week 8 — Prove it
Whiteboard both pipelines from memory, recorded, then fill every gap where I hedged. A blank-file coding test with docs allowed and nothing else. The write-up, including limitations stated plainly. Interview rehearsal, recorded. And a one-page internal proposal for a real repetitive visual workflow at work — because studying alone doesn't convert into anything.

**Gate:** the full checklist below.

## Final checklist

- [ ] Training loop from a blank file
- [ ] Every major tensor shape in my pipeline explained cold
- [ ] Build, train, and debug a CNN
- [ ] Transfer learning with a defended freeze/fine-tune choice
- [ ] Detection understood well enough to know when not to use it
- [ ] A real failure analysis, acted on
- [ ] I can state what my model costs when it's wrong
- [ ] Model served behind an API in a container
- [ ] One project I can talk about for twenty minutes without notes

## Progress

Updated as I go. Week entries link to the journal in the root README once they're done.

- **Week 1 — Tensors, data, baseline:** starting 8/17

## After this

The eight weeks are phase one. What follows is harder projects rather than more curriculum, and a shift toward the production side — experiment tracking, cloud deployment, monitoring, retraining. I start Georgia Tech's OMSCS in Spring 2027, and from that point this repo's independent work and the coursework are meant to be one budget, not two.

## Resources I'm actually using

Deliberately short. I'd rather go through a few things properly than collect bookmarks.

- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) — tensors, datasets, autograd, optimization
- [Stanford CS231n](https://cs231n.stanford.edu/) — lectures on linear classifiers, optimization, CNNs, architectures, detection
- [NumPy absolute beginners guide](https://numpy.org/doc/stable/user/absolute_beginners) — API reps, not concepts
- [TorchVision models](https://docs.pytorch.org/vision/stable/models/resnet.html) and the [detection fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [FastAPI](https://fastapi.tiangolo.com/tutorial/) and [Docker](https://docs.docker.com/get-started/)