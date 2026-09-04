# neural_net

This is where I learn machine learning in public.

Everything here is built from scratch or rebuilt from memory. If a file is in this repo, I can explain every line in it. That's the rule I'm holding myself to — running code is not understanding code, and a notebook I copied is worth nothing to me.

I'm a machine learning intern at Sandhills Global, finishing a math degree at UNL, and starting an OMSCS master's in the machine learning specialization. I started this repo to stop treating deep learning as a library I call and start treating it as machinery I understand.

## Three tracks

**`agent/` — a grounded question-answering agent over the IETF RFC corpus. Current focus.**

An LLM agent that answers questions about internet standards, cites the exact section it used, refuses when the corpus doesn't support an answer, and checks whether the document it found has been superseded. Hand-written agent loop, hybrid retrieval, an eval suite that gates CI, and end-to-end tracing.

The design question driving it: *the hard problem in a normative corpus is not finding the text, it's knowing whether the text you found is still in force.* That can't be answered by similarity search — it's a graph query over document metadata — which is why this is an agent with tools and not a retrieval pipeline. See [`agent/README.md`](agent/README.md) and [`ROADMAP.md`](ROADMAP.md).

Started 2026-09-04. First measured retrieval baseline lands at the end of week 1; the results table in `agent/README.md` is where it goes.

**`cv/` — applied computer vision. Paused.**

The drills are live and still run daily: [`reps.py`](cv/reps.py) is a retrieval drill for tensor API fluency and [`call_box.md`](cv/call_box.md) is the bounded curriculum behind it. The project work — classification, detection, segmentation, serving — is not built. [`cv/README.md`](cv/README.md) says exactly where it stopped and why.

**`micrograd/`, `makemore/` — fundamentals, built from scratch, no frameworks. Done.**

`micrograd/` is a scalar autograd engine and a small MLP on top of it, following Karpathy's Zero to Hero. [`engine.py`](micrograd/engine.py) is the `Value` class and the backward pass, [`nn.py`](micrograd/nn.py) is Neuron/Layer/MLP, and the notebooks are where I worked it out before cleaning it up.

`makemore/` is a character-level language model over ~32k names — a bigram count model. The journal covers how it works and the broadcasting bug I nearly shipped building it.

**Supporting.**

[`journal.md`](journal.md) — what I got wrong, and what the gap taught me. Newest first.
[`docs/debugging_playbook.md`](docs/debugging_playbook.md) — symptom → hypothesis → confirmation → rule.
[`experiments/experiments.csv`](experiments/experiments.csv) — the run ledger. Seed, the one thing that changed, the number that came out.

## Setup

I use [uv](https://docs.astral.sh/uv/). The Python version is pinned in `.python-version` and the lockfile is committed, so this reproduces exactly:

```bash
uv sync                          # creates .venv and installs everything
uv run python cv/reps.py         # run anything without activating
uv run jupyter lab               # notebooks
uv run pytest                    # tests
```

If you'd rather activate the environment normally, `source .venv/bin/activate` after `uv sync` works fine. Adding a dependency is `uv add <package>`, which updates both `pyproject.toml` and `uv.lock`.

### Hardware

I develop on an RTX 3050 6GB laptop GPU (compute capability 8.6). `pyproject.toml` pins torch and torchvision to the `cu130` wheel index so `uv sync` installs a CUDA build rather than the CPU-only default from PyPI.

**This is not required to run anything here.** Delete the `[[tool.uv.index]]` and `[tool.uv.sources]` blocks at the bottom of `pyproject.toml` and `uv sync` falls back to the CPU wheel, which is also what CI uses. Everything works either way — the GPU changes how long things take, not what comes out:

```bash
uv run rfcagent status          # prints the device actually in use, and the batch sizes
RFCAGENT_DEVICE=cpu uv run ...  # force the CPU path on a machine that has a GPU
```

Measured on this machine, `bge-small-en-v1.5` encoding: **2,988 texts/s on the GPU against 412 texts/s on CPU, a 7.3× speedup.** That's the difference between a full-corpus index build taking two minutes and taking twenty, which is the difference between running the week-1 chunking ablation twice and running it once and hoping.

Two things the GPU does *not* do, worth being explicit about because it's tempting to assume otherwise:

- **It does not change any quality metric.** recall@5 is the same number on either device. The device is therefore excluded from `Settings.fingerprint()` and gets its own column in the experiment ledger instead — quality is keyed by the config hash, latency by the config hash *and* the device.
- **It does not make results reproducible by itself.** A seed alone isn't enough on CUDA: cuDNN benchmarks algorithms at runtime and some reductions accumulate nondeterministically. `set_seed()` pins cuDNN to deterministic algorithms for exactly that reason.

## How I work

Four rules, and they're most of the reason this repo is worth anything to me:

**Type it, don't paste it.** Every important line gets typed by hand. Pasting a working implementation teaches me nothing except that it works.

**Predict before running.** Before any `.shape`, matmul, or model output I write down what I expect. When I'm wrong, that gap is the actual lesson, and it goes in the journal.

**Rebuild closed-book.** Following along doesn't count. I close the tutorial and rebuild the core idea from memory before moving on. If I can't, I didn't learn it.

**AI is a tutor before it's a code generator.** Conceptual hints first, pseudocode if I'm genuinely stuck, the solution last — and then I close it and write it myself.

Every run gets a row in [`experiments/experiments.csv`](experiments/experiments.csv) with its seed. If I report a number, I can reproduce it from a clean clone.

## What is not here

Stated plainly, because a portfolio that only advertises is not worth reading:

- The CV projects (classify, detect, segment, serve) are **not built**. The scaffolding for them used to be in this repo as empty directories; I removed it, because empty directories that promise results are worse than an honest gap.
- The agent track is **days old**. Everything in `agent/` is dated and its README distinguishes what is measured from what is planned.
- No employer data appears anywhere in this repo, and no dataset without an open license.
