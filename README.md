# neural_net

This is where I learn machine learning in public.

Everything here is built from scratch or rebuilt from memory. If a file is in this repo, I can explain every line in it. That's the rule I'm holding myself to — running code is not understanding code, and a notebook I copied is worth nothing to me.

I'm an ML engineer at Sandhills Global and I just finished up a math degree at UNL. I started this repo to stop treating deep learning as a library I call and start treating it as machinery I understand.

## What's in here

**`micrograd/`** — a scalar autograd engine and a small MLP built on top of it, following
Karpathy's Zero to Hero. `engine.py` is the `Value` class and backward pass, `nn.py` is
Neuron/Layer/MLP, and the notebooks are where I worked it out before cleaning it up.

**`makemore/`** — character-level language models over ~32k names. Currently a bigram count
model; the write-up in the journal below covers how it works and the broadcasting bug I nearly
shipped building it.

**`cv/`** — computer vision. Tensors through to a deployed inference service. This is the
current focus and the plan is in [ROADMAP.md](ROADMAP.md).

**`docs/`** — the debugging playbook, and dataset cards as I write them.

**`experiments/`** — every training run I've done, logged with the seed and the one thing that
changed.

## Setup

I use [uv](https://docs.astral.sh/uv/). Python version is pinned in `.python-version` and the
lockfile is committed, so this should reproduce exactly:

```bash
uv sync                          # creates .venv and installs everything
uv run python cv/utils.py        # run anything without activating
uv run jupyter lab               # notebooks
uv run pytest                    # tests
```

If you'd rather activate the environment the normal way, `source .venv/bin/activate` after
`uv sync` works fine.

Adding a dependency is `uv add <package>`, which updates both `pyproject.toml` and `uv.lock`.
The default torch wheel is CPU-only on Linux — there's a commented block in `pyproject.toml`
for pointing at a CUDA index instead.

## How I work

Four rules, and they're most of the reason this repo is worth anything to me:

**Type it, don't paste it.** Every important line gets typed by hand. Pasting a working
implementation teaches me nothing except that it works.

**Predict before running.** Before any `.shape`, matmul, or model output I write down what I
expect. When I'm wrong, that gap is the actual lesson, and it goes in the journal.

**Rebuild closed-book.** Following along doesn't count. I close the tutorial and rebuild the
core idea from memory before moving on. If I can't, I didn't learn it.

**AI is a tutor before it's a code generator.** Conceptual hints first, pseudocode if I'm
genuinely stuck, the solution last — and then I close it and write it myself.

Every training script calls `set_seed()` and every run gets a row in
[`experiments/experiments.csv`](experiments/experiments.csv). If I report a number I can
reproduce it from a clean clone.