# The Call Box

The complete, closed set of calls I am learning cold — and the explicit list of what I am *not*.

## How to use this

This is a **curriculum map, not a lookup table.** The difference matters:

- **Closed while building.** If this file is open in a second pane during a build block, it is doing the same damage a tutorial does — removing the retrieval attempt that is the entire mechanism.
- **Open on Sunday.** Once a week, to check off what's gone cold and pick the next tier.
- **Cold means:** I can write the call, with correct argument order and correct output shape, from a blank file, without hesitating. Not "I'd recognize it." Not "I know roughly what it does."

Everything here is one of three things:

| Status | Meaning |
|---|---|
| **Cold** | Recall from memory, no lookup. ~75 calls total across three tiers. |
| **Warm** | Know it exists and what problem it solves. Look up the signature every time, forever. That's fine. |
| **Never** | Don't even try. Look it up, copy it, move on. |

The bounded size is the point. This is finishable.

---

## Tier A — cold by end of Week 2

The inner loop. These appear in every project, every day. Drilled by `reps.py` tier A.

### Create and inspect

| Call | For | The gotcha | Rep |
|---|---|---|---|
| `torch.tensor(data, dtype=)` | Build from python data | Copies. `torch.as_tensor` doesn't. | — |
| `torch.randn / rand / zeros / ones` | Test tensors | `randn` is normal, `rand` is uniform [0,1) | a01 |
| `torch.arange / linspace` | Index and sweep vectors | `arange` excludes the stop; `linspace` includes it | a15 |
| `torch.Generator().manual_seed(s)` | Reproducible draws | Pass as `generator=`; global seed is separate | a15 |
| `.shape` / `.ndim` / `.numel()` | Inspect | `.shape` is a `torch.Size`, a tuple subclass | all |
| `.dtype` | Inspect | Labels must be `int64`, inputs `float32`. Silent `float64` from numpy is a classic. | a06 |
| `.device` / `.to(device)` | Move | On a **tensor** returns a new one; on a **module** it's in-place. | — |
| `.item()` | Tensor → python scalar | Scalars only. Forces a GPU sync — never per-element in a loop. | a14 |

### Reshape

| Call | For | The gotcha | Rep |
|---|---|---|---|
| `.view(shape)` | Reinterpret | Fails on non-contiguous. That failure is informative — read it. | a04 |
| `.reshape(shape)` | Reinterpret, safely | Silently copies when it can't view. Convenient, hides cost. | a04 |
| `.permute(dims)` | Reorder axes | Result is non-contiguous. `.view` after it will fail. | a05 |
| `.transpose(d0, d1)` | Swap two axes | Same contiguity issue | a05 |
| `.contiguous()` | Fix the above | Only after permute/transpose, never reflexively | a05 |
| `.squeeze(dim)` | Drop a size-1 axis | **Always pass `dim`.** Bare `.squeeze()` drops *all* size-1 dims — kills batch dim when N=1. | a09 |
| `.unsqueeze(dim)` | Add a size-1 axis | The broadcasting workhorse | a09, a12 |
| `.flatten(start_dim)` | Collapse trailing axes | `start_dim=1` keeps the batch | a04 |
| `torch.cat(list, dim)` | Join along existing axis | Shapes must match on all *other* dims | b08 |
| `torch.stack(list, dim)` | Join along a **new** axis | cat vs stack is the single most common batching confusion | b08 |
| `.expand` vs `.repeat` | Broadcast vs copy | `expand` is a free view; `repeat` allocates | — |

### Reduce

| Call | For | The gotcha | Rep |
|---|---|---|---|
| `.sum(dim, keepdim)` | Reduce | `keepdim=True` when the result feeds a broadcast | a02 |
| `.mean(dim, keepdim)` | Reduce | Multi-axis: `dim=(0,2,3)` | a01, a02 |
| `.std(dim, unbiased=)` | Reduce | `unbiased=True` is the default and is *not* what normalization wants | a03 |
| `.max(dim)` | Reduce | With `dim`, returns `(values, indices)`, not a tensor. Without, returns a scalar. | a06 |
| `.argmax(dim)` | Predicted class | This is what accuracy is built on | a06 |
| `.topk(k, dim)` | Top-k | Returns `(values, indices)`, sorted descending | a08 |
| `.any(dim)` / `.all(dim)` | Boolean reduce | Assertion glue | c06 |

### Index

| Call | For | The gotcha | Rep |
|---|---|---|---|
| Boolean mask `x[mask]` | Filter rows | Mask shape must match the leading dims | a10 |
| `x[torch.arange(n), idx]` | One element per row | The idiom for "pick per-row." Learn it as a unit. | a11 |
| `.gather(dim, index)` | Same, generalized | `index` must have the **same ndim** as the source — hence the `.unsqueeze(1)` | a09 |
| `.masked_fill(mask, value)` | Conditional overwrite | Non-in-place variant; `_` suffix mutates | — |

### Math

| Call | For | The gotcha | Rep |
|---|---|---|---|
| Broadcast arithmetic | Everything | Right-align, size 1 stretches, missing dims prepend. Memorize the rule, not the cases. | a03, a12 |
| `@` / `torch.matmul` | Matmul | `@` broadcasts batch dims; `.mm` doesn't | a13 |
| `F.softmax(x, dim)` | Probabilities | **Always pass `dim`.** Never feed to `cross_entropy`. | — |
| `F.log_softmax(x, dim)` | Log-probs | Stable; `log(softmax(x))` is not | a07 |
| `torch.exp / log / sqrt / clamp` | Elementwise | `clamp(min=0)` is the intersection-area idiom | c05, c08 |
| Max-subtraction trick | Numerical stability | `x - x.max(dim, keepdim).values` before `exp`. Not a call — a reflex. | a07 |

### Train

| Call | For | The gotcha | Rep |
|---|---|---|---|
| `nn.Module` + `__init__` + `forward` | Model | `super().__init__()` first or nothing registers | c02 |
| `nn.Linear(in, out)` | Dense layer | Weight is stored `(out, in)` — trips everyone once | a15 |
| `nn.Sequential(...)` | Stack | Indexable: `model[-1]` is the head | c02, c04 |
| `nn.ReLU()` vs `F.relu()` | Nonlinearity | Module for `Sequential`, functional for `forward` | c02 |
| `F.cross_entropy(logits, labels)` | Loss | Takes **raw logits**. Labels are `int64` class indices, not one-hot. | a07, a14 |
| `F.mse_loss(pred, target)` | Loss | Shapes must match exactly — broadcasting here is a silent bug | a15 |
| `optim.SGD / Adam(params, lr=)` | Optimizer | Pass `model.parameters()`, not the model | a14 |
| `.zero_grad()` → `.backward()` → `.step()` | The step | **Order.** Zero before backward, step after. Skipping zero accumulates silently. | a14 |
| `torch.no_grad()` | Disable autograd | Saves memory in eval. Orthogonal to `.eval()`. | b06 |
| `.detach()` | Break the graph | For logging a loss without holding the graph alive | — |
| `torch.manual_seed(s)` | Global seed | Necessary, not sufficient — see b01 | a15 |

---

## Tier B — cold by end of Week 3

Data plumbing, evaluation, reproducibility. Drilled by `reps.py` tier B.

| Call | For | The gotcha | Rep |
|---|---|---|---|
| `Dataset` + `__len__` + `__getitem__` | Custom data | Return tensors, not PIL images | b09 |
| `DataLoader(ds, batch_size, shuffle)` | Batching | `shuffle=True` train, `False` val — always | b09 |
| `collate_fn` | Custom batching | Needed the moment a sample isn't `(tensor, int)` | b08 |
| `num_workers` / `worker_init_fn` | Parallel loading | Lambdas aren't picklable. Workers need their own seeds. | — |
| `drop_last` / `pin_memory` | Batching detail | `drop_last=True` when BatchNorm would see a size-1 batch | — |
| `model.train()` / `model.eval()` | Mode | Affects **Dropout and BatchNorm only**. Does *not* disable grad. Pair with `no_grad()`. | b06 |
| `torch.save` / `torch.load(weights_only=True)` | Persistence | Save `state_dict()`, not the model object | c10 |
| `.state_dict()` / `.load_state_dict()` | Persistence | Save the **optimizer** too or resuming is fiction | c10 |
| `torch.randperm(n, generator=)` | Shuffling | The seeded-split primitive | b02 |
| `torch.bincount(x, minlength=)` | Counts | `minlength` or empty classes vanish | b04 |
| `.unique(return_counts=True)` | Class discovery | — | b02 |
| `.nonzero(as_tuple=True)` | Mask → indices | The tuple form is the one you want | b02 |
| `torch.where(cond, a, b)` | Vectorized if | — | c07 |
| `.index_put_` / `.scatter_add_` | Accumulate at indices | `accumulate=True` or you overwrite | b05 |
| `.float()` / `.long()` / `.bool()` | Casting | `.long()` for labels, always | b08 |
| `sum(p.numel() for p in model.parameters())` | Model size | Filter on `requires_grad` for the trainable count | b10 |
| `transforms.Compose / ToTensor / Normalize / Resize` | Preprocessing | `ToTensor` does HWC→CHW **and** rescales to [0,1]. Both. | b07 |
| `Image.open(p).convert("RGB")` | Loading | The convert is not optional — grayscale JPEGs exist in every real dataset | — |

---

## Tier C — cold by end of Week 6

Convolutions, transfer, detection. Drilled by `reps.py` tier C.

| Call | For | The gotcha | Rep |
|---|---|---|---|
| `nn.Conv2d(in, out, k, stride, padding)` | Conv | `padding=k//2` preserves size for odd `k` | c02 |
| Output-size formula | Shape math | `floor((in + 2p - d(k-1) - 1)/s) + 1`. In your head. | c01 |
| `nn.MaxPool2d(k)` | Downsample | Halves spatial size at `k=2` | c02 |
| `nn.AdaptiveAvgPool2d(1)` | Global pool | This is what makes input size free. The reason modern heads work. | c02 |
| `nn.BatchNorm2d(c)` | Normalization | Breaks on batch size 1 in train mode. Behaves differently in eval. | — |
| `nn.Dropout(p)` | Regularization | Inactive in eval — which is why `.eval()` matters | — |
| `nn.Flatten()` | Conv → linear | Or `AdaptiveAvgPool2d` + `Flatten` | c02 |
| `p.requires_grad_(False)` | Freezing | Freeze **before** constructing the optimizer, or it still tracks them | c03 |
| `model.fc = nn.Linear(in_f, k)` | Head swap | Read `in_features` off the old head; don't hardcode | c04 |
| `torchvision.models.resnet18(weights=)` | Pretrained | `weights=` not `pretrained=`; the latter is deprecated | — |
| `RandomResizedCrop / RandomHorizontalFlip / ColorJitter` | Augmentation | Train transform only. Never on val. | — |
| Box format `(x1, y1, x2, y2)` | Detection | vs `(cx, cy, w, h)`. Conversion bugs are most of detection debugging. | c08 |
| IoU by hand | Detection | `clamp(min=0)` on the intersection or negatives leak through | c08 |
| Greedy NMS by hand | Detection | Sort descending, keep, suppress, repeat | c09 |

---

## Warm — know it exists, look up the signature forever

No shame in this column. These appear once per project, not once per loop.

`torch.einsum` · `einops.rearrange` · `torch.cdist` · `clip_grad_norm_` · LR schedulers (`CosineAnnealingLR`, `ReduceLROnPlateau`) · `torch.amp.autocast` + `GradScaler` · `persistent_workers` / `prefetch_factor` · `torch.compile` · `torch.profiler` · `torchmetrics` · sklearn (`StratifiedShuffleSplit`, `classification_report`, `roc_auc_score`) · Albumentations · ONNX export · `torch.jit` · weight init schemes · `register_buffer` / `register_forward_hook`

## Never — look it up, copy it, move on

Default argument values · ImageNet normalization constants · model zoo names and their exact head attribute (`.fc` vs `.classifier` vs `.head`) · argparse boilerplate · matplotlib incantations · CUDA version tables · anything in a `setup.py`

---

## Direction

**The sorting principle is frequency, not difficulty.** `einsum` is harder than `.gather` and belongs in the warm column anyway, because I'll write `.gather` a hundred times this year and `einsum` twice. Anything in the inner loop of a training run goes cold. Anything that appears once at the top of a script gets looked up. When something migrates from once-a-project to once-a-day, promote it.

**Sequence and why:**

1. **Tier A first, alone.** Shapes and the training loop. Nothing about images. If the loop isn't automatic, every CNN bug will read as a CNN bug when it's actually a loop bug, and I'll debug the wrong layer of the stack for a week.
2. **Tier B before touching a real dataset.** Splits and eval come before models. A model trained on a leaked split produces a number that feels like progress and isn't. The assertions are load-bearing precisely because the failure is silent.
3. **Tier C only once A and B are cold.** Conv arithmetic on top of shaky reshape fluency is two unknowns in one bug.

**Tier gates — a tier is done when:**
- 15/15 (or 10/10) cold, three consecutive days
- I can write the calls in a *different* context than the drill — same op, new problem
- No 90-second stalls in the last two sessions

**What comes after the call box.** This is worth being clear-eyed about, because finishing a finite list feels like arrival and isn't. Three ceilings, in order:

1. **API fluency** — this document. Roughly three weeks. Purely mechanical; reps solve it.
2. **Debugging fluency** — reading a shape error and knowing which line, spotting a silent broadcast, diagnosing a flat loss curve. Not drillable. Comes from volume of broken runs, which is why the experiment log matters more than it looks.
3. **Judgment** — which experiment is worth running, when a result is real, when to stop. Years. This is the actual job.

The call box is the price of entry to ceiling 2, not the destination. Its whole value is that it's finite and can be closed out fast, so the years-long thing can start sooner.

**One trap to name.** The temptation, once this list is written down, is to keep expanding it — add another twenty calls, make it exhaustive, feel productive. Don't. A call box that grows is a call box that never goes cold. If something genuinely earns promotion from warm to cold, promote it and *drop something else*. Fixed size.