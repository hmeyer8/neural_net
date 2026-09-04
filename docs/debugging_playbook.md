# Debugging Playbook

Built by breaking my own working code on purpose and diagnosing each failure from symptoms
alone. Format is fixed: **symptom -> hypothesis -> confirmation -> rule.**

Filled in during Week 2. Added to permanently after that.

---

## The sanity check that comes before all of this

**Overfit a single batch to ~100% accuracy.** If a model cannot memorize eight examples, the
problem is the pipeline, not the architecture, the learning rate, or the amount of data. Nothing
else is worth debugging until this passes.

---

## 1. Loss doesn't decrease at all

- **Symptom:**
- **Hypothesis:**
- **How I confirmed it:**
- **Rule:**

## 2. Loss explodes to NaN

- **Symptom:**
- **Hypothesis:**
- **How I confirmed it:**
- **Rule:**

## 3. Loss decreases absurdly slowly

- **Symptom:**
- **Hypothesis:**
- **How I confirmed it:**
- **Rule:**

## 4. Training accuracy climbs, validation flat

- **Symptom:**
- **Hypothesis:**
- **How I confirmed it:**
- **Rule:**

## 5. Validation accuracy suspiciously higher than training

- **Symptom:**
- **Hypothesis:**
- **How I confirmed it:**
- **Rule:**

## 6. Model performs worse at inference than at validation

- **Symptom:**
- **Hypothesis:**
- **How I confirmed it:**
- **Rule:**

---

## Silent failures

The dangerous category. Code runs, nothing throws, results are wrong.

| Failure | How it hides | How to catch it |
|---|---|---|
| Missing `keepdim` before a broadcast divide | Shape-compatible, so no error. Divides along the wrong axis. | Assert the result sums to what it should. |
| Softmax applied before `CrossEntropyLoss` | Trains, just badly. | PyTorch's cross-entropy takes raw logits. |
| Missing `model.eval()` at validation | BatchNorm and dropout stay in training mode. | Assert `not model.training` inside the eval loop. |
| Augmenting the validation set | Val metric is noisy and pessimistic. | Separate transform pipelines, always. |
| Near-duplicates across train and test | Metric looks great, generalization doesn't exist. | Hash or embed and check overlap. |
| Normalization stats from the full dataset | Test statistics leak into training. | Compute on the train split only. |

---

## Silent failures specific to a GPU

Added 2026-09-04, when this repo moved from a CPU-only torch wheel to `cu130` on an RTX 3050. Every
one of these runs clean and returns plausible numbers.

| Failure | How it hides | How to catch it |
|---|---|---|
| Torch installed as the CPU wheel while you believe you're on GPU | Everything works, just ~7x slower. Nothing errors, and "it felt slow" is not a diagnosis. | `torch.__version__` must end in `+cuXXX`, not `+cpu`. `rfcagent status` prints it. |
| Model on GPU, input tensors on CPU | Raises rather than hides *if* they meet in an op — but a model that never sees a tensor (an empty batch, a short-circuited branch) silently does nothing. | Assert `next(model.parameters()).device` matches where the batch is built. |
| Half the pipeline on GPU, half on CPU | Two modules each call `cuda.is_available()` independently and disagree after an env change. Latency numbers become meaningless while quality looks fine. | Resolve the device once, pass it down. Never re-ask at the call site. |
| Comparing latency across devices in one table | GPU and CPU rows look like a fair ablation and are not. | Record the device as its own column; only compare within a device. |
| Seeded run that still won't reproduce on CUDA | cuDNN benchmarks algorithms at runtime; some reductions accumulate nondeterministically. Near-tied retrieval hits swap order and read as a regression. | `torch.backends.cudnn.deterministic = True`, `benchmark = False` before any reported run. |
| Batch size tuned for a datacenter GPU on a 6 GB laptop | OOM three hours into an index build, after the useful work is lost. | Size the batch for VRAM shared with a desktop compositor, not for the card's spec sheet. Checkpoint long builds. |
| VRAM not released between stages | Index build then rerank in one process; the second stage OOMs while `nvidia-smi` shows memory held by the first. | `del model; torch.cuda.empty_cache()` between stages, or run stages as separate processes. |