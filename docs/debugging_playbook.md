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