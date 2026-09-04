"""
reps.py — daily retrieval drill for PyTorch/tensor API fluency.

WHAT THIS IS
    A prompt file. Every function below is a spec with an empty body. You fill
    the bodies from memory, run the file, and it tells you what passed.
    It is NOT a tutorial and it is NOT a reference. There are no solutions in
    this file and there never should be.

THE PROTOCOL
    1.  Ten minutes. Every morning. Before anything else.
    2.  Run ONE tier per session (see TIERS below). Tier A until it is 15/15.
    3.  Before writing a line, write the shape you expect as a comment above it.
        Wrong predictions are the point. Log them.
    4.  Stuck for 90 seconds -> official docs. NOT an AI, NOT this file's checks,
        NOT yesterday's answers. Mark the item in your journal; re-drill tomorrow.
    5.  When the session ends, throw your answers away:
            git checkout -- reps.py
        The file must be back to prompts-only before tomorrow. If you keep your
        answers, tomorrow is reading practice, not retrieval practice.
    6.  Paste the score line into reps.md. Watch the trend, not the day.

RUNNING IT
    python reps.py           # runs tier A
    python reps.py A         # same
    python reps.py A B       # tiers A and B
    python reps.py all       # everything

TIERS
    A  (15)  Core. Daily, weeks 1-8. This is the surface area that covers
             ~90% of everything you will write. Do not move on until it is 15/15
             cold, three days running.
    B  (10)  Data, evaluation, reproducibility. Unlock at the Week 1 gate.
    C  (10)  Convolutions, transfer learning, detection. Unlock in Week 4.

A NOTE ON THE CHECKS AT THE BOTTOM
    The checks verify behavior; they do not contain the implementations. Where a
    check compares against a library function (F.cross_entropy, torch.topk,
    torch.cdist), that function is the *reference*, not the method — knowing it
    exists tells you nothing about how to write the thing by hand, which is the
    actual rep. Reading the check block to figure out an answer is the same as
    looking up the answer. Don't.
"""

import sys
import time
import tempfile
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# TIER A — CORE (daily)
# =============================================================================

def a01_per_channel_mean(x):
    """Mean over batch and spatial dims, per channel.

    in:  x (N, C, H, W) float32
    out:    (C,) float32
    """
    raise NotImplementedError


def a02_per_channel_mean_keepdim(x):
    """Same value as a01, but shaped so `x - result` broadcasts cleanly.

    in:  x (N, C, H, W) float32
    out:    (1, C, 1, 1) float32
    """
    raise NotImplementedError


def a03_standardize_per_feature(x, eps=1e-8):
    """Zero-mean, unit-variance per COLUMN. Population std (unbiased=False).

    in:  x (N, D) float32
    out:    (N, D) float32, each column mean ~0 and std ~1
    """
    raise NotImplementedError


def a04_flatten_images(x):
    """Flatten each image to a vector, keeping the batch dim.

    in:  x (N, C, H, W)
    out:    (N, C*H*W)
    """
    raise NotImplementedError


def a05_nhwc_to_nchw(x):
    """Channels-last -> channels-first. Result must be contiguous.

    in:  x (N, H, W, C)
    out:    (N, C, H, W), out[n, c, h, w] == x[n, h, w, c], out.is_contiguous()
    """
    raise NotImplementedError


def a06_accuracy(logits, labels):
    """Top-1 accuracy. No torch.nn.functional.

    in:  logits (N, K) float32, labels (N,) int64
    out: python float in [0, 1]
    """
    raise NotImplementedError


def a07_manual_cross_entropy(logits, labels):
    """Mean cross-entropy, computed by hand.

    Numerically stable. Must match F.cross_entropy(logits, labels).
    BANNED: F.cross_entropy, F.nll_loss, F.log_softmax, F.softmax, nn.*Loss.

    in:  logits (N, K) float32, labels (N,) int64
    out: scalar tensor
    """
    raise NotImplementedError


def a08_topk_classes(logits, k):
    """Indices of the k highest-scoring classes per row, highest first.

    in:  logits (N, K) float32, k int
    out:        (N, k) int64
    """
    raise NotImplementedError


def a09_gather_true_logit(logits, labels):
    """The logit of the correct class for each row. Use .gather.

    in:  logits (N, K) float32, labels (N,) int64
    out:        (N,) float32
    """
    raise NotImplementedError


def a10_select_classes(logits, labels, keep):
    """Keep only the rows whose label is in `keep`. Preserve original order.

    in:  logits (N, K), labels (N,) int64, keep list[int]
    out: (logits_subset (M, K), labels_subset (M,))
    """
    raise NotImplementedError


def a11_one_hot(labels, num_classes):
    """One-hot encode. BANNED: F.one_hot, torch.eye indexing tricks are fine.

    in:  labels (N,) int64, num_classes int
    out:        (N, num_classes) float32
    """
    raise NotImplementedError


def a12_pairwise_sq_dists(a, b):
    """Squared Euclidean distance between every row of a and every row of b.

    Broadcasting only. No python loops, no torch.cdist.

    in:  a (N, D) float32, b (M, D) float32
    out:   (N, M) float32
    """
    raise NotImplementedError


def a13_linear_forward(x, W, bias):
    """A linear layer, by hand.

    in:  x (N, D) float32, W (D, K) float32, bias (K,) float32
    out:   (N, K) float32
    """
    raise NotImplementedError


def a14_train_step(model, x, y, optimizer):
    """Exactly one optimization step: forward, loss, zero, backward, step.

    Cross-entropy loss. Get the order right — this is the one that costs you
    silent bugs for a week when you get it wrong.

    in:  model nn.Module, x (N, D), y (N,) int64, optimizer torch.optim.Optimizer
    out: python float — the loss BEFORE the step
    """
    raise NotImplementedError


def a15_fit_linear(xs, ys, steps=300, lr=0.05, seed=0):
    """The whole loop, from nothing. Fit ys ~= w*xs + b.

    Build an nn.Linear(1, 1), an optimizer, an MSE loss, and train it. Seed
    first so the run is reproducible.

    in:  xs (N, 1) float32, ys (N, 1) float32
    out: (final_loss float, w float, b float)
    """
    raise NotImplementedError


# =============================================================================
# TIER B — DATA, EVAL, REPRODUCIBILITY (unlock at the Week 1 gate)
# =============================================================================

def b01_set_all_seeds(seed):
    """Make the run reproducible: python, numpy if present, torch CPU + CUDA.

    in:  seed int
    out: None
    """
    raise NotImplementedError


def b02_stratified_indices(labels, train_frac, seed):
    """Split indices into train/val, preserving per-class proportions.

    Deterministic given the seed. Disjoint. Union covers every index exactly
    once. No scikit-learn.

    in:  labels (N,) int64, train_frac float, seed int
    out: (train_idx (T,) int64, val_idx (V,) int64)
    """
    raise NotImplementedError


def b03_assert_no_leakage(train_items, val_items):
    """Raise AssertionError if any item appears in both splits. Else return None.

    in:  train_items, val_items — sequences of hashables (e.g. file paths)
    out: None, or raises AssertionError
    """
    raise NotImplementedError


def b04_class_counts(labels, num_classes):
    """Count of each class. Classes with zero members must appear as 0.

    in:  labels (N,) int64, num_classes int
    out:        (num_classes,) int64
    """
    raise NotImplementedError


def b05_confusion_matrix(preds, labels, num_classes):
    """out[i, j] = number of samples with true class i predicted as j.

    No scikit-learn. No python loop over samples.

    in:  preds (N,) int64, labels (N,) int64, num_classes int
    out:       (num_classes, num_classes) int64
    """
    raise NotImplementedError


def b06_evaluate(model, batches):
    """Evaluate without training. Leaves the model in eval mode.

    Sets eval mode, disables grad, accumulates mean cross-entropy weighted by
    batch size, and top-1 accuracy over all samples.

    in:  model nn.Module, batches iterable of (x (n, D), y (n,) int64)
    out: (mean_loss float, accuracy float)
    """
    raise NotImplementedError


def b07_normalize(x, mean, std):
    """Per-channel normalize a batch of images. Broadcasting, no loops.

    in:  x (N, 3, H, W) float32 in [0, 1]; mean, std each a 3-element sequence
    out:   (N, 3, H, W) float32, (x - mean) / std per channel
    """
    raise NotImplementedError


def b08_collate_with_paths(batch):
    """Collate a list of samples into a batch, carrying the path through.

    in:  batch — list of (image (C, H, W), label int, path str)
    out: (images (B, C, H, W), labels (B,) int64, paths tuple[str, ...])
    """
    raise NotImplementedError


class B09TinyDataset(torch.utils.data.Dataset):
    """A minimal map-style Dataset over an in-memory list.

    __init__: store `items`, a list of (feature_list, label int)
    __len__:  number of items
    __getitem__(i): returns (features float32 tensor (D,), label int)
    """
    def __init__(self, items):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


def b10_count_parameters(model, trainable_only=True):
    """Total number of scalar parameters.

    in:  model nn.Module, trainable_only bool
    out: python int
    """
    raise NotImplementedError


# =============================================================================
# TIER C — CONV, TRANSFER, DETECTION (unlock in Week 4)
# =============================================================================

def c01_conv_output_shape(h, w, kernel, stride=1, padding=0, dilation=1):
    """Output spatial size of a conv/pool. Pure arithmetic — no torch.

    You will need this in your head, not in a notebook.

    out: (h_out int, w_out int)
    """
    raise NotImplementedError


def c02_build_tiny_cnn(num_classes):
    """A small CNN that accepts ANY input spatial size.

    Two conv blocks (conv -> relu -> maxpool), then a global pool, then a linear
    head. The global pool is what makes the input size free.

    in:  num_classes int
    out: nn.Module mapping (N, 3, H, W) -> (N, num_classes)
    """
    raise NotImplementedError


def c03_freeze_backbone(model, unfreeze_last_n):
    """Freeze everything, then unfreeze the last n parameter tensors.

    in:  model nn.Module, unfreeze_last_n int
    out: the same model, mutated
    """
    raise NotImplementedError


def c04_replace_head(model, num_classes):
    """Swap the final nn.Linear for a fresh one with the same in_features.

    in:  model — an nn.Sequential whose last module is nn.Linear
    out: the same model, mutated
    """
    raise NotImplementedError


def c05_unnormalize(x, mean, std):
    """Invert b07 and clamp to [0, 1] for display.

    in:  x (N, 3, H, W) normalized; mean, std 3-element sequences
    out:   (N, 3, H, W) in [0, 1]
    """
    raise NotImplementedError


def c06_topk_accuracy(logits, labels, k):
    """Fraction of rows whose true class is among the top k predictions.

    in:  logits (N, K), labels (N,) int64, k int
    out: python float
    """
    raise NotImplementedError


def c07_per_class_accuracy(preds, labels, num_classes):
    """Recall per class. Classes with no true samples -> nan.

    in:  preds (N,) int64, labels (N,) int64, num_classes int
    out:       (num_classes,) float32
    """
    raise NotImplementedError


def c08_iou(boxes_a, boxes_b):
    """Pairwise intersection-over-union. Boxes are (x1, y1, x2, y2).

    Broadcasting, no loops, no torchvision.

    in:  boxes_a (N, 4) float32, boxes_b (M, 4) float32
    out:         (N, M) float32
    """
    raise NotImplementedError


def c09_nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression. No torchvision.

    Repeatedly take the highest-scoring remaining box, keep it, and discard
    every remaining box whose IoU with it exceeds the threshold.

    in:  boxes (N, 4) float32, scores (N,) float32, iou_threshold float
    out: (K,) int64 — kept indices, in the order they were kept
    """
    raise NotImplementedError


def c10_checkpoint_roundtrip(model, optimizer, path):
    """Save model + optimizer state to `path`, then load it into fresh objects.

    Build the fresh model with the same architecture, load the state dicts, and
    return them. This is the rep that saves you when a run dies at epoch 40.

    in:  model nn.Module, optimizer torch.optim.Optimizer, path str
    out: (loaded_model nn.Module, loaded_optimizer torch.optim.Optimizer)
    """
    raise NotImplementedError


# =============================================================================
# CHECKS — behavior only. Do not read these to find an answer.
# =============================================================================

def _g(seed=0):
    return torch.Generator().manual_seed(seed)


def _check_a01():
    x = torch.randn(8, 3, 5, 5, generator=_g(1))
    out = a01_per_channel_mean(x)
    assert out.shape == (3,), f"shape {tuple(out.shape)}"
    assert torch.allclose(out, x.mean(dim=(0, 2, 3)), atol=1e-6)


def _check_a02():
    x = torch.randn(8, 3, 5, 5, generator=_g(2))
    out = a02_per_channel_mean_keepdim(x)
    assert out.shape == (1, 3, 1, 1), f"shape {tuple(out.shape)}"
    assert (x - out).shape == x.shape
    assert torch.allclose(out.flatten(), x.mean(dim=(0, 2, 3)), atol=1e-6)


def _check_a03():
    x = torch.randn(64, 4, generator=_g(3)) * 7 + 3
    out = a03_standardize_per_feature(x)
    assert out.shape == x.shape, f"shape {tuple(out.shape)}"
    assert torch.allclose(out.mean(0), torch.zeros(4), atol=1e-5), "columns not zero-mean"
    assert torch.allclose(out.std(0, unbiased=False), torch.ones(4), atol=1e-4), "columns not unit-std"


def _check_a04():
    x = torch.randn(6, 3, 4, 5, generator=_g(4))
    out = a04_flatten_images(x)
    assert out.shape == (6, 60), f"shape {tuple(out.shape)}"
    assert torch.equal(out[2], x[2].reshape(-1))


def _check_a05():
    x = torch.randn(2, 4, 5, 3, generator=_g(5))
    out = a05_nhwc_to_nchw(x)
    assert out.shape == (2, 3, 4, 5), f"shape {tuple(out.shape)}"
    assert out.is_contiguous(), "result is not contiguous"
    assert out[1, 2, 3, 4] == x[1, 3, 4, 2], "axes permuted incorrectly"


def _check_a06():
    logits = torch.tensor([[3.0, 1.0], [0.5, 2.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1, 1, 1])
    out = a06_accuracy(logits, labels)
    assert isinstance(out, float), f"returned {type(out).__name__}, expected float"
    assert abs(out - 0.75) < 1e-6, f"got {out}, expected 0.75"


def _check_a07():
    logits = torch.randn(16, 7, generator=_g(7)) * 4
    labels = torch.randint(0, 7, (16,), generator=_g(8))
    out = a07_manual_cross_entropy(logits, labels)
    ref = F.cross_entropy(logits, labels)
    assert torch.is_tensor(out) and out.ndim == 0, "expected a scalar tensor"
    assert torch.allclose(out, ref, atol=1e-5), f"got {out.item():.6f}, reference {ref.item():.6f}"
    big = torch.tensor([[500.0, 0.0, -500.0]])
    assert torch.isfinite(a07_manual_cross_entropy(big, torch.tensor([0]))), "not numerically stable"


def _check_a08():
    logits = torch.randn(10, 8, generator=_g(9))
    out = a08_topk_classes(logits, 3)
    assert out.shape == (10, 3), f"shape {tuple(out.shape)}"
    assert out.dtype == torch.int64, f"dtype {out.dtype}"
    assert torch.equal(out, logits.topk(3, dim=1).indices)


def _check_a09():
    logits = torch.randn(12, 5, generator=_g(10))
    labels = torch.randint(0, 5, (12,), generator=_g(11))
    out = a09_gather_true_logit(logits, labels)
    assert out.shape == (12,), f"shape {tuple(out.shape)}"
    expected = torch.stack([logits[i, labels[i]] for i in range(12)])
    assert torch.allclose(out, expected)


def _check_a10():
    logits = torch.randn(10, 4, generator=_g(12))
    labels = torch.tensor([0, 3, 1, 2, 0, 1, 3, 3, 2, 0])
    sub_logits, sub_labels = a10_select_classes(logits, labels, [0, 1])
    keep_rows = [0, 2, 4, 5, 9]
    assert sub_logits.shape == (5, 4), f"shape {tuple(sub_logits.shape)}"
    assert torch.equal(sub_labels, labels[keep_rows]), "order not preserved"
    assert torch.equal(sub_logits, logits[keep_rows])


def _check_a11():
    labels = torch.tensor([2, 0, 1, 2])
    out = a11_one_hot(labels, 4)
    assert out.shape == (4, 4), f"shape {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"dtype {out.dtype}"
    assert torch.equal(out, F.one_hot(labels, 4).float())


def _check_a12():
    a = torch.randn(6, 3, generator=_g(13))
    b = torch.randn(9, 3, generator=_g(14))
    out = a12_pairwise_sq_dists(a, b)
    assert out.shape == (6, 9), f"shape {tuple(out.shape)}"
    assert torch.allclose(out, torch.cdist(a, b) ** 2, atol=1e-3)


def _check_a13():
    x = torch.randn(5, 3, generator=_g(15))
    W = torch.randn(3, 7, generator=_g(16))
    bias = torch.randn(7, generator=_g(17))
    out = a13_linear_forward(x, W, bias)
    assert out.shape == (5, 7), f"shape {tuple(out.shape)}"
    assert torch.allclose(out, x @ W + bias, atol=1e-5)


def _check_a14():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    x = torch.randn(16, 4, generator=_g(18))
    y = torch.randint(0, 3, (16,), generator=_g(19))
    before = model.weight.detach().clone()
    loss = a14_train_step(model, x, y, opt)
    assert isinstance(loss, float), f"returned {type(loss).__name__}, expected float"
    assert not torch.equal(model.weight.detach(), before), "weights did not change — no .step()?"
    grad_after = model.weight.grad.detach().clone()
    a14_train_step(model, x, y, opt)
    assert not torch.equal(model.weight.grad.detach(), grad_after * 2), \
        "gradients appear to be accumulating — missing .zero_grad()?"


def _check_a15():
    xs = torch.linspace(-2, 2, 64).unsqueeze(1)
    ys = 3.0 * xs + 2.0
    loss, w, b = a15_fit_linear(xs, ys, steps=800, lr=0.05, seed=0)
    assert loss < 1e-2, f"final loss {loss:.5f} — did not converge"
    assert abs(w - 3.0) < 0.1, f"w = {w:.4f}, expected ~3.0"
    assert abs(b - 2.0) < 0.1, f"b = {b:.4f}, expected ~2.0"
    loss2, _, _ = a15_fit_linear(xs, ys, steps=800, lr=0.05, seed=0)
    assert abs(loss - loss2) < 1e-9, "same seed gave a different result — not reproducible"


def _check_b01():
    b01_set_all_seeds(1234)
    a = torch.randn(5)
    b01_set_all_seeds(1234)
    b = torch.randn(5)
    assert torch.equal(a, b), "reseeding did not reproduce the draw"


def _check_b02():
    labels = torch.tensor([0] * 40 + [1] * 20 + [2] * 40)
    tr, va = b02_stratified_indices(labels, 0.8, seed=7)
    assert tr.dtype == torch.int64 and va.dtype == torch.int64
    assert len(set(tr.tolist()) & set(va.tolist())) == 0, "splits overlap"
    assert sorted(tr.tolist() + va.tolist()) == list(range(100)), "splits do not cover every index once"
    for c, n in [(0, 40), (1, 20), (2, 40)]:
        got = int((labels[tr] == c).sum())
        assert abs(got - 0.8 * n) <= 1, f"class {c}: {got} train samples, expected ~{0.8 * n:.0f}"
    tr2, _ = b02_stratified_indices(labels, 0.8, seed=7)
    assert torch.equal(tr, tr2), "same seed gave a different split"


def _check_b03():
    b03_assert_no_leakage(["a", "b"], ["c", "d"])
    try:
        b03_assert_no_leakage(["a", "b"], ["b", "c"])
    except AssertionError:
        return
    raise AssertionError("did not raise on an overlapping item")


def _check_b04():
    labels = torch.tensor([0, 0, 2, 2, 2])
    out = b04_class_counts(labels, 4)
    assert out.shape == (4,), f"shape {tuple(out.shape)}"
    assert out.tolist() == [2, 0, 3, 0], f"got {out.tolist()}"


def _check_b05():
    preds = torch.tensor([0, 1, 1, 2, 0])
    labels = torch.tensor([0, 1, 2, 2, 1])
    out = b05_confusion_matrix(preds, labels, 3)
    assert out.shape == (3, 3), f"shape {tuple(out.shape)}"
    assert out.tolist() == [[1, 0, 0], [1, 1, 0], [0, 1, 1]], f"got {out.tolist()}"


def _check_b06():
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    model.train()
    xs = torch.randn(20, 3, generator=_g(20))
    ys = torch.randint(0, 2, (20,), generator=_g(21))
    batches = [(xs[:8], ys[:8]), (xs[8:16], ys[8:16]), (xs[16:], ys[16:])]
    loss, acc = b06_evaluate(model, batches)
    with torch.no_grad():
        full = model(xs)
    assert abs(loss - F.cross_entropy(full, ys).item()) < 1e-4, f"loss {loss:.6f} looks wrong"
    assert abs(acc - (full.argmax(1) == ys).float().mean().item()) < 1e-6, f"acc {acc:.6f} looks wrong"
    assert not model.training, "model was left in train mode"


def _check_b07():
    x = torch.rand(4, 3, 6, 6, generator=_g(22))
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    out = b07_normalize(x, mean, std)
    assert out.shape == x.shape, f"shape {tuple(out.shape)}"
    expected = (x[0, 1, 2, 3] - mean[1]) / std[1]
    assert abs(out[0, 1, 2, 3].item() - expected.item()) < 1e-6, "channel stats applied to the wrong axis"


def _check_b08():
    batch = [(torch.zeros(3, 4, 4) + i, i % 2, f"/img/{i}.jpg") for i in range(5)]
    imgs, labels, paths = b08_collate_with_paths(batch)
    assert imgs.shape == (5, 3, 4, 4), f"shape {tuple(imgs.shape)}"
    assert labels.dtype == torch.int64 and labels.tolist() == [0, 1, 0, 1, 0]
    assert tuple(paths) == tuple(f"/img/{i}.jpg" for i in range(5)), "paths lost or reordered"


def _check_b09():
    items = [([1.0, 2.0], 0), ([3.0, 4.0], 1), ([5.0, 6.0], 1)]
    ds = B09TinyDataset(items)
    assert len(ds) == 3, f"len {len(ds)}"
    feats, label = ds[1]
    assert torch.is_tensor(feats) and feats.dtype == torch.float32, "features must be a float32 tensor"
    assert feats.shape == (2,), f"shape {tuple(feats.shape)}"
    assert torch.allclose(feats, torch.tensor([3.0, 4.0])) and label == 1
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    xb, yb = next(iter(loader))
    assert xb.shape == (2, 2), "does not survive a DataLoader"


def _check_b10():
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))
    assert b10_count_parameters(model) == 4 * 5 + 5 + 5 * 2 + 2
    for p in model[0].parameters():
        p.requires_grad_(False)
    assert b10_count_parameters(model, trainable_only=True) == 5 * 2 + 2
    assert b10_count_parameters(model, trainable_only=False) == 4 * 5 + 5 + 5 * 2 + 2


def _check_c01():
    cases = [
        ((32, 32, 3, 1, 1, 1), (32, 32)),
        ((32, 32, 3, 2, 1, 1), (16, 16)),
        ((28, 28, 5, 1, 0, 1), (24, 24)),
        ((64, 64, 3, 1, 2, 2), (64, 64)),
        ((7, 11, 2, 2, 0, 1), (3, 5)),
    ]
    for args, expected in cases:
        got = c01_conv_output_shape(*args)
        assert tuple(got) == expected, f"{args} -> {tuple(got)}, expected {expected}"


def _check_c02():
    model = c02_build_tiny_cnn(7)
    assert isinstance(model, nn.Module)
    for size in (32, 64, 97):
        out = model(torch.randn(2, 3, size, size))
        assert out.shape == (2, 7), f"input {size}x{size} -> {tuple(out.shape)}, expected (2, 7)"


def _check_c03():
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    c03_freeze_backbone(model, unfreeze_last_n=2)
    flags = [p.requires_grad for p in model.parameters()]
    assert flags == [False, False, True, True], f"got {flags}"


def _check_c04():
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    c04_replace_head(model, 11)
    head = model[-1]
    assert isinstance(head, nn.Linear)
    assert head.in_features == 6 and head.out_features == 11, \
        f"head is {head.in_features}->{head.out_features}, expected 6->11"
    assert model(torch.randn(2, 4)).shape == (2, 11)


def _check_c05():
    x = torch.rand(2, 3, 5, 5, generator=_g(23))
    mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    norm = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    out = c05_unnormalize(norm, mean, std)
    assert torch.allclose(out, x, atol=1e-5), "not the inverse of b07"
    assert out.min() >= 0.0 and out.max() <= 1.0, "not clamped to [0, 1]"


def _check_c06():
    logits = torch.tensor([
        [5.0, 1.0, 0.0, 2.0],
        [0.0, 1.0, 5.0, 2.0],
        [0.0, 5.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 5.0],
    ])
    labels = torch.tensor([3, 2, 0, 3])
    assert abs(c06_topk_accuracy(logits, labels, 1) - 0.5) < 1e-6
    assert abs(c06_topk_accuracy(logits, labels, 2) - 0.75) < 1e-6
    assert abs(c06_topk_accuracy(logits, labels, 4) - 1.0) < 1e-6


def _check_c07():
    preds = torch.tensor([0, 0, 1, 1, 1])
    labels = torch.tensor([0, 1, 1, 1, 0])
    out = c07_per_class_accuracy(preds, labels, 3)
    assert out.shape == (3,), f"shape {tuple(out.shape)}"
    assert abs(out[0].item() - 0.5) < 1e-6, f"class 0 -> {out[0].item()}, expected 0.5"
    assert abs(out[1].item() - (2 / 3)) < 1e-6, f"class 1 -> {out[1].item()}, expected 0.667"
    assert torch.isnan(out[2]), "empty class should be nan"


def _check_c08():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 5.0, 15.0, 15.0],
        [20.0, 20.0, 30.0, 30.0],
    ])
    out = c08_iou(a, b)
    assert out.shape == (1, 3), f"shape {tuple(out.shape)}"
    expected = torch.tensor([[1.0, 25.0 / 175.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-5), f"got {out.tolist()}"


def _check_c09():
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [20.0, 20.0, 30.0, 30.0],
        [0.0, 0.0, 9.0, 9.0],
    ])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    out = c09_nms(boxes, scores, 0.5)
    assert out.dtype == torch.int64, f"dtype {out.dtype}"
    assert out.tolist() == [0, 2], f"got {out.tolist()}, expected [0, 2]"


def _check_c10():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = F.mse_loss(model(torch.randn(8, 4, generator=_g(24))), torch.zeros(8, 3))
    loss.backward()
    opt.step()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        loaded_model, loaded_opt = c10_checkpoint_roundtrip(model, opt, path)
        assert os.path.exists(path), "nothing was written to path"
        assert loaded_model is not model, "returned the original model, not a fresh one"
        for k, v in model.state_dict().items():
            assert torch.allclose(v, loaded_model.state_dict()[k]), f"weights differ at {k}"
        assert loaded_opt.state_dict()["param_groups"][0]["lr"] == 0.01, "optimizer state not restored"


CHECKS = {
    "A": [
        ("a01 per_channel_mean", _check_a01),
        ("a02 per_channel_mean_keepdim", _check_a02),
        ("a03 standardize_per_feature", _check_a03),
        ("a04 flatten_images", _check_a04),
        ("a05 nhwc_to_nchw", _check_a05),
        ("a06 accuracy", _check_a06),
        ("a07 manual_cross_entropy", _check_a07),
        ("a08 topk_classes", _check_a08),
        ("a09 gather_true_logit", _check_a09),
        ("a10 select_classes", _check_a10),
        ("a11 one_hot", _check_a11),
        ("a12 pairwise_sq_dists", _check_a12),
        ("a13 linear_forward", _check_a13),
        ("a14 train_step", _check_a14),
        ("a15 fit_linear", _check_a15),
    ],
    "B": [
        ("b01 set_all_seeds", _check_b01),
        ("b02 stratified_indices", _check_b02),
        ("b03 assert_no_leakage", _check_b03),
        ("b04 class_counts", _check_b04),
        ("b05 confusion_matrix", _check_b05),
        ("b06 evaluate", _check_b06),
        ("b07 normalize", _check_b07),
        ("b08 collate_with_paths", _check_b08),
        ("b09 TinyDataset", _check_b09),
        ("b10 count_parameters", _check_b10),
    ],
    "C": [
        ("c01 conv_output_shape", _check_c01),
        ("c02 build_tiny_cnn", _check_c02),
        ("c03 freeze_backbone", _check_c03),
        ("c04 replace_head", _check_c04),
        ("c05 unnormalize", _check_c05),
        ("c06 topk_accuracy", _check_c06),
        ("c07 per_class_accuracy", _check_c07),
        ("c08 iou", _check_c08),
        ("c09 nms", _check_c09),
        ("c10 checkpoint_roundtrip", _check_c10),
    ],
}


def run(tiers):
    start = time.time()
    passed = skipped = failed = 0
    attempted_total = 0

    for tier in tiers:
        print(f"\n  TIER {tier}")
        print("  " + "-" * 56)
        for name, check in CHECKS[tier]:
            try:
                check()
            except NotImplementedError:
                print(f"  ....  {name}")
                skipped += 1
            except AssertionError as e:
                print(f"  FAIL  {name}  --  {e}")
                failed += 1
                attempted_total += 1
            except Exception as e:
                print(f"  ERR   {name}  --  {type(e).__name__}: {e}")
                failed += 1
                attempted_total += 1
            else:
                print(f"  PASS  {name}")
                passed += 1
                attempted_total += 1

    total = sum(len(CHECKS[t]) for t in tiers)
    elapsed = time.time() - start
    print("\n  " + "=" * 56)
    print(f"  {passed}/{total} passed   |   {failed} failed   |   {skipped} not attempted")
    print(f"  elapsed {elapsed:.1f}s")
    print("\n  paste into reps.md:")
    print(f"  | {time.strftime('%Y-%m-%d')} | {'+'.join(tiers)} | {passed}/{total} | "
          f"attempted {attempted_total} | notes: |")
    print()
    return failed == 0 and skipped == 0


if __name__ == "__main__":
    args = [a.upper() for a in sys.argv[1:]] or ["A"]
    if args == ["ALL"]:
        args = ["A", "B", "C"]
    unknown = [a for a in args if a not in CHECKS]
    if unknown:
        sys.exit(f"unknown tier(s): {', '.join(unknown)}. use A, B, C, or all")
    ok = run(args)
    sys.exit(0 if ok else 1)