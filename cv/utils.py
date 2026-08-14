"""Shared helpers. Drop this in as cv/utils.py — every training script imports set_seed."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1337) -> int:
    """Seed python, numpy and torch. Returns the seed so it can be logged."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_run(
    path: str,
    run_id: str,
    project: str,
    change: str,
    seed: int,
    train_metric: str,
    val_metric: str,
    notes: str = "",
) -> None:
    """Append one row to experiments/experiments.csv. Called at the end of every run."""
    import csv
    from datetime import date

    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [date.today().isoformat(), run_id, project, change, seed,
             train_metric, val_metric, notes]
        )