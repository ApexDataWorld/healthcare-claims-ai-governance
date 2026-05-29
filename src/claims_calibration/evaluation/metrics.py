from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


EPS = 1e-8


def calibration_metrics(correctness: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, float]:
    y = correctness.astype(int)
    p = np.clip(probabilities.astype(float), EPS, 1.0 - EPS)
    predicted = p >= 0.5
    accuracy = float(np.mean(predicted == y))
    brier = float(np.mean((p - y) ** 2))
    nll = -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    bin_table = reliability_bins(y, p, bins)
    ece = float(np.sum((bin_table["count"] / len(y)) * np.abs(bin_table["accuracy"] - bin_table["confidence"])))
    mce = float(np.max(np.abs(bin_table["accuracy"] - bin_table["confidence"])))
    return {
        "accuracy": accuracy,
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "nll": nll,
    }


def reliability_bins(correctness: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y = correctness.astype(int)
    p = np.clip(probabilities.astype(float), EPS, 1.0 - EPS)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for idx in range(bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        count = int(mask.sum())
        if count:
            confidence = float(p[mask].mean())
            accuracy = float(y[mask].mean())
        else:
            confidence = float((lower + upper) / 2.0)
            accuracy = float("nan")
        rows.append(
            {
                "bin": idx + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    table = pd.DataFrame(rows)
    table["accuracy"] = table["accuracy"].fillna(table["confidence"])
    return table


def bootstrap_metric_intervals(
    correctness: np.ndarray,
    probabilities: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(correctness)
    samples: list[dict[str, float]] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        samples.append(metric_fn(correctness[idx], probabilities[idx]))
    boot = pd.DataFrame(samples)
    rows = []
    for metric in boot.columns:
        rows.append(
            {
                "metric": metric,
                "mean": float(boot[metric].mean()),
                "ci_lower": float(boot[metric].quantile(0.025)),
                "ci_upper": float(boot[metric].quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)
