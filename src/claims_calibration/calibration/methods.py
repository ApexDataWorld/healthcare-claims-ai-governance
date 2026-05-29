from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-8


def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), EPS, 1.0 - EPS)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = _clip(probabilities)
    return np.log(probabilities / (1.0 - probabilities))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class RawCalibrator:
    name: str = "raw"

    def fit(self, confidences: np.ndarray, correctness: np.ndarray) -> "RawCalibrator":
        return self

    def predict(self, confidences: np.ndarray) -> np.ndarray:
        return _clip(confidences)


@dataclass
class TemperatureScalingCalibrator:
    temperature: float = 1.0
    name: str = "temperature_scaling"

    def fit(self, confidences: np.ndarray, correctness: np.ndarray) -> "TemperatureScalingCalibrator":
        logits = _logit(confidences)
        y = correctness.astype(float)

        def objective(temperature: float) -> float:
            probs = _clip(_sigmoid(logits / temperature))
            return -float(np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))

        grid = np.linspace(0.05, 10.0, 400)
        losses = np.array([objective(float(value)) for value in grid])
        center = float(grid[int(np.argmin(losses))])
        fine = np.linspace(max(0.05, center - 0.1), min(10.0, center + 0.1), 200)
        fine_losses = np.array([objective(float(value)) for value in fine])
        self.temperature = float(fine[int(np.argmin(fine_losses))])
        return self

    def predict(self, confidences: np.ndarray) -> np.ndarray:
        return _clip(_sigmoid(_logit(confidences) / self.temperature))


@dataclass
class PlattCalibrator:
    name: str = "platt_scaling"

    def __post_init__(self) -> None:
        self.intercept_ = 0.0
        self.coef_ = 1.0

    def fit(self, confidences: np.ndarray, correctness: np.ndarray) -> "PlattCalibrator":
        x = _logit(confidences)
        y = correctness.astype(float)
        x_mean = float(x.mean())
        x_std = float(x.std() or 1.0)
        x_scaled = (x - x_mean) / x_std
        intercept = 0.0
        coef = 0.0
        learning_rate = 0.05
        l2 = 0.01
        for _ in range(2000):
            probs = _sigmoid(intercept + coef * x_scaled)
            error = probs - y
            grad_intercept = float(error.mean())
            grad_coef = float((error * x_scaled).mean() + l2 * coef)
            intercept -= learning_rate * grad_intercept
            coef -= learning_rate * grad_coef
        self.intercept_ = intercept - coef * x_mean / x_std
        self.coef_ = coef / x_std
        return self

    def predict(self, confidences: np.ndarray) -> np.ndarray:
        return _clip(_sigmoid(self.intercept_ + self.coef_ * _logit(confidences)))


@dataclass
class IsotonicCalibrator:
    name: str = "isotonic_regression"

    def __post_init__(self) -> None:
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, confidences: np.ndarray, correctness: np.ndarray) -> "IsotonicCalibrator":
        order = np.argsort(confidences)
        x = confidences.astype(float)[order]
        y = correctness.astype(float)[order]
        weights = np.ones_like(y)
        means = list(y)
        counts = list(weights)
        starts = list(range(len(y)))
        ends = list(range(len(y)))
        idx = 0
        while idx < len(means) - 1:
            if means[idx] <= means[idx + 1]:
                idx += 1
                continue
            total = counts[idx] + counts[idx + 1]
            merged = (means[idx] * counts[idx] + means[idx + 1] * counts[idx + 1]) / total
            means[idx] = merged
            counts[idx] = total
            ends[idx] = ends[idx + 1]
            del means[idx + 1], counts[idx + 1], starts[idx + 1], ends[idx + 1]
            idx = max(idx - 1, 0)
        fitted = np.empty_like(y)
        for mean, start, end in zip(means, starts, ends):
            fitted[start : end + 1] = mean
        unique_x, unique_indices = np.unique(x, return_index=True)
        self.x_ = unique_x
        self.y_ = fitted[unique_indices]
        return self

    def predict(self, confidences: np.ndarray) -> np.ndarray:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before predict().")
        return _clip(np.interp(confidences.astype(float), self.x_, self.y_))


def make_calibrator(name: str):
    calibrators = {
        "raw": RawCalibrator,
        "temperature_scaling": TemperatureScalingCalibrator,
        "platt_scaling": PlattCalibrator,
        "isotonic_regression": IsotonicCalibrator,
    }
    if name not in calibrators:
        raise ValueError(f"Unknown calibration method: {name}")
    return calibrators[name]()
