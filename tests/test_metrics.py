import numpy as np

from claims_calibration.evaluation.metrics import calibration_metrics, reliability_bins


def test_calibration_metrics_include_required_values():
    y = np.array([1, 1, 0, 1, 0, 1])
    p = np.array([0.9, 0.8, 0.4, 0.7, 0.2, 0.6])
    metrics = calibration_metrics(y, p, bins=5)
    assert set(metrics) == {"accuracy", "ece", "mce", "brier", "nll"}
    assert 0 <= metrics["ece"] <= 1
    assert 0 <= metrics["brier"] <= 1


def test_reliability_bins_has_expected_bin_count():
    table = reliability_bins(np.array([1, 0, 1]), np.array([0.8, 0.3, 0.9]), bins=4)
    assert len(table) == 4
    assert table["count"].sum() == 3
