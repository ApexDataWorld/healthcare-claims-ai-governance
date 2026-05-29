import numpy as np

from claims_calibration.calibration.methods import make_calibrator


def test_all_calibrators_return_probabilities():
    confidences = np.array([0.55, 0.62, 0.71, 0.83, 0.91, 0.97, 0.67, 0.77])
    correctness = np.array([0, 1, 1, 1, 1, 1, 0, 1])
    for method in ["raw", "temperature_scaling", "platt_scaling", "isotonic_regression"]:
        calibrator = make_calibrator(method).fit(confidences, correctness)
        probabilities = calibrator.predict(confidences)
        assert probabilities.shape == confidences.shape
        assert np.all(probabilities > 0)
        assert np.all(probabilities < 1)
