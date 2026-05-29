from pathlib import Path

import pandas as pd

from claims_calibration.pipeline import run_pipeline


def test_pipeline_generates_required_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_config = Path(__file__).resolve().parents[1] / "configs" / "synthetic_baseline.yaml"
    outputs = run_pipeline(source_config)
    for path in outputs.values():
        assert path.exists()

    claims = pd.read_csv(outputs["data"])
    assert len(claims) == 1200
    risk = pd.read_csv(outputs["risk"])
    assert set(risk["risk_tier"]) == {"low", "medium", "high", "mandatory"}
