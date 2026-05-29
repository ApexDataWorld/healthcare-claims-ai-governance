from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "data/synthetic/synthetic_claims.csv",
    "results/tables/calibration_metrics.csv",
    "results/tables/bootstrap_confidence_intervals.csv",
    "results/tables/risk_tier_escalation.csv",
    "results/tables/subgroup_calibration.csv",
    "results/tables/calibration_diagnostics.csv",
    "results/tables/subgroup_diagnostics.csv",
    "results/tables/reliability_bins_platt.csv",
    "results/tables/governance_triggers.csv",
    "results/figures/fig1_reliability_diagram.png",
    "results/figures/fig2_risk_coverage_curve.png",
    "results/figures/fig3_subgroup_ece.png",
]


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not Path(path).exists()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))

    claims = pd.read_csv("data/synthetic/synthetic_claims.csv")
    if len(claims) != 1200:
        raise SystemExit(f"Expected 1,200 synthetic claims, found {len(claims)}")

    calibration = pd.read_csv("results/tables/calibration_metrics.csv")
    required_methods = {"raw", "temperature_scaling", "platt_scaling", "isotonic_regression"}
    if set(calibration["method"]) != required_methods:
        raise SystemExit("Calibration table does not include all required methods.")

    risk = pd.read_csv("results/tables/risk_tier_escalation.csv")
    if set(risk["risk_tier"]) != {"low", "medium", "high", "mandatory"}:
        raise SystemExit("Risk-tier table does not include the four required tiers.")
    mandatory = risk[risk["risk_tier"] == "mandatory"].iloc[0]
    if mandatory["coverage"] != 0 or mandatory["escalation_rate"] != 1:
        raise SystemExit("Mandatory-review tier must have zero coverage and full escalation.")

    reliability_bins = pd.read_csv("results/tables/reliability_bins_platt.csv")
    expected_reliability_columns = {"bin", "lower", "upper", "count", "confidence", "accuracy", "abs_gap"}
    if not expected_reliability_columns.issubset(reliability_bins.columns):
        raise SystemExit("Reliability bin diagnostic table is missing required columns.")
    if not (reliability_bins["count"] > 0).any():
        raise SystemExit("Reliability bin diagnostic table has no populated bins.")

    subgroup_diagnostics = pd.read_csv("results/tables/subgroup_diagnostics.csv")
    group_b = subgroup_diagnostics[
        (subgroup_diagnostics["subgroup_type"] == "member_group")
        & (subgroup_diagnostics["subgroup"] == "group_b")
    ]
    if group_b.empty:
        raise SystemExit("Subgroup diagnostics must include member_group/group_b.")
    if group_b.iloc[0]["calibration_direction"] != "underconfident":
        raise SystemExit("Expected member_group/group_b to be marked underconfident.")

    for figure in Path("results/figures").glob("*.png"):
        if figure.stat().st_size < 1000:
            raise SystemExit(f"Figure appears empty or invalid: {figure}")

    print("All required reproducibility artifacts verified.")


if __name__ == "__main__":
    main()
