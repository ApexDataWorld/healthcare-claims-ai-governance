from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from claims_calibration.evaluation.metrics import calibration_metrics
from claims_calibration.governance.monitoring import subgroup_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Example monitoring check for calibrated claims predictions.")
    parser.add_argument("--predictions", default="results/tables/test_predictions.csv")
    parser.add_argument("--probability-col", default=None)
    parser.add_argument("--ece-threshold", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    probability_col = args.probability_col
    if probability_col is None:
        probability_columns = [col for col in df.columns if col.startswith("p_") and col != "p_raw"]
        probability_col = probability_columns[0] if probability_columns else "p_raw"

    metrics = calibration_metrics(df["correct"].to_numpy(), df[probability_col].to_numpy(), bins=10)
    subgroup = subgroup_calibration(df, probability_col, ["claim_type", "policy_category", "provider_type"], 10)
    print(f"Monitoring column: {probability_col}")
    print(metrics)
    print("Governance alert:", metrics["ece"] > args.ece_threshold)
    print("Highest subgroup ECE rows:")
    print(subgroup.sort_values("ece", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
