from __future__ import annotations

import pandas as pd

from claims_calibration.evaluation.metrics import calibration_metrics


def subgroup_calibration(df: pd.DataFrame, probability_col: str, subgroup_cols: list[str], bins: int) -> pd.DataFrame:
    rows = []
    for column in subgroup_cols:
        for value, group in df.groupby(column):
            metrics = calibration_metrics(group["correct"].to_numpy(), group[probability_col].to_numpy(), bins)
            rows.append({"subgroup_type": column, "subgroup": value, "cases": len(group), **metrics})
    return pd.DataFrame(rows)


def governance_trigger_table(
    calibration_row: dict,
    risk_table: pd.DataFrame,
    subgroup_table: pd.DataFrame,
    monitoring_config: dict,
) -> pd.DataFrame:
    triggers = []
    triggers.append(
        _trigger(
            "aggregate_ece",
            calibration_row["ece"],
            monitoring_config["ece_target"],
            calibration_row["ece"] > monitoring_config["ece_target"],
            "Pause threshold expansion and review calibration sample.",
        )
    )
    triggers.append(
        _trigger(
            "aggregate_mce",
            calibration_row["mce"],
            monitoring_config["mce_target"],
            calibration_row["mce"] > monitoring_config["mce_target"],
            "Investigate worst calibration bin and consider recalibration.",
        )
    )
    triggers.append(
        _trigger(
            "aggregate_brier",
            calibration_row["brier"],
            monitoring_config["brier_target"],
            calibration_row["brier"] > monitoring_config["brier_target"],
            "Review probability quality and recent claim mix.",
        )
    )
    triggers.append(
        _trigger(
            "aggregate_nll",
            calibration_row["nll"],
            monitoring_config["nll_target"],
            calibration_row["nll"] > monitoring_config["nll_target"],
            "Check for high-confidence errors and model drift.",
        )
    )

    for _, row in risk_table.iterrows():
        if row["risk_tier"] == "mandatory" or pd.isna(row["accepted_accuracy"]):
            continue
        triggers.append(
            _trigger(
                f"{row['risk_tier']}_accepted_accuracy",
                row["accepted_accuracy"],
                monitoring_config["accepted_accuracy_floor"],
                row["accepted_accuracy"] < monitoring_config["accepted_accuracy_floor"],
                "Raise tier threshold; audit accepted cases.",
            )
        )

    aggregate_ece = calibration_row["ece"]
    max_gap = float((subgroup_table["ece"] - aggregate_ece).abs().max())
    triggers.append(
        _trigger(
            "max_subgroup_ece_gap",
            max_gap,
            monitoring_config["subgroup_ece_gap"],
            max_gap > monitoring_config["subgroup_ece_gap"],
            "Open equity review and assess subgroup-specific thresholds.",
        )
    )
    return pd.DataFrame(triggers)


def _trigger(metric: str, observed: float, threshold: float, triggered: bool, action: str) -> dict:
    return {
        "metric": metric,
        "observed": float(observed),
        "threshold": float(threshold),
        "triggered": bool(triggered),
        "recommended_action": action if triggered else "Continue routine monitoring.",
    }
