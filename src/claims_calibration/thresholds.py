from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_table(
    df: pd.DataFrame,
    probability_col: str,
    cost_model: dict,
    grid_min: float,
    grid_max: float,
    grid_step: float,
    minimum_accepted_accuracy: float,
    minimum_thresholds: dict | None = None,
) -> pd.DataFrame:
    rows = []
    thresholds = np.round(np.arange(grid_min, grid_max + grid_step, grid_step), 4)
    for tier, group in df.groupby("risk_tier", sort=False):
        if tier == "mandatory":
            rows.append(_tier_row(tier, group, probability_col, 1.0, cost_model[tier]))
            continue

        tier_minimum_threshold = float((minimum_thresholds or {}).get(tier, grid_min))
        tier_thresholds = thresholds[thresholds >= tier_minimum_threshold]
        candidates = []
        for threshold in tier_thresholds:
            row = _tier_row(tier, group, probability_col, float(threshold), cost_model[tier])
            candidates.append(row)
        candidate_df = pd.DataFrame(candidates)
        if isinstance(minimum_accepted_accuracy, dict):
            accuracy_floor = float(minimum_accepted_accuracy.get(tier, 1.0))
        else:
            accuracy_floor = float(minimum_accepted_accuracy)
        eligible = candidate_df[candidate_df["expected_accepted_accuracy"].fillna(1.0) >= accuracy_floor]
        if eligible.empty:
            best = candidate_df.sort_values(["expected_loss", "threshold"]).iloc[0].to_dict()
        else:
            best = eligible.sort_values(["expected_loss", "threshold"]).iloc[0].to_dict()
        rows.append(best)
    order = {"low": 0, "medium": 1, "high": 2, "mandatory": 3}
    return pd.DataFrame(rows).sort_values("risk_tier", key=lambda s: s.map(order)).reset_index(drop=True)


def _tier_row(tier: str, group: pd.DataFrame, probability_col: str, threshold: float, costs: dict) -> dict:
    accepted = (group[probability_col] >= threshold) & (group["risk_tier"] != "mandatory")
    accepted_count = int(accepted.sum())
    escalated_count = int((~accepted).sum())
    cases = int(len(group))
    if accepted_count:
        accepted_accuracy = float(group.loc[accepted, "correct"].mean())
        expected_accepted_accuracy = float(group.loc[accepted, probability_col].mean())
        wrong_accepted = int((1 - group.loc[accepted, "correct"]).sum())
        expected_wrong_accepted = float((1.0 - group.loc[accepted, probability_col]).sum())
    else:
        accepted_accuracy = np.nan
        expected_accepted_accuracy = np.nan
        wrong_accepted = 0
        expected_wrong_accepted = 0.0
    expected_loss = expected_wrong_accepted * costs["wrong_cost"] + escalated_count * costs["review_cost"]
    return {
        "risk_tier": tier,
        "cases": cases,
        "threshold": threshold,
        "coverage": accepted_count / cases,
        "accepted_accuracy": accepted_accuracy,
        "expected_accepted_accuracy": expected_accepted_accuracy,
        "escalation_rate": escalated_count / cases,
        "expected_loss": float(expected_loss),
        "wrong_accepted": wrong_accepted,
        "expected_wrong_accepted": expected_wrong_accepted,
    }


def risk_coverage_curve(df: pd.DataFrame, probability_col: str) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.5, 0.99, 50):
        accepted = df[probability_col] >= threshold
        rows.append(
            {
                "threshold": threshold,
                "coverage": float(accepted.mean()),
                "accepted_accuracy": float(df.loc[accepted, "correct"].mean()) if accepted.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
