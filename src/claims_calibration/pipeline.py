from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from claims_calibration.calibration.methods import make_calibrator
from claims_calibration.data import generate_synthetic_claims, split_claims
from claims_calibration.evaluation.metrics import bootstrap_metric_intervals, calibration_metrics, reliability_bins
from claims_calibration.governance.monitoring import governance_trigger_table, subgroup_calibration
from claims_calibration.plotting import save_reliability_diagram, save_risk_coverage_curve, save_subgroup_ece
from claims_calibration.thresholds import risk_coverage_curve, threshold_table


def load_config(path: str | Path) -> dict:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except ModuleNotFoundError:
        return _load_simple_yaml(Path(path))


def _load_simple_yaml(path: Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            text = line.strip()
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            current = stack[-1][1]
            if not value:
                current[key] = {}
                stack.append((indent, current[key]))
            else:
                current[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str):
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def run_pipeline(config_path: str | Path = "configs/synthetic_baseline.yaml") -> dict[str, Path]:
    config = load_config(config_path)
    root = Path.cwd()
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    data_dir = root / "data" / "synthetic"
    for directory in (tables_dir, figures_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed = int(config["experiment"]["random_seed"])
    data = generate_synthetic_claims(config["dataset"]["total_claims"], seed)
    data = split_claims(
        data,
        seed=seed + 1,
        calibration_fraction=config["dataset"]["calibration_fraction"],
        test_fraction=config["dataset"]["test_fraction"],
    )
    data_path = data_dir / "synthetic_claims.csv"
    data.to_csv(data_path, index=False)

    calibration = data[data["split"] == "calibration"].copy()
    test = data[data["split"] == "test"].copy()
    metric_rows = []
    calibrators = {}
    for method in config["calibration"]["methods"]:
        calibrator = make_calibrator(method)
        calibrator.fit(calibration["raw_confidence"].to_numpy(), calibration["correct"].to_numpy())
        probabilities = calibrator.predict(test["raw_confidence"].to_numpy())
        test[f"p_{method}"] = probabilities
        metrics = calibration_metrics(test["correct"].to_numpy(), probabilities, config["calibration"]["ece_bins"])
        metric_rows.append({"method": method, **metrics})
        calibrators[method] = calibrator

    calibration_table = pd.DataFrame(metric_rows).sort_values("ece").reset_index(drop=True)
    calibration_path = tables_dir / "calibration_metrics.csv"
    calibration_table.to_csv(calibration_path, index=False)
    best_method = str(calibration_table.iloc[0]["method"])
    best_probability_col = f"p_{best_method}"

    diagnostic_rows = []
    for method in config["calibration"]["methods"]:
        probability_col = f"p_{method}"
        method_row = calibration_table.loc[calibration_table["method"] == method].iloc[0].to_dict()
        mean_probability = float(test[probability_col].mean())
        observed_accuracy = float(test["correct"].mean())
        diagnostic_rows.append(
            {
                "method": method,
                "mean_probability": mean_probability,
                "observed_accuracy": observed_accuracy,
                "mean_minus_accuracy": mean_probability - observed_accuracy,
                "ece": method_row["ece"],
                "mce": method_row["mce"],
                "brier": method_row["brier"],
                "nll": method_row["nll"],
            }
        )
    diagnostics_path = tables_dir / "calibration_diagnostics.csv"
    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_path, index=False)

    boot = bootstrap_metric_intervals(
        test["correct"].to_numpy(),
        test[best_probability_col].to_numpy(),
        lambda y, p: calibration_metrics(y, p, config["calibration"]["ece_bins"]),
        int(config["bootstrap"]["resamples"]),
        int(config["bootstrap"]["random_seed"]),
    )
    boot.insert(0, "method", best_method)
    boot_path = tables_dir / "bootstrap_confidence_intervals.csv"
    boot.to_csv(boot_path, index=False)

    risk_table = threshold_table(
        test,
        best_probability_col,
        config["cost_model"],
        config["thresholds"]["grid_min"],
        config["thresholds"]["grid_max"],
        config["thresholds"]["grid_step"],
        config["thresholds"]["minimum_accepted_accuracy"],
        config["thresholds"].get("minimum_thresholds"),
    )
    risk_path = tables_dir / "risk_tier_escalation.csv"
    risk_table.to_csv(risk_path, index=False)

    subgroup_table = subgroup_calibration(
        test,
        best_probability_col,
        ["claim_type", "policy_category", "provider_type", "member_group"],
        config["calibration"]["ece_bins"],
    )
    subgroup_path = tables_dir / "subgroup_calibration.csv"
    subgroup_table.to_csv(subgroup_path, index=False)

    subgroup_diagnostics_path = tables_dir / "subgroup_diagnostics.csv"
    _subgroup_diagnostics(
        test,
        best_probability_col,
        ["claim_type", "policy_category", "provider_type", "member_group"],
        config["calibration"]["ece_bins"],
    ).to_csv(subgroup_diagnostics_path, index=False)

    trigger_table = governance_trigger_table(
        calibration_table.iloc[0].to_dict(),
        risk_table,
        subgroup_table,
        config["monitoring"],
    )
    triggers_path = tables_dir / "governance_triggers.csv"
    trigger_table.to_csv(triggers_path, index=False)

    curve = risk_coverage_curve(test, best_probability_col)
    curve_path = tables_dir / "risk_coverage_curve.csv"
    curve.to_csv(curve_path, index=False)

    reliability_bins_path = tables_dir / "reliability_bins_platt.csv"
    reliability_table = reliability_bins(
        test["correct"].to_numpy(),
        test[best_probability_col].to_numpy(),
        config["calibration"]["ece_bins"],
    )
    reliability_table["abs_gap"] = (reliability_table["accuracy"] - reliability_table["confidence"]).abs()
    reliability_table.to_csv(reliability_bins_path, index=False)

    enriched_test_path = tables_dir / "test_predictions.csv"
    test.to_csv(enriched_test_path, index=False)

    subgroup_fig_path = figures_dir / "fig1_subgroup_ece.png"
    reliability_path = figures_dir / "fig2_reliability_diagram.png"
    coverage_path = figures_dir / "fig3_risk_coverage_curve.png"
    best_calibration_row = calibration_table.loc[calibration_table["method"] == best_method].iloc[0]
    save_reliability_diagram(
        test["correct"].to_numpy(),
        test[best_probability_col].to_numpy(),
        reliability_path,
        config["calibration"]["ece_bins"],
    )
    shutil.copyfile(reliability_path, figures_dir / "reliability_diagram.png")

    save_risk_coverage_curve(
        curve,
        coverage_path,
        risk_table=risk_table,
        accuracy_floor=config["monitoring"]["accepted_accuracy_floor"],
    )
    shutil.copyfile(coverage_path, figures_dir / "risk_coverage_curve.png")

    save_subgroup_ece(
        subgroup_table,
        subgroup_fig_path,
        aggregate_ece=float(best_calibration_row["ece"]),
        subgroup_gap_threshold=config["monitoring"]["subgroup_ece_gap"],
        top_n=12,
    )
    shutil.copyfile(subgroup_fig_path, figures_dir / "subgroup_ece.png")

    return {
        "data": data_path,
        "calibration": calibration_path,
        "calibration_diagnostics": diagnostics_path,
        "bootstrap": boot_path,
        "risk": risk_path,
        "subgroup": subgroup_path,
        "subgroup_diagnostics": subgroup_diagnostics_path,
        "triggers": triggers_path,
        "curve": curve_path,
        "reliability_bins": reliability_bins_path,
        "predictions": enriched_test_path,
        "reliability_figure": reliability_path,
        "coverage_figure": coverage_path,
        "subgroup_figure": subgroup_fig_path,
    }


def _subgroup_diagnostics(
    df: pd.DataFrame,
    probability_col: str,
    subgroup_cols: list[str],
    bins: int,
) -> pd.DataFrame:
    rows = []
    for column in subgroup_cols:
        for value, group in df.groupby(column):
            metrics = calibration_metrics(group["correct"].to_numpy(), group[probability_col].to_numpy(), bins)
            accuracy = float(group["correct"].mean())
            mean_probability = float(group[probability_col].mean())
            difference = accuracy - mean_probability
            if difference > 0.01:
                direction = "underconfident"
            elif difference < -0.01:
                direction = "overconfident"
            else:
                direction = "near_calibrated"
            rows.append(
                {
                    "subgroup_type": column,
                    "subgroup": value,
                    "cases": int(len(group)),
                    "accuracy": accuracy,
                    "mean_probability": mean_probability,
                    "ece": metrics["ece"],
                    "calibration_direction": direction,
                }
            )
    return pd.DataFrame(rows)
