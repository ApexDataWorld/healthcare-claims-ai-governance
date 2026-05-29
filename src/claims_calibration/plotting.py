from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from claims_calibration.evaluation.metrics import reliability_bins

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))


def save_reliability_diagram(correct, probabilities, output_path: Path, bins: int = 10) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = reliability_bins(correct, probabilities, bins)
    plot_df = table[table["count"] > 0].copy()

    # Use populated range only, because Platt-scaled probabilities are compressed
    # into the high-confidence region. Numeric ECE is still computed on standard
    # 10 equal-width bins over [0, 1].
    xmin = max(0.0, min(plot_df["confidence"].min(), plot_df["accuracy"].min()) - 0.05)
    xmax = min(1.0, max(plot_df["confidence"].max(), plot_df["accuracy"].max()) + 0.02)
    ymin = max(0.0, min(plot_df["confidence"].min(), plot_df["accuracy"].min()) - 0.05)
    ymax = min(1.0, max(plot_df["confidence"].max(), plot_df["accuracy"].max()) + 0.02)

    # For this paper, force the cleaner manuscript range if values are high-confidence.
    if xmin > 0.80:
        xmin, xmax = 0.86, 0.96
        ymin, ymax = 0.86, max(0.96, ymax)

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=220)

    ax.plot(
        [xmin, xmax],
        [xmin, xmax],
        linestyle="--",
        color="#8c8c8c",
        linewidth=1.6,
        label="Perfect calibration",
        zorder=1,
    )

    sizes = 30 + plot_df["count"].to_numpy() * 2.4
    ax.scatter(
        plot_df["confidence"],
        plot_df["accuracy"],
        s=sizes,
        color="#1f4e79",
        edgecolor="white",
        linewidth=0.8,
        label="Platt-scaled bin (size proportional to n)",
        zorder=3,
    )

    for _, row in plot_df.iterrows():
        if row["count"] >= 20:
            ax.annotate(
                f"n={int(row['count'])}",
                (row["confidence"], row["accuracy"]),
                textcoords="offset points",
                xytext=(10, 8),
                fontsize=9,
                color="#333333",
            )

    ax.set_title("Reliability diagram (Platt-scaled calibrator, n=300)", fontsize=13)
    ax.set_xlabel("Mean calibrated confidence", fontsize=11)
    ax.set_ylabel("Observed correctness", fontsize=11)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_risk_coverage_curve(
    curve: pd.DataFrame,
    output_path: Path,
    risk_table: pd.DataFrame | None = None,
    accuracy_floor: float = 0.95,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = curve.dropna(subset=["coverage", "accepted_accuracy"]).copy()
    plot_df = plot_df.sort_values("coverage")

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=220)

    ax.plot(
        plot_df["coverage"],
        plot_df["accepted_accuracy"],
        color="#1b7837",
        marker="o",
        markersize=3.0,
        linewidth=2.0,
        label="Sweep over threshold [0.50, 0.99]",
        zorder=2,
    )

    if risk_table is not None:
        colors = {
            "low": "#d73027",
            "medium": "#9467bd",
            "high": "#ff7f0e",
        }
        labels = {
            "low": "Low tau=0.90",
            "medium": "Medium tau=0.91",
            "high": "High tau=0.92",
        }

        for tier in ["low", "medium", "high"]:
            row = risk_table[risk_table["risk_tier"] == tier]
            if row.empty:
                continue
            row = row.iloc[0]
            if pd.isna(row["accepted_accuracy"]):
                continue
            ax.scatter(
                row["coverage"],
                row["accepted_accuracy"],
                s=80,
                color=colors[tier],
                label=labels[tier],
                zorder=4,
            )

    ax.axhline(
        accuracy_floor,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.3,
        label=f"Accuracy floor {accuracy_floor:.2f}",
        zorder=1,
    )

    ax.set_title("Risk-coverage tradeoff with per-tier operating points", fontsize=13)
    ax.set_xlabel("Coverage (fraction of cases accepted)", fontsize=11)
    ax.set_ylabel("Accepted-case accuracy", fontsize=11)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.90, 1.00)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _pretty_subgroup_label(subgroup_type: str, subgroup: str, cases: int) -> str:
    type_map = {
        "policy_category": "Policy",
        "member_group": "Member group",
        "provider_type": "Provider",
        "claim_type": "Claim type",
    }

    value_map = {
        "medicaid_managed": "Medicaid managed care",
        "medicare_advantage": "Medicare Advantage",
        "commercial": "Commercial",
        "exchange": "Exchange",
        "group_a": "Member group A",
        "group_b": "Member group B",
        "group_c": "Member group C",
        "group_d": "Member group D",
        "independent_practice": "Independent practice",
        "health_system": "Health system",
        "specialty_group": "Specialty group",
        "facility": "Facility",
        "office_visit": "Office visit",
        "behavioral_health": "Behavioral health",
        "preventive": "Preventive",
        "imaging": "Imaging",
        "procedure": "Procedure",
        "pharmacy": "Pharmacy",
    }

    prefix = type_map.get(subgroup_type, subgroup_type.replace("_", " ").title())
    value = value_map.get(str(subgroup), str(subgroup).replace("_", " ").title())
    return f"{prefix}: {value} (n={cases})"


def save_subgroup_ece(
    subgroup_table: pd.DataFrame,
    output_path: Path,
    aggregate_ece: float,
    subgroup_gap_threshold: float = 0.05,
    top_n: int = 12,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    alert_line = aggregate_ece + subgroup_gap_threshold

    plot_df = subgroup_table.sort_values("ece", ascending=False).head(top_n).copy()
    plot_df["label"] = [
        _pretty_subgroup_label(row["subgroup_type"], row["subgroup"], int(row["cases"]))
        for _, row in plot_df.iterrows()
    ]

    colors = ["#b2182b" if value >= alert_line else "#9e9e9e" for value in plot_df["ece"]]

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=220)

    ax.barh(plot_df["label"], plot_df["ece"], color=colors)
    ax.axvline(
        alert_line,
        color="#b2182b",
        linestyle="--",
        linewidth=1.6,
        zorder=2,
    )

    for y, value in enumerate(plot_df["ece"]):
        ax.text(
            value + 0.0015,
            y,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )

    ax.set_title("Subgroup calibration monitoring (top 12 by ECE)", fontsize=13)
    ax.set_xlabel("Expected calibration error (ECE)", fontsize=11)
    ax.set_xlim(0, max(0.105, float(plot_df["ece"].max()) + 0.02))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    legend_items = [
        Patch(facecolor="#b2182b", label="Above trigger threshold"),
        Patch(facecolor="#9e9e9e", label="Within tolerance"),
        Line2D(
            [0],
            [0],
            color="#b2182b",
            linestyle="--",
            linewidth=1.6,
            label=f"Aggregate ECE + gap threshold = {alert_line:.3f}",
        ),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
