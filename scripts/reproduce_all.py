from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

from claims_calibration.pipeline import run_pipeline


def main() -> None:
    outputs = run_pipeline("configs/synthetic_baseline.yaml")
    print("Reproduction complete.")
    print("Primary manuscript artifacts:")
    for key in (
        "data",
        "calibration",
        "bootstrap",
        "risk",
        "subgroup",
        "triggers",
        "reliability_figure",
        "coverage_figure",
        "subgroup_figure",
    ):
        print(f"- {outputs[key]}")


if __name__ == "__main__":
    main()
