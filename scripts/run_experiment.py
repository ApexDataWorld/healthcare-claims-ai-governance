from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from claims_calibration.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the healthcare claims calibration experiment.")
    parser.add_argument("--config", default="configs/synthetic_baseline.yaml")
    args = parser.parse_args()

    outputs = run_pipeline(args.config)
    print("Generated reproducibility artifacts:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
