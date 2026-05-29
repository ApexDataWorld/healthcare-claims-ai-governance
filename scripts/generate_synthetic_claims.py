from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from claims_calibration.data import generate_synthetic_claims, split_claims


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic healthcare insurance claims.")
    parser.add_argument("--claims", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--output", default="data/synthetic/synthetic_claims.csv")
    args = parser.parse_args()

    df = generate_synthetic_claims(args.claims, args.seed)
    df = split_claims(df, seed=args.seed + 1, calibration_fraction=0.25, test_fraction=0.25)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} synthetic claims to {output}")


if __name__ == "__main__":
    main()
