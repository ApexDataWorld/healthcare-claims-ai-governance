#!/usr/bin/env bash
set -euo pipefail

# Idempotent scaffold helper for the JAMIA reproducibility package.
# The repository implementation lives in tracked source files; this script only
# creates expected directories for fresh clones or clean worktrees.

REPO_NAME="healthcare-claims-ai-governance"

mkdir -p \
  .github/workflows \
  configs \
  data/synthetic \
  docs \
  results/figures \
  results/tables \
  scripts \
  src/claims_calibration/calibration \
  src/claims_calibration/evaluation \
  src/claims_calibration/governance \
  tests

touch data/synthetic/.gitkeep results/figures/.gitkeep results/tables/.gitkeep

cat <<EOF
${REPO_NAME} scaffold directories are ready.

Next steps:
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python scripts/reproduce_all.py
  .venv/bin/python scripts/verify_outputs.py
EOF
