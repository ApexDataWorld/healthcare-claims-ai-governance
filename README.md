# Aggregate Calibration Is Not Enough: Subgroup Monitoring and Pre-Registered Governance Triggers for LLM-Style Decision Support in Healthcare Claims

[![CI](https://github.com/ApexDataWorld/healthcare-claims-ai-governance/actions/workflows/ci.yml/badge.svg)](https://github.com/ApexDataWorld/healthcare-claims-ai-governance/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

This repository contains the reproducible artifact package for the manuscript **"Aggregate Calibration Is Not Enough: Subgroup Monitoring and Pre-Registered Governance Triggers for LLM-Style Decision Support in Healthcare Claims."** The project implements a synthetic, deterministic healthcare-claims decision-support experiment evaluating post-hoc probability calibration, risk-tiered expected-loss thresholding, subgroup calibration monitoring, and pre-registered governance triggers. The goal is to show how aggregate calibration can pass while tier-level accepted accuracy and subgroup calibration checks still surface operational governance risks.

## Important Scope Boundary

This repository uses a deterministic synthetic healthcare-claims cohort. It does not contain protected health information, production payer data, clinical data, member records, or proprietary claims data. The empirical results validate the reproducible calibration, subgroup monitoring, thresholding, and governance-trigger workflow under controlled synthetic assumptions. They should not be interpreted as production payer validation, clinical validation, or deployment evidence.

## One-Command Reproduction

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/reproduce_all.py
```

The command regenerates synthetic data, calibration results, risk-tier escalation tables, bootstrap intervals, subgroup equity monitoring tables, governance trigger reports, and publication-ready figures.

To verify that manuscript artifacts exist and pass basic quality checks:

```bash
.venv/bin/python scripts/verify_outputs.py
```

## Repository Structure

```text
configs/                 Experiment and monitoring configuration
data/synthetic/          Deterministic synthetic claims data
docs/                    Governance, monitoring, and risk-tier documentation
scripts/                 Reproduction, generation, verification, and monitoring entry points
src/claims_calibration/  Core package
tests/                   Unit and integration tests
results/tables/          Publication-ready CSV tables
results/figures/         Publication-ready PNG figures
```

## Core Contributions

- Compares raw confidence, temperature scaling, Platt scaling, and isotonic regression on a held-out synthetic claims test split.
- Reports calibration metrics including expected calibration error, maximum calibration error, Brier score, and negative log-likelihood.
- Selects risk-tiered thresholds using expected-loss optimization under accepted-accuracy floors.
- Computes subgroup expected calibration error across claim type, policy category, provider type, and member group strata.
- Produces pre-registered governance triggers that map statistical signals to recommended human-review actions.
- Provides reproducible scripts, generated tables, generated figures, and verification checks.

## Key Outputs

After reproduction, inspect:

- `results/tables/calibration_metrics.csv`
- `results/tables/calibration_diagnostics.csv`
- `results/tables/bootstrap_confidence_intervals.csv`
- `results/tables/risk_tier_escalation.csv`
- `results/tables/subgroup_calibration.csv`
- `results/tables/subgroup_diagnostics.csv`
- `results/tables/reliability_bins_platt.csv`
- `results/tables/governance_triggers.csv`
- `results/figures/fig1_reliability_diagram.png`
- `results/figures/fig2_risk_coverage_curve.png`
- `results/figures/fig3_subgroup_ece.png`

## Governance Position

This package is for synthetic validation and reproducible methods development. It is not a production claims adjudication system and does not process protected health information. Real-world use would require approved de-identified data, label-quality assessment with trained reviewers, prospective validation, compliance review, and ongoing monitoring.

See [docs/GOVERNANCE.md](docs/GOVERNANCE.md), [docs/MONITORING.md](docs/MONITORING.md), and [docs/DEPLOYMENT_PATHWAY.md](docs/DEPLOYMENT_PATHWAY.md).

## Cite this work

Gupta, S. (2026). *Aggregate Calibration Is Not Enough: Subgroup Monitoring and Pre-Registered Governance Triggers for LLM-Style Decision Support in Healthcare Claims*. Manuscript transferred to and under review at JAMIA Open.

Author note: Master of Statistics, North Carolina State University.

Repository: https://github.com/ApexDataWorld/healthcare-claims-ai-governance

```bibtex
@misc{gupta2026aggregatecalibration,
  author = {Gupta, Saurabh},
  title = {Aggregate Calibration Is Not Enough: Subgroup Monitoring and Pre-Registered Governance Triggers for LLM-Style Decision Support in Healthcare Claims},
  year = {2026},
  note = {Manuscript transferred to and under review at JAMIA Open. Author note: Master of Statistics, North Carolina State University.},
  howpublished = {\url{https://github.com/ApexDataWorld/healthcare-claims-ai-governance}}
}
```
