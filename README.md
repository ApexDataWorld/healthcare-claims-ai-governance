# Calibration-Aware LLM Decision Support for Healthcare Insurance Claims Processing

[![CI](https://github.com/ApexDataWorld/healthcare-claims-ai-governance/actions/workflows/ci.yml/badge.svg)](https://github.com/ApexDataWorld/healthcare-claims-ai-governance/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

This repository is a reproducible research package for the manuscript:

**Calibration-Aware Large Language Model Decision Support for Healthcare Insurance Claims Processing: Uncertainty Quantification, Risk-Tiered Escalation, and Human Oversight**

It implements a focused healthcare informatics decision-support framework for synthetic healthcare insurance claims. The package converts raw model confidence into calibrated probabilities, risk-tiered escalation decisions, expected-loss thresholds, bootstrap confidence intervals, subgroup calibration checks, and governance monitoring triggers.

The repository is intentionally scoped to claims decision support and reproducible evaluation, with production infrastructure topics kept outside the manuscript package.

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

## Methods Implemented

- Synthetic healthcare insurance claims generator for 1,200 claims
- Calibration methods: raw confidence, temperature scaling, Platt scaling, isotonic regression
- Metrics: accuracy, ECE, MCE, Brier score, negative log-likelihood
- Risk-tiered abstention and human escalation
- Expected-loss threshold optimization
- Bootstrap confidence intervals
- Subgroup calibration and equity monitoring
- Governance monitoring triggers
- Publication-ready tables and figures

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
