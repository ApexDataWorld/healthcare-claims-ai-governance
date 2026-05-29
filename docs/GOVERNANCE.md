# Governance Framework

This repository implements a decision-support layer for healthcare insurance claims operations. It is designed for human oversight, not autonomous claims adjudication.

## Scope

The framework covers calibration, uncertainty quantification, risk-tiered escalation, expected-loss thresholding, bootstrap confidence intervals, subgroup monitoring, and governance triggers.

It does not cover protected health information processing, HIPAA implementation, claims ETL, regulatory certification, enterprise deployment infrastructure, or clinical decision support optimization.

## Human Oversight

The calibration model estimates whether model recommendations are likely to be correct. Threshold policy remains a governance decision. Claims operations, compliance, legal, clinical review, analytics, and leadership should approve thresholds before any real-world pilot.

Mandatory-review cases are always escalated regardless of model confidence.

## Governance Action Triggers

The package generates `results/tables/governance_triggers.csv`. Example triggers include:

- ECE above 0.05
- MCE above 0.10
- accepted-case accuracy below 95% in any automated tier
- Brier score or NLL outside target
- subgroup ECE gap above 0.05

Triggered metrics should pause threshold expansion, increase manual audit, or start recalibration review depending on severity.

## Required Stakeholders

- Claims Operations: owns workflow design, risk-tier definitions, review capacity, and threshold policy.
- Compliance and Legal: reviews auditability, member impact, regulatory obligations, and documentation.
- Clinical or Medical Review: reviews medically sensitive escalations and medical necessity criteria.
- Data and Analytics: owns monitoring, calibration, validation studies, and subgroup analysis.
- Executive Leadership: sets risk tolerance and approves major changes.

## Audit Trail

For real-world pilots, every model-assisted decision should retain claim identifier, model recommendation, raw confidence, calibrated probability, risk tier, threshold, escalation decision, reviewer action, override status, and policy version.
