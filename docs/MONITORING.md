# Monitoring

Calibration metrics measured during validation can drift when claim mix changes, policy language changes, coding guidance evolves, prompts change, model versions change, or member and provider populations shift.

## Dashboard Metrics

Track metrics weekly or monthly and stratify by risk tier and subgroup:

- Expected Calibration Error
- Maximum Calibration Error
- Brier score
- Negative log-likelihood
- accepted-case accuracy
- coverage
- escalation rate
- human override rate
- subgroup ECE and escalation differences

## Recalibration Procedure

1. Collect newly reviewed claims with human adjudicator outcomes.
2. Compute calibration metrics with the current calibrator.
3. Confirm whether drift exceeds governance triggers.
4. Refit calibration candidates on recent data.
5. Test on a holdout set.
6. Compare old and new calibrators.
7. Reoptimize risk-tier thresholds.
8. Obtain governance approval.
9. Deploy updated calibration and threshold policy.
10. Document the change in the audit trail.
