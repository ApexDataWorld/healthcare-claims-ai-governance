# Risk-Tier Definitions

Risk tiers represent operational sensitivity, financial impact, policy ambiguity, appeal likelihood, documentation completeness, and governance requirements. They do not represent clinical severity.

## Low Risk

Routine claims with clear policy applicability, complete documentation, lower financial impact, and low appeal likelihood. Examples include preventive care, in-network office visits, and standard procedures with clear coverage.

## Medium Risk

Claims with moderate complexity, moderate financial impact, or standard policy ambiguity. Examples include basic medical necessity determinations, new providers, out-of-network services with clear coverage, and moderately complex documentation.

## High Risk

Claims with high financial impact, significant policy complexity, unclear documentation, or high appeal likelihood. Examples include experimental treatments, high-cost procedures, conflicting diagnostic information, and claims from providers with weak documentation histories.

## Mandatory Review

Cases that governance, compliance, policy, or regulatory requirements mandate for human review regardless of model confidence. Examples include regulatory exceptions, precedent-setting coverage decisions, or corporate coverage policy exceptions.

The synthetic data uses these tiers to demonstrate calibration-aware thresholding. Real organizations should define tiers through formal governance review.
