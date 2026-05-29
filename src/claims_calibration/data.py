from __future__ import annotations

import numpy as np
import pandas as pd


RISK_TIERS = ("low", "medium", "high", "mandatory")
CLAIM_TYPES = ("preventive", "office_visit", "imaging", "procedure", "pharmacy", "behavioral_health")
POLICY_CATEGORIES = ("commercial", "medicare_advantage", "medicaid_managed", "exchange")
PROVIDER_TYPES = ("health_system", "independent_practice", "specialty_group", "facility")
MEMBER_GROUPS = ("group_a", "group_b", "group_c", "group_d")


def generate_synthetic_claims(total_claims: int = 1200, seed: int = 20260503) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tier_probs = np.array([0.20, 0.50, 0.20, 0.10])
    risk_tier = rng.choice(RISK_TIERS, size=total_claims, p=tier_probs)
    claim_type = rng.choice(CLAIM_TYPES, size=total_claims, p=[0.16, 0.24, 0.16, 0.18, 0.16, 0.10])
    policy_category = rng.choice(POLICY_CATEGORIES, size=total_claims, p=[0.48, 0.24, 0.16, 0.12])
    provider_type = rng.choice(PROVIDER_TYPES, size=total_claims, p=[0.38, 0.27, 0.22, 0.13])
    member_group = rng.choice(MEMBER_GROUPS, size=total_claims, p=[0.42, 0.27, 0.20, 0.11])

    tier_complexity = {"low": 0.15, "medium": 0.40, "high": 0.68, "mandatory": 0.78}
    tier_amount_mean = {"low": 180, "medium": 850, "high": 3800, "mandatory": 5200}
    complexity = np.array([tier_complexity[t] for t in risk_tier])
    documentation = np.clip(1.0 - complexity + rng.normal(0, 0.14, total_claims), 0.05, 1.0)
    claim_amount = np.array([rng.lognormal(np.log(tier_amount_mean[t]), 0.55) for t in risk_tier])
    policy_ambiguity = np.clip(complexity + rng.normal(0, 0.12, total_claims), 0.0, 1.0)
    prior_auth = rng.binomial(1, np.clip(0.2 + complexity * 0.55, 0.0, 0.95), total_claims)

    approval_logit = (
        1.1
        + 1.8 * documentation
        - 1.35 * policy_ambiguity
        - 0.00012 * claim_amount
        + 0.35 * prior_auth
        + np.where(claim_type == "preventive", 0.9, 0.0)
        + np.where(claim_type == "procedure", -0.35, 0.0)
        + np.where(policy_category == "medicaid_managed", -0.25, 0.0)
    )
    approval_probability = 1.0 / (1.0 + np.exp(-approval_logit))
    label_approved = rng.binomial(1, approval_probability)

    model_skill = 3.15 - 0.65 * complexity + 0.35 * documentation
    correctness_probability = np.clip(1.0 / (1.0 + np.exp(-model_skill)), 0.58, 0.97)
    is_correct = rng.binomial(1, correctness_probability)
    model_recommendation = np.where(is_correct == 1, label_approved, 1 - label_approved)

    base_confidence = 0.56 + 0.36 * correctness_probability + 0.10 * (1.0 - policy_ambiguity)
    overconfidence = 0.055 + 0.055 * complexity
    raw_confidence = np.clip(base_confidence + overconfidence + rng.normal(0, 0.055, total_claims), 0.51, 0.995)

    appeal_likelihood = np.clip(0.03 + 0.32 * policy_ambiguity + 0.00003 * claim_amount, 0.0, 0.8)
    override_probability = np.clip(0.03 + (1.0 - raw_confidence) * 0.34 + complexity * 0.08, 0.0, 0.45)

    df = pd.DataFrame(
        {
            "claim_id": [f"CLM-{seed}-{i:04d}" for i in range(total_claims)],
            "risk_tier": risk_tier,
            "claim_type": claim_type,
            "policy_category": policy_category,
            "provider_type": provider_type,
            "member_group": member_group,
            "claim_amount": np.round(claim_amount, 2),
            "documentation_completeness": np.round(documentation, 4),
            "policy_ambiguity": np.round(policy_ambiguity, 4),
            "prior_authorization": prior_auth,
            "label_approved": label_approved,
            "model_recommendation": model_recommendation,
            "correct": (model_recommendation == label_approved).astype(int),
            "raw_confidence": np.round(raw_confidence, 6),
            "appeal_likelihood": np.round(appeal_likelihood, 4),
            "simulated_override_probability": np.round(override_probability, 4),
        }
    )
    return df


def split_claims(df: pd.DataFrame, seed: int, calibration_fraction: float, test_fraction: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = df.copy()
    shuffled["_random"] = rng.random(len(shuffled))
    shuffled = shuffled.sort_values("_random").drop(columns="_random").reset_index(drop=True)
    n = len(shuffled)
    cal_end = int(n * calibration_fraction)
    test_end = cal_end + int(n * test_fraction)
    shuffled["split"] = "train"
    shuffled.loc[: cal_end - 1, "split"] = "calibration"
    shuffled.loc[cal_end : test_end - 1, "split"] = "test"
    return shuffled
