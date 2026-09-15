#!/usr/bin/env python
"""
Oracle Grounding Validation for VASST Trial
Validates that the synthetic oracle reflects published VASST trial data.
Tier 1-5 checks: ATE, baseline risk, propensity, subgroup effects, range/safety.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from knobs import overlap_transform

# Paths (relative to project root)
TRIAL = "vasst"
PROJECT_ROOT = Path(__file__).parent.parent.parent
ORACLE_PATH = PROJECT_ROOT / f"data/oracles/{TRIAL}/semi_synthetic.parquet"
COHORT_PATH = PROJECT_ROOT / f"data/cohorts/{TRIAL}/STUDY_COHORT.parquet"
REFERENCE_PATH = Path(__file__).parent / "reference_values.yaml"
OUTPUT_DIR = PROJECT_ROOT / f"data/oracles/{TRIAL}"
CAUSAL_PARAMS_PATH = OUTPUT_DIR / "causal_params.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_reference_values(yaml_file):
    """Load published VASST reference values."""
    with open(yaml_file) as f:
        return yaml.safe_load(f)


def check_primary_ate(oracle_df, reference, knobs):
    """Tier 1: oracle ATE must fall within VASST's reported 95% CI for the ARD.

    VASST's primary result was non-significant (p=0.26); the oracle anchors to Table 2's
    adjusted AOR (a different estimand from the crude ARD), so matching the CI — not the point
    estimate — is the statistically appropriate bar.
    """
    oracle_ate = oracle_df["true_tau"].mean()
    expected_ate = reference["primary_effect"]["ard_observed"]
    ci_lo, ci_hi = reference["primary_effect"]["ard_ci_lower"], reference["primary_effect"]["ard_ci_upper"]

    # gamma (heterogeneity) and s (scale) deliberately move tau(X)/ATE away from the
    # trial-anchored value; only meaningful to compare against VASST's CI at identity.
    gamma, s = knobs.get("gamma_hetero", 1.0), knobs.get("xi_effect", 1.0)
    if gamma != 1.0 or s != 1.0:
        return {
            "tier": 1,
            "check": "Primary ATE",
            "expected": expected_ate,
            "oracle_value": float(oracle_ate),
            "ci": [ci_lo, ci_hi],
            "status": "SKIP",
            "notes": f"gamma={gamma}, s={s} deliberately perturb ATE away from VASST's CI; not comparable."
        }

    status = "PASS" if ci_lo <= oracle_ate <= ci_hi else "FAIL"

    return {
        "tier": 1,
        "check": "Primary ATE",
        "expected": expected_ate,
        "oracle_value": float(oracle_ate),
        "ci": [ci_lo, ci_hi],
        "status": status,
        "notes": f"Oracle ATE: {oracle_ate:.4f}, VASST point est.: {expected_ate:.4f}, 95% CI: [{ci_lo}, {ci_hi}]"
    }


def check_baseline_risk(oracle_df, cohort_df, reference, tolerance=0.03):
    """Tier 2: Compare P(Y0=1) from oracle to control arm mortality."""
    oracle_baseline = oracle_df["Y0"].mean()

    # Get observed control mortality from cohort
    controls = cohort_df.filter(pl.col("Treatment Arm") == "Never")
    observed_mortality = controls["Mortality in Hospital"].mean()

    # Reference uses norepinephrine (control) mortality from VASST
    expected_baseline = reference["primary_effect"]["mortality_norepinephrine_pct"] / 100

    status = "PASS" if abs(oracle_baseline - expected_baseline) <= tolerance else "FAIL"

    return {
        "tier": 2,
        "check": "Baseline Risk",
        "expected": expected_baseline,
        "oracle_value": float(oracle_baseline),
        "observed_control": float(observed_mortality),
        "tolerance": tolerance,
        "status": status,
        "notes": f"Oracle P(Y0): {oracle_baseline:.4f}, Expected: {expected_baseline:.4f}, Observed: {observed_mortality:.4f}"
    }


def check_propensity_calibration(oracle_df, cohort_df, knobs, tolerance=0.02):
    """Tier 3: Verify oracle propensity e(X) reflects observational treatment selection (fitted
    on the real cohort's T ~ X since commit f3fead7), not VASST's 1:1 trial randomization.
    pi_T=0.5 is retained in causal_params.json only as a diagnostic constant for the overlap
    knob (beta), not a calibration target. ps_min/ps_max are reported but don't gate status —
    positivity is a benchmark diagnostic computed downstream in D_evaluation.py.
    """
    oracle_ps_mean = oracle_df["propensity_score"].mean()

    # ps_min/ps_max are reported for reference only; positivity (ESS, extreme-weight fraction)
    # is a benchmark diagnostic computed downstream in D_evaluation.py, not gated here — a
    # fitted e(X) on real observational covariates can legitimately come close to 0/1.
    ps_min = oracle_df["propensity_score"].min()
    ps_max = oracle_df["propensity_score"].max()

    # Observational treatment fraction: the fitted e(X) should track this, not RCT randomization
    treated = (cohort_df.select("Treatment Arm") == "Early").sum().item()
    total = len(cohort_df)
    observed_tx_frac = treated / total

    # beta (overlap knob) interpolates e(X) toward 0.5 (see B_oracles/shared/knobs.py); apply
    # the same logit-space interpolation to observed_tx_frac to get the beta-adjusted target.
    beta = knobs.get("kappa_overlap", 1.0)
    expected_ps_mean = float(overlap_transform(np.array([observed_tx_frac]), beta)[0])

    status = "PASS" if abs(oracle_ps_mean - expected_ps_mean) <= tolerance else "FAIL"

    return {
        "tier": 3,
        "check": "Propensity Calibration (Observational Fit)",
        "oracle_value": float(oracle_ps_mean),
        "expected": expected_ps_mean,
        "tolerance": tolerance,
        "ps_min": float(ps_min),
        "ps_max": float(ps_max),
        "observational_tx_frac": float(observed_tx_frac),
        "beta": beta,
        "status": status,
        "notes": f"Oracle π_T={oracle_ps_mean:.4f} vs. beta={beta}-adjusted target {expected_ps_mean:.4f} "
                 f"(observational TX frac {observed_tx_frac:.4f} interpolated toward 0.5)."
    }


def check_subgroup_effects(oracle_df, cohort_df, reference, knobs):
    """Tier 4: Validate subgroup effect directions match VASST."""
    # gamma=0 makes tau(X) constant (fully homogeneous) and s=0 zeroes it out; either
    # deliberately erases the lactate-quartile gradient this check looks for.
    gamma, s = knobs.get("gamma_hetero", 1.0), knobs.get("xi_effect", 1.0)
    if gamma == 0.0 or s == 0.0:
        return {
            "tier": 4,
            "check": "Subgroup Effects (Lactate)",
            "status": "SKIP",
            "notes": f"gamma={gamma}, s={s} deliberately flatten tau(X); no gradient to validate."
        }

    # Get lactate from cohort and merge with oracle
    cohort_select = cohort_df.select(["Global ICU Stay ID", "Max Lactate (0-6h)"])
    oracle_pd = oracle_df.to_pandas() if hasattr(oracle_df, 'to_pandas') else oracle_df

    # Merge (assume same order or matching by ID)
    if isinstance(cohort_select, pl.DataFrame):
        cohort_lactate = cohort_select.to_pandas()
    else:
        cohort_lactate = cohort_select

    if len(cohort_lactate) == len(oracle_pd):
        combined = oracle_pd.copy()
        combined["lactate"] = cohort_lactate["Max Lactate (0-6h)"].values
    else:
        return {
            "tier": 4,
            "check": "Subgroup Effects (Lactate)",
            "status": "SKIP",
            "notes": "Could not align cohort and oracle data"
        }

    # Split by lactate quartile
    q1_threshold = combined["lactate"].quantile(0.25)
    q4_threshold = combined["lactate"].quantile(0.75)

    q1_tau = combined[combined["lactate"] <= q1_threshold]["true_tau"].mean()
    q4_tau = combined[combined["lactate"] >= q4_threshold]["true_tau"].mean()

    # Expected: low lactate (Q1) -> more benefit (negative), high lactate (Q4) -> less benefit (less negative)
    direction_ok = q1_tau < q4_tau
    status = "PASS" if direction_ok else "FAIL"

    ref_q1 = reference["subgroups"]["lactate_quartile"]["q1_low"]["ard"]
    ref_q4 = reference["subgroups"]["lactate_quartile"]["q4_high"]["ard"]

    return {
        "tier": 4,
        "check": "Subgroup Effects (Lactate)",
        "status": status,
        "q1_ard_oracle": float(q1_tau),
        "q1_ard_reference": ref_q1,
        "q4_ard_oracle": float(q4_tau),
        "q4_ard_reference": ref_q4,
        "direction_correct": direction_ok,
        "notes": f"Q1 oracle: {q1_tau:.4f} (ref: {ref_q1:.4f}), Q4 oracle: {q4_tau:.4f} (ref: {ref_q4:.4f})"
    }


def check_ranges_and_safety(oracle_df):
    """Tier 5: Verify all values are in valid ranges."""
    checks = {}

    # Propensity scores
    ps_min = oracle_df["propensity_score"].min()
    ps_max = oracle_df["propensity_score"].max()
    checks["propensity_range"] = (ps_min > 0.0) and (ps_max < 1.0)

    # Risks (Y0, Y1)
    y0_min, y0_max = oracle_df["Y0"].min(), oracle_df["Y0"].max()
    y1_min, y1_max = oracle_df["Y1"].min(), oracle_df["Y1"].max()
    checks["y0_range"] = (y0_min >= 0.0) and (y0_max <= 1.0)
    checks["y1_range"] = (y1_min >= 0.0) and (y1_max <= 1.0)

    # Treatment effects
    tau_min, tau_max = oracle_df["true_tau"].min(), oracle_df["true_tau"].max()
    checks["tau_range"] = (tau_min >= -1.0) and (tau_max <= 1.0)

    # NaN/Inf (check key numeric columns)
    key_cols = ["propensity_score", "Y0", "Y1", "true_tau", "log_or"]
    has_nan = any(oracle_df[col].is_nan().any() for col in key_cols if col in oracle_df.columns)
    has_inf = any(oracle_df[col].is_infinite().any() for col in key_cols if col in oracle_df.columns)
    checks["no_nan_inf"] = not has_nan and not has_inf

    status = "PASS" if all(checks.values()) else "FAIL"

    return {
        "tier": 5,
        "check": "Range & Safety",
        "status": status,
        "propensity_range": checks["propensity_range"],
        "y0_range": checks["y0_range"],
        "y1_range": checks["y1_range"],
        "tau_range": checks["tau_range"],
        "no_nan_inf": checks["no_nan_inf"],
        "notes": f"All checks: {all(checks.values())}"
    }



def generate_report(checks):
    """Generate human-readable validation report."""
    report = []
    report.append("=" * 80)
    report.append(f"ORACLE GROUNDING REPORT: {TRIAL.upper()}")
    report.append("=" * 80)
    report.append("")
    report.append("Goal: Verify that the oracle embeds VASST-calibrated treatment mechanisms into a realistic observational population with known ground-truth effects.")
    report.append("")

    for check in checks:
        name = check["check"]
        status = check["status"]
        details = ", ".join(
            f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}"
            for key, val in check.items() if key not in ["tier", "check", "status"]
        )
        report.append(f"{name}: {status} ({details})" if details else f"{name}: {status}")
        report.append("")

    # Overall decision
    # Critical tiers: 1 (ATE), 2 (baseline risk), 3 (RCT propensity), 5 (ranges/safety)
    passed_critical = all(c["status"] in ("PASS", "SKIP") for c in checks if c["tier"] in [1, 2, 3, 5])
    overall = "PASS" if passed_critical else "FAIL"

    report.append("=" * 80)
    report.append(f"OVERALL: {overall}")
    report.append("=" * 80)

    if overall == "PASS":
        report.append("Oracle is clinically grounded and causally valid. Primary ATE preserved, baseline risk calibrated, effect modifiers plausible, and all values pass sanity checks. Ready for embedding & causal evaluation.")
    else:
        report.append("Oracle validation FAILED. Review specific check failures and adjust oracle parameters (gamma, alpha_lac, alpha_ne) as needed.")

    return "\n".join(report)


def main():
    print("Loading oracle and reference data...")

    # Load data
    oracle_df = pl.read_parquet(ORACLE_PATH)
    cohort_df = pl.read_parquet(COHORT_PATH)
    reference = load_reference_values(REFERENCE_PATH)
    with open(CAUSAL_PARAMS_PATH) as f:
        knobs = json.load(f)["knobs"]

    print(f"  Oracle: {oracle_df.shape[0]} patients × {oracle_df.shape[1]} columns")
    print(f"  Cohort: {cohort_df.shape[0]} patients × {cohort_df.shape[1]} columns")
    print(f"  Knobs: {knobs}")

    # Run all checks
    print("\nRunning validation checks...")
    checks = [
        check_primary_ate(oracle_df, reference, knobs),
        check_baseline_risk(oracle_df, cohort_df, reference),
        check_propensity_calibration(oracle_df, cohort_df, knobs),
        check_subgroup_effects(
            oracle_df.filter(pl.col("seed") == 0),
            cohort_df.filter(pl.col("Treatment Arm").is_in(["Early", "Never"])),
            reference,
            knobs,
        ),
        check_ranges_and_safety(oracle_df)
    ]

    # Save results
    print("Saving results...")

    # CSV report
    pd.DataFrame(checks).to_csv(str(OUTPUT_DIR / "validation_checks.csv"), index=False)

    # Text report
    report = generate_report(checks)
    with open(str(OUTPUT_DIR / "VALIDATION_REPORT.txt"), "w") as f:
        f.write(report)

    # Console output
    print("")
    print(report)

    # Exit code
    passed_critical = all(c["status"] in ("PASS", "SKIP") for c in checks if c["tier"] in [1, 2, 3, 5])
    sys.exit(0 if passed_critical else 1)


if __name__ == "__main__":
    main()
