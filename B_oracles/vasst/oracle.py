# Semi-synthetic oracle generation for the causal embedding benchmark (VASST trial).
#
# Reads:  data/cohorts/vasst/STUDY_COHORT.parquet
# Writes: data/oracles/vasst/semi_synthetic.parquet
#         data/oracles/vasst/scaler.json           (mean/std for each continuous column)
#         data/oracles/vasst/feature_cols.json     (ordered feature column list after encoding)
#         data/oracles/vasst/causal_params.json    (fitted mu0_coef, tau_0, target/realized ATE, knobs)
#
# VASST 2008 (Russell et al., NEJM): real ICU covariates, synthetic T and Y(0)/Y(1).
# Treatment subset: Early vs Never (binary T=1/T=0); Delayed arm dropped for the MVP.
#
# propensity: fitted e(X) = P(T=1|X) via logistic regression, used for T_synthetic sampling
#             (the cohort is real-world observational ICU data, not VASST's own 1:1
#             randomization — pi_T is retained only as a diagnostic constant, printed but
#             not asserted against; see B_oracles/README.md).
# mu0_coef  : Y_obs ~ X logistic regression on controls, recalibrated to VASST's published
#             control-arm mortality.
# log_or    : tau_0 (VASST Table 2 adjusted AOR) plus an MLP surface fit to Table 3's
#             subgroup log-ORs (lactate quartile, NE-dose split) — evidence-anchored
#             heterogeneity. Oracle knobs (methods.tex Section 9) applied via
#             B_oracles/shared/knobs.py; identity (1.0) by default.

import json
import os
import sys

import numpy as np
import polars as pl
import yaml
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from config import SEEDS
from knobs import (
    FITTED_KNOBS,
    FittedOracle,
    apply_alignment_split,
    apply_knobs,
    build_covariate_role_map,
    draw_binary,
    knob_params,
    parse_oracle_args,
    setup_oracle_dir,
)
from oracle_io import (
    assemble_seed_df,
    should_log_seed,
    skip_if_exists,
    write_causal_params,
    write_feature_cols,
)
from oracle_fit import _MLP_KWARGS, fitted_propensity


def _cell_log_or(mortality_treated_pct: float, mortality_control_pct: float) -> float:
    return float(logit(mortality_treated_pct / 100) - logit(mortality_control_pct / 100))


def fit_treatment_effect_surface(
    X_std: np.ndarray,
    lactate_raw: np.ndarray,
    ne_dose_raw: np.ndarray,
    reference: dict,
) -> np.ndarray:
    """Evidence-anchored, flexible log-OR surface: MLP fit to per-patient subgroup log-ORs
    from VASST Table 3 (lactate quartile + NE-dose split proxy for vasopressor count)."""
    lactate_raw = np.where(np.isnan(lactate_raw), np.nanmedian(lactate_raw), lactate_raw)
    ne_dose_raw = np.where(np.isnan(ne_dose_raw), np.nanmedian(ne_dose_raw), ne_dose_raw)

    lac_ref = reference["subgroups"]["lactate_quartile"]
    lac_bin = np.digitize(lactate_raw, np.quantile(lactate_raw, [0.25, 0.5, 0.75]))
    lac_cells = [lac_ref["q1_low"], lac_ref["q2"], lac_ref["q3"], lac_ref["q4_high"]]
    lac_log_or = np.array(
        [_cell_log_or(c["vasopressin_mortality_pct"], c["norepinephrine_mortality_pct"]) for c in lac_cells]
    )[lac_bin]

    ne_ref = reference["subgroups"]["vasopressor_count"]
    ne_bin = (ne_dose_raw > np.median(ne_dose_raw)).astype(int)
    ne_cells = [ne_ref["one_agent"], ne_ref["two_plus_agents"]]
    ne_log_or = np.array(
        [_cell_log_or(c["vasopressin_mortality_pct"], c["norepinephrine_mortality_pct"]) for c in ne_cells]
    )[ne_bin]

    y_cell = lac_log_or + ne_log_or
    model = MLPRegressor(**_MLP_KWARGS)
    model.fit(X_std, y_cell)
    return model.predict(X_std)


args, knob_config = parse_oracle_args(
    skip_help="Skip regeneration if the output parquet already exists",
    knob_note="",
)

# ---------------------------------------------------------------------------
TRIAL = "vasst"
OUT_DIR = setup_oracle_dir(TRIAL, args.knobs_preset)

skip_if_exists(OUT_DIR, args.skip_existing)

# ---------------------------------------------------------------------------
# 1. Load cohort and restrict to Early vs Never
# ---------------------------------------------------------------------------
print("1. Loading cohort")
cohort = pl.read_parquet(f"data/cohorts/{TRIAL}/STUDY_COHORT.parquet")

cohort = cohort.filter(pl.col("Treatment Arm").is_in(["Early", "Never"]))
print(f"   Early+Never cohort: {len(cohort)} patients "
      f"({cohort['Treatment Arm'].value_counts().to_pandas().to_string(index=False)})")

# ---------------------------------------------------------------------------
# 2. Encode X
# ---------------------------------------------------------------------------
print("2. Encoding covariates")

# All continuous clinical variables (numeric, unsummarized)
CONTINUOUS_COLS = [
    "Admission Age (years)",
    "Admission Height (cm)",
    "Admission Weight (kg)",
    "Max SOFA (0-6h)",
    "Peak Norepinephrine (0-6h)",
    "Min MAP (0-6h)",
    "Max Lactate (0-6h)",
    "Max Creatinine (0-6h)",
    "Fluid Balance 0-6h (mL)",
    "Max Heart Rate (0-6h)",
    "Mean Temperature (0-6h)",
    "Min pH (0-6h)",
    "Min Bicarbonate (0-6h)",
    "APACHE II Score",
]

# Binary clinical indicators
BINARY_COLS = ["Ventilated at T_landmark"]

# Comorbidity flags (all binary)
COMORBIDITY_COLS = [
    "Congestive heart failure",
    "Chronic pulmonary disease",
    "Renal failure",
    "Diabetes",
    "Liver disease",
    "Alcohol abuse",
    "Drug abuse",
    "Cancer",
    "AIDS/HIV",
    "Ischemic heart disease",
]

# Categorical variables to encode
CATEGORICAL_COLS = ["Gender", "Source Dataset", "Ethnicity"]

# Cast types
df = cohort.with_columns(
    pl.col("Ventilated at T_landmark").cast(pl.Int8).alias("Ventilated at T_landmark"),
    pl.col("Gender").cast(pl.Utf8),
    pl.col("Ethnicity").cast(pl.Utf8),
    **{col: pl.col(col).cast(pl.Int8) for col in COMORBIDITY_COLS},
)

# One-hot encode Gender, Ethnicity, and Source Dataset; drop one level each to avoid multicollinearity
encoded_dfs = [
    df.select(["Global ICU Stay ID", "Treatment Arm"] + CONTINUOUS_COLS + BINARY_COLS + COMORBIDITY_COLS)
      .rename({"Global ICU Stay ID": "ID"})
]

for cat_col in ["Gender", "Ethnicity", "Source Dataset"]:
    dummies = df.select(cat_col).to_dummies(cat_col, separator="=")
    # Drop the last category to avoid perfect multicollinearity
    cols_to_drop = [c for c in dummies.columns if c.startswith(f"{cat_col}=")]
    if len(cols_to_drop) > 1:
        dummies = dummies.drop(cols_to_drop[-1])
    encoded_dfs.append(dummies)

df_encoded = pl.concat(encoded_dfs, how="horizontal")

FEATURE_COLS = CONTINUOUS_COLS + BINARY_COLS + COMORBIDITY_COLS
for cat_col in ["Gender", "Ethnicity", "Source Dataset"]:
    cat_cols = [c for c in df_encoded.columns if c.startswith(f"{cat_col}=")]
    FEATURE_COLS.extend(cat_cols)

print(f"   Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

write_feature_cols(OUT_DIR, FEATURE_COLS)

# ---------------------------------------------------------------------------
# 3. Standardise continuous columns; impute missing with column median pre-std
# ---------------------------------------------------------------------------
print("3. Standardising features")
X_raw = df_encoded.select(FEATURE_COLS).to_numpy().astype(np.float32)

# Impute NaNs with column median before standardising
col_medians = np.nanmedian(X_raw, axis=0)
nan_mask = np.isnan(X_raw)
X_raw[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

# Standardize continuous columns only; keep binary/categorical as-is
n_continuous = len(CONTINUOUS_COLS)
means = X_raw[:, :n_continuous].mean(axis=0)
stds  = X_raw[:, :n_continuous].std(axis=0)
stds[stds == 0] = 1.0  # avoid division by zero for constant columns

X_std = X_raw.copy()
X_std[:, :n_continuous] = (X_raw[:, :n_continuous] - means) / stds

scaler = {
    "continuous_cols": CONTINUOUS_COLS,
    "means": means.tolist(),
    "stds":  stds.tolist(),
}
with open(f"{OUT_DIR}/scaler.json", "w") as f:
    json.dump(scaler, f, indent=2)

print(f"   X_std shape: {X_std.shape}  (n_patients × n_features)")

# ---------------------------------------------------------------------------
# 3.5 Calibrate oracle parameters
# ---------------------------------------------------------------------------
print("3.5. Calibrating oracle parameters")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_values.yaml")) as f:
    reference = yaml.safe_load(f)

T_obs = (cohort["Treatment Arm"] == "Early").cast(pl.Int8).to_numpy().astype(np.int32)
Y_obs = cohort["Mortality in Hospital"].cast(pl.Int8).fill_null(0).to_numpy().astype(np.int32)

pi_T = 0.5

# Alignment split (pre-fit): at alpha_alignment=0, propensity and outcome models see disjoint
# covariate subsets (no true confounders); at alpha_alignment=1, both see the full set (today's
# behavior).
propensity_idx, outcome_idx = apply_alignment_split(len(FEATURE_COLS), knob_config.alpha_alignment)

# Fitted, covariate-dependent propensity e(X) = P(T=1|X), used for T_synthetic sampling below.
# VASST itself randomized 1:1 (pi_T above is retained only as a diagnostic, printed at Section
# 5), but the *cohort* here is real-world observational ICU data, not the trial's randomized allocation
# — a fitted e(X) reflecting actual treatment-selection patterns is more faithful, and is what
# the overlap knob (kappa_overlap, see B_oracles/shared/knobs.py) interpolates toward pi_T=0.5 from.
propensity_fitted = fitted_propensity(
    X_std[:, propensity_idx], T_obs, family=knob_config.assignment_family).astype(np.float64)
print(f"   fitted e(X): mean={propensity_fitted.mean():.3f}  (pi_T constant: {pi_T:.3f})")

VASST_CONTROL_MORTALITY = reference["primary_effect"]["mortality_norepinephrine_pct"] / 100
ctrl_mask = T_obs == 0
lr_Y = LogisticRegression(max_iter=1000, C=1.0)
lr_Y.fit(X_std[ctrl_mask][:, outcome_idx], Y_obs[ctrl_mask])
mu0_coef = np.concatenate([lr_Y.intercept_, lr_Y.coef_[0]]).astype(np.float64)

# Recalibrate intercept to match VASST control mortality across full cohort
X_outcome = X_std[:, outcome_idx].astype(np.float64)
current_mean_risk = (1 / (1 + np.exp(-mu0_coef[0] - X_outcome @ mu0_coef[1:]))).mean()
adjustment = np.log(VASST_CONTROL_MORTALITY / (1 - VASST_CONTROL_MORTALITY)) - np.log(current_mean_risk / (1 - current_mean_risk))
mu0_coef[0] += adjustment
_risk0_all = 1 / (1 + np.exp(-mu0_coef[0] - X_outcome @ mu0_coef[1:]))
print(f"   mu0_coef intercept={mu0_coef[0]:.4f}  mean P(Y0=1)={_risk0_all.mean():.3f} (target {VASST_CONTROL_MORTALITY:.3f})")

lactate_raw = df["Max Lactate (0-6h)"].to_numpy().astype(np.float64)
ne_dose_raw = df["Peak Norepinephrine (0-6h)"].to_numpy().astype(np.float64)
log_or_shape_raw = fit_treatment_effect_surface(X_outcome, lactate_raw, ne_dose_raw, reference)
log_or_shape = log_or_shape_raw - log_or_shape_raw.mean()
tau_0 = float(np.log(reference["primary_effect"]["or_adjusted"]))
log_or = tau_0 + log_or_shape

risk0_safe = np.clip(_risk0_all, 1.0e-8, 1.0 - 1.0e-8)
risk1_all = expit(logit(risk0_safe) + log_or)
realized_ATE = float((risk1_all - risk0_safe).mean())
target_ATE = reference["primary_effect"]["ard_observed"]
print(f"   τ_0={tau_0:.4f} (Table 2 AOR)  realized ARD={realized_ATE:.4f}  (VASST point est.: {target_ATE:.4f})")

# ---------------------------------------------------------------------------
# 3.6 Build FittedOracle and apply post-fit oracle-parameterization knobs (identity by default;
#     alpha_alignment was already applied pre-fit above, via the propensity_idx/outcome_idx split)
# ---------------------------------------------------------------------------
fitted_oracle = FittedOracle(
    outcome_type="binary",
    mu0=risk0_safe,
    mu1=risk1_all,
    propensity=propensity_fitted,
    sigma0=None,
    sigma1=None,
    tau_fitted=(risk1_all - risk0_safe).astype(np.float32),
    ate_fitted=realized_ATE,
    tau_0=tau_0,
    pi_T=pi_T,
    covariate_role_map=build_covariate_role_map(len(FEATURE_COLS), propensity_idx, outcome_idx),
)

print(f"   Applying knob preset '{args.knobs_preset}': {knob_config}")
fitted_oracle = apply_knobs(fitted_oracle, knob_config)

causal_params = {
    "n":            int(X_std.shape[0]),
    "pi_T":         pi_T,
    "mu0_coef":     mu0_coef.tolist(),
    "tau_0":        tau_0,          # VASST Table 2 adjusted AOR, log-odds scale
    "target_ATE":   target_ATE,     # VASST published (unadjusted) ATE point estimate
    "realized_ATE": realized_ATE,
    "outcome_type": "binary",
    "ood_sources":  ["eICU-CRD", "AmsterdamUMCdb"],
    **knob_params(knob_config, FITTED_KNOBS),
}
write_causal_params(OUT_DIR, causal_params)
print(f"   Saved {OUT_DIR}/causal_params.json")

# ---------------------------------------------------------------------------
# 4. Generate oracle draws across seed replications
# ---------------------------------------------------------------------------
print(f"4. Generating {len(SEEDS)} oracle draws across seed replications")

log_or_final = (logit(np.clip(fitted_oracle.mu1, 1e-8, 1 - 1e-8)) - logit(np.clip(fitted_oracle.mu0, 1e-8, 1 - 1e-8))).astype(np.float32)
ids = df_encoded.select("ID").to_series().to_numpy()
source_dataset = df["Source Dataset"].to_numpy()
extra_cols = {
    "T_observed":       T_obs.astype(np.int8),
    "Y_observed":       Y_obs.astype(np.int8),
    "log_or":           log_or_final,
    "propensity_score": fitted_oracle.propensity.astype(np.float32),
    # deterministic control risk E[Y|X,A=0] (mu1 = mu0 + true_tau); noise-free regression
    # target for the Bayes-reweighted representation-validity RICB in D_evaluation.
    "mu0":              fitted_oracle.mu0.astype(np.float32),
}

oracle_dfs = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    draws = draw_binary(fitted_oracle, rng)
    oracle_dfs.append(assemble_seed_df(
        seed, len(ids), draws["T_synthetic"], draws["Y_obs_synthetic"], draws["Y0"], draws["Y1"],
        fitted_oracle.tau_fitted, source_dataset,
        X_std, FEATURE_COLS, ids=ids, extra_cols=extra_cols,
    ))

    if should_log_seed(seed):
        print(f"   seed={seed:2d}  ATE={fitted_oracle.ate_fitted:.4f}  "
              f"mean(Y0)={draws['Y0'].mean():.3f}  mean(Y1)={draws['Y1'].mean():.3f}  "
              f"mean(T)={draws['T_synthetic'].mean():.3f}")

# ---------------------------------------------------------------------------
# 5. Sanity checks (on first seed)
# ---------------------------------------------------------------------------
print("5. Running sanity checks on seed=0")
ps_mean = fitted_oracle.propensity.mean()
ps_min, ps_max = fitted_oracle.propensity.min(), fitted_oracle.propensity.max()
# Positivity is measured as a benchmark diagnostic in D_evaluation.py (ESS, extreme-weight
# fraction), not gated here: per B_oracles/README.md, kappa_overlap<1 interpolates e(X) toward
# 0.5 (better positivity) while kappa_overlap>1 pushes it further away (worse positivity) by
# design, and a fitted e(X) on real observational covariates can legitimately come close to
# 0/1 for some patients even at baseline.
print(f"   Fitted e(X): mean={ps_mean:.3f}  range=[{ps_min:.3f}, {ps_max:.3f}]  "
      f"(observational cohort, not VASST's {pi_T:.3f} randomization ratio — see module docstring)")

# gamma_hetero/xi_effect deliberately move ate_fitted away from realized_ATE post-knob
# (apply_heterogeneity/apply_scale in knobs.py recompute it); only meaningful to check equality
# at identity.
if knob_config.gamma_hetero == 1.0 and knob_config.xi_effect == 1.0:
    assert abs(fitted_oracle.ate_fitted - realized_ATE) < 0.005, (
        f"Fitted ATE ({fitted_oracle.ate_fitted:.4f}) doesn't match the calibrated value "
        f"({realized_ATE:.4f})."
    )

ci_lo, ci_hi = reference["primary_effect"]["ard_ci_lower"], reference["primary_effect"]["ard_ci_upper"]
assert ci_lo <= realized_ATE <= ci_hi, (
    f"Realized ATE ({realized_ATE:.4f}) falls outside VASST's reported 95% CI [{ci_lo}, {ci_hi}]."
)

print("   All sanity checks passed.")

# ---------------------------------------------------------------------------
# 6. Write Parquet (all seeds, assembled in the loop above)
# ---------------------------------------------------------------------------
print(f"6. Writing {OUT_DIR}/semi_synthetic.parquet")
oracle_df = pl.concat(oracle_dfs)
oracle_df.write_parquet(f"{OUT_DIR}/semi_synthetic.parquet")
print(f"   Written {len(oracle_df)} rows × {oracle_df.width} columns "
      f"({len(SEEDS)} seeds × {len(ids)} patients).")
print("   Done.")
