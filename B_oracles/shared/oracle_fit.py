# Generic in-sample T-learner oracle fitting: real covariates X, T; potential outcomes
# Y(0)/Y(1) fit from a trial's own two arms (no published subgroup table to anchor against,
# used by RCTBench and JOBS). tau_0 is the trial's covariate-adjusted effect; heterogeneity
# is a flexible per-arm MLP fit, shrunk by a sample-size-derived factor toward tau_0 so small
# trials don't get noise written into "ground truth" tau (auto-derived from n/d, unlike
# knobs.py's researcher-settable gamma knob, which shrinks the same way for a different reason).

import os
import sys
import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knobs import FittedOracle, apply_alignment_split, build_covariate_role_map

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_MLP_KWARGS = {"hidden_layer_sizes": (8,), "alpha": 1.0, "max_iter": 2000, "random_state": 42}

# shrinkage = clip(min(n_control, n_treated) / (SHRINKAGE_OBS_PER_PARAM * d), SHRINKAGE_MIN, SHRINKAGE_MAX)
# "~5 observations per free parameter before the per-arm shape term is trusted at full
# weight" (the events-per-variable rule of thumb, relaxed since the shape term is already
# L2-regularized). `shape` is de-meaned below, so this rescales CATE spread only, never the
# ATE level. Without it, n~40 trials write per-arm MLP noise into true_tau.
SHRINKAGE_OBS_PER_PARAM = 5.0
SHRINKAGE_MIN = 0.1
SHRINKAGE_MAX = 1.0


def _shrinkage_factor(n_control: int, n_treated: int, d: int) -> float:
    return float(np.clip(min(n_control, n_treated) / (SHRINKAGE_OBS_PER_PARAM * d), SHRINKAGE_MIN, SHRINKAGE_MAX))


def fitted_propensity(X_std: np.ndarray, T: np.ndarray, family: str = "logit") -> np.ndarray:
    """Fitted, covariate-dependent e(X) = P(T=1|X), used for T_synthetic sampling.

    family="logit": LogisticRegression (default, today's oracle).
    family="tree":  RandomForestClassifier -- a non-smooth decision boundary that is not
                    congenial to the logistic-propensity balance/overlap diagnostics
                    (congeniality-triangulation independence check).
    """
    if family == "tree":
        model = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=max(5, len(T) // 100),
            random_state=42, n_jobs=-1,
        ).fit(X_std, T)
    else:
        model = LogisticRegression(C=1.0, max_iter=1000).fit(X_std, T)
    return np.clip(model.predict_proba(X_std)[:, 1], 1e-8, 1 - 1e-8)


def fit_binary_oracle(X_std: np.ndarray, T: np.ndarray, Y: np.ndarray, alpha_alignment: float = 1.0,
                      assignment_family: str = "logit") -> FittedOracle:
    ctrl, trt = T == 0, T == 1
    d = X_std.shape[1]

    # Alignment split (pre-fit, see apply_alignment_split). tau_0's adjustment regression
    # below intentionally still uses the full X_std — it's a calibration/unbiasedness step
    # (randomized), not part of the response surface under test.
    propensity_idx, outcome_idx = apply_alignment_split(d, alpha_alignment)

    risk0_model = MLPClassifier(**_MLP_KWARGS).fit(X_std[ctrl][:, outcome_idx], Y[ctrl])
    risk1_model = MLPClassifier(**_MLP_KWARGS).fit(X_std[trt][:, outcome_idx], Y[trt])
    risk0 = np.clip(risk0_model.predict_proba(X_std[:, outcome_idx])[:, 1], 1e-8, 1 - 1e-8)
    risk1_raw = np.clip(risk1_model.predict_proba(X_std[:, outcome_idx])[:, 1], 1e-8, 1 - 1e-8)

    shape = (logit(risk1_raw) - logit(risk0))
    shape = shape - shape.mean()

    adj = LogisticRegression(C=1.0, max_iter=1000).fit(np.column_stack([T, X_std]), Y)
    tau_0 = float(adj.coef_[0][0])

    shrinkage = _shrinkage_factor(ctrl.sum(), trt.sum(), d)
    log_or = tau_0 + shrinkage * shape  # mean(log_or) == tau_0 exactly (shape is de-meaned)
    risk1 = expit(logit(risk0) + log_or)
    # true_tau.mean() != tau_0 in general: expit is nonlinear, so averaging risk1-risk0 after
    # the transform doesn't reproduce tau_0 (only mean(log_or) does). Compare against the
    # trial's raw diff-in-means ATE instead when sanity-checking this.
    true_tau = (risk1 - risk0).astype(np.float32)

    propensity = fitted_propensity(X_std[:, propensity_idx], T, family=assignment_family)
    print(f"  shrinkage={shrinkage:.3f}", end="  ")

    return FittedOracle(
        outcome_type="binary",
        mu0=risk0.astype(np.float64),
        mu1=risk1.astype(np.float64),
        propensity=propensity,
        sigma0=None,
        sigma1=None,
        tau_fitted=true_tau,
        ate_fitted=float(true_tau.mean()),
        tau_0=tau_0,
        pi_T=float(T.mean()),
        covariate_role_map=build_covariate_role_map(d, propensity_idx, outcome_idx),
    )


def fit_continuous_oracle(X_std: np.ndarray, T: np.ndarray, Y: np.ndarray, alpha_alignment: float = 1.0,
                          assignment_family: str = "logit") -> FittedOracle:
    ctrl, trt = T == 0, T == 1
    d = X_std.shape[1]

    propensity_idx, outcome_idx = apply_alignment_split(d, alpha_alignment)

    mu0_model = MLPRegressor(**_MLP_KWARGS).fit(X_std[ctrl][:, outcome_idx], Y[ctrl])
    mu1_model = MLPRegressor(**_MLP_KWARGS).fit(X_std[trt][:, outcome_idx], Y[trt])
    mu0 = mu0_model.predict(X_std[:, outcome_idx])
    mu1_raw = mu1_model.predict(X_std[:, outcome_idx])

    shape = mu1_raw - mu0
    shape = shape - shape.mean()

    adj = Ridge(alpha=10.0).fit(np.column_stack([T, X_std]), Y)
    tau_0 = float(adj.coef_[0])

    shrinkage = _shrinkage_factor(ctrl.sum(), trt.sum(), d)
    true_tau = (tau_0 + shrinkage * shape).astype(np.float32)
    mu1 = mu0 + true_tau

    sigma0 = max(float((Y[ctrl] - mu0[ctrl]).std()), 1e-3)
    sigma1 = max(float((Y[trt] - mu1[trt]).std()), 1e-3)

    propensity = fitted_propensity(X_std[:, propensity_idx], T, family=assignment_family)
    print(f"  shrinkage={shrinkage:.3f}", end="  ")

    return FittedOracle(
        outcome_type="continuous",
        mu0=mu0.astype(np.float64),
        mu1=mu1.astype(np.float64),
        propensity=propensity,
        sigma0=sigma0,
        sigma1=sigma1,
        tau_fitted=true_tau,
        ate_fitted=float(true_tau.mean()),
        tau_0=tau_0,
        pi_T=float(T.mean()),
        covariate_role_map=build_covariate_role_map(d, propensity_idx, outcome_idx),
    )
