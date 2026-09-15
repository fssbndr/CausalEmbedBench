# Evaluation functions for the causal embedding benchmark.
#
# Each function accepts plain numpy arrays and returns a flat dict of metrics.
# No I/O — all persistence is handled by D_evaluation.py.
#
# Causal-effect tasks
#   1. Outcome prediction     AUROC/AUPRC (binary) or R^2/RMSE (continuous)
#   2. ATE / ATT              ATE_hat/ATE_error/ATE_SE, ATT_hat/ATT_error (AIPW, REF_BASE)
#   3. CATE/PEHE              PEHE_{learner}_{base} for the 10 learners x active bases,
#                             plus two-level R-/DR-loss-selected PEHE over all learner x base,
#                             tau_recovery_R2 and REF_BASE CATE calibration
#   4. Counterfactual RMSE    CF_RMSE
#   5. Policy value           policy_value/oracle_policy_value/regret + regret_weighted
#                             + policy_accuracy (binary only)
#
# Representation-sufficiency diagnostics — cheap by-products of the fits
# above, checking the assumptions under which replacing X with Z=Phi(X) is
# valid for causal estimation:
#   Balance                (evaluate_balance; Z-only, no model fit, always available)
#                           SMD_mean_abs/SMD_max_abs, Mahalanobis_D
#   Overlap                (from the fitted e(Z) in evaluate_treatment_tasks)
#                           ps_mean/std/quantiles, ESS_*_frac, frac_extreme
#   Propensity preservation ps_AUROC (residual T-information in Z after fitting
#                           e(Z); high AUROC hints at overfitting)
#   Representation validity  RICB_true_* / HET_loss_* -- oracle-only measures of
#                           Melnychuk et al. 2024's Def. 1 (confounding left after
#                           X -> Z / heterogeneity Z cannot resolve). Emitted by
#                           evaluate_cate; references, not deployable diagnostics.

import warnings

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    LogisticRegression,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# Base learner constructors
# ------------------------------------------------------------------------------
# CV-tuned alpha/C (not fixed) so regularization scales with embedding dim.
ALPHA_GRID = np.logspace(-3, 5, 9)
C_GRID     = np.logspace(-4, 4, 9)

def _lr_clf():
    return make_pipeline(
        StandardScaler(),
        # default scoring (accuracy) degenerates under class imbalance
        LogisticRegressionCV(Cs=C_GRID, max_iter=1000, random_state=42, scoring="neg_log_loss"),
    )

def _lr_reg():
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHA_GRID),
    )

def _lr_clf_fixed(C: float):
    """LogisticRegression at a pre-tuned C, no internal grid search."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=1000, random_state=42),
    )

# Quadratic features + the same CV'd linear head as "lr"
def _poly2_clf():
    return make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        StandardScaler(),
        LogisticRegressionCV(Cs=C_GRID, max_iter=1000, random_state=42, scoring="neg_log_loss"),
    )

def _poly2_reg():
    return make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=ALPHA_GRID),
    )

def _clf_or_fixed(base: str, key: str, model_fn, lr_hparams: dict):
    """Pre-tuned LR at lr_hparams[key] if available for this base, else a fresh model_fn(base)."""
    if lr_hparams is not None and base == "lr":
        return _lr_clf_fixed(lr_hparams[key])
    return model_fn(base)

def _tune_C(X: np.ndarray, y: np.ndarray) -> float:
    """Fit LogisticRegressionCV once, return the selected C."""
    pipe = _lr_clf()
    pipe.fit(X, y)
    return float(pipe.named_steps["logisticregressioncv"].C_[0])

# GBM picks its own iteration count via early stopping (held-out 10% split), so no
# per-embedding grid search.
_GBM_KWARGS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, early_stopping=True,
                   validation_fraction=0.1, n_iter_no_change=15, random_state=42)

def _gbm_clf():
    return HistGradientBoostingClassifier(**_GBM_KWARGS)

def _gbm_reg():
    return HistGradientBoostingRegressor(**_GBM_KWARGS)

def _rf_clf():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)

def _rf_reg():
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)


def _clf(base: str):
    return {"lr": _lr_clf, "poly2": _poly2_clf, "gbm": _gbm_clf, "rf": _rf_clf}[base]()

def _reg(base: str):
    return {"lr": _lr_reg, "poly2": _poly2_reg, "gbm": _gbm_reg, "rf": _rf_reg}[base]()

def _fit_weighted(estimator, X, y, sample_weight):
    """estimator.fit(X, y, sample_weight=...), routed through Pipeline's step__param syntax if needed."""
    if isinstance(estimator, Pipeline):
        return estimator.fit(X, y, **{f"{estimator.steps[-1][0]}__sample_weight": sample_weight})
    return estimator.fit(X, y, sample_weight=sample_weight)


def _rmse(a) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))


def has_degenerate_arm(T: np.ndarray, Y: np.ndarray, outcome_type: str = "binary") -> bool:
    """True if either treatment arm has zero variance in Y (e.g. no events at
    all among the treated). Individual-CF tasks fit arm-conditional outcome
    models (mu0, mu1); with no CV-fold rearrangement can supply a second class
    that doesn't exist anywhere in the arm, so such jobs must be skipped
    upstream rather than attempted."""
    if outcome_type != "binary":
        return False
    return (len(np.unique(Y[T == 0])) < 2) or (len(np.unique(Y[T == 1])) < 2)


def _strat_key(T: np.ndarray, Y: np.ndarray, outcome_type: str) -> np.ndarray:
    """Joint (T, Y) stratification key so CV folds preserve both outcome
    classes within each treatment arm — stratifying on T alone can leave a
    fold's control (or treated) arm with single-class Y, which crashes
    classifier fitting downstream."""
    if outcome_type != "binary":
        return T.astype(int)
    return T.astype(int) * 2 + Y.astype(int)


def _binary_splits(E_X: np.ndarray, strata: np.ndarray, cv: int, random_state: int = 42):
    """Stratified CV splits that never hold out a (T,Y) singleton — a stratum
    with only 1 member is always in test for exactly one fold, leaving that
    fold's training arm with a single class. Pin such points to every fold's
    training set instead of splitting them."""
    counts = np.bincount(strata)
    singleton = counts[strata] < 2
    pinned, free = np.where(singleton)[0], np.where(~singleton)[0]
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for tr, te in kf.split(free, strata[free]):
        yield np.concatenate([free[tr], pinned]), free[te]


BASES     = ["lr", "poly2", "gbm"]  # rf also wired in _clf/_reg but not run
LEARNERS  = ["S", "Lo", "T", "X", "RA", "Z", "F", "U", "R", "DR"]

# Fixed estimator for the representation references (RICB_true / HET_loss /
# tau_recovery_R2 / policy / CF_RMSE / calibration / decon_bias): those are properties
# of the embedding, so a --bases-chosen class would confound estimator bias with
# representation bias (Melnychuk et al. 2024). --bases drives only the PEHE_{l}_{b} grid.
REF_BASE = "gbm"
CATE_COLS = [f"PEHE_{l}_{b}" for l in LEARNERS for b in BASES]
# Per-tier PEHE column subsets. The leaderboard aggregates CATE_COLS into the bare
# family names (global mean over tiers) and each subset below into a _{base} companion.
LR_CATE_COLS    = [f"PEHE_{l}_lr" for l in LEARNERS]
GBM_CATE_COLS   = [f"PEHE_{l}_gbm" for l in LEARNERS]
POLY2_CATE_COLS = [f"PEHE_{l}_poly2" for l in LEARNERS]

# Meta-learner families for stratified PEHE_{plugin,pseudo}_mean. X grouped with plug-in
# (final step combines arm imputations, not a single pseudo-outcome regression).
PLUGIN_LEARNERS = ["S", "Lo", "T", "X"]
PSEUDO_LEARNERS = ["RA", "Z", "F", "U", "R", "DR"]
assert sorted(PLUGIN_LEARNERS + PSEUDO_LEARNERS) == sorted(LEARNERS)

OVERLAP_COLS = ["ESS_trt_frac", "ESS_ctrl_frac", "frac_extreme", "ps_AUROC",
                "ps_AUROC_raw", "ps_AUROC_delta", "ps_corr", "ps_spearman",
                "decon_bias", "cf_variance", "overlap_dinf"]
BALANCE_COLS = ["SMD_mean_abs", "SMD_max_abs", "Mahalanobis_D", "Energy_dist"]

# Melnychuk Def. 1 oracle-only references, emitted by evaluate_cate: (ii) RICB = tau^phi -
# (mu1^phi - mu0^phi), (i) HET_loss = tau^phi - true_tau. The RICB plug-in regresses the
# deterministic mu_a(X) arm-wise on Z when the oracle exposes mu0(X), else the noisy Y.
# Cross-trial pooling uses RICB_true_mean_norm (signed ATE-level bias / SD(tau^x)) and
# HET_loss_norm (RMS / SD(tau^x); its mean is ~0 by the tower rule). PCR_oracle =
# Var(e(Z))/Var(e_oracle) -- ~1 means RICB is non-discriminating for that Z (balancing
# score); NaN for real-assignment trials.
VALIDITY_COLS = ["RICB_true_rmse", "RICB_true_mean", "RICB_true_mean_norm", "PCR_oracle",
                 "HET_loss_rmse", "HET_loss_norm"]


# ------------------------------------------------------------------------------
# Task 1: Outcome prediction
# ------------------------------------------------------------------------------
def evaluate_prediction(E_X: np.ndarray, Y: np.ndarray, *, outcome_type: str = "binary", cv: int = 5) -> dict:
    """Cross-validated prediction of Y from Z. Linear probe (primary -- the field's
    linear-evaluation convention, "how linearly decodable is Y") plus a REF_BASE
    probe (_flex suffix, "total decodable signal"): a linear probe under-counts
    nonlinear signal. Binary: AUROC/AUPRC. Continuous: R^2/RMSE."""
    is_binary = outcome_type == "binary"
    # too few minority events for the fold count -> single-class train folds
    if is_binary and np.bincount(Y.astype(int), minlength=2).min() < cv:
        return {f"{m}{s}": np.nan for m in ("AUROC", "AUPRC") for s in ("", "_flex")}

    kf = (StratifiedKFold if is_binary else KFold)(n_splits=cv, shuffle=True, random_state=42)
    y  = Y.astype(int) if is_binary else Y
    kw = {"method": "predict_proba"} if is_binary else {}

    def _probe(model):
        p = cross_val_predict(model, E_X, y, cv=kf, n_jobs=1, **kw)
        if is_binary:
            return {"AUROC": float(roc_auc_score(Y, p[:, 1])), "AUPRC": float(average_precision_score(Y, p[:, 1]))}
        return {"R2": float(r2_score(Y, p)), "RMSE": float(np.sqrt(mean_squared_error(Y, p)))}

    lin  = _probe(_lr_clf() if is_binary else _lr_reg())
    flex = _probe(_clf(REF_BASE) if is_binary else _reg(REF_BASE))
    return {**lin, **{f"{k}_flex": v for k, v in flex.items()}}


# ------------------------------------------------------------------------------
# Propensity / overlap geometry -- a deliberate LR probe on e(Z)
# ------------------------------------------------------------------------------
def evaluate_treatment_tasks(
    E_X: np.ndarray, T: np.ndarray, Y: np.ndarray, lr_folds: list, ref: tuple, *, cv: int = 5,
) -> dict:
    """Propensity/overlap-geometry diagnostics on the LR e(Z) -- a deliberate linear
    probe (ps_AUROC is 'how linearly separable is T in Z'). ATE/ATT and every
    CATE-dependent metric live in evaluate_cate on the REF_BASE nuisances.

    `ref` = fit_reference_nuisance(...) output (ps_ref, m0_ref, m1_ref) -- raw-X e(X)/m_a(X)
    for the ps-vs-raw comparison and decon_bias.
    """
    ps_ref, m0_ref, m1_ref = ref
    n = len(Y)
    ps_hat_raw = np.empty(n, dtype=np.float64)   # unclipped LR e(Z)
    for _, test_idx, *_, mps in lr_folds:
        ps_hat_raw[test_idx] = mps.predict_proba(E_X[test_idx])[:, 1]
    ps_lr_dinf = np.clip(ps_hat_raw, 0.01, 0.99)

    trt_mask, ctrl_mask = T == 1, T == 0
    n_trt  = trt_mask.sum()
    n_ctrl = ctrl_mask.sum()

    def _ess(weights):
        s = weights.sum()
        return float(s ** 2 / (weights ** 2).sum()) if s > 0 else 0.0

    w_trt  = 1.0 / ps_hat_raw[trt_mask]
    w_ctrl = 1.0 / (1.0 - ps_hat_raw[ctrl_mask])
    ess_trt  = _ess(w_trt)  / n_trt   # as fraction of arm size
    ess_ctrl = _ess(w_ctrl) / n_ctrl

    frac_extreme = float(np.mean((ps_hat_raw < 0.05) | (ps_hat_raw > 0.95)))

    # Residual T-information in Z (also a coarse leakage smell-test for
    # supervised embeddings: ps_AUROC(Z) >> ps_AUROC(raw) hints at overfitting to T).
    ps_auroc = float(roc_auc_score(T, ps_hat_raw))

    # e(Z) vs the raw-X reference propensity: delta sign separates overlap collapse
    # (e(Z) toward 0.5) from T-leakage (e(Z) sharper than e(X_raw)).
    ps_auroc_raw = float(roc_auc_score(T, ps_ref))
    ps_corr = float(np.corrcoef(ps_hat_raw, ps_ref)[0, 1])
    ps_spearman = float(spearmanr(ps_hat_raw, ps_ref).correlation)

    # Deconfounding score (D'Amour & Franks 2021, "Deconfounding Scores", Prop. 2):
    # identified |ATE bias| from reducing X -> Z, = |E[ Cov(m_t(X), e(X) | Z) / e_t(Z) ]|
    # over t, each conditional covariance being the mean product of the Z-residuals of
    # m_t(X_raw) and e(X_raw).
    # REF_BASE projection (a linear one under-residualises a curved Z -> nuisance map).
    kf_d = KFold(cv, shuffle=True, random_state=42)
    _rr = _reg(REF_BASE)
    r_e, r_m0, r_m1 = (v - cross_val_predict(_rr, E_X, v, cv=kf_d) for v in (ps_ref, m0_ref, m1_ref))
    decon_bias = float(abs(np.mean(r_m1 * r_e / ps_lr_dinf + r_m0 * r_e / (1 - ps_lr_dinf))))

    # Overlap divergence (Zhang, Bellot & van der Schaar 2020): cf_variance = cross-arm
    # outcome-model predictive variance in Z-space (their L_var / counterfactual-variance
    # regulariser); overlap_dinf = 99th pct of the arm density ratio p(z|T=1)/p(z|T=0)
    # from e(Z) (their D_inf).
    # ponytail: BayesianRidge std is a linear-probability proxy for binary Y.
    def _cf_std(src, dst):
        br = make_pipeline(StandardScaler(), BayesianRidge()).fit(E_X[src], Y[src])
        return (br.predict(E_X[dst], return_std=True)[1] ** 2).mean()
    cf_variance = float(np.mean([_cf_std(ctrl_mask, trt_mask), _cf_std(trt_mask, ctrl_mask)]))

    dr = (ps_lr_dinf / (1 - ps_lr_dinf)) * ((1 - T.mean()) / T.mean())   # p(z|T=1) / p(z|T=0), LR e(Z)
    overlap_dinf = float(max(np.quantile(dr[ctrl_mask], 0.99),
                             np.quantile(1.0 / dr[trt_mask], 0.99)))

    return {
        "ESS_trt_frac": ess_trt, "ESS_ctrl_frac": ess_ctrl, "frac_extreme": frac_extreme,
        "ps_AUROC": ps_auroc, "ps_AUROC_raw": ps_auroc_raw,
        "ps_AUROC_delta": ps_auroc - ps_auroc_raw,
        "ps_corr": ps_corr, "ps_spearman": ps_spearman,
        "decon_bias": decon_bias, "cf_variance": cf_variance, "overlap_dinf": overlap_dinf,
    }


# ------------------------------------------------------------------------------
# Representation-sufficiency diagnostics
# ------------------------------------------------------------------------------
ENERGY_DIST_MAX_N = 2000  # per arm; pairwise distances are O(n^2), cap for large cohorts

def _energy_distance(a: np.ndarray, b: np.ndarray, rng_seed: int = 0) -> float:
    """Two-sample energy distance; subsamples at ENERGY_DIST_MAX_N to cap pairwise O(n^2) cost."""
    rng = np.random.default_rng(rng_seed)
    if len(a) > ENERGY_DIST_MAX_N:
        a = a[rng.choice(len(a), ENERGY_DIST_MAX_N, replace=False)]
    if len(b) > ENERGY_DIST_MAX_N:
        b = b[rng.choice(len(b), ENERGY_DIST_MAX_N, replace=False)]
    cross    = cdist(a, b).mean()
    within_a = pdist(a).mean() if len(a) > 1 else 0.0
    within_b = pdist(b).mean() if len(b) > 1 else 0.0
    return float(2 * cross - within_a - within_b)


def evaluate_balance(E_X: np.ndarray, T: np.ndarray) -> dict:
    """Per-dimension SMD and Mahalanobis distance between arms. No model fit,
    works even with degenerate-Y arms."""
    trt, ctrl = T == 1, T == 0
    mean1, mean0 = E_X[trt].mean(axis=0), E_X[ctrl].mean(axis=0)
    diff = mean1 - mean0
    var1, var0 = E_X[trt].var(axis=0, ddof=1), E_X[ctrl].var(axis=0, ddof=1)
    pooled_sd = np.sqrt((var1 + var0) / 2)
    smd = np.divide(diff, pooled_sd, out=np.zeros_like(pooled_sd), where=pooled_sd > 1e-12)

    n1, n0 = trt.sum(), ctrl.sum()
    cov1 = np.atleast_2d(np.cov(E_X[trt], rowvar=False))
    cov0 = np.atleast_2d(np.cov(E_X[ctrl], rowvar=False))
    pooled_cov = ((n1 - 1) * cov1 + (n0 - 1) * cov0) / (n1 + n0 - 2)
    # pinv (not inv): pooled_cov can be singular/ill-conditioned when d approaches n
    # or Z-dimensions are collinear — pinv degrades gracefully instead of raising.
    # Its SVD can still hit LAPACK non-convergence on an ill-scaled cov; NaN the
    # (secondary) diagnostic rather than kill the whole eval.
    try:
        mahalanobis = float(np.sqrt(max(diff @ np.linalg.pinv(pooled_cov) @ diff, 0.0)))
    except np.linalg.LinAlgError:
        mahalanobis = np.nan

    energy_dist = _energy_distance(E_X[trt], E_X[ctrl])

    return {
        "SMD_mean_abs": float(np.mean(np.abs(smd))),
        "SMD_max_abs": float(np.max(np.abs(smd))),
        "Mahalanobis_D": mahalanobis,
        "Energy_dist": energy_dist,
    }


def fit_reference_nuisance(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    *, outcome_type: str = "binary", cv: int = 3, random_state: int = 42,
) -> tuple:
    """Cross-fitted raw-X nuisances, fit once per seed, reused across embeddings.
    e(X) stays LR (oracle T is logit; only the ps_AUROC_raw ref); m0/m1 use REF_BASE
    (they feed decon_bias)."""
    n = len(T)
    ps, m0, m1 = np.empty(n), np.empty(n), np.empty(n)
    reg = outcome_type != "binary"
    pred = (lambda m, Xt: m.predict(Xt)) if reg else (lambda m, Xt: m.predict_proba(Xt)[:, 1])
    for tr, te in StratifiedKFold(cv, shuffle=True, random_state=random_state).split(X, T.astype(int)):
        mps = _lr_clf(); mps.fit(X[tr], T[tr].astype(int))
        ps[te] = mps.predict_proba(X[te])[:, 1]
        for m, arm in ((m0, T[tr] == 0), (m1, T[tr] == 1)):
            c = _reg(REF_BASE) if reg else _clf(REF_BASE)
            c.fit(X[tr][arm], Y[tr][arm])
            m[te] = pred(c, X[te])
    return ps, m0, m1


def fit_lr_nuisance_folds(
    E_X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    *, outcome_type: str = "binary", cv: int = 5, random_state: int = 42
) -> list:
    """Cross-fit LR mu0/mu1/propensity models once for reuse across evaluation tasks.

    Returns a list of (train_idx, test_idx, m0, m1, mps) tuples (one per fold),
    enabling both evaluate_treatment_tasks and evaluate_cate's LR base learner to
    reuse the same fold splits and fitted models instead of refitting.
    """
    n = len(Y)
    strata = _strat_key(T, Y, outcome_type)
    if outcome_type == "binary":
        splits = _binary_splits(E_X, strata, cv)
    else:
        splits = KFold(n_splits=cv, shuffle=True, random_state=random_state).split(E_X)

    folds = []
    for train_idx, test_idx in splits:
        E_tr, E_te = E_X[train_idx], E_X[test_idx]
        T_tr, Y_tr = T[train_idx], Y[train_idx]
        ctrl = T_tr == 0
        trt  = T_tr == 1

        if outcome_type == "binary":
            m0, m1 = _lr_clf(), _lr_clf()
            m0.fit(E_tr[ctrl], Y_tr[ctrl])
            m1.fit(E_tr[trt],  Y_tr[trt])
        else:
            m0, m1 = _lr_reg(), _lr_reg()
            m0.fit(E_tr[ctrl], Y_tr[ctrl])
            m1.fit(E_tr[trt],  Y_tr[trt])

        mps = _lr_clf()
        mps.fit(E_tr, T_tr.astype(int))

        folds.append((train_idx, test_idx, m0, m1, mps))

    return folds


# ------------------------------------------------------------------------------
# Task 3: CATE / PEHE  — 10 meta-learners × active base learners (lr; raw_tuned adds gbm)
# ------------------------------------------------------------------------------
def _eval_fold(
    train_idx: np.ndarray, test_idx: np.ndarray,
    E_X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    lr_models: dict,   # {"m0", "m1", "mps"} pre-fitted LR models, reused for base="lr"
    lr_hparams: dict,  # {"mmx", "ms", "lo"} pre-tuned C, reused for base="lr"
    *, outcome_type: str = "binary", bases: list = None,
) -> tuple:
    """{(learner, base): held-out tau_hat} on one fold."""
    bases = bases or BASES
    E_tr, E_te = E_X[train_idx], E_X[test_idx]
    T_tr, Y_tr = T[train_idx], Y[train_idx]
    n_te = len(test_idx)
    ctrl = T_tr == 0
    trt  = T_tr == 1
    out: dict = {}

    for base in bases:
        is_binary = outcome_type == "binary"
        model_fn = _clf if is_binary else _reg
        pred = (lambda m, X: m.predict_proba(X)[:, 1]) if is_binary else (lambda m, X: m.predict(X))

        # ── Nuisance 2: propensity e(x) = E[T|X] (classifier; T binary). Per base
        # so a flexible outcome model isn't fed an LR propensity -- family mixing
        # breaks the pseudo-outcome / DR guarantees.
        mps = lr_models["mps"] if base == "lr" else _clf(base).fit(E_tr, T_tr.astype(int))
        ps_te = np.clip(mps.predict_proba(E_te)[:, 1], 0.01, 0.99)
        ps_tr = np.clip(mps.predict_proba(E_tr)[:, 1], 0.01, 0.99)

        # ── Nuisance 1: mu0 = E[Y|X,T=0]  mu1 = E[Y|X,T=1] ────────────────────
        if base == "lr":
            m0, m1 = lr_models["m0"], lr_models["m1"]
        else:
            m0 = model_fn(base)
            m0.fit(E_tr[ctrl], Y_tr[ctrl])
            m1 = model_fn(base)
            m1.fit(E_tr[trt], Y_tr[trt])

        mu0_te = pred(m0, E_te)
        mu0_tr = pred(m0, E_tr)
        mu1_te = pred(m1, E_te)
        mu1_tr = pred(m1, E_tr)

        # ── Nuisance 3: m(x) = E[Y|X] marginal (for R / U) ────────────────────
        mmx = _clf_or_fixed(base, "mmx", model_fn, lr_hparams)
        mmx.fit(E_tr, Y_tr.astype(int) if is_binary else Y_tr)
        mx_tr = pred(mmx, E_tr)

        # ── S-Learner: single model with T as feature ─────────────────────────
        X_aug = np.column_stack([E_tr, T_tr])
        ms = _clf_or_fixed(base, "ms", model_fn, lr_hparams)
        ms.fit(X_aug, Y_tr.astype(int) if is_binary else Y_tr)
        out[("S", base)] = (
            pred(ms, np.column_stack([E_te, np.ones(n_te)]))
            - pred(ms, np.column_stack([E_te, np.zeros(n_te)]))
        )

        # ── Lo-Learner: single model with T and X·T interaction features ──────
        XT_tr = np.column_stack([E_tr, T_tr, E_tr * T_tr[:, None]])
        lo = _clf_or_fixed(base, "lo", model_fn, lr_hparams)
        lo.fit(XT_tr, Y_tr.astype(int) if is_binary else Y_tr)
        XT_te_1 = np.column_stack([E_te, np.ones(n_te),  E_te])
        XT_te_0 = np.column_stack([E_te, np.zeros(n_te), np.zeros_like(E_te)])
        out[("Lo", base)] = pred(lo, XT_te_1) - pred(lo, XT_te_0)

        # ── T-Learner: separate models per arm ────────────────────────────────
        out[("T", base)] = mu1_te - mu0_te

        # ── X-Learner: two-stage with propensity weighting ────────────────────
        xg1, xg0 = _reg(base), _reg(base)
        xg1.fit(E_tr[trt],  Y_tr[trt]  - mu0_tr[trt])
        xg0.fit(E_tr[ctrl], mu1_tr[ctrl] - Y_tr[ctrl])
        out[("X", base)] = ps_te * xg0.predict(E_te) + (1 - ps_te) * xg1.predict(E_te)        

        # ── RA-Learner: regression-adjusted — one model on the X-Learner's
        # pseudo-outcome directly, instead of two arm models + propensity combo
        ra = np.where(T_tr == 1, Y_tr - mu0_tr, mu1_tr - Y_tr)

        # ── Z-Learner: Horvitz-Thompson transformed-outcome pseudo-outcome ────
        z = (T_tr / ps_tr - (1 - T_tr) / (1 - ps_tr)) * Y_tr

        # ── F-Learner: inverse-propensity-weighted pseudo-outcome ─────────────
        f = (T_tr - ps_tr) / (ps_tr * (1 - ps_tr)) * Y_tr

        # ── U-Learner: unweighted pseudo-outcome ──────────────────────────────
        rho   = T_tr - ps_tr
        rho_s = np.where(np.abs(rho) < 0.01, np.sign(rho + 1e-9) * 0.01, rho)
        y_R   = (Y_tr - mx_tr) / rho_s
        
        # ── R-Learner: weighted regression (weight = rho^2) ───────────────────
        rr = _reg(base)
        _fit_weighted(rr, E_tr, y_R, rho ** 2)
        out[("R", base)] = rr.predict(E_te)

        # ── DR-Learner: doubly-robust pseudo-outcome ──────────────────────────
        phi = (mu1_tr - mu0_tr
               + T_tr       * (Y_tr - mu1_tr) / ps_tr
               - (1 - T_tr) * (Y_tr - mu0_tr) / (1 - ps_tr))

        # ── Shared pseudo-outcomes ───────────────────────────────────────────
        pseudo_outcomes = {"RA": ra, "Z": z, "F": f, "U": y_R, "DR": phi}
        for name, pseudo in pseudo_outcomes.items():
            m = _reg(base)
            m.fit(E_tr, pseudo)
            out[(name, base)] = m.predict(E_te)

    return test_idx, out


def _cate_calibration(tau_hat: np.ndarray, true_tau: np.ndarray, n_bins: int = 10) -> tuple:
    """GRF-style calibration vs. the oracle. cal_alpha/cal_beta: OLS
    true_tau ~ [1, tau_hat-mean] (ideal beta~1). ECE_CATE: mean abs gap between
    E[true_tau] and E[tau_hat] over n_bins quantile bins of tau_hat. NaNs if no spread."""
    th = tau_hat - tau_hat.mean()
    if not np.any(np.abs(th) > 1e-12):
        return np.nan, np.nan, np.nan
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(th), th]), true_tau, rcond=None)
    cal_alpha, cal_beta = float(coef[0]), float(coef[1])

    q = np.quantile(tau_hat, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    bin_idx = np.digitize(tau_hat, q[1:-1])
    ece, n = 0.0, len(tau_hat)
    for k in range(n_bins):
        m = bin_idx == k
        if m.any():
            ece += (m.sum() / n) * abs(true_tau[m].mean() - tau_hat[m].mean())
    return cal_alpha, cal_beta, float(ece)


def evaluate_cate(
    E_X: np.ndarray, T: np.ndarray, Y: np.ndarray, lr_folds: list, oracle: dict,
    *, outcome_type: str = "binary", cv: int = 5, bases: list = None,
) -> dict:
    """`oracle`: ground-truth for these rows -- {"tau": tau^x (required), "Y0", "Y1",
    "ATE", "mu0" (control risk E[Y|X,A=0]), "ps" (e(X))}. Y0/Y1+ATE enable CF_RMSE /
    ATE metrics; mu0(+ps) switch RICB to the noise-free plug-in (see VALIDITY_COLS).

    10 meta-learners (S/Lo/T/X/RA/Z/F/U/R/DR), folds run serially (kept flat --
    the caller's Parallel pool over (seed x embedding) jobs is the only parallelism
    here, so a second nested pool per fold would risk oversubscribing whatever core
    count that outer pool is sized off).

    lr_folds must be fit_lr_nuisance_folds(E_X, T, Y, ..., cv=cv)'s output (same cv,
    same random_state) -- its per-fold m0/m1/mps are reused for base="lr" instead of
    refitting. Likewise, mmx/ms/lo's C is tuned once on the full data rather than
    re-searched per fold (accepted simplification, not leakage-free nested CV).

    Meta-learner definitions
    ─────────────────────────
    S   Single model: τ(x) = µ(x,1) − µ(x,0)             [Künzel et al. 2019]
    Lo  S-Learner + explicit x·t interaction terms       [Lo 2002]
    T   Two models:   τ(x) = µ_1(x) − µ_0(x)             [Künzel et al. 2019]
    X   Two-stage with propensity combination            [Künzel et al. 2019]
    RA  Regression-adjusted: one model on X-Learner's
        pseudo-outcome, no propensity combination        [Curth & van der Schaar 2021]
    Z   Horvitz-Thompson transformed-outcome pseudo-out  [Curth & van der Schaar 2021]
    F   Inverse-propensity-weighted pseudo-outcome       [Künzel et al. 2019]
    U   Unweighted pseudo-outcome (R without weights)    [Künzel et al. 2019]
    R   Robinson decomposition, weighted pseudo-outcome  [Nie & Wager 2021]
    DR  Doubly-robust pseudo-outcome regression          [Kennedy 2023]

    Returns PEHE_{learner}_{base} per pair, plus PEHE, tau_hat_mean, two-level
    PEHE_{r,dr}selected + lr-reference signals, and the REF_BASE metrics RICB_true_* / HET_loss_* /
    tau_recovery_R2 / cal_* / ECE_CATE / true_ATE+ATE_hat/error/SE / ATT_* /
    policy_value / regret* / policy_accuracy / CF_RMSE (needs Y0/Y1 + true_ATE).
    """
    true_tau = oracle["tau"]
    Y0, Y1, true_ATE = oracle.get("Y0"), oracle.get("Y1"), oracle.get("ATE")
    mu0_true, ps_true = oracle.get("mu0"), oracle.get("ps")

    bases = bases or BASES
    n  = len(Y)
    if outcome_type == "binary":
        splits = list(_binary_splits(E_X, _strat_key(T, Y, outcome_type), cv))
    else:
        splits = list(KFold(n_splits=cv, shuffle=True, random_state=42).split(E_X))

    is_binary = outcome_type == "binary"
    Y_int = Y.astype(int) if is_binary else Y
    lr_hparams = {
        "mmx": _tune_C(E_X, Y_int),
        "ms":  _tune_C(np.column_stack([E_X, T]), Y_int),
        "lo":  _tune_C(np.column_stack([E_X, T, E_X * T[:, None]]), Y_int),
    } if "lr" in bases and is_binary else None

    # REF_BASE (a stated constant, not a --bases value) OOF nuisances on Z for the
    # representation metrics -- tau_phi = REF_BASE regression of the known true_tau on Z,
    # mu*_ref = REF_BASE per-arm models, ps_z = REF_BASE propensity (e(Z) need not be
    # logistic even when e(X) is).
    mk = _clf if is_binary else _reg
    predp = (lambda m, X: m.predict_proba(X)[:, 1]) if is_binary else (lambda m, X: m.predict(X))
    tau_phi  = np.empty(n, dtype=np.float64)
    mu0_ref  = np.empty(n, dtype=np.float64)
    mu1_ref  = np.empty(n, dtype=np.float64)
    m_ref    = np.empty(n, dtype=np.float64)   # E[Y|Z], for R-/DR-loss selection
    ps_z     = np.empty(n, dtype=np.float64)

    # RICB plug-in: when the oracle exposes the deterministic control risk mu0(X) (mu1 =
    # mu0 + true_tau), regress those noise-free targets arm-wise on Z instead of the
    # Bernoulli Y -- the gap then isolates within-Z confounding, not plug-in variance
    # (deep-research verdict; replaces the two-fold noise-floor heuristic).
    have_det = mu0_true is not None
    if have_det:
        mu_det = {0: np.asarray(mu0_true, dtype=np.float64)}
        mu_det[1] = mu_det[0] + true_tau
        m0_det, m1_det = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)

    fold_results = []
    for (tr, te), (_, _, m0, m1, mps) in zip(splits, lr_folds):
        lr_models = {"m0": m0, "m1": m1, "mps": mps}
        fold_results.append(_eval_fold(
            tr, te, E_X, T, Y, outcome_type=outcome_type, lr_models=lr_models,
            lr_hparams=lr_hparams, bases=bases,
        ))
        ps_z[te]    = _clf(REF_BASE).fit(E_X[tr], T[tr].astype(int)).predict_proba(E_X[te])[:, 1]
        tau_phi[te] = _reg(REF_BASE).fit(E_X[tr], true_tau[tr]).predict(E_X[te])
        for arr, idx in ((m_ref, tr), (mu0_ref, tr[T[tr] == 0]), (mu1_ref, tr[T[tr] == 1])):
            arr[te] = predp(mk(REF_BASE).fit(E_X[idx], Y[idx]), E_X[te])
        if have_det:
            for a, arr in ((0, m0_det), (1, m1_det)):
                idx = tr[T[tr] == a]
                arr[te] = _reg(REF_BASE).fit(E_X[idx], mu_det[a][idx]).predict(E_X[te])

    tau_hats = {l: {b: np.empty(n, dtype=np.float64) for b in bases} for l in LEARNERS}
    for test_idx, fold_out in fold_results:
        for (l, b), vals in fold_out.items():
            tau_hats[l][b][test_idx] = vals

    result: dict = {}
    for l in LEARNERS:
        for b in bases:
            result[f"PEHE_{l}_{b}"] = _rmse(tau_hats[l][b] - true_tau)

    result["PEHE"]         = result["PEHE_T_lr"]
    result["tau_hat_mean"] = tau_hats["T"]["lr"].mean()

    # Melnychuk et al. 2024 representation-validity gaps (tau_phi = E[tau^x | Z]):
    #   RICB     = tau_phi - (mu1^phi - mu0^phi)  -- confounding left after X -> Z
    #   HET_loss = tau_phi - tau^x                -- heterogeneity Z cannot resolve
    plugin = mu1_ref - mu0_ref                    # arm-wise E[Y|Z,A=a]: Task 2 / policy / calibration
    result["HET_loss_rmse"] = _rmse(tau_phi - true_tau)  # E[HET_loss] ~ 0 by the tower rule

    ricb = tau_phi - ((m1_det - m0_det) if have_det else plugin)
    result["RICB_true_rmse"], result["RICB_true_mean"] = _rmse(ricb), np.mean(ricb)
    # Oracle-referenced propensity concentration Var(e(Z))/Var(e_oracle); ~1 => Z keeps the
    # true e, so RICB ~ 0 by construction (balancing score) and is non-discriminating for
    # that Z. NaN for real-assignment trials.
    result["PCR_oracle"] = (float(np.var(ps_z) / np.var(ps_true))
                            if ps_true is not None and np.var(ps_true) > 0 else np.nan)

    # Task 2 -- ATE/ATT on the REF_BASE nuisances (mu*_ref, ps_z), so the errors
    # reflect Z dropping confounders, not estimator misspecification. Both AIPW /
    # doubly-robust (consistent if either mu_a or e is right; dominates Hájek IPW);
    # ATE_SE = SE of the influence function, no bootstrap.
    ps  = np.clip(ps_z, 0.01, 0.99)
    trt = T == 1
    aipw   = plugin + T*(Y - mu1_ref)/ps - (1-T)*(Y - mu0_ref)/(1-ps)
    resid0 = Y - mu0_ref
    ate_hat  = aipw.mean()
    att_hat  = (T*resid0 - (1-T)*(ps/(1-ps))*resid0).sum() / T.sum()
    true_att = true_tau[trt].mean()
    result["true_ATE"], result["ATE_hat"] = true_ATE, ate_hat
    result["ATE_error"] = abs(ate_hat - true_ATE)
    result["ATE_SE"]    = aipw.std(ddof=1) / np.sqrt(n)
    result["true_ATT"], result["ATT_hat"] = true_att, att_hat
    result["ATT_error"] = abs(att_hat - true_att)

    # Policy value / regret: REF_BASE T-learner sign, doubly-robust value estimator
    # (Dudík et al. 2011) on the REF_BASE mu_a(Z) / e(Z).
    pi_hat, pi_ora = (plugin < 0).astype(float), (true_tau < 0).astype(float)
    mu_obs = np.where(trt, mu1_ref, mu0_ref)

    def _dr_value(pi):
        return np.mean(np.where(pi > 0.5, mu1_ref, mu0_ref)
                       + (T == pi) * (Y - mu_obs) / np.where(pi > 0.5, ps, 1 - ps))

    pv, pvo = _dr_value(pi_hat), _dr_value(pi_ora)
    result["policy_value"], result["oracle_policy_value"] = pv, pvo
    result["regret"] = pv - pvo
    result["regret_weighted"] = np.mean(np.abs(true_tau) * (pi_hat != pi_ora))
    result["policy_accuracy"] = np.mean(pi_hat == pi_ora)

    # Counterfactual RMSE: predict the unobserved arm with the ref outcome model.
    if Y0 is not None:
        result["CF_RMSE"] = _rmse(np.where(trt, mu0_ref, mu1_ref) - np.where(trt, Y0, Y1))

    # ── Oracle-free selection over the full (learner, base) pool, two-level: best
    #    learner per base, then best base (Mahajan et al. 2024). REF_BASE nuisances.
    #    R-loss mean(((Y-m) - tau*(T-e))^2)  /  DR-loss mean((tau - phi_AIPW)^2)
    if "lr" in bases:
        resid_y, resid_t = Y.astype(float) - m_ref, T.astype(float) - ps
        _rl  = lambda th: np.mean((resid_y - th * resid_t) ** 2)
        _drl = lambda th: np.mean((th - aipw) ** 2)

        def _two_level(loss, tag):                        # tag in {"r", "dr"}
            win = {}                                      # base -> (learner, its loss)
            for b in bases:
                d = {l: loss(tau_hats[l][b]) for l in LEARNERS}
                lw = min(d, key=d.get)
                win[b] = (lw, d[lw])
                result[f"{tag}selected_{b}"] = lw         # winning meta-learner in tier b
            bb = min(bases, key=lambda b: win[b][1])
            return f"{win[bb][0]}_{bb}"                   # "{learner}_{base}"

        result["rselected"]  = _two_level(_rl, "r")
        result["drselected"] = _two_level(_drl, "dr")
        result["PEHE_rselected"]  = result[f"PEHE_{result['rselected']}"]
        result["PEHE_drselected"] = result[f"PEHE_{result['drselected']}"]
        # Held-out R-/DR-loss(Z) are NOT exported: their magnitude carries Var(Y|Z), so
        # they gauge Z's prognostic quality not its causal sufficiency and have no
        # cross-representation validity (Melnychuk et al. 2024). They feed the two-level
        # selection above and nothing else.

    # CATE calibration (ref T-learner) + heterogeneity recovery (complement of
    # HET_loss: 1 - HET_loss_rmse^2 / Var(tau^x) -- no extra fit).
    result["cal_alpha"], result["cal_beta"], result["ECE_CATE"] = _cate_calibration(plugin, true_tau)
    var_tau = np.var(true_tau)
    tau_sd = np.sqrt(var_tau)
    result["tau_recovery_R2"] = 1.0 - result["HET_loss_rmse"] ** 2 / var_tau if var_tau > 0 else np.nan
    # Scale-free RICB / HET: metric / SD(tau^x) so pooled cross-trial curves track
    # compression not effect-size scale. tau_sd also normalises E_leaderboard's PEHE families.
    _sf = lambda x: x / tau_sd if tau_sd else np.nan
    result["tau_sd"] = tau_sd
    result["HET_loss_norm"] = _sf(result["HET_loss_rmse"])
    result["RICB_true_mean_norm"] = _sf(result["RICB_true_mean"])
    return result
