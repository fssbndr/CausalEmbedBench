"""Oracle parameterization (knobs) for semi-synthetic data generation, plus the generic
sampling helpers ("FittedOracle -> synthetic T/Y0/Y1") shared across all trials.

Five knobs sweep benchmark difficulty: overlap (kappa_overlap), heterogeneity
(gamma_hetero), effect scale (xi_effect), noise (eta_noise), alignment (alpha_alignment).
kappa_overlap/gamma_hetero/xi_effect/eta_noise are post-fit transforms of an already-fitted
FittedOracle and default to identity (1.0).
alpha_alignment is pre-fit: it partitions covariates between the propensity and outcome
models before fitting (see apply_alignment_split), so it has no post-fit "apply" function
here.
"""

import argparse
import os
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import yaml
from scipy.special import expit, logit

_GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle_knob_grid.yaml")
FITTED_KNOBS = ["kappa_overlap", "gamma_hetero", "xi_effect", "eta_noise", "alpha_alignment"]
PUBLISHED_KNOBS = ["gamma_hetero", "xi_effect", "eta_noise"]


@dataclass
class FittedOracle:
    """A fitted semi-synthetic DGP, pre-knob-application. Knob functions transform this
    to produce difficulty variants; sample_* functions draw synthetic data from it."""
    outcome_type: str  # "binary" | "continuous"
    mu0: np.ndarray  # E[Y|W=0,X], shape (n,)
    mu1: np.ndarray  # E[Y|W=1,X], shape (n,)
    propensity: np.ndarray  # e(X) = P(W=1|X), shape (n,)
    tau_fitted: np.ndarray  # tau(X) = mu1 - mu0, shape (n,)
    ate_fitted: float
    tau_0: float  # trial-level covariate-adjusted ATE
    pi_T: float  # constant propensity, retained as a diagnostic/sanity-check value
    sigma0: float | None = None  # residual SD, continuous outcomes only
    sigma1: float | None = None
    covariate_role_map: dict[int, str] = field(default_factory=dict)  # {feature_idx: "propensity"|"outcome"|"both"}


@dataclass
class OracleKnobConfig:
    """Oracle parameterization knob values. All default to identity (1.0)."""
    kappa_overlap:   float = 1.0  # overlap: [0,inf), 1=fitted data, <1=better overlap (toward random), >1=worse overlap (positivity pushed further than the fitted data actually has)
    gamma_hetero:    float = 1.0  # heterogeneity: [0,inf), 0=homogeneous, 1=fitted CATE spread, >1=exaggerated spread
    xi_effect:       float = 1.0  # effect scale: [0,inf), scales magnitude of tau(X)
    eta_noise:       float = 1.0  # noise: [0,inf), scales outcome residual SD (sqrt of variance scale factor)
    alpha_alignment: float = 1.0  # alignment: [0,1], 0=disjoint propensity/outcome covariate sets (no true confounders), 1=fully shared (structural ceiling, no >1 direction exists)
    # NOT a difficulty knob -- a DGP-variant axis for the congeniality-triangulation study
    # (methodological hardening round 2). It rides the preset mechanism purely for plumbing
    # reuse (resolve_out_dir namespacing, path parsing, BLOCK_KEY). Identity = today's oracle.
    assignment_family: str = "logit"  # "logit" (fitted LogisticRegression e(X)) | "tree" (RandomForest e(X), non-smooth boundary) | "published" (recorded for IHDP/NEWS, no model)


def _load_grid(grid_path: str | None = None) -> dict:
    with open(grid_path or _GRID_PATH) as f:
        return yaml.safe_load(f)


def load_knob_config(preset: str = "baseline", grid_path: str | None = None) -> OracleKnobConfig:
    """Load a named preset from oracle_knob_grid.yaml. Raises KeyError if unknown."""
    grid = _load_grid(grid_path)
    if preset not in grid:
        raise KeyError(f"Unknown oracle knob preset '{preset}'. Available: {sorted(grid)}")
    return OracleKnobConfig(**grid[preset])


def known_presets(grid_path: str | None = None) -> list[str]:
    """Every preset name in oracle_knob_grid.yaml, 'baseline' first."""
    grid = _load_grid(grid_path)
    return ["baseline"] + sorted(p for p in grid if p != "baseline")


def resolve_out_dir(trial: str, preset: str, base: str = "data/oracles") -> str:
    """baseline -> flat {base}/{trial} (backward-compatible with C_embeddings.py/
    D_evaluation.py/E_leaderboard.py, which read that path directly); any other preset ->
    namespaced {base}/{trial}/knob_variants/{preset} so it can't clobber or be mistaken
    for the default."""
    return f"{base}/{trial}" if preset == "baseline" else f"{base}/{trial}/knob_variants/{preset}"


def output_exists(out_dir: str) -> bool:
    return os.path.exists(f"{out_dir}/semi_synthetic.parquet")


def setup_oracle_dir(trial: str, preset: str) -> str:
    """resolve_out_dir + ensure it exists. RCTBench resolves its own per-trial dir instead."""
    out_dir = resolve_out_dir(trial, preset)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def parse_oracle_args(skip_help: str, knob_note: str = "") -> tuple[argparse.Namespace, OracleKnobConfig]:
    """Standard --skip-existing/--knobs-preset CLI shared by every oracle.py. skip_help
    is trial-specific ("regeneration" vs. RCTBench's per-trial "trials"); knob_note is an
    optional trailing sentence documenting which knobs actually apply for that trial."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true", help=skip_help)
    parser.add_argument(
        "--knobs-preset", default="baseline",
        help="Named oracle-knob preset from B_oracles/shared/oracle_knob_grid.yaml "
             "(default: baseline = identity)." + (f" {knob_note}" if knob_note else ""),
    )
    args = parser.parse_args()
    return args, load_knob_config(args.knobs_preset)


def overlap_transform(propensity: np.ndarray, kappa_overlap: float) -> np.ndarray:
    """e_kappa(X) = sigmoid(kappa_overlap*logit(e(X))). 1=unchanged, 0=0.5 everywhere
    (full randomization), >1=pushed further from 0.5 than the fitted data (worse overlap)."""
    if kappa_overlap == 1.0:
        return propensity
    e = np.clip(propensity, 1e-8, 1 - 1e-8)
    e_kappa = expit(kappa_overlap * logit(e))
    return e_kappa.astype(propensity.dtype)


def apply_overlap(oracle: FittedOracle, kappa_overlap: float) -> FittedOracle:
    """See overlap_transform."""
    if kappa_overlap == 1.0:
        return oracle
    return replace(oracle, propensity=overlap_transform(oracle.propensity, kappa_overlap))


def _replace_tau(oracle: FittedOracle, new_tau: np.ndarray) -> FittedOracle:
    """Recompute mu1/tau_fitted/ate_fitted from a new per-unit tau(X), holding mu0 fixed.
    Shared tail of apply_heterogeneity/apply_scale, which differ only in how new_tau is
    derived from the oracle's existing tau_fitted."""
    return replace(
        oracle,
        mu1=(oracle.mu0 + new_tau).astype(oracle.mu1.dtype),
        tau_fitted=new_tau.astype(oracle.tau_fitted.dtype),
        ate_fitted=float(new_tau.mean()),
    )


def apply_heterogeneity(oracle: FittedOracle, gamma_hetero: float) -> FittedOracle:
    """tau_gamma(X) = gamma_hetero*tau(X) + (1-gamma_hetero)*ATE. gamma_hetero=1 -> unchanged,
    gamma_hetero=0 -> constant ATE everywhere (fully homogeneous). Adjusts mu1 to match."""
    if gamma_hetero == 1.0:
        return oracle
    tau_gamma = gamma_hetero * oracle.tau_fitted + (1 - gamma_hetero) * oracle.ate_fitted
    return _replace_tau(oracle, tau_gamma)


def apply_scale(oracle: FittedOracle, xi_effect: float) -> FittedOracle:
    """tau_xi(X) = xi_effect*tau(X). xi_effect=1 -> unchanged, xi_effect=0 -> no treatment
    effect. Adjusts mu1."""
    if xi_effect == 1.0:
        return oracle
    return _replace_tau(oracle, xi_effect * oracle.tau_fitted)


def apply_noise(oracle: FittedOracle, eta_noise: float) -> FittedOracle:
    """sigma'^2 = eta_noise*sigma^2, i.e. SD scales by sqrt(eta_noise). eta_noise=1 ->
    unchanged, eta_noise=0 -> noiseless. No-op for binary outcomes (sigma0/sigma1 are None)."""
    if eta_noise == 1.0:
        return oracle
    scale = np.sqrt(eta_noise)
    return replace(
        oracle,
        sigma0=oracle.sigma0 * scale if oracle.sigma0 is not None else None,
        sigma1=oracle.sigma1 * scale if oracle.sigma1 is not None else None,
    )


def apply_knobs(oracle: FittedOracle, knob_config: "OracleKnobConfig") -> FittedOracle:
    """Overlap -> heterogeneity -> scale -> noise. alpha_alignment is pre-fit, not part of
    this chain. For IHDP/NEWS, overlap acts on a propensity field that's discarded anyway."""
    oracle = apply_overlap(oracle, knob_config.kappa_overlap)
    oracle = apply_heterogeneity(oracle, knob_config.gamma_hetero)
    oracle = apply_scale(oracle, knob_config.xi_effect)
    oracle = apply_noise(oracle, knob_config.eta_noise)
    return oracle


def resample_published_outcome(
    seed: int, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, yf: np.ndarray, ycf: np.ndarray,
    knob_config: "OracleKnobConfig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared IHDP/NEWS per-seed logic. At identity, returns the published (mu0, mu1, yf)
    unchanged; otherwise applies the knob chain and resamples Y_obs_synthetic through the
    reconstructed noise model, keeping Y0/Y1 as the noise-free post-knob mu0/mu1."""
    if knob_config == OracleKnobConfig():
        return mu0, mu1, yf

    sigma0, sigma1 = empirical_noise_sigma(t, yf, ycf, mu0, mu1)
    fitted = FittedOracle(
        outcome_type="continuous", mu0=mu0, mu1=mu1,
        propensity=np.full(len(t), t.mean()),  # diagnostic only; T is never resampled
        sigma0=sigma0, sigma1=sigma1,
        tau_fitted=(mu1 - mu0).astype(np.float32), ate_fitted=float((mu1 - mu0).mean()),
        tau_0=float((mu1 - mu0).mean()), pi_T=float(t.mean()),
    )
    fitted = apply_knobs(fitted, knob_config)
    y0, y1 = fitted.mu0, fitted.mu1
    rng = np.random.default_rng(seed)
    y0_noisy, y1_noisy = sample_continuous_outcomes(fitted.mu0, fitted.mu1, fitted.sigma0, fitted.sigma1, rng)
    y_obs = np.where(t == 1, y1_noisy, y0_noisy).astype(np.float32)
    return y0, y1, y_obs


def knob_params(knob_config: "OracleKnobConfig", active_knobs: list[str], assignment_family: str | None = None) -> dict:
    """The knobs/active_knobs/active_assignment_family causal_params.json keys. Pass
    "published" for IHDP/NEWS; otherwise defaults to knob_config's own assignment_family."""
    return {
        "knobs": asdict(knob_config),
        "active_knobs": active_knobs,
        "active_assignment_family": assignment_family or knob_config.assignment_family,
    }


def apply_alignment_split(
    n_features: int, alpha_alignment: float
) -> tuple[np.ndarray, np.ndarray]:
    """Partition covariate indices between the propensity and outcome models (pre-fit,
    unlike the four knobs above). alpha_alignment=1 -> both models see every covariate
    (identity; structural ceiling, no >1 direction exists). alpha_alignment=0 -> disjoint
    sets (no true confounders). 0<alpha_alignment<1 interpolates: a fraction
    alpha_alignment of covariates is shared (true confounders), the rest split exclusively.

    Returns (propensity_indices, outcome_indices).
    """
    all_indices = np.arange(n_features)

    if alpha_alignment == 1.0:
        return all_indices, all_indices

    if alpha_alignment == 0.0:
        # Disjoint *feeder* sets only. Real covariates in the two halves stay
        # correlated, so residual confounding still leaks through that dependence --
        # no_alignment is a confounding floor, not a true zero-confounder DGP.
        split_point = n_features // 2
        return all_indices[:split_point], all_indices[split_point:]

    rng = np.random.default_rng(seed=42)  # deterministic split for a given n_features
    n_shared = int(alpha_alignment * n_features)
    n_exclusive = n_features - n_shared
    n_prop_exclusive = n_exclusive // 2

    shuffled = rng.permutation(all_indices)
    propensity_exclusive = shuffled[:n_prop_exclusive]
    outcome_exclusive = shuffled[n_prop_exclusive:n_exclusive]
    shared = shuffled[n_exclusive:]

    return (
        np.sort(np.concatenate([propensity_exclusive, shared])),
        np.sort(np.concatenate([outcome_exclusive, shared])),
    )


def build_covariate_role_map(n_features: int, propensity_idx: np.ndarray, outcome_idx: np.ndarray) -> dict[int, str]:
    """Build a FittedOracle.covariate_role_map from apply_alignment_split's output."""
    prop_set, out_set = set(propensity_idx.tolist()), set(outcome_idx.tolist())
    return {
        i: ("both" if i in prop_set and i in out_set else "propensity" if i in prop_set else "outcome")
        for i in range(n_features)
    }


def sample_treatment(propensity: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """T ~ Bernoulli(e(X))."""
    return rng.binomial(1, np.clip(propensity, 1e-8, 1 - 1e-8)).astype(np.float32)


def sample_binary_outcomes(risk0: np.ndarray, risk1: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Y0 ~ Bernoulli(risk0), Y1 ~ Bernoulli(risk1)."""
    r0, r1 = np.clip(risk0, 1e-8, 1 - 1e-8), np.clip(risk1, 1e-8, 1 - 1e-8)
    return rng.binomial(1, r0).astype(np.float32), rng.binomial(1, r1).astype(np.float32)


def sample_continuous_outcomes(mu0: np.ndarray, mu1: np.ndarray, sigma0: float, sigma1: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Y0 ~ N(mu0, sigma0^2), Y1 ~ N(mu1, sigma1^2)."""
    n = len(mu0)
    y0 = (mu0 + rng.normal(0, sigma0, size=n)).astype(np.float32)
    y1 = (mu1 + rng.normal(0, sigma1, size=n)).astype(np.float32)
    return y0, y1


def draw_binary(fitted_oracle: FittedOracle, rng: np.random.Generator) -> dict:
    """Sample T/Y0/Y1/Y_obs from a (possibly knob-transformed) binary-outcome FittedOracle."""
    T = sample_treatment(fitted_oracle.propensity, rng)
    Y0, Y1 = sample_binary_outcomes(fitted_oracle.mu0, fitted_oracle.mu1, rng)
    Y_obs = np.where(T == 1, Y1, Y0).astype(np.float32)
    return {"T_synthetic": T, "Y0": Y0, "Y1": Y1, "Y_obs_synthetic": Y_obs}


def draw_continuous(fitted_oracle: FittedOracle, rng: np.random.Generator, resample_treatment: bool = True) -> dict:
    """Sample Y0/Y1/Y_obs from a (possibly knob-transformed) continuous-outcome FittedOracle.
    If resample_treatment is False, fitted_oracle.propensity is used as the realized T
    directly (for trials where T must stay exactly the published/observed assignment)."""
    T = sample_treatment(fitted_oracle.propensity, rng) if resample_treatment else fitted_oracle.propensity
    Y0, Y1 = sample_continuous_outcomes(fitted_oracle.mu0, fitted_oracle.mu1, fitted_oracle.sigma0, fitted_oracle.sigma1, rng)
    Y_obs = np.where(T == 1, Y1, Y0).astype(np.float32)
    return {"T_synthetic": T, "Y0": Y0, "Y1": Y1, "Y_obs_synthetic": Y_obs}


def empirical_noise_sigma(t: np.ndarray, yf: np.ndarray, ycf: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> tuple[float, float]:
    """Recover residual noise SD from a published factual/counterfactual pair (e.g. IHDP,
    NEWS): reconstruct each unit's realized Y0/Y1 from (t, yf, ycf), then measure their
    spread around the noise-free mu0/mu1."""
    y0 = np.where(t == 0, yf, ycf)
    y1 = np.where(t == 1, yf, ycf)
    return float((y0 - mu0).std()), float((y1 - mu1).std())
