# Embedding helpers for the causal embedding benchmark.
# Each function accepts a 2-D float32 numpy array X_std (n_patients × n_features)
# and returns (embedding: np.ndarray shape (n, d), fitted_model).

import re

import numpy as np
import torch
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.random_projection import GaussianRandomProjection

from utils.methods.autoencoder import AE, VAE
from utils.methods.BNN import BNN
from utils.methods.CEVAE import CEVAE
from utils.methods.CFRISW import CFRISW
from utils.methods.CFRNet import CFRNet
from utils.methods.DCN import DCN
from utils.methods.DeepTreat import DeepTreat
from utils.methods.DragonNet import DragonNet
from utils.methods.DRCFR import DRCFR
from utils.methods.NICE import NICE
from utils.methods.SITE import SITE
from utils.methods.TEDVAE import TEDVAE

# Embedding dimension configuration: standardized relative-capacity grid
RELATIVE_RATIOS = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]

# Single source of truth for which methods C_embeddings.py fits -- paired with
# embedding_filename()/parse_embedding_name() below so producers (C_embeddings.py)
# and consumers (cleanup_embeddings.py) can't drift out of sync on what "complete"
# means for a (trial, preset) pair.
UNSUPERVISED_METHODS = ["pca", "fa", "fastica", "ae", "vae", "randomproj"]
SUPERVISED_METHODS = [
    "bnn",
    "cevae",
    "cfrisw",
    "cfrnet",
    "dcn",
    "deeptreat",
    "dragonnet",
    "drcfr",
    "nice",
    "site",
    "tarnet",
    "tedvae",
]


def k_grid_for_trial(d: int, baseline: int | None = None) -> list[tuple[int, float]]:
	"""
	Relative-capacity-ratio grid: rho = k/d in {1/16, 1/8, 1/4, 1/2, 1, 2, 4}.

	For each rho, k = round(baseline * rho). Processed coarsest-first so a fine
	rho that rounds down to an already-claimed k (baseline too small to tell them
	apart) is the one dropped, not the coarser rho it collides with.
	"""
	if baseline is None:
		baseline = d
	seen_k = set()
	grid = []
	for rho in sorted(RELATIVE_RATIOS, reverse=True):
		k = round(baseline * rho)
		if k < 1 or k in seen_k:
			continue
		seen_k.add(k)
		grid.append((k, rho))
	return sorted(grid)


_EMBEDDING_NAME_RE = re.compile(r"^(?P<method>.+)_d(?P<k>\d+)_rho(?P<rho>[\d.]+)$")


def embedding_filename(method: str, k: int, rho: float) -> str:
	"""Canonical embedding-parquet filename for a (method, k, rho) grid point. The single
	writer paired with parse_embedding_name() below, so producers (C_embeddings.py) and
	consumers (cleanup_embeddings.py, E_leaderboard.py) can't drift out of sync on the
	naming convention."""
	return f"{method}_d{k}_rho{rho:g}.parquet"


def parse_embedding_name(name: str) -> tuple[str, int, float] | None:
	"""Inverse of embedding_filename(). `name` is the parquet stem (no directory, no
	.parquet). Returns None for "raw" and anything else that isn't in the canonical
	{method}_d{k}_rho{rho} form -- callers special-case "raw" themselves since it has no
	(k, rho)."""
	match = _EMBEDDING_NAME_RE.match(name)
	if not match:
		return None
	return match.group("method"), int(match.group("k")), float(match.group("rho"))


# ------------------------------------------------------------------------------
# Linear manifolds
# ------------------------------------------------------------------------------

def fit_pca(X: np.ndarray, d: int):
    model = PCA(n_components=d, random_state=42)
    E = model.fit_transform(X).astype(np.float32)
    explained = model.explained_variance_ratio_.sum()
    print(f"   PCA d={d}: explained variance = {explained:.3f}")
    return E, model


def fit_fa(X: np.ndarray, d: int):
    model = FactorAnalysis(n_components=d, random_state=42, max_iter=1000)
    E = model.fit_transform(X).astype(np.float32)
    print(f"   FactorAnalysis d={d}: fitted and transformed")
    return E, model


def fit_fastica(X: np.ndarray, d: int):
    model = FastICA(n_components=d, random_state=42, max_iter=2000, tol=1e-5)
    E = model.fit_transform(X).astype(np.float32)
    print(f"   FastICA d={d}: fitted and transformed")
    return E, model


def fit_random_projection(X: np.ndarray, d: int):
    """Random Gaussian projection baseline (control for overcomplete evaluation).

    Proves that performance gains at k > d stem from learned causal structure,
    not arbitrary feature space expansion. Fixed random seed ensures
    deterministic results across runs (different seed per trial is drawn once
    at seed-0 time and reused for all seeds).
    """
    model = GaussianRandomProjection(n_components=d, random_state=42)
    E = model.fit_transform(X).astype(np.float32)
    print(f"   RandomProjection d={d}: fitted and transformed")
    return E, model


# ------------------------------------------------------------------------------
# Autoencoders (unsupervised)
# ------------------------------------------------------------------------------

def fit_ae(X: np.ndarray, d: int, epochs: int = 300, lr: float = 1e-3):
    """
    Fit an Autoencoder and return (embedding, model).

    Unsupervised dimensionality reduction via reconstruction loss.
    Deterministic latents (contrast with VAE).
    """
    model = AE(x_dim=X.shape[1], d=d, h=64, nh=3, lr=lr)
    model.fit(X, epochs=epochs, batch_size=100, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   AE d={d}: fitted and transformed")
    return E, model


def fit_vae(X: np.ndarray, d: int, epochs: int = 300, lr: float = 1e-3, beta: float = 1.0):
    """
    Fit a Variational Autoencoder and return (embedding, model).

    Unsupervised dimensionality reduction via reconstruction + KL regularization.
    Stochastic latents that encourage q(z|x) ≈ N(0,1). Middle ground between AE and CEVAE.
    """
    model = VAE(x_dim=X.shape[1], d=d, h=64, nh=3, lr=lr)
    model.fit(X, epochs=epochs, batch_size=100, log_every=100, beta=beta)

    E = model.transform(X).astype(np.float32)
    print(f"   VAE d={d}: fitted and transformed")
    return E, model


# ------------------------------------------------------------------------------
# Causal embeddings (supervised on T, Y)
# ------------------------------------------------------------------------------

def fit_cevae(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int, epochs: int = 300, outcome_type: str = "binary"):
    """
    Fit a CEVAE (Louizos et al. 2017) and return (embedding, model).

    Unlike PCA/UMAP/AE, CEVAE uses treatment T and outcome Y during training so
    the latent space explicitly captures confounding structure. At inference time
    z is inferred from X alone via the soft-mixed posterior.
    """
    model = CEVAE(x_dim=X.shape[1], d=d, outcome_type=outcome_type)
    model = torch.compile(model, mode="reduce-overhead")
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   CEVAE d={d}: fitted and transformed")
    return E, model


def fit_cfrisw(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int = 200, epochs: int = 300, alpha: float = 1.0, outcome_type: str = "binary"):
    """
    Fit CFRISW (Hassanpour & Greiner 2019) and return (embedding, model).

    Like CFRNet but with context-aware importance-sampling weights computed from a
    logistic propensity network fit on the learned representation. Training alternates
    per batch: reweighted factual loss + IPM vs. propensity network fitting.

    Not torch.compile'd: the per-batch alternating two-optimizer scheme is incompatible
    with graph capture and would just force repeated graph breaks.
    """
    model = CFRISW(x_dim=X.shape[1], h=200, nh=3, d=d, alpha=alpha, outcome_type=outcome_type)
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   CFRISW (d={d}): fitted and transformed")
    return E, model


def fit_cfrnet(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int = 200, epochs: int = 300, alpha: float = 0.0, outcome_type: str = "binary"):
    """
    Fit CFRNet (Shalit et al. 2016) and return (embedding, model).

    Learns a balanced representation Φ(x) via ITE-relevant loss + optional IPM penalty.
    - alpha=0 -> TARNet: no balance regularization
    - alpha>0 -> CFRNet: with linear MMD balance between treated/control distributions

    Args:
        d: representation (bottleneck) dimension (default 200)
        outcome_type: "binary" (log loss) or "continuous" (squared loss), per trial
    """
    model = CFRNet(x_dim=X.shape[1], h=200, nh=3, d=d, alpha=alpha, outcome_type=outcome_type)
    model = torch.compile(model, mode="reduce-overhead")
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    name = "CFRNet" if alpha > 0 else "TARNet"
    print(f"   {name} (d={d}): fitted and transformed")
    return E, model


def fit_dragonnet(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int, epochs: int = 300, beta: float = 1.0, outcome_type: str = "binary"):
    """
    Fit DragonNet (Shi, Blei & Veitch 2019) and return (embedding, model).

    Shares CFRNet's balanced-representation idea but adds a propensity head off
    the same trunk and a targeted-regularization loss term (TMLE-style bias
    correction), jointly optimizing representation, outcome heads, and propensity.
    """
    model = DragonNet(x_dim=X.shape[1], h=200, nh=3, d=d, beta=beta, outcome_type=outcome_type)
    model = torch.compile(model, mode="reduce-overhead")
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   DragonNet (d={d}): fitted and transformed")
    return E, model


def fit_site(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int, epochs: int = 200, outcome_type: str = "binary"):
    """
    Fit SITE (Yao et al. 2018) and return (embedding, model).

    Extends the CFRNet/TARNet representation with two similarity-preserving
    terms computed on six propensity-selected "hard" points per batch: a PDDM
    loss that matches representation similarity to propensity-based ground
    truth, and a mid-point distance loss that locally balances treated/control.

    Not torch.compile'd: per-batch hard-point selection needs a numpy
    propensity model and dynamic indexing, which would just force repeated
    graph breaks.
    """
    model = SITE(x_dim=X.shape[1], h=200, nh=3, d=d, outcome_type=outcome_type)
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   SITE (d={d}): fitted and transformed")
    return E, model


def fit_tedvae(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int, epochs: int = 300, outcome_type: str = "binary"):
    """
    Fit TEDVAE (Zhang, Liu & Li 2021) and return (embedding, model).

    Disentangles the latent representation into a true confounder z, a
    treatment-only factor zt, and an outcome-only factor zy. The returned
    embedding is the posterior mean of z alone (d = latent_dim), matching
    CEVAE's convention of exposing only the confounder-relevant latent.
    zt/zy dims scale with d so the side factors can absorb non-confounding
    variation without inflating the reported embedding dimension.
    """
    model = TEDVAE(x_dim=X.shape[1], latent_dim=d, latent_dim_t=max(1, d // 2), latent_dim_y=max(1, d // 2), outcome_type=outcome_type)
    model = torch.compile(model, mode="reduce-overhead")
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   TEDVAE d={d}: fitted and transformed")
    return E, model


def fit_bnn(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int, epochs: int = 300, alpha: float = 1.0, outcome_type: str = "binary"):
    """
    Fit BNN (Johansson, Shalit & Sontag 2016) and return (embedding, model).

    Predecessor of CFRNet: representation Φ(x) fed into a single outcome head
    with t concatenated as a feature (not split t=0/t=1 heads), balanced via
    a linear discrepancy penalty between treated/control representations.
    """
    model = BNN(x_dim=X.shape[1], h=200, d_r=2, d_o=2, d=d, alpha=alpha, outcome_type=outcome_type)
    model = torch.compile(model, mode="reduce-overhead")
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   BNN (d={d}): fitted and transformed")
    return E, model


def fit_dcn(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int, epochs: int = 300, gamma: float = 1.0, outcome_type: str = "binary"):
    """
    Fit DCN (Alaa, Weisz & van der Schaar 2017) and return (embedding, model).

    Shared trunk + per-arm idiosyncratic heads (TARNet-shaped), regularized by
    propensity-dependent dropout instead of a balance/IPM loss, trained with
    an alternating per-epoch schedule (treated-only, then control-only epochs).

    Not torch.compile'd: propensity dropout needs a numpy sklearn model lookup
    mid-forward, and the arm-alternating training loop varies which head's
    parameters get gradients each epoch — both would force repeated graph
    breaks, same rationale as SITE.
    """
    model = DCN(x_dim=X.shape[1], h=200, ls=2, li=1, d=d, gamma=gamma, outcome_type=outcome_type)
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   DCN (d={d}): fitted and transformed")
    return E, model


def fit_deeptreat(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int = 200, epochs: int = 300, lambda1: float = 1.0, lambda2: float = 1.0, outcome_type: str = "binary"):
    """
    Fit DeepTreat (Atan, Jordon & Van Der Schaar 2018) and return (embedding, model).

    Bias-removing autoencoder with IPW-reweighted ITE heads and propensity-based
    debiasing loss. Propensity model refit per epoch (practical simplification of
    the paper's per-update refitting).

    Not torch.compile'd: per-epoch propensity refitting via sklearn inside the loop
    requires numpy round-trips, incompatible with graph capture.
    """
    model = DeepTreat(x_dim=X.shape[1], h=200, nh=3, d=d, li=1, lambda1=lambda1, lambda2=lambda2, outcome_type=outcome_type)
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   DeepTreat (d={d}): fitted and transformed")
    return E, model


def fit_drcfr(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int = 200, epochs: int = 300, alpha: float = 1.0, beta: float = 1.0, outcome_type: str = "binary"):
    """
    Fit DRCFR (Hassanpour & Greiner 2020) and return (embedding, model).

    Disentangles representation into Γ(x) (treatment-only), Δ(x) (confounders),
    Υ(x) (outcome-only). Outcome heads use concat(Δ, Υ) only. Reweighting from Δ,
    logging-policy head π₀(t|Γ, Δ) adds cross-entropy term with weight β.

    Not torch.compile'd: per-batch alternating two-optimizer scheme (reweighting net
    updates separately from main factors) is incompatible with graph capture.
    """
    model = DRCFR(x_dim=X.shape[1], h=200, nh=3, d=d, alpha=alpha, beta=beta, outcome_type=outcome_type)
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   DRCFR (d={d}): fitted and transformed")
    return E, model


def fit_nice(X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int, epochs: int = 300, outcome_type: str = "binary"):
    """
    Fit NICE (Shi, Veitch & Blei 2021) and return (embedding, model).

    DragonNet-style backbone trained with Invariant Risk Minimization (IRM)
    across environments (manufactured internally by shuffling and splitting
    the pooled training set) instead of plain ERM, so the representation is
    pushed toward one where the outcome/treatment relationship is stable
    across environments rather than exploiting spurious associations.

    Not torch.compile'd: the IRM penalty's autograd.grad(..., create_graph=True)
    double-backward is fragile under torch.compile's graph capture.
    """
    model = NICE(x_dim=X.shape[1], h=200, nh=3, d=d, outcome_type=outcome_type)
    model.fit(X, T, Y, epochs=epochs, log_every=100)

    E = model.transform(X).astype(np.float32)
    print(f"   NICE (d={d}): fitted and transformed")
    return E, model


