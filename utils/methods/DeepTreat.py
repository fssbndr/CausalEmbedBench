"""
PyTorch implementation of DeepTreat (Atan, Jordon & Van Der Schaar 2018).

Same backbone as autoencoder.py's AE, but adds IPW-reweighted ITE heads and
a debiasing term λ_1 that pushes the learned representation toward treatment-
uninformativeness via logistic-regression propensity feedback.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam

from .base import EPSILON, FC, BaseEstimator, select_by_treatment


class DeepTreat(BaseEstimator):
    def __init__(self, x_dim, h=200, nh=3, d=200, li=1, lambda1=1.0, lambda2=1.0, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            h: hidden layer size (encoder, decoder, ITE heads)
            nh: number of hidden layers (encoder and decoder)
            d: representation (bottleneck) dimension
            li: number of hidden layers in ITE heads per arm
            lambda1: debiasing cross-entropy weight (§ Bias removing auto-encoder)
            lambda2: IPW outcome loss weight (§ Extension: Treatment Effect Estimation)
            lr: learning rate for Adam optimizer
            outcome_type: "binary" (log loss) or "continuous" (squared loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.h = h
        self.nh = nh
        self.d = d
        self.li = li
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # BIAS REMOVING AUTOENCODER (§ Bias removing auto-encoder)
        # ──────────────────────────────────────────────────────────────────────
        self.encoder = FC([x_dim] + [h] * (nh - 1) + [d], activation=nn.ReLU)
        self.decoder = FC([d] + [h] * (nh - 1) + [x_dim], activation=nn.ReLU)

        # ──────────────────────────────────────────────────────────────────────
        # ITE REGRESSION HEADS (§ Extension: Treatment Effect Estimation)
        # ──────────────────────────────────────────────────────────────────────
        self.y0_head = FC([d] + [h] * li + [1], activation=nn.ReLU)
        self.y1_head = FC([d] + [h] * li + [1], activation=nn.ReLU)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.propensity_scaler = None
        self.propensity_clf = None

    def encode(self, x):
        """
        Encode x -> representation Φ(x).

        Args:
            x: (n, x_dim) covariates

        Returns:
            r: (n, d) representation
        """
        return self.encoder(x)

    def decode(self, r):
        """
        Decode representation r -> reconstruction x̂.

        Args:
            r: (n, d) representation

        Returns:
            x_recon: (n, x_dim) reconstructed input
        """
        return self.decoder(r)

    def _propensity_logits(self, r):
        """
        Logits from fitted logistic-regression propensity model applied to representation.
        (§ Bias removing auto-encoder)

        Args:
            r: (n, d) representation (torch tensor)

        Returns:
            logits: (n,) logits for BCE loss
        """
        if self.propensity_clf is None:
            return torch.zeros(r.shape[0], device=r.device)
        # scale r the same way the scaler was fit, then apply the linear classifier --
        # two plain steps, rather than folding the scaler into an equivalent coef/intercept
        scaled_r = (r - torch.tensor(self.propensity_scaler.mean_, dtype=r.dtype, device=r.device)) \
            / torch.tensor(self.propensity_scaler.scale_, dtype=r.dtype, device=r.device)
        coef = torch.tensor(self.propensity_clf.coef_[0], dtype=r.dtype, device=r.device)
        intercept = torch.tensor(self.propensity_clf.intercept_[0], dtype=r.dtype, device=r.device)
        return (scaled_r @ coef) + intercept

    def _ipw_weights(self, r, t):
        """
        IPW weights 1 / Pr̂(t|Φ) from fitted propensity model (detached).
        (§ Extension: Treatment Effect Estimation)

        Args:
            r: (n, d) representation (torch tensor)
            t: (n,) binary treatment (torch tensor)

        Returns:
            w: (n,) IPW weights
        """
        with torch.no_grad():
            logits = self._propensity_logits(r)
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, min=EPSILON, max=1 - EPSILON)
            w = t / probs + (1 - t) / (1 - probs)
        return w

    def loss(self, x, t, y):
        """
        Joint objective combining reconstruction, debiasing, and IPW outcome loss.
        (§ Bias removing auto-encoder + § Extension: Treatment Effect Estimation)

        Combines (Eq. 3) and IPW-reweighted MSE/BCE:
        - L1: reconstruction MSE + λ_1·ℓ_ce(Pr(T), Pr(T|Φ))
        - L2: λ_2·IPW-weighted outcome loss

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment
            y: (n,) outcome

        Returns:
            total_loss: scalar
        """
        r = self.encode(x)
        x_recon = self.decode(r)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # Debiasing loss: cross-entropy between marginal Pr(T) and model Pr(T|Φ)
        debias_loss = 0.0
        if self.lambda1 > 0 and self.propensity_clf is not None:
            logits = self._propensity_logits(r)
            debias_loss = F.binary_cross_entropy_with_logits(logits, t, reduction='mean')

        # IPW outcome loss
        y0_pred = self.y0_head(r).squeeze(-1)
        y1_pred = self.y1_head(r).squeeze(-1)
        pred_y = select_by_treatment(t, y0_pred, y1_pred)

        w = self._ipw_weights(r, t)
        if self.outcome_type == "binary":
            per_sample_loss = F.binary_cross_entropy_with_logits(pred_y, y, reduction='none')
        else:
            per_sample_loss = F.mse_loss(pred_y, y, reduction='none')
        outcome_loss = (w * per_sample_loss).mean()

        total_loss = recon_loss + self.lambda1 * debias_loss + self.lambda2 * outcome_loss

        return total_loss

    def _before_fit(self, x, t, y):
        """Stash treatment for per-epoch propensity refitting (§ Bias removing auto-encoder).

        Args:
            x: (n, x_dim) covariates (numpy array, unused)
            t: (n,) binary treatment (numpy array)
            y: (n,) outcome (numpy array, unused)
        """
        self._t_np = t.astype(np.float32)

    def _before_epoch(self, x):
        """Refit propensity on current representations (Algorithm 1, lines 593-607).

        Args:
            x: (n, d) tensorized representations (refit computes these from full x)
        """
        r_current = self.encode(x).detach().cpu().numpy()
        self.propensity_scaler = StandardScaler().fit(r_current)
        r_scaled = self.propensity_scaler.transform(r_current)
        self.propensity_clf = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 20), max_iter=1000, scoring="neg_log_loss"
        ).fit(r_scaled, self._t_np)
