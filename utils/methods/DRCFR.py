"""
PyTorch implementation of DRCFR (Hassanpour & Greiner 2020).

Same backbone as CFRISW.py, but disentangles representation into Γ, Δ, Υ with
outcome heads using concat(Δ, Υ) only. Adds logging-policy head π_0(t|Γ, Δ) with
weight β; alternates per batch.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .base import EPSILON, FC, BaseEstimator, linear_mmd


class DRCFR(BaseEstimator):
    def __init__(self, x_dim, h=200, nh=3, d=200, alpha=1.0, beta=1.0, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            h: hidden layer size (all factor encoders and outcome heads)
            nh: number of hidden layers
            d: dimension per factor (Γ, Δ, Υ each produce d-dim latents)
            alpha: imbalance regularization weight, applied to Υ only (Eq. 4; dr_cfr.py line 158)
            beta: logging-policy cross-entropy weight for π_0(t|Γ,Δ) (Eq. 5; dr_cfr.py lines 166-167)
            lr: learning rate for Adam optimizers
            outcome_type: "binary" (log loss) or "continuous" (squared loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.h = h
        self.nh = nh
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # DISENTANGLED FACTORS — three separate encoders
        # ──────────────────────────────────────────────────────────────────────
        # Γ(x): treatment-only factor (not used in outcome heads)
        self.gamma_encoder = FC([x_dim] + [h] * (nh - 1) + [d], activation=nn.ELU)

        # Δ(x): confounder factor (used in both outcome and reweighting)
        self.delta_encoder = FC([x_dim] + [h] * (nh - 1) + [d], activation=nn.ELU)

        # Υ(x): outcome-only factor (used in outcome, balanced via MMD)
        self.upsilon_encoder = FC([x_dim] + [h] * (nh - 1) + [d], activation=nn.ELU)

        # ──────────────────────────────────────────────────────────────────────
        # OUTCOME HEADS (split y0_head/y1_head) — take concat(Δ, Υ) only (Γ excluded)
        # ──────────────────────────────────────────────────────────────────────
        self.y0_head = FC([2 * d] + [h] * nh + [1], activation=nn.ELU)
        self.y1_head = FC([2 * d] + [h] * nh + [1], activation=nn.ELU)

        # ──────────────────────────────────────────────────────────────────────
        # PROPENSITY NETWORKS
        # ──────────────────────────────────────────────────────────────────────
        # π(t|Δ(x)): reweighting net, logistic on Δ only (like CFRISW's π_0 but on Δ)
        self.pi_reweight = nn.Linear(d, 1)

        # π_0(t|Γ,Δ): logging-policy net, logistic on concat(Γ, Δ)
        # Has its own cross-entropy term in the joint objective (dr_cfr.py lines 166-167)
        self.pi_logging = nn.Linear(2 * d, 1)

        # Three optimizers: one for factors+outcome, one for reweighting, one for logging
        # (Logging is trained jointly with factors/outcome; reweighting alternates per CFRISW pattern)
        self.optimizer_main = Adam(
            list(self.gamma_encoder.parameters()) +
            list(self.delta_encoder.parameters()) +
            list(self.upsilon_encoder.parameters()) +
            list(self.y0_head.parameters()) +
            list(self.y1_head.parameters()) +
            list(self.pi_logging.parameters()),
            lr=lr
        )
        self.optimizer_reweight = Adam(self.pi_reweight.parameters(), lr=lr)

    def encode(self, x):
        """
        Encode x -> disentangled factors (Γ(x), Δ(x), Υ(x)).

        Args:
            x: (n, x_dim) covariates

        Returns:
            gamma: (n, d) treatment-only factor
            delta: (n, d) confounder factor
            upsilon: (n, d) outcome-only factor
        """
        gamma = self.gamma_encoder(x)
        delta = self.delta_encoder(x)
        upsilon = self.upsilon_encoder(x)
        return gamma, delta, upsilon

    def _reweight_logits(self, delta):
        """
        Logits from reweighting propensity network π(t|Δ(x)) (dr_cfr.py _build_treatment_graph(), lines 329-337).

        Args:
            delta: (n, d) confounder factor

        Returns:
            logits: (n,) logits for BCE loss
        """
        return self.pi_reweight(delta).squeeze(-1)

    def _logging_logits(self, gamma, delta):
        """
        Logits from logging-policy propensity network π_0(t|Γ,Δ) (dr_cfr.py _build_treatment_graph(), lines 329-337).

        Args:
            gamma: (n, d) treatment-only factor
            delta: (n, d) confounder factor

        Returns:
            logits: (n,) logits for BCE loss
        """
        combined = torch.cat([gamma, delta], dim=-1)
        return self.pi_logging(combined).squeeze(-1)

    def predict(self, x, t):
        """
        Predict outcome: h_t(concat(Δ(x), Υ(x))) for each sample's treatment.
        Logit if outcome_type='binary', raw regression value if 'continuous'.

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment

        Returns:
            pred_y: (n,) predicted outcomes
        """
        gamma, delta, upsilon = self.encode(x)
        r_outcome = torch.cat([delta, upsilon], dim=-1)
        y0_pred = self.y0_head(r_outcome).squeeze(-1)
        y1_pred = self.y1_head(r_outcome).squeeze(-1)
        return t * y1_pred + (1 - t) * y0_pred

    def _reweight_importance_weights(self, delta, t):
        """
        Importance-sampling weights ω_i computed from reweighting network π(t|Δ)
        (Eq. 2; dr_cfr.py lines 116-124, adapted from CFRISW to use Δ instead of full φ).

        Computed with reweighting-network parameters detached (no gradient flow).

        Args:
            delta: (n, d) confounder factor (torch tensor)
            t: (n,) binary treatment (torch tensor)

        Returns:
            w: (n,) importance weights
        """
        with torch.no_grad():
            logits = self._reweight_logits(delta)
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, min=EPSILON, max=1 - EPSILON)

            u = torch.clamp(t.mean(), min=EPSILON, max=1 - EPSILON)
            odds_ratio = t / u + (1 - t) / (1 - u)
            odds_propensity = (1 - probs) / probs
            w = 1.0 + odds_ratio * odds_propensity

        return w

    def loss(self, x, t, y):
        """
        Joint objective (dr_cfr.py lines 136-170): factual loss (reweighted) +
        imbalance loss (on Υ) + logging-policy cross-entropy.

        Combines:
        - ω-weighted factual loss: (1/N)Σ ω_i · L[y_i, h^{t_i}(Δ, Υ)] (Eq. 3; dr_cfr.py lines 136-148)
        - Imbalance loss on Υ: α · disc({Υ_i}_{t_i=0}, {Υ_i}_{t_i=1}) (Eq. 4; dr_cfr.py line 158)
        - Logging-policy term: β · (1/N)Σ -log[π_0(t_i|Γ, Δ)] (Eq. 5; dr_cfr.py lines 166-167)

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment
            y: (n,) outcome

        Returns:
            total_loss: scalar
        """
        gamma, delta, upsilon = self.encode(x)

        # Outcome predictions from concat(Δ, Υ)
        r_outcome = torch.cat([delta, upsilon], dim=-1)
        y0_pred = self.y0_head(r_outcome).squeeze(-1)
        y1_pred = self.y1_head(r_outcome).squeeze(-1)
        pred_y = t * y1_pred + (1 - t) * y0_pred

        # Reweighted factual loss (with reweighting held fixed)
        w = self._reweight_importance_weights(delta, t)
        if self.outcome_type == "binary":
            per_sample_loss = F.binary_cross_entropy_with_logits(pred_y, y, reduction='none')
        else:
            per_sample_loss = F.mse_loss(pred_y, y, reduction='none')
        factual_loss = (w * per_sample_loss).mean()

        # Imbalance loss on Υ only (MMD between treated/control Υ distributions)
        imbalance_loss = linear_mmd(upsilon, t)

        # Logging-policy cross-entropy plus the reference's L2 penalty on the
        # logging network's weights (Eq. 6's Reg(h0,h1,π0) term for π0;
        # dr_cfr.py lines 335-337; TF l2_loss = sum(w^2)/2, hence the 0.5 factor;
        # W only, no bias)
        logging_logits = self._logging_logits(gamma, delta)
        logging_l2 = 0.5 * 1e-3 * self.pi_logging.weight.pow(2).sum()
        logging_loss = F.binary_cross_entropy_with_logits(logging_logits, t, reduction='mean') + logging_l2

        total_loss = factual_loss + self.alpha * imbalance_loss + self.beta * logging_loss

        return total_loss

    def _reweight_loss(self, delta, t):
        """
        Cross-entropy loss for reweighting network π(t|Δ) (dr_cfr.py _build_treatment_graph(), lines 329-337).

        Trains π to predict observed treatment from the confounder factor
        (with factor parameters detached).

        Args:
            delta: (n, d) confounder factor (torch tensor, detached)
            t: (n,) observed binary treatment

        Returns:
            loss: scalar
        """
        logits = self._reweight_logits(delta)
        return F.binary_cross_entropy_with_logits(logits, t, reduction='mean')

    def _step(self, x_batch, t_batch, y_batch, optimizer):
        """Two-step alternating update: factors+outcome+logging, then reweighting (dr_cfr.py lines 160-170).

        Args:
            x_batch: (b, x_dim) batch covariates
            t_batch: (b,) batch treatment
            y_batch: (b,) batch outcome
            optimizer: unused (uses self.optimizer_main and self.optimizer_reweight)

        Returns:
            loss_value: scalar loss from step 1 (joint factual + imbalance + logging term)
        """
        self.optimizer_main.zero_grad()
        total_loss = self.loss(x_batch, t_batch, y_batch)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gamma_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.delta_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.upsilon_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.y0_head.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.y1_head.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.pi_logging.parameters(), max_norm=1.0)
        self.optimizer_main.step()

        self.optimizer_reweight.zero_grad()
        _, delta, _ = self.encode(x_batch)
        delta = delta.detach()
        reweight_loss = self._reweight_loss(delta, t_batch)
        reweight_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi_reweight.parameters(), max_norm=1.0)
        self.optimizer_reweight.step()

        return total_loss.item()

    def transform(self, x):
        """
        Apply learned DRCFR embedding: concat(Δ(x), Υ(x)) — the outcome-relevant factors.

        Γ(x) is excluded since it carries only treatment-specific information
        and does not contribute to outcome prediction (by design).

        Args:
            x: (n, x_dim) covariates (numpy array)

        Returns: (n, 2*d) embedding (outcome-relevant disentangled factors, numpy array)
        """
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            gamma, delta, upsilon = self.encode(x)
            # Concatenate Δ and Υ only (Γ excluded)
            embedding = torch.cat([delta, upsilon], dim=-1)
        return embedding.detach().cpu().numpy().astype(np.float32)
