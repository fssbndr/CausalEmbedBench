"""
PyTorch implementation of CFRISW (Hassanpour & Greiner 2019).

Same backbone as CFRNet.py, but weighs samples via propensity network π_0(t|Φ(x)).
Training alternates per batch: trunk/heads + IPM in step 1, π_0 only in step 2.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .base import EPSILON, FC, BaseEstimator, linear_mmd, select_by_treatment


class CFRISW(BaseEstimator):
    def __init__(self, x_dim, h=200, nh=3, d=200, alpha=1.0, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            h: hidden layer size (encoder trunk and outcome heads)
            nh: number of hidden layers
            d: representation (bottleneck) dimension
            alpha: IPM regularization weight (linear MMD, same as CFRNet)
            lr: learning rate for Adam optimizers
            outcome_type: "binary" (log loss) or "continuous" (squared loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.h = h
        self.nh = nh
        self.d = d
        self.alpha = alpha
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # REPRESENTATION (encoder) — shared trunk producing Φ(x)
        # ──────────────────────────────────────────────────────────────────────
        self.encoder = FC([x_dim] + [h] * (nh - 1) + [d], activation=nn.ELU)

        # ──────────────────────────────────────────────────────────────────────
        # OUTPUT HEADS (split_output=True) — same as CFRNet
        # ──────────────────────────────────────────────────────────────────────
        self.y0_head = FC([d] + [h] * nh + [1], activation=nn.ELU)
        self.y1_head = FC([d] + [h] * nh + [1], activation=nn.ELU)

        # ──────────────────────────────────────────────────────────────────────
        # PROPENSITY HEAD t_head = π_0(t|Φ(x)) — logistic on representation (§ 3)
        # ──────────────────────────────────────────────────────────────────────
        self.t_head = nn.Linear(d, 1)

        # Two optimizers: one for trunk+outcome, one for propensity (Algorithm 1)
        self.optimizer_main = Adam(list(self.encoder.parameters()) + list(self.y0_head.parameters()) + list(self.y1_head.parameters()), lr=lr)
        self.optimizer_propensity = Adam(self.t_head.parameters(), lr=lr)

    def _propensity_logits(self, r):
        """
        Logits from propensity head t_head = π_0(t|Φ(x)) (§ 3 Context-aware Importance Weighting).

        Args:
            r: (n, d) representation

        Returns:
            logits: (n,) logits for BCE loss
        """
        return self.t_head(r).squeeze(-1)

    def _importance_weights(self, r, t):
        """
        Importance-sampling weights ω_i = 1 + (P(t_i)/P(¬t_i)) · (1-π_0(t_i|φ_i))/π_0(t_i|φ_i)
        (Eq. 4-5, derived from Bayes' theorem on the propensity network).

        Computed with propensity parameters detached (no gradient flow).

        Args:
            r: (n, d) representation (torch tensor)
            t: (n,) binary treatment (torch tensor)

        Returns:
            w: (n,) importance weights
        """
        with torch.no_grad():
            logits = self._propensity_logits(r)
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, min=EPSILON, max=1 - EPSILON)

            # ω = 1 + (t / (1 - u)) · (1 - π) / π + (1 - t) / u · (1 - π) / π
            # where u = batch propensity rate. Simplify:
            # ω_i = 1 + (P(t_i) / P(¬t_i)) · (1 - π_0(t_i|φ_i)) / π_0(t_i|φ_i)
            u = torch.clamp(t.mean(), min=EPSILON, max=1 - EPSILON)
            odds_ratio = t / u + (1 - t) / (1 - u)
            odds_propensity = (1 - probs) / probs
            w = 1.0 + odds_ratio * odds_propensity

        return w

    def loss(self, x, t, y):
        """
        Factual loss + optional IPM penalty (§ 3 Context-aware Importance Weighting).

        Step 1 of the alternating scheme: weighted BCE/L2 loss + linear MMD
        regularization, with importance weights computed from propensity (held fixed).

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment
            y: (n,) binary outcome

        Returns:
            total_loss: scalar
        """
        r = self.encode(x)

        # Factual loss with importance weights
        y0_pred = self.y0_head(r).squeeze(-1)
        y1_pred = self.y1_head(r).squeeze(-1)
        pred_y = select_by_treatment(t, y0_pred, y1_pred)

        w = self._importance_weights(r, t)

        if self.outcome_type == "binary":
            per_sample_loss = F.binary_cross_entropy_with_logits(pred_y, y, reduction='none')
        else:
            per_sample_loss = F.mse_loss(pred_y, y, reduction='none')

        factual_loss = (w * per_sample_loss).mean()

        # IPM penalty (linear MMD)
        total_loss = factual_loss
        if self.alpha > 0:
            total_loss = factual_loss + self.alpha * linear_mmd(r, t)

        return total_loss

    def _propensity_loss(self, r, t):
        """
        Cross-entropy loss for propensity network (Eq. 6).

        Trains π_0 to predict observed treatment from the learned representation
        (with representation parameters detached).

        Args:
            r: (n, d) representation (torch tensor, detached)
            t: (n,) observed binary treatment

        Returns:
            loss: scalar
        """
        logits = self._propensity_logits(r)
        return F.binary_cross_entropy_with_logits(logits, t, reduction='mean')

    def _step(self, x_batch, t_batch, y_batch, optimizer):
        """Two-step alternating update: trunk+outcome, then propensity (§ 3).

        Args:
            x_batch: (b, x_dim) batch covariates
            t_batch: (b,) batch treatment
            y_batch: (b,) batch outcome
            optimizer: unused (uses self.optimizer_main and self.optimizer_propensity)

        Returns:
            loss_value: scalar loss from step 1 (factual + IPM term)
        """
        self.optimizer_main.zero_grad()
        total_loss = self.loss(x_batch, t_batch, y_batch)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.y0_head.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.y1_head.parameters(), max_norm=1.0)
        self.optimizer_main.step()

        self.optimizer_propensity.zero_grad()
        r = self.encode(x_batch).detach()
        prop_loss = self._propensity_loss(r, t_batch)
        prop_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.t_head.parameters(), max_norm=1.0)
        self.optimizer_propensity.step()

        return total_loss.item()
