"""
PyTorch implementation of TARNet / CFRNet (Shalit, Johansson & Sontag 2017).

Same balanced-representation idea as BNN.py, but with split t=0/t=1 outcome heads
in place of one shared head, and a linear MMD balance penalty weighted by α in
place of BNN's discrepancy term: α=0 is TARNet (no penalty), α>0 is CFRNet.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .base import FC, BaseEstimator, ipw_weights, linear_mmd, select_by_treatment


class CFRNet(BaseEstimator):
    def __init__(self, x_dim, h=200, nh=3, d=200, alpha=0.0, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            h: hidden layer size (encoder trunk and outcome heads)
            nh: number of hidden layers
            d: representation (bottleneck) dimension
            alpha: IPM regularization weight (0 -> TARNet, >0 -> CFRNet)
            lr: learning rate for Adam optimizer
            outcome_type: "binary" (log loss, cfr_net.py FLAGS.loss='log') or
                          "continuous" (squared loss, cfr_net.py FLAGS.loss='l2')
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.h = h
        self.nh = nh
        self.d = d
        self.alpha = alpha
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # REPRESENTATION (encoder) — cfr_net.py lines 94-126
        # ──────────────────────────────────────────────────────────────────────
        # nh-1 layers of width h, then bottleneck to d, ELU activation
        self.encoder = FC([x_dim] + [h] * (nh - 1) + [d], activation=nn.ELU)

        # ──────────────────────────────────────────────────────────────────────
        # OUTPUT HEADS (split_output=True) — cfr_net.py lines 257-278
        # ──────────────────────────────────────────────────────────────────────
        # y0_head: d -> nh layers of h -> 1 (for t=0, control)
        self.y0_head = FC([d] + [h] * nh + [1], activation=nn.ELU)

        # y1_head: d -> nh layers of h -> 1 (for t=1, treated)
        self.y1_head = FC([d] + [h] * nh + [1], activation=nn.ELU)

        self.optimizer = Adam(self.parameters(), lr=lr)

    def loss(self, x, t, y):
        """
        Factual loss + optional IPM penalty (cfr_net.py lines 136-204).

        Computes inverse-propensity-weighted BCE loss + optional linear MMD regularization.
        Inlines prediction to avoid double-encoding: phi(x) is computed once per batch.

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment
            y: (n,) binary outcome

        Returns:
            total_loss: scalar
        """
        # Encode once; reuse for both prediction and IPM
        r = self.encode(x)

        # Factual loss: weighted BCE (binary) or squared error (continuous) using
        # split outcome heads (cfr_net.py lines 138-156, FLAGS.loss 'log'/'l2')
        # Inline predict() to avoid calling phi(x) twice per batch
        y0_pred = self.y0_head(r).squeeze(-1)
        y1_pred = self.y1_head(r).squeeze(-1)
        pred_y = select_by_treatment(t, y0_pred, y1_pred)

        w = ipw_weights(t)  # inverse propensity weights, batch treatment rate stabilized

        if self.outcome_type == "binary":
            per_sample_loss = F.binary_cross_entropy_with_logits(pred_y, y, reduction='none')
        else:
            per_sample_loss = F.mse_loss(pred_y, y, reduction='none')

        factual_loss = (w * per_sample_loss).mean()

        # IPM penalty: linear MMD (util.py mmd2_lin lines 103-117)
        # Only applied when alpha > 0 (CFRNet); alpha=0 gives TARNet
        # MMD^2_lin = sum((2*u*μ_1 - 2*(1-u)*μ_0)^2); balance: push treated/control
        # representations close in expectation. Zero if either arm is empty in the batch.
        total_loss = factual_loss
        if self.alpha > 0:
            total_loss = factual_loss + self.alpha * linear_mmd(r, t)

        return total_loss

