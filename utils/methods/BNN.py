"""
PyTorch implementation of BNN (Johansson, Shalit & Sontag 2016).

Concatenates treatment t onto representation Φ(x) and predicts through a single
shared output head, not split t=0/t=1 heads. A linear discrepancy penalty balances
Φ(x) between treated and control.
"""

import torch
from torch import nn
from torch.optim import RMSprop

from .base import FC, BaseEstimator, linear_mmd


class BNN(BaseEstimator):
    """
    Uses the "BNN-2-2" variant (d_r=2, d_o=2), reported as the best-performing
    configuration in the paper.
    """
    def __init__(self, x_dim, h=25, d_r=2, d_o=2, d=25, alpha=1.0, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            h: hidden layer size (representation and outcome-head layers)
            d_r: number of ReLU representation layers (BNN-2-2: d_r=2)
            d_o: number of ReLU outcome-head layers after concatenating t (BNN-2-2: d_o=2)
            d: representation (bottleneck) dimension
            alpha: linear discrepancy balance-penalty weight
            lr: learning rate for RMSprop optimizer
            outcome_type: "binary" (log loss) or "continuous" (squared loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.h = h
        self.d_r = d_r
        self.d_o = d_o
        self.d = d
        self.alpha = alpha
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # REPRESENTATION Φ(x) — d_r ReLU layers to bottleneck d (§"Deep neural networks")
        # ──────────────────────────────────────────────────────────────────────
        self.encoder = FC([x_dim] + [h] * (d_r - 1) + [d], activation=nn.ReLU)

        # ──────────────────────────────────────────────────────────────────────
        # OUTCOME HEAD h([Φ(x), t]) — single shared head, t concatenated (§"Deep neural networks")
        # ──────────────────────────────────────────────────────────────────────
        self.head = FC([d + 1] + [h] * d_o + [1], activation=nn.ReLU)

        self.optimizer = RMSprop(self.parameters(), lr=lr, weight_decay=1e-3)

    def predict(self, x, t):
        """
        Predict outcome: h(Φ(x), t). Logit if outcome_type='binary', raw regression
        value if 'continuous'.

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment

        Returns:
            pred_y: (n,) predicted outcomes
        """
        r = self.encode(x)
        return self.head(torch.cat([r, t.unsqueeze(-1)], dim=-1)).squeeze(-1)

    def loss(self, x, t, y):
        """
        Factual loss + linear discrepancy balance penalty (§"Balancing counterfactual
        regression"; closed form for the balance term in §"Linear discrepancy").

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment
            y: (n,) outcome

        Returns:
            total_loss: scalar
        """
        r = self.encode(x)
        pred_y = self.head(torch.cat([r, t.unsqueeze(-1)], dim=-1)).squeeze(-1)

        factual_loss = self._y_nll(pred_y, y)

        total_loss = factual_loss
        if self.alpha > 0:
            total_loss = factual_loss + self.alpha * linear_mmd(r, t)

        return total_loss

