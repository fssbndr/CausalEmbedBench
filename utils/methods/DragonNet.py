"""
PyTorch implementation of DragonNet (Shi, Blei & Veitch 2019).

Same shared-trunk/split-head shape as CFRNet.py, plus a propensity head off Φ(x)
and a targeted-regularization term weighted by β that applies the TMLE trick
(propensity-weighted perturbation) to reduce ATE bias.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam

from .base import EPSILON, FC, BaseEstimator, select_by_treatment


class DragonNet(BaseEstimator):
    def __init__(self, x_dim, h=200, nh=3, d=200, beta=1.0, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            h: hidden layer size (encoder trunk)
            nh: number of hidden layers in the trunk
            d: representation (bottleneck) dimension
            beta: targeted-regularization weight (0 -> plain DragonNet loss)
            lr: learning rate for the Adam phase of training
            outcome_type: "binary" (log loss) or "continuous" (squared loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.h = h
        self.nh = nh
        self.d = d
        self.beta = beta
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # REPRESENTATION (shared trunk) — models.py lines 119-121
        # ──────────────────────────────────────────────────────────────────────
        self.encoder = FC([x_dim] + [h] * (nh - 1) + [d], activation=nn.ELU)

        # ──────────────────────────────────────────────────────────────────────
        # PROPENSITY HEAD (off the shared trunk, not raw x) — models.py line 124
        # ──────────────────────────────────────────────────────────────────────
        self.t_head = FC([d, 1])

        # ──────────────────────────────────────────────────────────────────────
        # OUTCOME HEADS (split y0_head/y1_head, 100-wide per reference) — models.py lines 127-138
        # ──────────────────────────────────────────────────────────────────────
        self.y0_head = FC([d, 100, 100, 1], activation=nn.ELU)
        self.y1_head = FC([d, 100, 100, 1], activation=nn.ELU)

        # ──────────────────────────────────────────────────────────────────────
        # TARGETED-REGULARIZATION FREE PARAMETER (EpsilonLayer) — models.py lines 59-75
        # Distinct from utils.utils.EPSILON, which is a fixed numerical-stability constant.
        # ──────────────────────────────────────────────────────────────────────
        self.epsilon = nn.Parameter(torch.randn(1) * 0.01)

        self.optimizer_adam = Adam(self.parameters(), lr=lr)

    def encode(self, x):
        """
        Encode x -> representation Φ(x).

        Args:
            x: (n, x_dim) covariates

        Returns:
            r: (n, d) representation
        """
        return self.encoder(x)

    def propensity(self, x):
        """
        Predict propensity score t_hat(Φ(x)) = sigmoid(t_head(Φ(x))).

        Args:
            x: (n, x_dim) covariates

        Returns:
            propensity: (n,) predicted P(T=1|x) in (0, 1)
        """
        r = self.encode(x)
        return torch.sigmoid(self.t_head(r).squeeze(-1))

    def loss(self, x, t, y):
        """
        Regression loss + propensity loss + optional targeted regularization
        (models.py lines 10-104).

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment
            y: (n,) outcome

        Returns:
            total_loss: scalar
        """
        r = self.encode(x)
        y0_logit = self.y0_head(r).squeeze(-1)
        y1_logit = self.y1_head(r).squeeze(-1)
        t_logit = self.t_head(r).squeeze(-1)

        # Regression loss: per-head loss masked by treatment (models.py lines 21-29)
        if self.outcome_type == "binary":
            loss0 = F.binary_cross_entropy_with_logits(y0_logit, y, reduction='none')
            loss1 = F.binary_cross_entropy_with_logits(y1_logit, y, reduction='none')
        else:
            loss0 = F.mse_loss(y0_logit, y, reduction='none')
            loss1 = F.mse_loss(y1_logit, y, reduction='none')
        regression_loss = ((1 - t) * loss0 + t * loss1).mean()

        # Propensity loss (models.py lines 12-16)
        propensity_loss = F.binary_cross_entropy_with_logits(t_logit, t, reduction='mean')

        total_loss = regression_loss + propensity_loss

        # Targeted regularization: TMLE-style perturbation along the clever
        # covariate h(t,g) = t/g - (1-t)/(1-g) (models.py lines 78-104)
        if self.beta > 0:
            pred_y = select_by_treatment(t, y0_logit, y1_logit)
            y_pred = torch.sigmoid(pred_y) if self.outcome_type == "binary" else pred_y

            g = torch.clamp(torch.sigmoid(t_logit), min=EPSILON, max=1 - EPSILON)
            clever_covariate = t / g - (1 - t) / (1 - g)

            y_pert = y_pred + self.epsilon * clever_covariate
            targeted_reg = ((y - y_pert) ** 2).mean()

            total_loss = total_loss + self.beta * targeted_reg

        return total_loss

    def fit(self, x, t, y, epochs=300, batch_size=100, log_every=0):
        """
        Fit DragonNet via two-phase SGD: Adam warmup, then SGD fine-tuning
        (ihdp_main.py lines 72-101, ratio ~1:3 between phases).

        Args:
            x: (n, x_dim) covariates (numpy array)
            t: (n,) binary treatment (numpy array)
            y: (n,) outcome (numpy array)
            epochs: total number of epochs, split ~1:3 between the Adam and SGD phases
            batch_size: batch size for SGD
            log_every: print loss every N epochs per phase (0 = silent)

        Returns: self (for sklearn-like chaining)
        """
        x = torch.tensor(x, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        self.train()

        epochs_adam = max(1, epochs // 4)
        epochs_sgd = max(0, epochs - epochs_adam)

        self._run_sgd(x, t, y, self.optimizer_adam, epochs_adam, batch_size, log_every)

        if epochs_sgd > 0:
            optimizer_sgd = SGD(self.parameters(), lr=1e-5, momentum=0.9, nesterov=True)
            self._run_sgd(x, t, y, optimizer_sgd, epochs_sgd, batch_size, log_every)

        return self
