"""
PyTorch implementation of NICE (Shi, Veitch & Blei 2021).

Same backbone as DragonNet.py, but replaces ERM with Invariant Risk Minimization: a
per-environment gradient penalty weighted by penalty_weight pushes Φ(x) toward one
invariant across environments, filtering out non-causal associations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression, Ridge

from .base import FC, BaseEstimator, select_by_treatment


class NICE(BaseEstimator):
    def __init__(self, x_dim, h=200, nh=3, d=100, n_env=3, l2_regularizer_weight=0.001,
                 penalty_weight=10000.0, penalty_anneal_iters=100, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            h: hidden layer size (trunk)
            nh: number of hidden layers in the trunk
            d: representation (bottleneck) dimension (reference hypo_dim=100)
            n_env: number of environments to split the pooled training data into
            l2_regularizer_weight: weight decay on all parameters (main.py FLAGS.l2_regularizer_weight)
            penalty_weight: IRM penalty weight after annealing kicks in (main.py FLAGS.penalty_weight)
            penalty_anneal_iters: steps before penalty_weight switches from 1 to penalty_weight
            lr: learning rate for Adam optimizer
            outcome_type: "binary" (log loss) or "continuous" (squared loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.d = d
        self.n_env = n_env
        self.l2_regularizer_weight = l2_regularizer_weight
        self.penalty_weight = penalty_weight
        self.penalty_anneal_iters = penalty_anneal_iters
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # REPRESENTATION (shared trunk) — main.py MLP.lin1/lin1_1/lin1_2, ReLU
        # ──────────────────────────────────────────────────────────────────────
        self.encoder = FC([x_dim] + [h] * (nh - 1) + [d], activation=nn.ReLU)

        # ──────────────────────────────────────────────────────────────────────
        # PROPENSITY HEAD (tarnet: off raw x, not shared representation)
        # ──────────────────────────────────────────────────────────────────────
        self.t_head = FC([x_dim, 1])

        # ──────────────────────────────────────────────────────────────────────
        # OUTCOME HEADS (split y0_head/y1_head) — main.py lin2_*/lin3_*/lin4_*
        # ──────────────────────────────────────────────────────────────────────
        self.y0_head = FC([d, d, d, 1], activation=nn.ReLU)
        self.y1_head = FC([d, d, d, 1], activation=nn.ReLU)

        self.optimizer = Adam(self.parameters(), lr=lr)

    def encode(self, x):
        """
        Encode x -> representation Φ(x).

        Trunk output has a trailing ReLU (main.py: `x = F.relu(x)` again after
        lin1_2, on top of FC's own ReLU-before-lin1_2 -- so an extra ReLU here
        matches the reference exactly).

        Args:
            x: (n, x_dim) covariates

        Returns:
            r: (n, d) representation
        """
        return F.relu(self.encoder(x))

    def propensity(self, x):
        """
        Predict propensity score P(T=1|x) = sigmoid(t_head(x)).

        Args:
            x: (n, x_dim) covariates

        Returns:
            propensity: (n,) predicted P(T=1|x) in (0, 1)
        """
        return torch.sigmoid(self.t_head(x).squeeze(-1))

    def _irm_penalty(self, y_logit, y):
        """
        IRMv1 gradient penalty (main.py penalty(), lines 172-180).

        Scales the outcome loss by a dummy scale=1 parameter and measures the
        squared gradient of that loss w.r.t. the scale: large if the optimal
        classifier differs across environments (i.e. representation isn't
        yet invariant).

        Args:
            y_logit: (n,) predicted outcome logits for one environment
            y: (n,) outcome for the same environment

        Returns:
            penalty: scalar squared gradient norm
        """
        scale = torch.tensor(1.0, requires_grad=True, dtype=y_logit.dtype)
        loss = self._y_nll(y_logit * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return (grad ** 2).sum()

    def _make_envs(self, x, t, y):
        """
        Split pooled (x, t, y) into n_env environments by sorting on the covariate
        with weakest fitted association with y (proxy for non-causal parent), then
        cutting into contiguous chunks (NICE Sec 5.3 real-data construction).
        """
        x_np, y_np = x.cpu().numpy(), y.cpu().numpy()
        if self.outcome_type == "binary":
            coefs = LogisticRegression(max_iter=1000).fit(x_np, y_np).coef_[0]
        else:
            coefs = Ridge().fit(x_np, y_np).coef_
        col = np.argmin(np.abs(coefs) * x_np.std(axis=0))
        order = np.argsort(x_np[:, col])
        idx_chunks = np.array_split(order, self.n_env)
        return [(x[idx], t[idx], y[idx]) for idx in idx_chunks]

    def loss(self, envs, step):
        """
        IRM loss across environments (main.py lines 213-260): mean outcome NLL
        + L2 weight decay + annealed IRM penalty + mean propensity NLL.

        Args:
            envs: list of (x_e, t_e, y_e) per environment
            step: current training step (drives the penalty anneal schedule)

        Returns:
            total_loss: scalar
        """
        nlls, t_nlls, penalties = [], [], []
        for x_e, t_e, y_e in envs:
            r = self.encode(x_e)
            y0_logit = self.y0_head(r).squeeze(-1)
            y1_logit = self.y1_head(r).squeeze(-1)
            t_logit = self.t_head(x_e).squeeze(-1)
            y_logit = select_by_treatment(t_e, y0_logit, y1_logit)

            nlls.append(self._y_nll(y_logit, y_e))
            t_nlls.append(self._y_nll(t_logit, t_e) if self.outcome_type == "binary"
                           else F.binary_cross_entropy_with_logits(t_logit, t_e, reduction='mean'))
            penalties.append(self._irm_penalty(y_logit, y_e))

        train_nll = torch.stack(nlls).mean()
        train_t_nll = torch.stack(t_nlls).mean()
        train_penalty = torch.stack(penalties).mean()

        weight_norm = sum((w ** 2).sum() for w in self.parameters())

        penalty_weight = self.penalty_weight if step >= self.penalty_anneal_iters else 1.0

        total_loss = train_nll + self.l2_regularizer_weight * weight_norm
        total_loss = total_loss + penalty_weight * train_penalty
        total_loss = total_loss + train_t_nll

        if penalty_weight > 1.0:
            total_loss = total_loss / penalty_weight

        return total_loss

    def fit(self, x, t, y, epochs=100, log_every=0):
        """
        Fit NICE via full-batch IRM training (main.py lines 234-286).

        Args:
            x: (n, x_dim) covariates (numpy array)
            t: (n,) binary treatment (numpy array)
            y: (n,) outcome (numpy array)
            epochs: number of full-batch steps (reference: FLAGS.steps)
            log_every: print loss every N steps (0 = silent)

        Returns: self (for sklearn-like chaining)
        """
        x = torch.tensor(x, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        self.train()
        envs = self._make_envs(x, t, y)

        for step in range(epochs):
            self.optimizer.zero_grad()

            total_loss = self.loss(envs, step)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

            if log_every > 0 and (step + 1) % log_every == 0:
                print(f"   Step {step + 1}/{epochs}  loss={total_loss.item():.4f}")

        return self
