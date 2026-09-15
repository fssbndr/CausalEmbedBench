"""
PyTorch implementation of DCN (Alaa, Weisz & van der Schaar 2017).

Same shared-trunk/split-head shape as CFRNet.py's α=0 case, but balances arms via
propensity-dependent dropout (rate set by propensity-model entropy, offset by γ)
instead of an MMD penalty. Training alternates per epoch between the treated and
control head.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .base import EPSILON, FC, BaseEstimator, select_by_treatment


class DCN(BaseEstimator):
    def __init__(self, x_dim, h=200, ls=2, li=1, d=200, gamma=1.0, lr=1e-3,
                 propensity_layers=2, propensity_h=25, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            h: hidden layer size (shared trunk and idiosyncratic-head layers)
            ls: number of shared ReLU trunk layers
            li: number of idiosyncratic ReLU layers per arm head
            d: representation (bottleneck) dimension, i.e. shared-trunk output
            gamma: propensity-dropout offset (§ 3.2 Propensity-Dropout; gamma=1 ->
                zero dropout at propensity 0.5, up to ~0.5 dropout at the extremes)
            lr: learning rate for Adam (shared trunk/heads and the propensity net)
            propensity_layers: hidden-layer count for the propensity network
            propensity_h: hidden-layer width for the propensity network
            outcome_type: "binary" (log loss) or "continuous" (squared loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.h = h
        self.ls = ls
        self.li = li
        self.d = d
        self.gamma = gamma
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # SHARED TRUNK (representation) — ls ReLU layers to d (§ 3.1 Multitask Networks)
        # ──────────────────────────────────────────────────────────────────────
        self.encoder = FC([x_dim] + [h] * (ls - 1) + [d], activation=nn.ReLU)

        # ──────────────────────────────────────────────────────────────────────
        # IDIOSYNCRATIC HEADS (per-arm, only one updated per epoch) (§ 3.1 Multitask Networks)
        # ──────────────────────────────────────────────────────────────────────
        self.y0_head = FC([d] + [h] * li + [1], activation=nn.ReLU)
        self.y1_head = FC([d] + [h] * li + [1], activation=nn.ReLU)

        # ──────────────────────────────────────────────────────────────────────
        # PROPENSITY NETWORK — trained separately (§ 3.1 Multitask Networks),
        # 2 layers x 25 units per the experimental config (§ 4 Experiments)
        # ──────────────────────────────────────────────────────────────────────
        self.propensity_net = FC([x_dim] + [propensity_h] * propensity_layers + [1], activation=nn.ReLU)

        # excludes propensity_net's params -- kept separately trained, not just via no_grad()
        self.optimizer = Adam(
            list(self.encoder.parameters()) + list(self.y0_head.parameters()) + list(self.y1_head.parameters()),
            lr=lr,
        )

    def _fit_propensity_net(self, x_t, t_t, max_epochs=1000, patience=10, tol=1e-4):
        """
        Train the propensity network separately from the outcome model (§ 3.1
        Multitask Networks), stopping on loss plateau since the paper gives no
        epoch count; max_epochs is only a non-convergence safety cap.

        Args:
            x_t: (n, x_dim) covariates (torch tensor)
            t_t: (n,) binary treatment (torch tensor)
            max_epochs: safety cap on training iterations
            patience: epochs to wait for improvement before stopping
            tol: minimum loss improvement to reset patience
        """
        opt = Adam(self.propensity_net.parameters(), lr=self.lr)
        best_loss = float("inf")
        stale = 0
        for _ in range(max_epochs):
            opt.zero_grad()
            logits = self.propensity_net(x_t).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, t_t)
            loss.backward()
            opt.step()
            if loss.item() < best_loss - tol:
                best_loss = loss.item()
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    def _propensity_dropout(self, x, r):
        """
        Apply entropy-based propensity dropout to representation r (§ 3.2 Propensity-Dropout).

        DropoutProb(x) = 1 - gamma/2 - H(p(x))/2
        H = Shannon entropy of the fitted propensity p(x) = P(T=1|x). 
        Inverted dropout: scale kept units by 1/keep_prob so expected activation 
        is unchanged.

        Args:
            x: (n, x_dim) covariates (torch tensor)
            r: (n, d) representation tensor

        Returns:
            r_dropped: (n, d) representation with propensity dropout applied
        """
        with torch.no_grad():
            p = torch.sigmoid(self.propensity_net(x)).squeeze(-1).clamp(EPSILON, 1 - EPSILON)
        entropy = torch.distributions.Bernoulli(probs=p).entropy() / np.log(2)
        keep_prob = (self.gamma / 2 + entropy / 2).clamp(0.01, 1.0).unsqueeze(-1).expand_as(r)
        mask = torch.bernoulli(keep_prob)
        return r * mask / keep_prob

    def _arm_loss(self, x, y, head):
        """
        Factual loss for a single-arm batch, through the shared trunk with
        propensity dropout and the given arm's idiosyncratic head.

        Args:
            x: (n, x_dim) covariates (single-arm batch, tensor)
            y: (n,) outcome (single-arm batch)
            head: self.y0_head or self.y1_head, whichever arm this batch belongs to

        Returns:
            loss: scalar
        """
        r = self.encode(x)
        r = self._propensity_dropout(x, r)
        pred_y = head(r).squeeze(-1)

        return self._y_nll(pred_y, y)

    def fit(self, x, t, y, epochs=300, batch_size=100, log_every=0):
        """
        Fit DCN via alternating per-epoch SGD (§ 3.3 Training the Model): even
        epochs update the shared trunk + treated head on the treated arm only,
        odd epochs update the shared trunk + control head on the control arm
        only. Trains the propensity network once, separately, up front.

        Args:
            x: (n, x_dim) covariates (numpy array)
            t: (n,) binary treatment (numpy array)
            y: (n,) outcome (numpy array)
            epochs: number of training epochs (alternating between arms)
            batch_size: batch size for SGD
            log_every: print loss every N epochs (0 = silent)

        Returns: self (for sklearn-like chaining)
        """
        x_t = torch.tensor(x, dtype=torch.float32)
        t_t = torch.tensor(t, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        self._fit_propensity_net(x_t, t_t)

        idx_by_arm = {1: np.where(t > 0.5)[0], 0: np.where(t <= 0.5)[0]}
        heads = {0: self.y0_head, 1: self.y1_head}

        self.train()

        for epoch in range(epochs):
            arm = 1 if epoch % 2 == 0 else 0
            idx_arm = idx_by_arm[arm]
            head = heads[arm]
            n = len(idx_arm)
            if n == 0:
                continue

            epoch_loss = 0.0
            n_batches = 0

            perm = idx_arm[np.random.permutation(n)]
            for i in range(0, n, batch_size):
                batch_idx = perm[i : i + batch_size]
                x_batch, y_batch = x_t[batch_idx], y_t[batch_idx]

                self.optimizer.zero_grad()

                loss = self._arm_loss(x_batch, y_batch, head)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(1, n_batches)
            if log_every > 0 and (epoch + 1) % log_every == 0:
                print(f"   Epoch {epoch + 1}/{epochs}  arm={arm}  loss={epoch_loss:.4f}")

        return self
