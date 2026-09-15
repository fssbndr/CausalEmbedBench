"""
PyTorch implementation of CEVAE (Louizos et al. 2017).

VAE-based latent-confounder model (see autoencoder.py:VAE for base VAE mechanics).
Conditions the encoder on (x, y, t) via soft-mixed t=0/t=1 posterior heads during
training; at inference z is inferred from x alone.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .base import FC, BaseEstimator, select_by_treatment


class CEVAE(BaseEstimator):
    def __init__(self, x_dim, d=8, h=200, nh=3, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            d: latent dimension
            h: hidden layer size (matches original paper)
            nh: number of hidden layers (matches original paper)
            lr: learning rate for Adam optimizer
            outcome_type: "binary" (Bernoulli p(y|t,z), matches cevae_ihdp.py) or
                          "continuous" (Normal p(y|t,z), squared-error loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.d = d
        self.h = h
        self.nh = nh
        self.lr = lr

        # ──────────────────────────────────────────────────────────────
        # ENCODER (inference network q) — (cevae_ihdp.py lines 102-117)
        # ──────────────────────────────────────────────────────────────
        # q(t|x) — (cevae_ihdp.py line 103)
        self.qt_net = FC([x_dim, d, 1])

        # q(y|x,t) — shared trunk, split t=0/t=1 heads (cevae_ihdp.py lines 106-108)
        self.qy_shared = FC([x_dim] + [h] * (nh - 1))
        self.qy_t0 = FC([h, h, 1])
        self.qy_t1 = FC([h, h, 1])

        # q(z|x,y,t) — concat [x, qy], shared trunk, split t=0/t=1 DiagNormal heads (cevae_ihdp.py lines 111-117)
        self.qz_shared = FC([x_dim + 1] + [h] * (nh - 1))
        self.qz_t0_loc = FC([h, h, d])
        self.qz_t0_scale = FC([h, h, d])
        self.qz_t1_loc = FC([h, h, d])
        self.qz_t1_scale = FC([h, h, d])

        # ──────────────────────────────────────────────────────────────
        # DECODER (generative model p) — (cevae_ihdp.py lines 79-99)
        # ──────────────────────────────────────────────────────────────
        # p(x|z) — Bernoulli (binary feats) + learned-variance Normal (continuous
        # feats), collapsed here to a single fixed-variance-1 Normal decoder (MSE)
        # (cevae_ihdp.py lines 84-90)
        self.px_shared = FC([d] + [h] * (nh - 1))
        self.px_loc = FC([h, h, x_dim])

        # p(t|z) — Bernoulli (cevae_ihdp.py lines 93-94)
        self.pt_net = FC([d, h, 1])

        # p(y|t,z) — TARNet, separate networks for t=0 and t=1 (cevae_ihdp.py lines 97-99)
        self.py_t0 = FC([d] + [h] * nh + [1])
        self.py_t1 = FC([d] + [h] * nh + [1])

        self.optimizer = Adam(self.parameters(), lr=lr)

    def encode(self, x, y=None):
        """
        Encode (x, y) -> posterior q(z|x,y,t) via soft mixture.

        During training: y is actual outcome (forms true posterior with real confounding).
        During inference: y is predicted from x,t (embedding is function of x alone).

        Returns logits separately to avoid log-of-binary-data issues in auxiliary loss.

        Args:
            x: (n, x_dim) covariates
            y: (n,) outcome. If provided, use actual; else predict from x,t.

        Returns:
            muq, sigmaq, qt_logits, mu_qy: posterior means + encoder logits for loss
        """
        # q(t|x) — soft binary (continuous 0->1, cevae_ihdp.py lines 103-104)
        qt_logits = self.qt_net(x).squeeze(-1)
        qt = torch.sigmoid(qt_logits)

        # q(y|x,t) — two treatment-conditioned heads, soft-mixed (cevae_ihdp.py lines 106-109)
        hqy = self.qy_shared(x)
        mu_qy_t0 = self.qy_t0(hqy).squeeze(-1)
        mu_qy_t1 = self.qy_t1(hqy).squeeze(-1)

        # Soft-mixed mean (before sigmoid) — loc of q(y|x,t)
        mu_qy = select_by_treatment(qt, mu_qy_t0, mu_qy_t1)

        # Use actual y if provided (training), else predicted mean
        qy = y if y is not None else self._y_mean(mu_qy)

        # q(z|x,y,t) — concat [x, qy], two treatment-conditioned Normal heads (cevae_ihdp.py lines 111-117)
        hqz = self.qz_shared(torch.cat([x, qy.unsqueeze(-1)], dim=-1))

        # Treatment t=0 and t=1 posteriors
        muq_t0 = self.qz_t0_loc(hqz)
        sigmaq_t0 = F.softplus(self.qz_t0_scale(hqz)) + 1e-6

        muq_t1 = self.qz_t1_loc(hqz)
        sigmaq_t1 = F.softplus(self.qz_t1_scale(hqz)) + 1e-6

        # Soft mixture: q(z|x,y,t) = qt * q(z|x,y,t=1) + (1-qt) * q(z|x,y,t=0)
        w = qt.unsqueeze(-1)
        muq = w * muq_t1 + (1 - w) * muq_t0
        sigmaq = w * sigmaq_t1 + (1 - w) * sigmaq_t0

        return muq, sigmaq, qt_logits, mu_qy

    def decode(self, z, t):
        """
        Decode z -> p(x|z), p(t|z), p(y|t,z).

        Args:
            z: (n, d) latent samples
            t: (n,) treatment (for selecting outcome head)

        Returns:
            mu_x, logits_t, logits_y: decoder outputs
        """
        # p(x|z) — collapsed to a single fixed-variance-1 Normal decoder here;
        # reference splits binary/continuous feats (cevae_ihdp.py lines 84-90)
        hx = self.px_shared(z)
        mu_x = self.px_loc(hx)

        # p(t|z) — Bernoulli (cevae_ihdp.py lines 93-94)
        logits_t = self.pt_net(z).squeeze(-1)

        # p(y|t,z) — TARNet: separate networks for t=0 and t=1 (cevae_ihdp.py lines 97-99)
        mu2_t0 = self.py_t0(z).squeeze(-1)
        mu2_t1 = self.py_t1(z).squeeze(-1)
        logits_y = select_by_treatment(t, mu2_t0, mu2_t1)

        return mu_x, logits_t, logits_y

    def loss(self, x, t, y, muq, sigmaq, qt_logits, mu_qy, mu_x, logits_t, logits_y):
        """
        ELBO + auxiliary loss (cevae_ihdp.py line 120, line 137).

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment
            y: (n,) outcome
            muq, sigmaq: (n, d) encoder posterior mean/scale for z
            qt_logits: (n,) encoder q(t|x) logits
            mu_qy: (n,) encoder q(y|x,t) soft-mixed mean (pre-activation)
            mu_x: (n, x_dim) decoder p(x|z) mean
            logits_t: (n,) decoder p(t|z) logits
            logits_y: (n,) decoder p(y|t,z) logits

        Returns:
            total_loss: scalar
            components: dict of the individual loss terms, for logging

        Note:
            L = E[log p(x|z)] + E[log p(t|z)] + E[log p(y|t,z)]
              - KL(q(z|x,t,y) || p(z))
              + E[log q(t|x)] + E[log q(y|x,t)]
        """
        # E[log p(x|z)] — Normal with fixed variance 1, so ≡ MSE
        px_loss = 0.5 * torch.mean((x - mu_x) ** 2)

        # E[log p(t|z)] — Bernoulli
        pt_loss = F.binary_cross_entropy_with_logits(logits_t, t, reduction='mean')

        # E[log p(y|t,z)] — Bernoulli (binary) or Normal, fixed variance 1 ≡ MSE (continuous)
        py_loss = self._y_nll(logits_y, y)

        # -KL(q(z|x,t,y) || p(z)) where p(z) = N(0, I)
        kl_loss = 0.5 * torch.mean(
            torch.sum(1 + torch.log(sigmaq ** 2) - muq ** 2 - sigmaq ** 2, dim=-1)
        )

        # Auxiliary: E[log q(t|x)] using logits directly (avoid log-of-binary issues)
        aux_t_loss = F.binary_cross_entropy_with_logits(qt_logits, t, reduction='mean')

        # Auxiliary: E[log q(y|x,t)] using logits directly
        aux_y_loss = self._y_nll(mu_qy, y)

        # Total: ELBO + aux
        total_loss = px_loss + pt_loss + py_loss - kl_loss + aux_t_loss + aux_y_loss

        return total_loss, {
            'px': px_loss.item(),
            'pt': pt_loss.item(),
            'py': py_loss.item(),
            'kl': -kl_loss.item(),
            'aux_t': aux_t_loss.item(),
            'aux_y': aux_y_loss.item(),
        }

    def fit(self, x, t, y, epochs=100, batch_size=100, log_every=0):
        """
        Fit the CEVAE via SGD with reparameterized sampling.

        Args:
            x: (n, x_dim) covariates (numpy array)
            t: (n,) binary treatment (numpy array)
            y: (n,) binary outcome (numpy array)
            epochs: number of training epochs
            batch_size: batch size for SGD
            log_every: print loss every N epochs (0 = silent)

        Returns: self (for sklearn-like chaining)
        """
        x = torch.tensor(x, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        self.train()
        n = x.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x_batch, t_batch, y_batch = x[idx], t[idx], y[idx]

                self.optimizer.zero_grad()

                # Encode: q(z|x,y,t) with actual y (training data provides ground truth)
                muq, sigmaq, qt_logits, mu_qy = self.encode(x_batch, y_batch)

                # Reparameterize: z ~ q(z|x,t,y)
                eps = torch.randn_like(muq)
                z = muq + sigmaq * eps

                # Decode: p(x|z), p(t|z), p(y|t,z)
                mu_x, logits_t, logits_y = self.decode(z, t_batch)

                # ELBO + auxiliary
                total_loss, _ = self.loss(
                    x_batch, t_batch, y_batch, muq, sigmaq, qt_logits, mu_qy,
                    mu_x, logits_t, logits_y
                )

                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            epoch_loss /= max(1, n_batches)
            if log_every > 0 and (epoch + 1) % log_every == 0:
                print(f"   Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

        return self

    def transform(self, x):
        """
        Apply learned CEVAE embedding: z = q(z|x) via soft mixture.

        At inference: no treatment/outcome needed. Predicts Y from X,
        then returns soft-mixed posterior mean.

        Args:
            x: (n, x_dim) covariates (numpy array)

        Returns: (n, d) latent embedding (posterior mean muq, numpy array)
        """
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            muq, _, _, _ = self.encode(x)
        return muq.detach().cpu().numpy().astype(np.float32)
