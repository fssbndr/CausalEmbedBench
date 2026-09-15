"""
PyTorch implementation of TEDVAE (Zhang, Liu & Li 2021).

Same VAE-based latent-confounder approach as CEVAE.py, but disentangles the encoder
into three independent blocks instead of one shared posterior: z (true confounder),
zt (treatment-only), zy (outcome-only).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .base import EPSILON, FC, BaseEstimator, select_by_treatment


class TEDVAE(BaseEstimator):
    def __init__(self, x_dim, latent_dim=20, latent_dim_t=10, latent_dim_y=10,
                 h=200, nh=3, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            latent_dim: dimension of z, the true confounder block (used as the embedding)
            latent_dim_t: dimension of zt, the treatment-only factor block
            latent_dim_y: dimension of zy, the outcome-only factor block
            h: hidden layer size
            nh: number of hidden layers per trunk
            lr: learning rate for Adam optimizer
            outcome_type: "binary" (log loss) or "continuous" (squared loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.latent_dim_t = latent_dim_t
        self.latent_dim_y = latent_dim_y
        self.h = h
        self.nh = nh
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # GUIDE (inference network q) — tedvae_gpu.py Guide, lines 295-332
        # ──────────────────────────────────────────────────────────────────────
        # Three independent trunks x -> (loc, scale), one per latent block
        # (no cross-conditioning between z/zt/zy, unlike CEVAE's z|x,y,t).
        self.qz_trunk = FC([x_dim] + [h] * (nh - 1))
        self.qz_loc = FC([h, h, latent_dim])
        self.qz_scale = FC([h, h, latent_dim])

        self.qzt_trunk = FC([x_dim] + [h] * (nh - 1))
        self.qzt_loc = FC([h, h, latent_dim_t])
        self.qzt_scale = FC([h, h, latent_dim_t])

        self.qzy_trunk = FC([x_dim] + [h] * (nh - 1))
        self.qzy_loc = FC([h, h, latent_dim_y])
        self.qzy_scale = FC([h, h, latent_dim_y])

        # ──────────────────────────────────────────────────────────────────────
        # AUXILIARY q(t|z,zt), q(y|t,z,zy) — loss-shaping only, not generative
        # (tedvae_gpu.py Guide.t_dist/y_dist, lines 303-332, 357-371; is_auxiliary=True)
        # ──────────────────────────────────────────────────────────────────────
        self.qt_net = FC([latent_dim + latent_dim_t, 1])
        self.qy_trunk = nn.Sequential(FC([latent_dim + latent_dim_y, h]), nn.ELU())
        self.qy0_net = FC([h, 1])
        self.qy1_net = FC([h, 1])

        # ──────────────────────────────────────────────────────────────────────
        # MODEL (generative model p) — tedvae_gpu.py Model, lines 421-523
        # ──────────────────────────────────────────────────────────────────────
        # p(x|z,zt,zy) — single fixed-variance Normal decoder (MSE); collapses
        # the reference's Bernoulli logits (binary feats) / DiagNormal
        # (continuous feats) split (tedvae_gpu.py lines 448-453).
        self.px_loc = FC([latent_dim + latent_dim_t + latent_dim_y, h, h, x_dim])

        # p(t|z,zt) — separate weights from the guide's qt_net (tedvae_gpu.py line 460)
        self.pt_net = FC([latent_dim + latent_dim_t, 1])

        # p(y|t,z,zy) — TARNet-style split heads, separate weights from guide's
        self.py0_net = FC([latent_dim + latent_dim_y, h, h, 1])
        self.py1_net = FC([latent_dim + latent_dim_y, h, h, 1])

        self.optimizer = Adam(self.parameters(), lr=lr)

    @staticmethod
    def _kl_diag_normal(mu, sigma):
        """Analytic KL(N(mu, sigma^2) || N(0, I)), summed over latent dims, averaged over batch."""
        return 0.5 * torch.mean(
            torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2, dim=-1)
        )

    def encode(self, x):
        """
        Encode x -> posteriors q(z|x), q(zt|x), q(zy|x).

        Args:
            x: (n, x_dim) covariates

        Returns:
            muz, sigmaz, mut, sigmat, muy, sigmay: posterior means/scales per block
        """
        hz = self.qz_trunk(x)
        muz = self.qz_loc(hz)
        sigmaz = F.softplus(self.qz_scale(hz)) + EPSILON

        ht = self.qzt_trunk(x)
        mut = self.qzt_loc(ht)
        sigmat = F.softplus(self.qzt_scale(ht)) + EPSILON

        hy = self.qzy_trunk(x)
        muy = self.qzy_loc(hy)
        sigmay = F.softplus(self.qzy_scale(hy)) + EPSILON

        return muz, sigmaz, mut, sigmat, muy, sigmay

    def decode(self, z, zt, zy, t):
        """
        Decode (z, zt, zy) -> p(x|z,zt,zy), p(t|z,zt), p(y|t,z,zy).

        Args:
            z: (n, latent_dim) true confounder sample
            zt: (n, latent_dim_t) treatment-only factor sample
            zy: (n, latent_dim_y) outcome-only factor sample
            t: (n,) binary treatment (for selecting outcome head)

        Returns:
            mu_x, logits_t, logits_y: decoder outputs
        """
        mu_x = self.px_loc(torch.cat([z, zt, zy], dim=-1))
        logits_t = self.pt_net(torch.cat([z, zt], dim=-1)).squeeze(-1)

        zy_in = torch.cat([z, zy], dim=-1)
        y0 = self.py0_net(zy_in).squeeze(-1)
        y1 = self.py1_net(zy_in).squeeze(-1)
        logits_y = select_by_treatment(t, y0, y1)

        return mu_x, logits_t, logits_y

    def loss(self, x, t, y):
        """
        ELBO + 100x auxiliary t/y loss (tedvae_gpu.py TraceCausalEffect_ELBO, lines 390-417).

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment
            y: (n,) outcome

        Returns:
            total_loss: scalar
        """
        muz, sigmaz, mut, sigmat, muy, sigmay = self.encode(x)

        # Reparameterized sampling per block
        z = muz + sigmaz * torch.randn_like(muz)
        zt = mut + sigmat * torch.randn_like(mut)
        zy = muy + sigmay * torch.randn_like(muy)

        # Generative likelihoods (model heads)
        mu_x, logits_t, logits_y = self.decode(z, zt, zy, t)
        recon_x = 0.5 * F.mse_loss(mu_x, x, reduction='mean')
        pt_loss = F.binary_cross_entropy_with_logits(logits_t, t, reduction='mean')
        py_loss = self._y_nll(logits_y, y)

        # KL(q||p), one term per disentangled block, prior N(0,I)
        kl = (
            self._kl_diag_normal(muz, sigmaz)
            + self._kl_diag_normal(mut, sigmat)
            + self._kl_diag_normal(muy, sigmay)
        )

        elbo_loss = recon_x + pt_loss + py_loss - kl

        # Auxiliary q(t|z,zt), q(y|t,z,zy) — weighted 100x (tedvae_gpu.py lines 410-411)
        aux_t_logits = self.qt_net(torch.cat([z, zt], dim=-1)).squeeze(-1)
        aux_t_loss = F.binary_cross_entropy_with_logits(aux_t_logits, t, reduction='mean')

        aux_hy = self.qy_trunk(torch.cat([z, zy], dim=-1))
        aux_y0 = self.qy0_net(aux_hy).squeeze(-1)
        aux_y1 = self.qy1_net(aux_hy).squeeze(-1)
        aux_y_logits = select_by_treatment(t, aux_y0, aux_y1)
        aux_y_loss = self._y_nll(aux_y_logits, y)

        total_loss = elbo_loss + 100.0 * (aux_t_loss + aux_y_loss)

        return total_loss


    def transform(self, x):
        """
        Apply learned TEDVAE embedding: posterior mean of z, the disentangled
        true-confounder block (analogous to CEVAE's transform returning muq).

        Args:
            x: (n, x_dim) covariates (numpy array)

        Returns: (n, latent_dim) confounder embedding (numpy array)
        """
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            hz = self.qz_trunk(x)
            muz = self.qz_loc(hz)
        return muz.detach().cpu().numpy().astype(np.float32)
