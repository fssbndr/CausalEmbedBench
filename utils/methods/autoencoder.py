"""
Autoencoder variants for unsupervised dimensionality reduction.

- AE: Standard autoencoder (deterministic latents, MSE reconstruction loss)
- VAE: Variational autoencoder (stochastic latents, ELBO loss with KL regularization)

Both use FC builder from utils and follow sklearn-like fit/transform interface.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .base import FC, BaseModel


class AE(BaseModel):
    """
    Standard Autoencoder with deterministic latents.

    Learns a latent representation z via reconstruction loss only (no KL regularization).
    Simpler than VAE: deterministic encoder, no stochasticity or sampling.
    """
    def __init__(self, x_dim, d=8, h=64, nh=3, lr=1e-3):
        """
        Args:
            x_dim: input (covariate) dimension
            d: latent dimension (bottleneck width)
            h: hidden layer size
            nh: number of hidden layers
            lr: learning rate for Adam optimizer
        """
        super().__init__()
        self.x_dim = x_dim
        self.d = d

        # Deterministic encoder: X -> z
        self.encoder =FC([x_dim] + [h] * nh + [d], activation=nn.ReLU)

        # Decoder: z -> X̂
        self.decoder = FC([d] + [h] * nh + [x_dim], activation=nn.ReLU)

        self.optimizer = Adam(self.parameters(), lr=lr)

    def decode(self, z):
        """
        Decode z -> reconstruction x̂.

        Args:
            z: (n, d) latent samples

        Returns:
            x_recon: (n, x_dim) reconstructed input
        """
        return self.decoder(z)

    def fit(self, x, epochs=200, batch_size=100, log_every=0):
        """
        Fit AE via SGD (deterministic forward pass, no sampling).

        Loss = MSE(x, decode(encode(x))).

        Args:
            x: (n, x_dim) covariates (numpy array)
            epochs: number of training epochs
            batch_size: batch size for SGD
            log_every: print loss every N epochs (0 = silent)

        Returns:
            self (for sklearn-like chaining)
        """
        x = torch.tensor(x, dtype=torch.float32)
        self.train()
        n = x.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x_batch = x[idx]

                self.optimizer.zero_grad()

                # Encoder: deterministic
                z = self.encode(x_batch)

                # Decoder
                x_recon = self.decode(z)

                # MSE loss
                loss = F.mse_loss(x_recon, x_batch, reduction='mean')

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(1, n_batches)
            if log_every > 0 and (epoch + 1) % log_every == 0:
                print(f"   Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

        return self

    def transform(self, x):
        """
        Apply learned AE embedding: deterministic encoder output.

        Args:
            x: (n, x_dim) covariates (numpy array)

        Returns:
            z: (n, d) latent embedding (numpy array)
        """
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
        return z.cpu().numpy().astype(np.float32)


class VAE(BaseModel):
    """
    Variational Autoencoder with stochastic latents.

    Learns a latent representation z ~ q(z|x) where q is a Gaussian approximation to
    the true posterior p(z|x,y). The ELBO loss combines reconstruction (likelihood) and
    KL divergence (prior matching). Beta controls the KL weight:
    - beta=0: pure reconstruction
    - beta=1: standard VAE (balance reconstruction and regularization)
    - beta>1: emphasizes prior matching (β-VAE)
    """
    def __init__(self, x_dim, d=8, h=64, nh=3, lr=1e-3):
        """
        Args:
            x_dim: input (covariate) dimension
            d: latent dimension (bottleneck width)
            h: hidden layer size in encoder/decoder trunks
            nh: number of hidden layers
            lr: learning rate for Adam optimizer
        """
        super().__init__()
        self.x_dim = x_dim
        self.d = d

        # Stochastic encoder: X -> (μ, log σ^2) of q(z|x)
        self.encoder_trunk = FC([x_dim] + [h] * nh, activation=nn.ReLU)
        self.encoder_mu = nn.Linear(h, d)
        self.encoder_logvar = nn.Linear(h, d)

        # Decoder: z -> X̂
        self.decoder = FC([d] + [h] * nh + [x_dim], activation=nn.ReLU)

        self.optimizer = Adam(self.parameters(), lr=lr)

    def encode(self, x):
        """
        Encode x -> latent distribution q(z|x).

        Args:
            x: (n, x_dim) covariates

        Returns:
            mu: (n, d) mean of q(z|x)
            logvar: (n, d) log variance of q(z|x)
        """
        h = self.encoder_trunk(x)
        return self.encoder_mu(h), self.encoder_logvar(h)

    def decode(self, z):
        """
        Decode z -> reconstruction x̂.

        Args:
            z: (n, d) latent samples

        Returns:
            x_recon: (n, x_dim) reconstructed input
        """
        return self.decoder(z)

    def fit(self, x, epochs=200, batch_size=100, log_every=0, beta=1.0):
        """
        Fit VAE via SGD with reparameterization trick.

        ELBO = E[log p(x|z)] - β·KL(q(z|x) || p(z)) where p(z) = N(0,I).

        Args:
            x: (n, x_dim) covariates (numpy array)
            epochs: number of training epochs
            batch_size: batch size for SGD
            log_every: print loss every N epochs (0 = silent)
            beta: KL weight (0 = pure reconstruction, 1 = standard VAE)

        Returns:
            self (for sklearn-like chaining)
        """
        x = torch.tensor(x, dtype=torch.float32)
        self.train()
        n = x.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x_batch = x[idx]

                self.optimizer.zero_grad()

                # Encoder: q(z|x)
                mu, logvar = self.encode(x_batch)

                # Reparameterization: z ~ q(z|x)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)

                # Decoder: p(x|z)
                x_recon = self.decode(z)

                # ELBO loss
                recon_loss = F.mse_loss(x_recon, x_batch, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(1, n_batches)
            if log_every > 0 and (epoch + 1) % log_every == 0:
                print(f"   Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

        return self

    def transform(self, x):
        """
        Apply learned VAE embedding: return latent posterior mean mu (deterministic at inference).

        Args:
            x: (n, x_dim) covariates (numpy array)

        Returns:
            mu: (n, d) posterior mean (latent embedding, numpy array)
        """
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu.cpu().numpy().astype(np.float32)