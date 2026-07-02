#!/usr/bin/env python3

"""
Define the base Variational Autoencoder (VAE) model and ELBO objective.

This file contains the encoder, decoder, probability distributions, plotting
function, and variational inference loss used by the standard VAE training script.
"""

import math
from typing import Any, Dict, Tuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

# Sum log-probabilities across all feature dimensions for each sample.
def reduce(x: Tensor) -> Tensor:
    """For each datapoint: sum over all non-batch dimensions."""
    return x.view(x.size(0), -1).sum(dim=1)

# Create and save diagnostic plots for the training process.
def make_vae_plots(
    training_data: Dict[str, list],
    validation_data: Dict[str, list],
    save_path: Path,
    figsize=(12, 4),
) -> None:
    """
    Plot:
      - ELBO
      - KL
      - log p(x|z)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)
    ax_elbo, ax_kl, ax_logpx = axes[0]

    ax_elbo.set_title(r"ELBO: $\mathcal{L}(\mathbf{x})$")
    ax_elbo.plot(training_data["elbo"], label="Training")
    ax_elbo.plot(validation_data["elbo"], label="Validation")
    ax_elbo.set_xlabel("Epoch")
    ax_elbo.legend()

    ax_kl.set_title(
        r"$\mathcal{D}_{\mathrm{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})\right)$"
    )
    ax_kl.plot(training_data["kl"], label="Training")
    ax_kl.plot(validation_data["kl"], label="Validation")
    ax_kl.set_xlabel("Epoch")
    ax_kl.legend()

    ax_logpx.set_title(r"$\log p_\theta(\mathbf{x}|\mathbf{z})$")
    ax_logpx.plot(training_data["log_px"], label="Training")
    ax_logpx.plot(validation_data["log_px"], label="Validation")
    ax_logpx.set_xlabel("Epoch")
    ax_logpx.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# Distribution
# =========================================================
class ReparameterizedDiagonalGaussian(Distribution):
    """
    A diagonal Gaussian N(mu, sigma^2 I) compatible with the reparameterization trick.
    """

    has_rsample = True

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        super().__init__()
        assert mu.shape == log_sigma.shape, (
            f"`mu` shape {mu.shape} and `log_sigma` shape {log_sigma.shape} must match"
        )
        self.mu = mu
        self.log_sigma = torch.clamp(log_sigma, min=-10.0, max=10.0)
        self.sigma = self.log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """eps ~ N(0, I)"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """Sample without gradients."""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """Sample with reparameterization trick."""
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z: Tensor) -> Tensor:
        """Elementwise log probability."""
        # Log probability of a diagonal Gaussian:
        # -0.5 * [((z - mu) / sigma)^2 + 2 log(sigma) + log(2*pi)]
        return -0.5 * (
            ((z - self.mu) ** 2) / (self.sigma ** 2)
            + 2 * self.log_sigma
            + math.log(2 * math.pi)
        )

# ------------------------------
# Negative Binomial distribution
# ------------------------------
class NegativeBinomial(Distribution):
    """
    Negative Binomial with mean `mu` and inverse-dispersion `theta` (>0).

    We implement:
      - log_prob(x)  : exact NB log-likelihood (used for training)
      - sample()     : robust NB(mu) approximation (used only for generation)
    """
    arg_constraints = {
        "mu": constraints.positive,
        "theta": constraints.positive,
    }
    support = constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, mu: Tensor, theta: Tensor, eps: float = 1e-8, validate_args=False):
        # Broadcast mu and theta to the same shape
        self.mu, self.theta = broadcast_all(mu, theta)
        self.eps = eps

        batch_shape = self.mu.size()
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size(), validate_args=validate_args)

    def _nb_base_dist(self):
        """
        Build the underlying torch.distributions.NegativeBinomial
        with (total_count, probs) parameterization.

        For consistency with:
          E[X]   = mu
          Var[X] = mu + mu^2 / theta
        we use:
          total_count = theta
          probs       = mu / (theta + mu)
        """
        probs = self.mu / (self.theta + self.mu + self.eps)
        return torch.distributions.NegativeBinomial(
            total_count=self.theta,
            probs=probs
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.round(value).clamp_min(0)  # Ensure non-negative integers
        base_dist = self._nb_base_dist()
        return base_dist.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        base_dist = self._nb_base_dist()
        return base_dist.sample(sample_shape)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def variance(self) -> torch.Tensor:
        return self.mu + (self.mu ** 2) / (self.theta + self.eps)


# =========================================================
# VAE Model
# =========================================================
class VariationalAutoencoder(nn.Module):
    """
    A Variational Autoencoder with
    - Negative Binomial likelihood p_theta(x|z), suitable for count data
    - Gaussian prior p(z) = N(0, I)
    - Gaussian approximate posterior q_phi(z|x) = N(z | mu_phi(x), sigma_phi(x)^2 I)
    """

    def __init__(self, input_shape: torch.Size, latent_features: int, fixed_log_sigma_x: float = 0.0, fixed_log_sigma_z: float = 0.0,) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = int(np.prod(input_shape))

        self.fixed_log_sigma_x = fixed_log_sigma_x
        self.fixed_log_sigma_z = fixed_log_sigma_z

        # Encoder: x -> posterior params
        self.encoder = nn.Sequential(
            nn.Linear(self.observation_features, 8000),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(8000, 500),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(500, latent_features),
        )

        # Decoder: z -> observation model params
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(500, 8000),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(8000, self.observation_features),
        )

        self.log_theta = torch.nn.Parameter(torch.zeros(self.observation_features))

        # Prior parameters for p(z)=N(0, I)
        self.register_buffer(
            "prior_params",
            torch.zeros(torch.Size([1, 2 * latent_features]))
        )

    def posterior(self, x: Tensor) -> Distribution:
        """q(z|x) with learned mean and fixed variance"""
        #h_x = self.encoder(x) # sample X goes through encoder to get params of q(z|x)
        #mu, log_sigma = h_x.chunk(2, dim=-1) # split last dimension into 2 parts: mu and log_sigma
        mu = self.encoder(x) # sample X goes through encoder to get mean of q(z|x)
        log_sigma = torch.full_like(mu, self.fixed_log_sigma_z) # fixed sigma=1 => log_sigma=0
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size: int = 1) -> Distribution:
        """p(z)"""
        prior_params = self.prior_params.expand(batch_size, -1)
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z: Tensor) -> Distribution:
        """
        p(x|z) as Negative Binomial.

        Decoder predicts the mean expression mu.
        Theta controls dispersion.
        """
        raw_mu = self.decoder(z)

        # Positive mean
        mu = torch.nn.functional.softplus(raw_mu) + 1e-4
        mu = mu.view(-1, *self.input_shape)

        # Positive dispersion, learned per gene
        theta = torch.nn.functional.softplus(self.log_theta) + 1e-4
        theta = theta.view(1, *self.input_shape).expand_as(mu)

        return NegativeBinomial(mu, theta)

    def forward(self, x: Tensor) -> Dict[str, Any]:
        """
        Compute q(z|x), sample z~q(z|x), and return p(x|z).
        """
        x_counts = x.view(x.size(0), -1)
        x_enc = torch.log1p(x_counts)

        qz = self.posterior(x_enc)  # latent distribution q(z|x) for each sample in the batch
        pz = self.prior(batch_size=x.size(0))
        z = qz.rsample()        # sample z from q(z|x) using reparameterization trick
        px = self.observation_model(z)  # compute p(x|z) for the sampled z (decoder)

        return {"px": px, "pz": pz, "qz": qz, "z": z}

    def sample_from_prior(self, batch_size: int = 128) -> Dict[str, Any]:
        """
        Sample z~p(z) and return p(x|z).
        """
        pz = self.prior(batch_size=batch_size)
        z = pz.rsample()
        px = self.observation_model(z)
        return {"px": px, "pz": pz, "z": z}


# =========================================================
# Variational Inference / ELBO
# =========================================================
class VariationalInference(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model: nn.Module, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Any]]:
        outputs = model(x)

        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        kl = log_qz - log_pz
        elbo = log_px - self.beta * kl

        loss = -elbo.mean()

        with torch.no_grad():
            diagnostics = {
                "elbo": elbo,
                "log_px": log_px,
                "kl": kl,
            }

        return loss, diagnostics, outputs

