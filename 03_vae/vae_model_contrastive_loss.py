#!/usr/bin/env python3

"""
Define the VAE model with an additional supervised contrastive loss.

This file contains the same core VAE components as the base model, plus a
contrastive objective that encourages samples with the same label to have similar
latent representations.
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
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
import torch.nn.functional as F

# Sum log-probabilities across all feature dimensions for each sample.
def reduce(x: Tensor) -> Tensor:
    """For each datapoint: sum over all non-batch dimensions."""
    return x.view(x.size(0), -1).sum(dim=1)

# Create and save diagnostic plots for the training process.
def make_vae_plots(
    training_data: Dict[str, list],
    validation_data: Dict[str, list],
    save_path: Path,
    figsize=(14, 8),
) -> None:
    """
    Plot training and validation curves:
      - ELBO
      - KL
      - log p(x|z)
      - VAE loss
      - contrastive loss
      - total loss
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)

    ax_elbo = axes[0, 0]
    ax_kl = axes[0, 1]
    ax_logpx = axes[0, 2]
    ax_vae_loss = axes[1, 0]
    ax_contrastive = axes[1, 1]
    ax_total_loss = axes[1, 2]

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

    ax_vae_loss.set_title("VAE loss")
    ax_vae_loss.plot(training_data["vae_loss"], label="Training")
    ax_vae_loss.plot(validation_data["vae_loss"], label="Validation")
    ax_vae_loss.set_xlabel("Epoch")
    ax_vae_loss.legend()

    ax_contrastive.set_title("Contrastive loss")
    ax_contrastive.plot(training_data["contrastive_loss"], label="Training")
    ax_contrastive.plot(validation_data["contrastive_loss"], label="Validation")
    ax_contrastive.set_xlabel("Epoch")
    ax_contrastive.legend()

    ax_total_loss.set_title("Total loss")
    ax_total_loss.plot(training_data["total_loss"], label="Training")
    ax_total_loss.plot(validation_data["total_loss"], label="Validation")
    ax_total_loss.set_xlabel("Epoch")
    ax_total_loss.legend()

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
    - Gaussian approximate posterior q_phi(z|x) = N(z | mu_phi(x), sigma_z^2 I)
    with fixed posterior variance
    """

    def __init__(self, input_shape: torch.Size, latent_features: int, fixed_log_sigma_x: float = 0.0, fixed_log_sigma_z: float = 0.0,) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = int(np.prod(input_shape))

        self.fixed_log_sigma_x = fixed_log_sigma_x # Not used in Negative Binomial likelihood, but kept for consistency with Gaussian likelihoods
        self.fixed_log_sigma_z = fixed_log_sigma_z

        # Encoder: x -> posterior params (mu, log_sigma is fixed)
        self.encoder = nn.Sequential(
            nn.Linear(self.observation_features, 8000),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(8000, 500),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(500, latent_features),
        )

        # Decoder: z -> observation model params
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 8000),
            nn.ReLU(),
            nn.Linear(8000, 500),
            nn.ReLU(),
            nn.Linear(500, self.observation_features),
        )

        self.raw_theta = torch.nn.Parameter(torch.zeros(self.observation_features))

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
        theta = torch.nn.functional.softplus(self.raw_theta) + 1e-4
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
# Supervised Contrastive Loss
# =========================================================
# Compute the supervised contrastive objective used together with the VAE loss.
def supervised_contrastive_loss(
    z: Tensor,
    labels: Tensor,
    temperature: float = 0.1
) -> Tensor:
    """
    Supervised contrastive loss.

    Pulls together samples with the same label and pushes apart samples
    with different labels.

    z: latent embeddings, shape [batch_size, latent_dim]
    labels: class labels, shape [batch_size]
    """

    device = z.device
    labels = labels.view(-1, 1)

    batch_size = z.size(0)

    # Normalize latent vectors so similarity = cosine similarity (norm = 1)
    z = F.normalize(z, dim=1)

    # Similarity matrix: [batch_size, batch_size]
    similarity_matrix = torch.matmul(z, z.T) / temperature

    # Mask: positives have same label, excluding self-comparisons
    labels_equal = torch.eq(labels, labels.T).float().to(device)

    # Remove diagonal: a sample should not be positive with itself
    self_mask = torch.eye(batch_size, device=device)
    positives_mask = labels_equal * (1.0 - self_mask)

    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Exclude self-comparisons from denominator
    exp_logits = torch.exp(logits) * (1.0 - self_mask)

    # Log-probability of positives
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Mean log-probability over positives
    positives_per_sample = positives_mask.sum(dim=1)

    # Avoid division by zero for classes appearing only once in batch
    valid_samples = positives_per_sample > 0

    mean_log_prob_pos = (positives_mask * log_prob).sum(dim=1) / (
        positives_per_sample + 1e-8
    )

    loss = -mean_log_prob_pos[valid_samples].mean()

    # If no positives exist in the batch, return zero loss
    if torch.isnan(loss):
        loss = torch.tensor(0.0, device=device)

    return loss


# =========================================================
# Variational Inference / ELBO
# =========================================================
class VariationalInference(nn.Module):
    def __init__(
        self, 
        beta: float = 1.0,
        lambda_contrastive: float = 0.0,
        contrastive_temperature: float = 0.1,
        use_posterior_mean: bool = True,
    ):
        super().__init__()
        self.beta = beta
        self.lambda_contrastive = lambda_contrastive
        self.contrastive_temperature = contrastive_temperature
        self.use_posterior_mean = use_posterior_mean

    def forward(self, model: nn.Module, x: Tensor, labels: Tensor) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Any]]:
        outputs = model(x)

        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        kl = log_qz - log_pz
        elbo = log_px - self.beta * kl

        vae_loss = -elbo.mean()

        contrastive_loss = torch.tensor(0.0, device=x.device)

        if labels is not None and self.lambda_contrastive > 0.0:
            if self.use_posterior_mean:
                z_contrastive = qz.mu
            else:
                z_contrastive = z

            contrastive_loss = supervised_contrastive_loss(
                z_contrastive,
                labels,
                temperature=self.contrastive_temperature
            )
        
        loss = vae_loss + self.lambda_contrastive * contrastive_loss

        with torch.no_grad():
            diagnostics = {
                "elbo": elbo,
                "log_px": log_px,
                "kl": kl,
                "vae_loss": vae_loss,
                "contrastive_loss": contrastive_loss,
                "total_loss": loss,
            }

        return loss, diagnostics, outputs

