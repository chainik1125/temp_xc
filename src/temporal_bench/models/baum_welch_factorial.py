"""Baum-Welch-friendly sparse factorial temporal autoencoder.

This model implements the "BW Approach 2" proposal:

1. Learn a shared latent dictionary over features.
2. Use a window-aware encoder to emit per-token observations in that space.
3. Run differentiable forward-backward for an independent 2-state HMM on each
   feature channel.
4. Decode only from the posterior support trajectory, optionally modulated by
   a continuous amplitude head.

The latent space itself is atemporal: feature k has the same meaning at every
position. What evolves through the window is the hidden support state for each
feature.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, TemporalAE


class BaumWelchFactorialAE(TemporalAE):
    """TemporalAE with a BW bottleneck over a learned shared latent space.

    Architecture:
        local = W_enc @ (x_t - b_dec) + b_enc
        obs_t = local_t + sum_s C[t, s] local_s
        gamma = forward_backward(obs_1:T)
        amp_t = softplus(local_t + b_amp)
        a_tilde_t = gamma_t(on) * amp_t
        x_hat_t = W_dec @ a_tilde_t + b_dec

    Notes:
        - The decoder sees only the BW posterior support, not the raw
          observations.
        - The HMM is factorial: each feature channel gets an independent 2-state
          chain.
        - HMM parameters are trained by backprop through forward-backward, which
          keeps the model compatible with the existing generic training loop.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        *,
        k: int | None = None,
        sparsity_weight: float = 1e-2,
        amplitude_weight: float = 1e-3,
        support_threshold: float = 0.5,
        min_scale: float = 1e-2,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.sparsity_weight = sparsity_weight
        self.amplitude_weight = amplitude_weight
        self.support_threshold = support_threshold
        self.min_scale = min_scale

        if self.k is not None and not (1 <= self.k <= self.d_sae):
            raise ValueError(f"k must be in [1, {self.d_sae}], got {self.k}")

        # Shared encoder/decoder dictionary.
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Zero-init context matrix lets the model start close to a positionwise
        # encoder while still allowing window-aware observation refinement.
        self.encoder_context = nn.Parameter(torch.zeros(T, T))

        # Continuous amplitude head that shares the encoder basis.
        self.b_amp = nn.Parameter(torch.zeros(d_sae))

        # Per-feature HMM parameters.
        self.logit_alpha = nn.Parameter(torch.full((d_sae,), 2.0))
        self.logit_beta = nn.Parameter(torch.full((d_sae,), -2.0))
        means = torch.stack(
            [torch.zeros(d_sae), torch.ones(d_sae)],
            dim=-1,
        )
        self.emission_mean = nn.Parameter(means)
        self.emission_log_scale = nn.Parameter(torch.zeros(d_sae, 2))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return window-aware observations, amplitudes, and local scores."""
        local = (x - self.b_dec) @ self.W_enc + self.b_enc
        context = torch.einsum("ts,bsm->btm", self.encoder_context, local)
        observations = local + context
        amplitude = F.softplus(local + self.b_amp)
        return observations, amplitude, local

    def _transition_probs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transition and stationary initial probabilities."""
        eps = 1e-5
        alpha = torch.sigmoid(self.logit_alpha).clamp(min=eps, max=1.0 - eps)
        beta = torch.sigmoid(self.logit_beta).clamp(min=eps, max=1.0 - eps)

        transition = torch.stack(
            [
                torch.stack([1.0 - beta, beta], dim=-1),
                torch.stack([1.0 - alpha, alpha], dim=-1),
            ],
            dim=-2,
        )

        denom = (1.0 - alpha + beta).clamp(min=eps)
        pi_on = (beta / denom).clamp(min=eps, max=1.0 - eps)
        initial = torch.stack([1.0 - pi_on, pi_on], dim=-1)
        return transition, initial

    def _emission_log_probs(self, observations: torch.Tensor) -> torch.Tensor:
        """Gaussian log p(obs | state) for each feature channel/state."""
        scales = F.softplus(self.emission_log_scale) + self.min_scale
        obs = observations.unsqueeze(-1)
        means = self.emission_mean.unsqueeze(0).unsqueeze(0)
        scales = scales.unsqueeze(0).unsqueeze(0)

        standardized = (obs - means) / scales
        log_norm = torch.log(scales) + 0.5 * math.log(2.0 * math.pi)
        return -0.5 * standardized.pow(2) - log_norm

    def _forward_backward(
        self,
        log_emission: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run differentiable forward-backward on independent 2-state chains."""
        B, T, m, n_states = log_emission.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"
        assert n_states == 2

        transition, initial = self._transition_probs()
        log_transition = transition.log().unsqueeze(0)
        log_initial = initial.log().unsqueeze(0)

        forward = [log_initial + log_emission[:, 0]]
        for t in range(1, T):
            scores = forward[-1].unsqueeze(-1) + log_transition
            next_forward = torch.logsumexp(scores, dim=-2) + log_emission[:, t]
            forward.append(next_forward)
        log_forward = torch.stack(forward, dim=1)

        log_z = torch.logsumexp(log_forward[:, -1], dim=-1)

        backward = [torch.zeros(B, m, 2, device=log_emission.device, dtype=log_emission.dtype)]
        for t in range(T - 2, -1, -1):
            scores = (
                log_transition
                + log_emission[:, t + 1].unsqueeze(-2)
                + backward[0].unsqueeze(-2)
            )
            backward.insert(0, torch.logsumexp(scores, dim=-1))
        log_backward = torch.stack(backward, dim=1)

        log_gamma = log_forward + log_backward - log_z.unsqueeze(1).unsqueeze(-1)
        gamma = F.softmax(log_gamma, dim=-1)

        if T == 1:
            xi = log_emission.new_empty(B, 0, m, 2, 2)
            return gamma, xi

        xi_terms = []
        normalizer = log_z.unsqueeze(-1).unsqueeze(-1)
        for t in range(T - 1):
            log_xi_t = (
                log_forward[:, t].unsqueeze(-1)
                + log_transition
                + log_emission[:, t + 1].unsqueeze(-2)
                + log_backward[:, t + 1].unsqueeze(-2)
                - normalizer
            )
            xi_terms.append(log_xi_t.exp())
        xi = torch.stack(xi_terms, dim=1)
        return gamma, xi

    def _select_latents(
        self,
        dense_latents: torch.Tensor,
        on_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply optional hard TopK selection to decoded latents."""
        if self.k is None:
            metric_latents = (on_prob >= self.support_threshold).float()
            return dense_latents, metric_latents

        _, topk_idx = dense_latents.topk(self.k, dim=-1)
        mask = torch.zeros_like(dense_latents)
        mask.scatter_(-1, topk_idx, 1.0)
        return dense_latents * mask, mask

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, T, d = x.shape
        assert d == self.d_in, f"Expected d_in={self.d_in}, got {d}"
        assert T == self.T, f"Expected T={self.T}, got {T}"

        observations, amplitude, local_scores = self._encode(x)
        log_emission = self._emission_log_probs(observations)
        state_posteriors, pairwise_posteriors = self._forward_backward(log_emission)

        on_prob = state_posteriors[..., 1]
        dense_latents = on_prob * amplitude
        latents, metric_latents = self._select_latents(dense_latents, on_prob)
        x_hat = latents @ self.W_dec + self.b_dec

        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        sparsity_loss = on_prob.mean()
        amplitude_loss = dense_latents.mean()
        loss = recon_loss
        loss = loss + self.sparsity_weight * sparsity_loss
        loss = loss + self.amplitude_weight * amplitude_loss

        l0 = metric_latents.sum(dim=-1).mean().item()

        transition, initial = self._transition_probs()

        return ModelOutput(
            x_hat=x_hat,
            latents=latents,
            loss=loss,
            metric_latents=metric_latents,
            metrics={
                "recon_loss": recon_loss.item(),
                "l0": l0,
                "mean_on_prob": on_prob.mean().item(),
                "sparsity_loss": sparsity_loss.item(),
                "amplitude_loss": amplitude_loss.item(),
                "alpha_mean": transition[:, 1, 1].mean().item(),
                "beta_mean": transition[:, 0, 1].mean().item(),
                "initial_on_mean": initial[:, 1].mean().item(),
            },
            aux={
                "observations": observations,
                "local_scores": local_scores,
                "amplitudes": amplitude,
                "dense_latents": dense_latents,
                "state_posteriors": state_posteriors,
                "pairwise_posteriors": pairwise_posteriors,
            },
        )

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        return self.W_dec.T

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()
