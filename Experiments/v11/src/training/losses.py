# src/training/losses.py
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.models.m2 import M2VAE, reparameterize


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL(q||p) with p = N(0, I); returns (B,)
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)


def bernoulli_loglik_with_logits(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    # returns (B,)
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    return -torch.sum(bce, dim=1)


def gaussian_loglik(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # log N(x | mu, diag(exp(logvar))) up to exact constant; returns (B,)
    # include constant term for comparability (optional)
    return -0.5 * torch.sum(
        (logvar + (x - mu).pow(2) / torch.exp(logvar) + torch.log(torch.tensor(2.0 * torch.pi, device=x.device))),
        dim=1,
    )


def elbo_labeled(
    model: M2VAE,
    *,
    x_fp: torch.Tensor,
    x_desc: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute negative ELBO (to minimize) for labeled batch."""
    K = model.dims.K
    y_oh = torch.nn.functional.one_hot(y, num_classes=K).float()

    mu, logvar = model.encode_z(x_fp, x_desc, y_oh)
    z = reparameterize(mu, logvar)

    fp_logits, desc_mu, desc_logvar = model.decode_x(z, y_oh)

    ll_fp = bernoulli_loglik_with_logits(x_fp, fp_logits)           # (B,)
    ll_desc = gaussian_loglik(x_desc, desc_mu, desc_logvar)          # (B,)
    recon = ll_fp + ll_desc

    kl = kl_standard_normal(mu, logvar)

    # log p(y) is constant for uniform prior; keep for completeness
    log_py = -torch.log(torch.tensor(float(K), device=x_fp.device))

    elbo = recon + log_py - kl
    loss = -torch.mean(elbo)

    info = {
        "loss_elbo": float(loss.detach().cpu()),
        "recon_fp": float(torch.mean(ll_fp).detach().cpu()),
        "recon_desc": float(torch.mean(ll_desc).detach().cpu()),
        "kl_z": float(torch.mean(kl).detach().cpu()),
    }
    return loss, info


def loss_unlabeled(
    model: M2VAE,
    *,
    x_fp: torch.Tensor,
    x_desc: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Unlabeled objective (negative of U(x) from M2)."""
    K = model.dims.K
    logits_y = model.q_y_logits(x_fp, x_desc)
    q = torch.softmax(logits_y, dim=1)  # (B,K)

    # compute per-class labeled loss terms in a vectorized way
    B = x_fp.shape[0]

    x_fp_rep = x_fp.unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)
    x_desc_rep = x_desc.unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)

    y_ids = torch.arange(K, device=x_fp.device).unsqueeze(0).repeat(B, 1).reshape(B * K)
    y_oh = torch.nn.functional.one_hot(y_ids, num_classes=K).float()

    mu, logvar = model.encode_z(x_fp_rep, x_desc_rep, y_oh)
    z = reparameterize(mu, logvar)
    fp_logits, desc_mu, desc_logvar = model.decode_x(z, y_oh)

    ll_fp = bernoulli_loglik_with_logits(x_fp_rep, fp_logits)
    ll_desc = gaussian_loglik(x_desc_rep, desc_mu, desc_logvar)
    recon = ll_fp + ll_desc
    kl = kl_standard_normal(mu, logvar)
    log_py = -torch.log(torch.tensor(float(K), device=x_fp.device))

    elbo = recon + log_py - kl  # (B*K,)
    elbo = elbo.reshape(B, K)

    # expected elbo under q(y|x)
    expected_elbo = torch.sum(q * elbo, dim=1)  # (B,)

    # entropy term: +H(q) in maximization => -H(q) in loss
    eps = 1e-8
    entropy = -torch.sum(q * torch.log(q + eps), dim=1)  # (B,)

    loss = -torch.mean(expected_elbo + entropy)

    info = {
        "loss_unl": float(loss.detach().cpu()),
        "entropy_qy": float(torch.mean(entropy).detach().cpu()),
        "expected_elbo": float(torch.mean(expected_elbo).detach().cpu()),
    }
    return loss, info


def loss_supervised_ce(
    model: M2VAE,
    *,
    x_fp: torch.Tensor,
    x_desc: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = model.q_y_logits(x_fp, x_desc)
    ce = F.cross_entropy(logits, y, reduction="mean")
    return ce, {"loss_ce": float(ce.detach().cpu())}
