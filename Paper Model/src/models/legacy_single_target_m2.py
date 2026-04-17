# src/models/m2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def one_hot(y: torch.Tensor, K: int) -> torch.Tensor:
    # y: (B,) int64
    return F.one_hot(y.clamp(min=0), num_classes=K).float()


@dataclass(frozen=True)
class M2Dims:
    d_fp: int
    d_desc: int
    K: int
    z_dim: int


class M2VAE(nn.Module):
    """Semi-supervised VAE (M2) with:
      - q(y|x): classifier
      - q(z|x,y): encoder
      - p(x|y,z): decoder (Bernoulli for fp, Gaussian for desc)
    """

    def __init__(
        self,
        dims: M2Dims,
        *,
        clf_hidden: Tuple[int, ...] = (512, 256),
        enc_hidden: Tuple[int, ...] = (512, 256),
        dec_hidden: Tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dims = dims
        d_x = dims.d_fp + dims.d_desc

        # q(y|x)
        self.qy = mlp(d_x, clf_hidden, dims.K, dropout=dropout)

        # q(z|x,y): outputs mu/logvar
        self.qz = mlp(d_x + dims.K, enc_hidden, 2 * dims.z_dim, dropout=dropout)

        # p(x|y,z): shared trunk + heads
        self.px_trunk = mlp(dims.z_dim + dims.K, dec_hidden, 512, dropout=dropout)
        self.px_fp = nn.Linear(512, dims.d_fp)          # logits
        self.px_desc_mu = nn.Linear(512, dims.d_desc)   # mean
        self.px_desc_lv = nn.Linear(512, dims.d_desc)   # log-variance

    def concat_x(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_fp, x_desc], dim=1)

    def q_y_logits(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> torch.Tensor:
        x = self.concat_x(x_fp, x_desc)
        return self.qy(x)

    def encode_z(self, x_fp: torch.Tensor, x_desc: torch.Tensor, y_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.concat_x(x_fp, x_desc)
        h = self.qz(torch.cat([x, y_onehot], dim=1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def decode_x(self, z: torch.Tensor, y_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.px_trunk(torch.cat([z, y_onehot], dim=1))
        fp_logits = self.px_fp(h)
        desc_mu = self.px_desc_mu(h)
        desc_logvar = self.px_desc_lv(h).clamp(min=-8.0, max=4.0)
        return fp_logits, desc_mu, desc_logvar
