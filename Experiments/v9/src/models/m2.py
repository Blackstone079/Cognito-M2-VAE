from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

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


class IFMMapper(nn.Module):
    """Independent Feature Mapping (IFM) style sinusoidal mapping for fingerprint inputs.

    Frequencies are learnable and Gaussian-initialized, following Xia et al. (NeurIPS 2023).
    """

    def __init__(self, in_dim: int, num_frequencies: int = 0, *, learnable: bool = True, include_raw: bool = False, init_std: float = 6.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_raw = bool(include_raw)
        self.init_std = float(init_std)
        if self.num_frequencies > 0:
            init_c = torch.randn(self.num_frequencies, dtype=torch.float32) * max(self.init_std, 1e-6)
            self.c = nn.Parameter(init_c, requires_grad=bool(learnable))
        else:
            self.register_parameter('c', None)

    @property
    def out_dim(self) -> int:
        if self.num_frequencies <= 0:
            return self.in_dim
        mapped = 2 * self.in_dim * self.num_frequencies
        return self.in_dim + mapped if self.include_raw else mapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_frequencies <= 0:
            return x
        c = self.c.view(1, 1, self.num_frequencies)
        v = (2.0 * torch.pi) * x.unsqueeze(-1) * c
        mapped = torch.cat([torch.sin(v), torch.cos(v)], dim=-1).reshape(x.shape[0], -1)
        if self.include_raw:
            return torch.cat([x, mapped], dim=1)
        return mapped


@dataclass(frozen=True)
class StructuredM2Dims:
    d_fp: int
    d_desc: int
    z_dim: int
    binary_tasks: Tuple[str, ...] = ('resp', 'ames')
    include_protox: bool = False
    protox_K: int = 6

    @property
    def y_dim(self) -> int:
        return (self.protox_K if self.include_protox else 0) + len(self.binary_tasks)


class StructuredM2VAE(nn.Module):
    """Structured multitask M2 VAE for optional protox + generic binary tasks."""

    def __init__(
        self,
        dims: StructuredM2Dims,
        *,
        clf_hidden: Tuple[int, ...] = (512, 256),
        enc_hidden: Tuple[int, ...] = (512, 256),
        dec_hidden: Tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        ifm_num_frequencies: int = 0,
        ifm_learnable: bool = True,
        ifm_include_raw: bool = False,
        ifm_init_std: float = 6.0,
        ifm_apply_to_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.include_protox = bool(dims.include_protox)
        self.binary_tasks = tuple(dims.binary_tasks)
        self.d_x = int(dims.d_fp + dims.d_desc)

        self.ifm = IFMMapper(
            dims.d_fp,
            num_frequencies=int(ifm_num_frequencies),
            learnable=bool(ifm_learnable),
            include_raw=bool(ifm_include_raw),
            init_std=float(ifm_init_std),
        )
        self.ifm_apply_to_encoder = bool(ifm_apply_to_encoder)
        d_qy = self.ifm.out_dim + dims.d_desc
        d_qz = d_qy if self.ifm_apply_to_encoder else self.d_x

        self.qy_trunk = mlp(d_qy, clf_hidden, clf_hidden[-1], dropout=dropout)
        h_dim = clf_hidden[-1]
        self.qy_protox = nn.Linear(h_dim, dims.protox_K) if self.include_protox else None
        self.qy_binary = nn.ModuleDict({task: nn.Linear(h_dim, 1) for task in self.binary_tasks})

        self.qz = mlp(d_qz + dims.y_dim, enc_hidden, 2 * dims.z_dim, dropout=dropout)
        self.px_trunk = mlp(dims.z_dim + dims.y_dim, dec_hidden, 512, dropout=dropout)
        self.px_fp = nn.Linear(512, dims.d_fp)
        self.px_desc_mu = nn.Linear(512, dims.d_desc)
        self.px_desc_lv = nn.Linear(512, dims.d_desc)

    def concat_x(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_fp, x_desc], dim=1)

    def map_qy_x(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> torch.Tensor:
        fp_mapped = self.ifm(x_fp)
        return torch.cat([fp_mapped, x_desc], dim=1)

    def map_qz_x(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> torch.Tensor:
        if self.ifm_apply_to_encoder:
            return self.map_qy_x(x_fp, x_desc)
        return self.concat_x(x_fp, x_desc)

    def q_y_logits(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.qy_trunk(self.map_qy_x(x_fp, x_desc))
        out: Dict[str, torch.Tensor] = {task: head(h).squeeze(1) for task, head in self.qy_binary.items()}
        if self.include_protox:
            assert self.qy_protox is not None
            out['protox'] = self.qy_protox(h)
        return out

    def encode_z(self, x_fp: torch.Tensor, x_desc: torch.Tensor, y_struct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.qz(torch.cat([self.map_qz_x(x_fp, x_desc), y_struct], dim=1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def decode_x(self, z: torch.Tensor, y_struct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.px_trunk(torch.cat([z, y_struct], dim=1))
        fp_logits = self.px_fp(h)
        desc_mu = self.px_desc_mu(h)
        desc_logvar = self.px_desc_lv(h).clamp(min=-8.0, max=4.0)
        return fp_logits, desc_mu, desc_logvar


def build_y_struct(
    binary_values: Dict[str, torch.Tensor],
    *,
    binary_task_order: Tuple[str, ...],
    include_protox: bool,
    protox: torch.Tensor | None = None,
    protox_K: int = 6,
) -> torch.Tensor:
    parts = []
    if include_protox:
        if protox is None:
            raise ValueError('protox tensor is required when include_protox=True')
        parts.append(F.one_hot(protox.long(), num_classes=protox_K).float())
    for task in binary_task_order:
        if task not in binary_values:
            raise KeyError(f'Missing binary value for task: {task}')
        parts.append(binary_values[task].float().unsqueeze(1))
    return torch.cat(parts, dim=1)
