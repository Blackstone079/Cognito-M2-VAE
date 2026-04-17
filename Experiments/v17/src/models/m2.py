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
    """Independent Feature Mapping for fingerprint inputs.

    Returns only the sinusoidal mapping. Any raw skip path is handled explicitly by
    the classifier so the raw and IFM branches can be ablated independently.
    """

    def __init__(self, in_dim: int, num_frequencies: int = 0, *, learnable: bool = True, init_std: float = 6.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_frequencies = int(num_frequencies)
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
        return 2 * self.in_dim * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_frequencies <= 0:
            return x
        c = self.c.view(1, 1, self.num_frequencies)
        v = (2.0 * torch.pi) * x.unsqueeze(-1) * c
        return torch.cat([torch.sin(v), torch.cos(v)], dim=-1).reshape(x.shape[0], -1)


class FusionMLP(nn.Module):
    """MLP that can return intermediate activations for diagnostics."""

    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        dims = [int(in_dim), *[int(h) for h in hidden_dims], int(out_dim)]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, *, return_hidden: bool = False):
        h = x
        hidden = []
        for i, layer in enumerate(self.linears):
            h = layer(h)
            if i < len(self.linears) - 1:
                h = F.relu(h)
                hidden.append(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        if return_hidden:
            return h, hidden
        return h


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor) -> None:
        self.mask.data.copy_(mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.mask, self.bias)


def _make_made_masks(dim: int, hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    in_deg = torch.arange(1, dim + 1)
    hidden_base = torch.arange(hidden_dim) % max(dim - 1, 1)
    hidden_deg = hidden_base + 1
    mask_in = (hidden_deg.unsqueeze(1) >= in_deg.unsqueeze(0)).float()
    out_deg = in_deg
    mask_out = (out_deg.unsqueeze(1) > hidden_deg.unsqueeze(0)).float()
    return mask_in, mask_out


class ConditionalIAFBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_dim: int, *, sigmoid_bias: float = 2.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.sigmoid_bias = float(sigmoid_bias)
        self.fc1 = MaskedLinear(self.dim, self.hidden_dim)
        self.fc2 = MaskedLinear(self.hidden_dim, 2 * self.dim)
        self.ctx1 = nn.Linear(self.context_dim, self.hidden_dim, bias=False)
        self.ctx2 = nn.Linear(self.context_dim, 2 * self.dim, bias=False)
        mask1, mask2 = _make_made_masks(self.dim, self.hidden_dim)
        self.fc1.set_mask(mask1)
        mask2_full = torch.cat([mask2, mask2], dim=0)
        self.fc2.set_mask(mask2_full)

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(z) + self.ctx1(context))
        m, logit_s = torch.chunk(self.fc2(h) + self.ctx2(context), 2, dim=1)
        sigma = torch.sigmoid(logit_s + self.sigmoid_bias)
        sigma = torch.clamp(sigma, min=1e-5, max=1.0 - 1e-5)
        z_new = sigma * z + (1.0 - sigma) * m
        log_det = torch.sum(torch.log(sigma), dim=1)
        return z_new, log_det


class IAFStack(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_flows: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.flows = nn.ModuleList([ConditionalIAFBlock(dim, context_dim, hidden_dim) for _ in range(int(num_flows))])

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        h = z
        for i, flow in enumerate(self.flows):
            if i % 2 == 1:
                h = torch.flip(h, dims=[1])
            h, log_det = flow(h, context)
            if i % 2 == 1:
                h = torch.flip(h, dims=[1])
            total_log_det = total_log_det + log_det
        return h, total_log_det


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
    """Structured multitask M2 VAE.

    v17 design:
    - encoder/decoder consume the frozen v13 log-count feature line
    - q(y|x) keeps the stronger dual-view classifier from v15/v16
    - optional small log-count branch partially restores classifier/generative alignment
    - optional IAF improves posterior flexibility without leaving the M2 family
    """

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
        fp_recon_distribution: str = 'bernoulli',
        classifier_fp_view: str = 'binary_presence',
        classifier_use_descriptor_branch: bool = True,
        classifier_use_logcount_branch: bool = False,
        classifier_logcount_branch_dim: int | None = None,
        iaf_num_flows: int = 0,
        iaf_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.include_protox = bool(dims.include_protox)
        self.binary_tasks = tuple(dims.binary_tasks)
        self.d_x = int(dims.d_fp + dims.d_desc)

        self.fp_recon_distribution = str(fp_recon_distribution).strip().lower()
        if self.fp_recon_distribution not in {'bernoulli', 'gaussian'}:
            raise ValueError(f'unknown fingerprint reconstruction distribution: {self.fp_recon_distribution}')

        self.classifier_fp_view = str(classifier_fp_view).strip().lower()
        if self.classifier_fp_view not in {'binary_presence', 'logcount'}:
            raise ValueError(f'unknown classifier_fp_view: {self.classifier_fp_view}')
        self.classifier_use_descriptor_branch = bool(classifier_use_descriptor_branch)
        self.classifier_use_logcount_branch = bool(classifier_use_logcount_branch)
        self.ifm_include_raw = bool(ifm_include_raw)
        self.ifm_apply_to_encoder = bool(ifm_apply_to_encoder)

        clf_hidden = tuple(int(h) for h in clf_hidden)
        if len(clf_hidden) == 0:
            raise ValueError('clf_hidden must contain at least one hidden width')
        branch_dim = int(clf_hidden[0])
        fusion_hidden = tuple(int(h) for h in clf_hidden[1:])
        fusion_out = int(clf_hidden[-1])
        logcount_branch_dim = int(classifier_logcount_branch_dim or max(branch_dim // 2, 64))

        self.ifm = IFMMapper(dims.d_fp, num_frequencies=int(ifm_num_frequencies), learnable=bool(ifm_learnable), init_std=float(ifm_init_std))

        self.qy_fp_ifm_branch = nn.Linear(self.ifm.out_dim, branch_dim)
        self.qy_fp_raw_branch = nn.Linear(dims.d_fp, branch_dim) if self.ifm_include_raw else None
        self.qy_desc_branch = nn.Linear(dims.d_desc, branch_dim) if self.classifier_use_descriptor_branch else None
        self.qy_fp_logcount_branch = nn.Linear(dims.d_fp, logcount_branch_dim) if self.classifier_use_logcount_branch else None

        fusion_in = branch_dim
        if self.qy_fp_raw_branch is not None:
            fusion_in += branch_dim
        if self.qy_desc_branch is not None:
            fusion_in += branch_dim
        if self.qy_fp_logcount_branch is not None:
            fusion_in += logcount_branch_dim
        self.qy_fusion = FusionMLP(fusion_in, fusion_hidden, fusion_out, dropout=dropout)

        self.qy_protox = nn.Linear(fusion_out, dims.protox_K) if self.include_protox else None
        self.qy_binary = nn.ModuleDict({task: nn.Linear(fusion_out, 1) for task in self.binary_tasks})

        d_qz = self.d_x if not self.ifm_apply_to_encoder else (self.ifm.out_dim + dims.d_desc)
        self.qz = mlp(d_qz + dims.y_dim, enc_hidden, 2 * dims.z_dim, dropout=dropout)
        self.iaf_num_flows = int(iaf_num_flows)
        self.iaf_hidden_dim = int(iaf_hidden_dim)
        self.iaf = IAFStack(dims.z_dim, d_qz + dims.y_dim, self.iaf_num_flows, self.iaf_hidden_dim) if self.iaf_num_flows > 0 else None
        self.px_trunk = mlp(dims.z_dim + dims.y_dim, dec_hidden, 512, dropout=dropout)
        if self.fp_recon_distribution == 'bernoulli':
            self.px_fp = nn.Linear(512, dims.d_fp)
            self.px_fp_mu = None
            self.px_fp_lv = None
        else:
            self.px_fp = None
            self.px_fp_mu = nn.Linear(512, dims.d_fp)
            self.px_fp_lv = nn.Linear(512, dims.d_fp)
        self.px_desc_mu = nn.Linear(512, dims.d_desc)
        self.px_desc_lv = nn.Linear(512, dims.d_desc)

    @property
    def uses_iaf(self) -> bool:
        return self.iaf is not None

    def concat_x(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_fp, x_desc], dim=1)

    def _classifier_fp_tensor(self, x_fp: torch.Tensor) -> torch.Tensor:
        if self.classifier_fp_view == 'binary_presence':
            return (x_fp > 0).to(dtype=x_fp.dtype)
        return x_fp

    def map_qy_x(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> torch.Tensor:
        fp_classifier = self._classifier_fp_tensor(x_fp)
        parts = [self.ifm(fp_classifier)]
        if self.ifm_include_raw:
            parts.append(fp_classifier)
        if self.classifier_use_descriptor_branch:
            parts.append(x_desc)
        if self.classifier_use_logcount_branch:
            parts.append(x_fp)
        return torch.cat(parts, dim=1)

    def map_qz_x(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> torch.Tensor:
        if self.ifm_apply_to_encoder:
            return self.map_qy_x(x_fp, x_desc)
        return self.concat_x(x_fp, x_desc)

    def classifier_forward(self, x_fp: torch.Tensor, x_desc: torch.Tensor, *, classifier_ablation: str = 'full', return_details: bool = False):
        mode = str(classifier_ablation).strip().lower()
        fp_classifier = self._classifier_fp_tensor(x_fp)
        fp_ifm_feat = self.ifm(fp_classifier)
        fp_ifm_branch = F.relu(self.qy_fp_ifm_branch(fp_ifm_feat))
        fp_raw_branch = F.relu(self.qy_fp_raw_branch(fp_classifier)) if self.qy_fp_raw_branch is not None else None
        desc_branch = F.relu(self.qy_desc_branch(x_desc)) if self.qy_desc_branch is not None else None
        fp_logcount_branch = F.relu(self.qy_fp_logcount_branch(x_fp)) if self.qy_fp_logcount_branch is not None else None

        if mode == 'no_ifm':
            fp_ifm_branch = torch.zeros_like(fp_ifm_branch)
        elif mode == 'no_fp_raw' and fp_raw_branch is not None:
            fp_raw_branch = torch.zeros_like(fp_raw_branch)
        elif mode == 'no_desc' and desc_branch is not None:
            desc_branch = torch.zeros_like(desc_branch)
        elif mode == 'no_logcount' and fp_logcount_branch is not None:
            fp_logcount_branch = torch.zeros_like(fp_logcount_branch)
        elif mode == 'fp_only' and desc_branch is not None:
            desc_branch = torch.zeros_like(desc_branch)
        elif mode == 'desc_only':
            fp_ifm_branch = torch.zeros_like(fp_ifm_branch)
            if fp_raw_branch is not None:
                fp_raw_branch = torch.zeros_like(fp_raw_branch)
            if fp_logcount_branch is not None:
                fp_logcount_branch = torch.zeros_like(fp_logcount_branch)
        elif mode not in {'full', '', 'none'}:
            raise ValueError(f'unknown classifier_ablation: {classifier_ablation}')

        parts = [fp_ifm_branch]
        if fp_raw_branch is not None:
            parts.append(fp_raw_branch)
        if desc_branch is not None:
            parts.append(desc_branch)
        if fp_logcount_branch is not None:
            parts.append(fp_logcount_branch)
        fusion_in = torch.cat(parts, dim=1)
        fusion_rep, fusion_hidden = self.qy_fusion(fusion_in, return_hidden=True)

        logits: Dict[str, torch.Tensor] = {task: head(fusion_rep).squeeze(1) for task, head in self.qy_binary.items()}
        if self.include_protox:
            assert self.qy_protox is not None
            logits['protox'] = self.qy_protox(fusion_rep)

        if not return_details:
            return logits
        details = {
            'classifier_fp_view': fp_classifier,
            'fp_ifm_feat': fp_ifm_feat,
            'fp_ifm_branch': fp_ifm_branch,
            'fp_raw_branch': fp_raw_branch,
            'desc_branch': desc_branch,
            'fp_logcount_branch': fp_logcount_branch,
            'fusion_input': fusion_in,
            'fusion_rep': fusion_rep,
        }
        for i, act in enumerate(fusion_hidden):
            details[f'fusion_hidden_{i}'] = act
        return logits, details

    def q_y_logits(self, x_fp: torch.Tensor, x_desc: torch.Tensor, *, classifier_ablation: str = 'full') -> Dict[str, torch.Tensor]:
        return self.classifier_forward(x_fp, x_desc, classifier_ablation=classifier_ablation, return_details=False)

    def encode_z_params(self, x_fp: torch.Tensor, x_desc: torch.Tensor, y_struct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        enc_input = torch.cat([self.map_qz_x(x_fp, x_desc), y_struct], dim=1)
        h = self.qz(enc_input)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar, enc_input if self.uses_iaf else None

    def encode_z(self, x_fp: torch.Tensor, x_desc: torch.Tensor, y_struct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar, _ = self.encode_z_params(x_fp, x_desc, y_struct)
        return mu, logvar

    def apply_flow(self, z0: torch.Tensor, flow_context: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.uses_iaf:
            return z0, torch.zeros(z0.shape[0], device=z0.device, dtype=z0.dtype)
        if flow_context is None:
            raise ValueError('flow_context is required when IAF is enabled')
        assert self.iaf is not None
        return self.iaf(z0, flow_context)

    def decode_x(self, z: torch.Tensor, y_struct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        h = self.px_trunk(torch.cat([z, y_struct], dim=1))
        if self.fp_recon_distribution == 'bernoulli':
            assert self.px_fp is not None
            fp_param = self.px_fp(h)
            fp_logvar = None
        else:
            assert self.px_fp_mu is not None and self.px_fp_lv is not None
            fp_param = self.px_fp_mu(h)
            fp_logvar = self.px_fp_lv(h).clamp(min=-8.0, max=4.0)
        desc_mu = self.px_desc_mu(h)
        desc_logvar = self.px_desc_lv(h).clamp(min=-8.0, max=4.0)
        return fp_param, fp_logvar, desc_mu, desc_logvar

    def inference_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend(list(self.ifm.parameters()))
        params.extend(list(self.qy_fp_ifm_branch.parameters()))
        if self.qy_fp_raw_branch is not None:
            params.extend(list(self.qy_fp_raw_branch.parameters()))
        if self.qy_desc_branch is not None:
            params.extend(list(self.qy_desc_branch.parameters()))
        if self.qy_fp_logcount_branch is not None:
            params.extend(list(self.qy_fp_logcount_branch.parameters()))
        params.extend(list(self.qy_fusion.parameters()))
        params.extend(list(self.qy_binary.parameters()))
        if self.qy_protox is not None:
            params.extend(list(self.qy_protox.parameters()))
        params.extend(list(self.qz.parameters()))
        if self.iaf is not None:
            params.extend(list(self.iaf.parameters()))
        return params

    def generative_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend(list(self.px_trunk.parameters()))
        if self.px_fp is not None:
            params.extend(list(self.px_fp.parameters()))
        if self.px_fp_mu is not None and self.px_fp_lv is not None:
            params.extend(list(self.px_fp_mu.parameters()))
            params.extend(list(self.px_fp_lv.parameters()))
        params.extend(list(self.px_desc_mu.parameters()))
        params.extend(list(self.px_desc_lv.parameters()))
        return params

    def diagnostic_parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        groups = {
            'qy_fp_ifm': list(self.ifm.parameters()) + list(self.qy_fp_ifm_branch.parameters()),
            'qy_fp_raw': list(self.qy_fp_raw_branch.parameters()) if self.qy_fp_raw_branch is not None else [],
            'qy_desc': list(self.qy_desc_branch.parameters()) if self.qy_desc_branch is not None else [],
            'qy_fp_logcount': list(self.qy_fp_logcount_branch.parameters()) if self.qy_fp_logcount_branch is not None else [],
            'qy_fusion': list(self.qy_fusion.parameters()) + list(self.qy_binary.parameters()) + (list(self.qy_protox.parameters()) if self.qy_protox is not None else []),
            'encoder': list(self.qz.parameters()),
            'iaf': list(self.iaf.parameters()) if self.iaf is not None else [],
            'decoder': list(self.px_trunk.parameters()) + (list(self.px_fp.parameters()) if self.px_fp is not None else []) + ((list(self.px_fp_mu.parameters()) + list(self.px_fp_lv.parameters())) if (self.px_fp_mu is not None and self.px_fp_lv is not None) else []) + list(self.px_desc_mu.parameters()) + list(self.px_desc_lv.parameters()),
        }
        return groups


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
            raise ValueError('protox tensor required when include_protox=True')
        parts.append(F.one_hot(protox.long(), num_classes=protox_K).float())
    for task in binary_task_order:
        parts.append(binary_values[task].float().unsqueeze(1))
    return torch.cat(parts, dim=1)
