from __future__ import annotations

from dataclasses import dataclass
import json
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.m2 import StructuredM2VAE, build_y_struct, reparameterize
from src.training.m2_metrics import binary_metrics, confusion_from_probs, multiclass_metrics
from .checkpoints import save_checkpoint


@dataclass(frozen=True)
class StructuredPriors:
    binary_pos: Dict[str, float]
    protox: torch.Tensor | None = None


@dataclass
class EvalOutputs:
    metrics: Dict[str, Dict[str, float]]
    predictions: List[Dict]
    confusion: Dict[str, np.ndarray]
    arrays: Dict[str, Dict[str, np.ndarray]]


@dataclass
class TrainResult:
    best_epoch: int
    best_score: float
    last_epoch: int
    checkpoint_used: str
    best_pretrain_epoch: int = -1
    best_pretrain_score: float = float('nan')
    best_joint_epoch: int = -1
    best_joint_score: float = float('nan')
    pretrain_epochs_used: int = 0


def _combo_tensors(device: str, priors: StructuredPriors, *, binary_tasks: tuple[str, ...], include_protox: bool) -> Dict[str, torch.Tensor]:
    binary_matrix = torch.tensor(list(product([0, 1], repeat=len(binary_tasks))), dtype=torch.long, device=device)
    protox_range = range(6) if include_protox else [None]
    rows = []
    log_prior = []
    protox_ix = []
    protox_raw = []
    p_protox = priors.protox.to(device) if (include_protox and priors.protox is not None) else None
    p_binary = {
        task: torch.tensor([1.0 - float(priors.binary_pos[task]), float(priors.binary_pos[task])], dtype=torch.float32, device=device)
        for task in binary_tasks
    }
    for p in protox_range:
        for combo in binary_matrix.tolist():
            combo_dict = {task: int(combo[i]) for i, task in enumerate(binary_tasks)}
            lp = torch.tensor(0.0, dtype=torch.float32, device=device)
            for task, val in combo_dict.items():
                lp = lp + torch.log(torch.clamp(p_binary[task][val], min=1e-8))
            if include_protox:
                assert p is not None and p_protox is not None
                protox_ix.append(int(p))
                protox_raw.append(int(p) + 1)
                lp = lp + torch.log(torch.clamp(p_protox[int(p)], min=1e-8))
            rows.append(combo)
            log_prior.append(lp)
    binary_rows = torch.tensor(rows, dtype=torch.long, device=device)
    y_parts = []
    if include_protox:
        y_parts.append(F.one_hot(torch.tensor(protox_ix, dtype=torch.long, device=device), num_classes=6).float())
    for i, _task in enumerate(binary_tasks):
        y_parts.append(binary_rows[:, i:i + 1].float())
    out = {
        'binary_matrix': binary_rows,
        'y_struct': torch.cat(y_parts, dim=1),
        'log_prior': torch.stack(log_prior),
    }
    if include_protox:
        out['protox_ix'] = torch.tensor(protox_ix, dtype=torch.long, device=device)
        out['protox_raw'] = torch.tensor(protox_raw, dtype=torch.long, device=device)
    return out


def _bernoulli_loglik_with_logits(x: torch.Tensor, logits: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction='none')
    ll = -torch.sum(bce, dim=1)
    if normalize:
        ll = ll / float(x.shape[1])
    return ll


def _gaussian_loglik(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    two_pi = torch.tensor(2.0 * np.pi, device=x.device, dtype=x.dtype)
    ll = -0.5 * torch.sum(logvar + (x - mu).pow(2) / torch.exp(logvar) + torch.log(two_pi), dim=1)
    if normalize:
        ll = ll / float(x.shape[1])
    return ll


def _kl_standard_normal_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


def _apply_free_bits_exact(kl_expected_per_sample_dim: torch.Tensor, free_bits_per_dim: float) -> tuple[torch.Tensor, torch.Tensor]:
    kl_mean_per_dim = kl_expected_per_sample_dim.mean(dim=0)
    if float(free_bits_per_dim) > 0.0:
        floor = torch.full_like(kl_mean_per_dim, float(free_bits_per_dim))
        kl_obj_per_dim = torch.maximum(floor, kl_mean_per_dim)
    else:
        kl_obj_per_dim = kl_mean_per_dim
    return kl_mean_per_dim, kl_obj_per_dim


def _task_supervised_losses(
    logits: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    binary_tasks: tuple[str, ...],
    include_protox: bool,
    pos_weights: Dict[str, float | None],
) -> Dict[str, torch.Tensor]:
    device = next(iter(logits.values())).device
    losses: Dict[str, torch.Tensor] = {}

    if include_protox:
        mask_protox = batch['mask_protox'].to(device) > 0.5
        y_protox = batch['y_protox'].to(device)
        if mask_protox.any():
            tgt = (y_protox[mask_protox] - 1).long()
            losses['protox'] = F.cross_entropy(logits['protox'][mask_protox], tgt)
        else:
            losses['protox'] = torch.tensor(0.0, device=device)

    for task in binary_tasks:
        mask = batch[f'mask_{task}'].to(device) > 0.5
        y = batch[f'y_{task}'].to(device)
        if mask.any():
            pos_weight = pos_weights.get(task)
            pw = None if pos_weight is None else torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
            losses[task] = F.binary_cross_entropy_with_logits(logits[task][mask], y[mask].float(), pos_weight=pw)
        else:
            losses[task] = torch.tensor(0.0, device=device)

    return losses


def structured_m2_batch_loss(
    model: StructuredM2VAE,
    batch: Dict[str, torch.Tensor],
    *,
    priors: StructuredPriors,
    device: str,
    beta_kl: float = 1.0,
    lambda_recon: float = 1.0,
    gen_weight: float = 1.0,
    alpha_protox: float = 1.0,
    alpha_binary: Dict[str, float] | None = None,
    normalize_recon: bool = False,
    pos_weights: Dict[str, float | None] | None = None,
    free_bits_per_dim: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    include_protox = bool(model.include_protox)
    binary_tasks = tuple(model.binary_tasks)
    alpha_binary = dict(alpha_binary or {})
    pos_weights = dict(pos_weights or {})

    x_fp = batch['x_fp'].to(device)
    x_desc = batch['x_desc'].to(device)
    logits = model.q_y_logits(x_fp, x_desc)
    q_pos = {task: torch.sigmoid(logits[task]) for task in binary_tasks}

    if include_protox:
        y_protox = batch['y_protox'].to(device)
        mask_protox = batch['mask_protox'].to(device)
        q_protox = torch.softmax(logits['protox'], dim=1)
    else:
        y_protox = None
        mask_protox = None
        q_protox = None

    y_true = {task: batch[f'y_{task}'].to(device) for task in binary_tasks}
    mask = {task: batch[f'mask_{task}'].to(device) for task in binary_tasks}

    combos = _combo_tensors(device, priors, binary_tasks=binary_tasks, include_protox=include_protox)
    B = x_fp.shape[0]
    C = combos['binary_matrix'].shape[0]

    x_fp_rep = x_fp.unsqueeze(1).repeat(1, C, 1).reshape(B * C, -1)
    x_desc_rep = x_desc.unsqueeze(1).repeat(1, C, 1).reshape(B * C, -1)
    y_struct_rep = combos['y_struct'].unsqueeze(0).repeat(B, 1, 1).reshape(B * C, -1)

    mu, logvar = model.encode_z(x_fp_rep, x_desc_rep, y_struct_rep)
    z = reparameterize(mu, logvar)
    fp_logits, desc_mu, desc_logvar = model.decode_x(z, y_struct_rep)
    ll_fp = _bernoulli_loglik_with_logits(x_fp_rep, fp_logits, normalize=normalize_recon)
    ll_desc = _gaussian_loglik(x_desc_rep, desc_mu, desc_logvar, normalize=normalize_recon)
    kl_per_dim = _kl_standard_normal_per_dim(mu, logvar).reshape(B, C, -1)
    recon_plus_prior = (lambda_recon * (ll_fp + ll_desc) + combos['log_prior'].repeat(B)).reshape(B, C)

    consistent = torch.ones((B, C), dtype=torch.bool, device=device)
    if include_protox:
        assert mask_protox is not None and y_protox is not None
        consistent &= ((mask_protox.unsqueeze(1) < 0.5) | (combos['protox_raw'].unsqueeze(0) == y_protox.unsqueeze(1)))
    for i, task in enumerate(binary_tasks):
        consistent &= ((mask[task].unsqueeze(1) < 0.5) | (combos['binary_matrix'][:, i].unsqueeze(0) == y_true[task].unsqueeze(1)))

    weights = torch.ones((B, C), dtype=torch.float32, device=device)
    if include_protox:
        missing_protox = mask_protox < 0.5
        if missing_protox.any():
            assert q_protox is not None
            weights[missing_protox] *= q_protox[missing_protox][:, combos['protox_ix']]
    else:
        missing_protox = None

    entropy = torch.zeros((B,), dtype=torch.float32, device=device)
    per_task_entropy = {}
    if include_protox and missing_protox is not None and missing_protox.any():
        ent = -torch.sum(q_protox * torch.log(torch.clamp(q_protox, min=1e-8)), dim=1)
        entropy = entropy + torch.where(missing_protox, ent, torch.zeros_like(entropy))
        per_task_entropy['protox'] = float(torch.mean(torch.where(missing_protox, ent, torch.zeros_like(entropy))).detach().cpu())
    for i, task in enumerate(binary_tasks):
        missing = mask[task] < 0.5
        probs = torch.where(combos['binary_matrix'][:, i].unsqueeze(0) == 1, q_pos[task].unsqueeze(1), 1.0 - q_pos[task].unsqueeze(1))
        if missing.any():
            weights[missing] *= probs[missing]
            ent = -(q_pos[task] * torch.log(torch.clamp(q_pos[task], min=1e-8)) + (1 - q_pos[task]) * torch.log(torch.clamp(1 - q_pos[task], min=1e-8)))
            entropy = entropy + torch.where(missing, ent, torch.zeros_like(entropy))
            per_task_entropy[task] = float(torch.mean(torch.where(missing, ent, torch.zeros_like(entropy))).detach().cpu())
        else:
            per_task_entropy[task] = 0.0

    weights = weights * consistent.float()
    weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)

    expected_recon_prior = torch.sum(weights * recon_plus_prior, dim=1)
    expected_kl_per_sample_dim = torch.sum(weights.unsqueeze(-1) * kl_per_dim, dim=1)
    kl_mean_per_dim, kl_obj_per_dim = _apply_free_bits_exact(expected_kl_per_sample_dim, free_bits_per_dim)
    kl_penalty = torch.sum(kl_obj_per_dim)
    expected_elbo = expected_recon_prior - beta_kl * torch.sum(expected_kl_per_sample_dim, dim=1)
    gen_loss = -(torch.mean(expected_recon_prior + entropy) - beta_kl * kl_penalty)

    sup = _task_supervised_losses(
        logits,
        batch,
        binary_tasks=binary_tasks,
        include_protox=include_protox,
        pos_weights=pos_weights,
    )
    loss = float(gen_weight) * gen_loss
    for task in binary_tasks:
        loss = loss + float(alpha_binary.get(task, 1.0)) * sup[task]
    if include_protox:
        loss = loss + float(alpha_protox) * sup['protox']

    info = {
        'loss': float(loss.detach().cpu()),
        'loss_gen': float(gen_loss.detach().cpu()),
        'expected_elbo': float(torch.mean(expected_elbo).detach().cpu()),
        'expected_recon_prior': float(torch.mean(expected_recon_prior).detach().cpu()),
        'expected_kl': float(torch.sum(kl_mean_per_dim).detach().cpu()),
        'expected_kl_objective': float(torch.sum(kl_obj_per_dim).detach().cpu()),
        'entropy_qy': float(torch.mean(entropy).detach().cpu()),
        'beta_kl': float(beta_kl),
        'gen_weight': float(gen_weight),
        'free_bits_per_dim': float(free_bits_per_dim),
    }
    for task in binary_tasks:
        info[f'loss_{task}'] = float(sup[task].detach().cpu())
        info[f'entropy_{task}'] = float(per_task_entropy.get(task, 0.0))
        info[f'mean_p_{task}'] = float(torch.mean(q_pos[task]).detach().cpu())
    if include_protox:
        info['loss_protox'] = float(sup['protox'].detach().cpu())
    return loss, info


@torch.no_grad()
def evaluate_structured(
    model: StructuredM2VAE,
    dl: DataLoader,
    device: str,
    priors: StructuredPriors,
    *,
    thresholds: Dict[str, float] | None = None,
) -> EvalOutputs:
    del priors
    thresholds = thresholds or {}
    include_protox = bool(model.include_protox)
    binary_tasks = tuple(model.binary_tasks)

    model.eval()
    protox_true = []
    protox_probs = []
    true_store = {task: [] for task in binary_tasks}
    prob_store = {task: [] for task in binary_tasks}
    preds_rows: List[Dict] = []

    thr = {task: float(thresholds.get(task, 0.5)) for task in binary_tasks}

    for batch in dl:
        x_fp = batch['x_fp'].to(device)
        x_desc = batch['x_desc'].to(device)
        logits = model.q_y_logits(x_fp, x_desc)
        prob_map = {task: torch.sigmoid(logits[task]) for task in binary_tasks}
        pred_map = {task: (prob_map[task] >= thr[task]).long() for task in binary_tasks}

        if include_protox:
            p_protox = torch.softmax(logits['protox'], dim=1)
            pred_protox = torch.argmax(p_protox, dim=1) + 1
            mask_protox = batch['mask_protox'].to(device) > 0.5
            y_protox_dev = batch['y_protox'].to(device)
            y_protox_for_z = torch.where(mask_protox, y_protox_dev, pred_protox)
        else:
            p_protox = None
            pred_protox = None
            y_protox_for_z = None
            mask_protox = None

        y_for_z = {}
        for task in binary_tasks:
            mask_task = batch[f'mask_{task}'].to(device) > 0.5
            y_task = batch[f'y_{task}'].to(device)
            y_for_z[task] = torch.where(mask_task, y_task, pred_map[task])
        y_struct = build_y_struct(
            y_for_z,
            binary_task_order=binary_tasks,
            include_protox=include_protox,
            protox=(y_protox_for_z.long() - 1) if include_protox and y_protox_for_z is not None else None,
            protox_K=model.dims.protox_K,
        )
        mu, _ = model.encode_z(x_fp, x_desc, y_struct)
        mu_cpu = mu.detach().cpu()

        batch_masks = {task: batch[f'mask_{task}'].cpu() > 0.5 for task in binary_tasks}
        batch_y = {task: batch[f'y_{task}'].cpu() for task in binary_tasks}
        prob_cpu = {task: prob_map[task].detach().cpu() for task in binary_tasks}
        pred_cpu = {task: pred_map[task].detach().cpu() for task in binary_tasks}

        if include_protox and p_protox is not None and pred_protox is not None:
            p_protox_cpu = p_protox.detach().cpu()
            pred_protox_cpu = pred_protox.detach().cpu()
            batch_y_protox = batch['y_protox'].cpu()
            batch_mask_protox = batch['mask_protox'].cpu() > 0.5
        else:
            p_protox_cpu = None
            pred_protox_cpu = None
            batch_y_protox = None
            batch_mask_protox = None

        for i in range(x_fp.shape[0]):
            ik = batch['inchi_key'][i]
            if isinstance(ik, (bytes, bytearray)):
                iks = ik.decode('ascii', errors='ignore')
            elif hasattr(ik, 'tobytes'):
                iks = ik.tobytes().decode('ascii', errors='ignore')
            else:
                iks = str(ik)
            row = {'inchi_key': iks}
            for task in binary_tasks:
                row[f'y_{task}_true'] = int(batch_y[task][i].item())
                row[f'{task}_known'] = int(batch_masks[task][i].item())
                row[f'{task}_pred'] = int(pred_cpu[task][i].item())
                row[f'p_{task}'] = float(prob_cpu[task][i].item())
            if include_protox and batch_y_protox is not None and batch_mask_protox is not None and pred_protox_cpu is not None and p_protox_cpu is not None:
                row.update({
                    'y_protox_true': int(batch_y_protox[i].item()),
                    'protox_known': int(batch_mask_protox[i].item()),
                    'protox_pred': int(pred_protox_cpu[i].item()),
                })
                for k in range(6):
                    row[f'p_protox_{k+1}'] = float(p_protox_cpu[i, k].item())
            for k in range(mu_cpu.shape[1]):
                row[f'z_mu_{k}'] = float(mu_cpu[i, k].item())
            preds_rows.append(row)

        if include_protox and batch_mask_protox is not None and batch_mask_protox.any() and batch_y_protox is not None and p_protox_cpu is not None:
            protox_true.append(batch_y_protox[batch_mask_protox].numpy())
            protox_probs.append(p_protox_cpu[batch_mask_protox].numpy())
        for task in binary_tasks:
            if batch_masks[task].any():
                true_store[task].append(batch_y[task][batch_masks[task]].numpy())
                prob_store[task].append(prob_cpu[task][batch_masks[task]].numpy())

    metrics: Dict[str, Dict[str, float]] = {}
    confusion: Dict[str, np.ndarray] = {}
    arrays: Dict[str, Dict[str, np.ndarray]] = {}

    if include_protox and protox_true:
        y = np.concatenate(protox_true)
        p = np.concatenate(protox_probs)
        arrays['protox'] = {'y_true': y, 'probs': p}
        metrics['protox'] = multiclass_metrics(y, p)
        confusion['protox'] = confusion_from_probs(y, p, task='protox')

    for task in binary_tasks:
        if true_store[task]:
            y = np.concatenate(true_store[task])
            p = np.concatenate(prob_store[task])
            arrays[task] = {'y_true': y, 'probs': p}
            metrics[task] = binary_metrics(y, p, threshold=thr[task])
            confusion[task] = confusion_from_probs(y, p, task=task, threshold=thr[task])
        else:
            metrics[task] = binary_metrics(np.array([]), np.array([]), threshold=thr[task])
            confusion[task] = np.zeros((0, 0), dtype=int)
            arrays[task] = {'y_true': np.array([]), 'probs': np.array([])}

    return EvalOutputs(metrics=metrics, predictions=preds_rows, confusion=confusion, arrays=arrays)




def _cycle_iter(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _merge_info(info_a: Dict[str, float], info_b: Dict[str, float] | None, unlabeled_weight: float) -> Dict[str, float]:
    out = dict(info_a)
    if info_b:
        for k, v in info_b.items():
            out[f'unlab_{k}'] = float(v)
            out[f'combined_{k}'] = float(info_a.get(k, 0.0) + unlabeled_weight * float(v))
    return out


def _cyclical_beta(joint_epoch: int, *, beta_max: float, cycle_length: int, ramp_ratio: float = 0.5, beta_min: float = 0.0) -> float:
    if cycle_length <= 0:
        return float(beta_max)
    pos = (joint_epoch - 1) % cycle_length
    frac = pos / float(max(cycle_length - 1, 1))
    ramp_ratio = float(max(1e-6, min(1.0, ramp_ratio)))
    phase = 1.0 if frac > ramp_ratio else frac / ramp_ratio
    return float(beta_min + phase * (beta_max - beta_min))


def _constant_prediction_guard(eval_out: EvalOutputs, *, binary_tasks: tuple[str, ...], auroc_eps: float = 0.02, prob_std_floor: float = 1e-4) -> Dict[str, Dict[str, float | bool]]:
    report: Dict[str, Dict[str, float | bool]] = {}
    triggered = True
    for task in binary_tasks:
        arr = eval_out.arrays.get(task, {})
        probs = np.asarray(arr.get('probs', np.array([])), dtype=float)
        metrics = eval_out.metrics.get(task, {})
        auroc = float(metrics.get('auroc', np.nan))
        prob_std = float(np.std(probs)) if probs.size else float('nan')
        near_random = (not np.isnan(auroc)) and abs(auroc - 0.5) <= auroc_eps
        near_constant = (not np.isnan(prob_std)) and prob_std <= prob_std_floor
        report[task] = {'auroc': auroc, 'prob_std': prob_std, 'near_random': near_random, 'near_constant': near_constant}
        triggered = triggered and near_random and near_constant
    report['_triggered'] = {'value': bool(triggered)}
    return report

def _score_from_metrics(metrics: Dict[str, Dict[str, float]], *, binary_tasks: tuple[str, ...], score_weights: Dict[str, float] | None = None) -> float:
    vals = []
    weights = []
    score_weights = score_weights or {}
    for task in binary_tasks:
        m = metrics.get(task, {})
        value = m.get('auprc', np.nan)
        if np.isnan(value):
            value = m.get('bal_acc', np.nan)
        if not np.isnan(value):
            vals.append(float(value))
            weights.append(float(score_weights.get(task, 1.0)))
    if not vals:
        return float('nan')
    w = np.asarray(weights, dtype=float)
    v = np.asarray(vals, dtype=float)
    return float(np.sum(w * v) / np.sum(w))


def train_structured_m2(
    *,
    model: StructuredM2VAE,
    train_all: DataLoader,
    train_pretrain_labeled: DataLoader | None,
    train_joint_labeled: DataLoader | None,
    train_unlabeled: DataLoader | None,
    val_all: DataLoader,
    device: str,
    priors: StructuredPriors,
    run_dir: Path,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    beta_kl: float = 1.0,
    lambda_recon: float = 1.0,
    gen_weight: float = 1.0,
    alpha_protox: float = 1.0,
    alpha_binary: Dict[str, float] | None = None,
    score_weights: Dict[str, float] | None = None,
    grad_clip: float = 5.0,
    patience: int = 10,
    normalize_recon: bool = False,
    pos_weights: Dict[str, float | None] | None = None,
    pretrain_supervised_epochs: int = 0,
    pretrain_supervised_epochs_min: int = 0,
    pretrain_supervised_epochs_max: int = 0,
    pretrain_transition_patience: int = 2,
    kl_warmup_epochs: int = 0,
    gen_warmup_epochs: int = 0,
    free_bits_per_dim: float = 0.0,
    joint_unlabeled_weight: float = 0.25,
    joint_steps_mode: str = 'labeled',
    joint_unlabeled_batches_per_step: int = 1,
    beta_schedule: str = 'cyclical',
    beta_cycle_length: int = 12,
    beta_cycle_ramp_ratio: float = 0.5,
    sanity_guard_epoch: int = 2,
    sanity_abort_on_constant_preds: bool = True,
    logger=None,
) -> TrainResult:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1.0
    best_epoch = -1
    best_pretrain_score = -1.0
    best_pretrain_epoch = -1
    best_joint_score = -1.0
    best_joint_epoch = -1
    bad = 0
    binary_tasks = tuple(model.binary_tasks)

    pre_min = int(pretrain_supervised_epochs_min or pretrain_supervised_epochs or 0)
    pre_max = int(pretrain_supervised_epochs_max or pretrain_supervised_epochs or 0)
    if pre_max < pre_min:
        pre_max = pre_min
    in_pretrain = pre_max > 0 and train_pretrain_labeled is not None
    pretrain_bad = 0
    pretrain_epochs_used = 0

    sanity_dir = run_dir / 'sanity'
    sanity_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        joint_epoch = max(1, epoch - pretrain_epochs_used)
        if in_pretrain:
            phase = 'pretrain'
            beta_kl_epoch = 0.0
            gen_weight_epoch = 0.0
            loader_name = 'train_pretrain_labeled'
        else:
            phase = 'joint'
            if str(beta_schedule).lower() == 'cyclical':
                beta_kl_epoch = _cyclical_beta(joint_epoch, beta_max=float(beta_kl), cycle_length=int(beta_cycle_length), ramp_ratio=float(beta_cycle_ramp_ratio), beta_min=0.0)
            else:
                beta_kl_epoch = float(beta_kl) * min(1.0, float(joint_epoch) / float(max(int(kl_warmup_epochs), 1))) if int(kl_warmup_epochs) > 0 else float(beta_kl)
            gen_weight_epoch = float(gen_weight) * min(1.0, float(joint_epoch) / float(max(int(gen_warmup_epochs), 1))) if int(gen_warmup_epochs) > 0 else float(gen_weight)
            loader_name = 'train_joint_alt'

        model.train()
        epoch_sums: Dict[str, float] = {}
        epoch_batches = 0

        if phase == 'pretrain':
            pbar = tqdm(train_pretrain_labeled, desc=f'epoch {epoch}/{epochs}', leave=False)
            for batch in pbar:
                loss, info = structured_m2_batch_loss(model, batch, priors=priors, device=device, beta_kl=beta_kl_epoch, lambda_recon=lambda_recon, gen_weight=gen_weight_epoch, alpha_protox=alpha_protox, alpha_binary=alpha_binary, normalize_recon=normalize_recon, pos_weights=pos_weights, free_bits_per_dim=free_bits_per_dim)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                epoch_batches += 1
                for k, v in info.items():
                    epoch_sums[k] = epoch_sums.get(k, 0.0) + float(v)
                if logger:
                    logger.log_metrics({'kind': 'batch_train', 'phase': phase, 'epoch': epoch, **info})
        else:
            labeled_loader = train_joint_labeled if train_joint_labeled is not None else train_pretrain_labeled
            if labeled_loader is None:
                raise RuntimeError('joint training requires a labeled loader')
            joint_steps_mode_norm = str(joint_steps_mode).lower().strip()
            labeled_steps = len(labeled_loader)
            unlabeled_steps = len(train_unlabeled) if train_unlabeled is not None else 0
            if joint_steps_mode_norm == 'labeled':
                n_steps = labeled_steps
            elif joint_steps_mode_norm == 'unlabeled':
                n_steps = unlabeled_steps if unlabeled_steps > 0 else labeled_steps
            elif joint_steps_mode_norm == 'max':
                n_steps = max(labeled_steps, unlabeled_steps) if unlabeled_steps > 0 else labeled_steps
            elif joint_steps_mode_norm == 'min':
                n_steps = min(labeled_steps, unlabeled_steps) if unlabeled_steps > 0 else labeled_steps
            elif joint_steps_mode_norm == 'all':
                n_steps = len(train_all)
            else:
                raise ValueError(f'unknown joint_steps_mode: {joint_steps_mode}')
            n_steps = max(int(n_steps), 1)
            labeled_iter = _cycle_iter(labeled_loader)
            unlabeled_iter = _cycle_iter(train_unlabeled) if train_unlabeled is not None else None
            n_unlabeled_batches = max(0, int(joint_unlabeled_batches_per_step))
            loader_name = f'train_joint_{joint_steps_mode_norm}'
            pbar = tqdm(range(n_steps), desc=f'epoch {epoch}/{epochs}', leave=False)
            for _ in pbar:
                batch_lab = next(labeled_iter)
                loss_lab, info_lab = structured_m2_batch_loss(model, batch_lab, priors=priors, device=device, beta_kl=beta_kl_epoch, lambda_recon=lambda_recon, gen_weight=gen_weight_epoch, alpha_protox=alpha_protox, alpha_binary=alpha_binary, normalize_recon=normalize_recon, pos_weights=pos_weights, free_bits_per_dim=free_bits_per_dim)
                total_loss = loss_lab
                info_unlab = None
                if unlabeled_iter is not None and float(joint_unlabeled_weight) > 0.0 and n_unlabeled_batches > 0:
                    unlab_infos = []
                    for _u in range(n_unlabeled_batches):
                        batch_unlab = next(unlabeled_iter)
                        loss_unlab, info_unlab_step = structured_m2_batch_loss(model, batch_unlab, priors=priors, device=device, beta_kl=beta_kl_epoch, lambda_recon=lambda_recon, gen_weight=gen_weight_epoch, alpha_protox=alpha_protox, alpha_binary=alpha_binary, normalize_recon=normalize_recon, pos_weights=pos_weights, free_bits_per_dim=free_bits_per_dim)
                        total_loss = total_loss + (float(joint_unlabeled_weight) / float(n_unlabeled_batches)) * loss_unlab
                        unlab_infos.append(info_unlab_step)
                    info_unlab = {}
                    for info_unlab_step in unlab_infos:
                        for k, v in info_unlab_step.items():
                            info_unlab[k] = info_unlab.get(k, 0.0) + float(v) / float(len(unlab_infos))
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                epoch_batches += 1
                merged = _merge_info(info_lab, info_unlab, float(joint_unlabeled_weight))
                merged['loss_total'] = float(total_loss.detach().item())
                merged['joint_epoch_steps'] = float(n_steps)
                merged['joint_unlabeled_batches_per_step'] = float(n_unlabeled_batches)
                for k, v in merged.items():
                    epoch_sums[k] = epoch_sums.get(k, 0.0) + float(v)
                if logger:
                    logger.log_metrics({'kind': 'batch_train', 'phase': phase, 'epoch': epoch, **merged})

        epoch_train = {k: (v / max(epoch_batches, 1)) for k, v in epoch_sums.items()}
        val_out = evaluate_structured(model, val_all, device, priors)
        score = _score_from_metrics(val_out.metrics, binary_tasks=binary_tasks, score_weights=score_weights)
        guard_report = _constant_prediction_guard(val_out, binary_tasks=binary_tasks)
        (sanity_dir / f'guard_epoch_{epoch:03d}.json').write_text(json.dumps({'epoch': epoch, 'phase': phase, 'report': guard_report}, indent=2), encoding='utf-8')

        if logger:
            logger.log(f'epoch {epoch}: phase={phase} loader={loader_name} beta_kl={beta_kl_epoch:.4f} gen_weight={float(gen_weight_epoch):.4f} free_bits_per_dim={float(free_bits_per_dim):.4f} train={epoch_train} val={val_out.metrics}')
            logger.log_metrics({'kind': 'epoch_train', 'phase': phase, 'epoch': epoch, 'loader': loader_name, **{f'train_{k}': float(v) for k, v in epoch_train.items()}})
            payload = {'kind': 'epoch_val', 'phase': phase, 'epoch': epoch, 'beta_kl_epoch': float(beta_kl_epoch), 'gen_weight_epoch': float(gen_weight_epoch), 'free_bits_per_dim_epoch': float(free_bits_per_dim), 'val_score': score}
            for task in binary_tasks:
                payload[f'val_{task}_auprc'] = val_out.metrics[task].get('auprc')
                payload[f'val_{task}_bal_acc'] = val_out.metrics[task].get('bal_acc')
                payload[f'val_{task}_brier'] = val_out.metrics[task].get('brier')
                payload[f'val_{task}_ece10'] = val_out.metrics[task].get('ece10')
                payload[f'guard_{task}_prob_std'] = guard_report[task].get('prob_std')
            logger.log_metrics(payload)

        save_checkpoint(run_dir / 'checkpoints' / 'last.pt', model=model, optimizer=opt, epoch=epoch, extra={'val': val_out.metrics, 'score': score, 'phase': phase})
        if phase == 'pretrain' and not np.isnan(score) and score > best_pretrain_score:
            best_pretrain_score = score
            best_pretrain_epoch = epoch
            save_checkpoint(run_dir / 'checkpoints' / 'best_pretrain.pt', model=model, optimizer=opt, epoch=epoch, extra={'val': val_out.metrics, 'score': score, 'phase': phase})
        if phase == 'joint' and not np.isnan(score) and score > best_joint_score:
            best_joint_score = score
            best_joint_epoch = epoch
            save_checkpoint(run_dir / 'checkpoints' / 'best_joint.pt', model=model, optimizer=opt, epoch=epoch, extra={'val': val_out.metrics, 'score': score, 'phase': phase})
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            save_checkpoint(run_dir / 'checkpoints' / 'best_overall.pt', model=model, optimizer=opt, epoch=epoch, extra={'val': val_out.metrics, 'score': score, 'phase': phase})
        else:
            bad += 1

        if phase == 'pretrain':
            pretrain_epochs_used = epoch
            pretrain_bad = 0 if (not np.isnan(score) and score >= best_pretrain_score) else (pretrain_bad + 1)
            if sanity_abort_on_constant_preds and epoch >= int(sanity_guard_epoch) and guard_report.get('_triggered', {}).get('value', False):
                raise RuntimeError(f'sanity guard triggered at epoch {epoch}: near-constant predictions during pretrain')
            if epoch >= pre_min and (epoch >= pre_max or pretrain_bad >= int(pretrain_transition_patience)):
                in_pretrain = False
                bad = 0
                if logger:
                    logger.log(f'transitioning to joint phase after epoch {epoch}; pretrain best epoch {best_pretrain_epoch}, score {best_pretrain_score:.4f}')
        else:
            if sanity_abort_on_constant_preds and epoch == pretrain_epochs_used + int(sanity_guard_epoch) and guard_report.get('_triggered', {}).get('value', False):
                raise RuntimeError(f'sanity guard triggered at epoch {epoch}: near-constant predictions during joint')
            if bad >= patience:
                if logger:
                    logger.log(f'early stop at epoch {epoch} (best epoch {best_epoch}, best score {best_score:.4f})')
                break

    ckpt_name = 'best_overall.pt' if (run_dir / 'checkpoints' / 'best_overall.pt').exists() else 'last.pt'
    return TrainResult(best_epoch=best_epoch, best_score=best_score, last_epoch=epoch, checkpoint_used=ckpt_name, best_pretrain_epoch=best_pretrain_epoch, best_pretrain_score=best_pretrain_score if best_pretrain_epoch >= 0 else float('nan'), best_joint_epoch=best_joint_epoch, best_joint_score=best_joint_score if best_joint_epoch >= 0 else float('nan'), pretrain_epochs_used=pretrain_epochs_used)

