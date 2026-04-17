# src/training/loops.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.m2 import M2VAE
from .losses import elbo_labeled, loss_unlabeled, loss_supervised_ce
from .metrics import classification_metrics
from .checkpoints import save_checkpoint


def _cycle(dl: DataLoader):
    while True:
        for batch in dl:
            yield batch


@torch.no_grad()
def evaluate_classifier(model: M2VAE, dl: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ys = []
    yp = []
    for b in dl:
        x_fp = b["x_fp"].to(device)
        x_desc = b["x_desc"].to(device)
        y = b["y"].to(device)

        logits = model.q_y_logits(x_fp, x_desc)
        pred = torch.argmax(logits, dim=1)

        ys.append(y.detach().cpu().numpy())
        yp.append(pred.detach().cpu().numpy())

    if len(ys) == 0:
        return {"acc": float("nan"), "bal_acc": float("nan"), "f1_macro": float("nan"), "f1_weighted": float("nan")}

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yp, axis=0)
    return classification_metrics(y_true, y_pred)


@dataclass
class TrainResult:
    best_epoch: int
    best_val_f1: float
    last_epoch: int


def train_m2(
    *,
    model: M2VAE,
    labeled_train: DataLoader,
    unlabeled_train: DataLoader,
    val_labeled: DataLoader,
    device: str,
    run_dir,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    alpha_ce: float = 1.0,
    lambda_unl: float = 1.0,
    grad_clip: float = 5.0,
    patience: int = 10,
    logger=None,
) -> TrainResult:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    has_unl = len(unlabeled_train.dataset) > 0

    lab_iter = _cycle(labeled_train)
    unl_iter = _cycle(unlabeled_train) if has_unl else None

    steps_per_epoch = max(len(labeled_train), len(unlabeled_train)) if has_unl else len(labeled_train)

    best_val = -1.0
    best_epoch = -1
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}/{epochs}", leave=False)

        for _ in pbar:
            b_lab = next(lab_iter)
            x_fp = b_lab["x_fp"].to(device)
            x_desc = b_lab["x_desc"].to(device)
            y = b_lab["y"].to(device)

            loss_elbo, info_elbo = elbo_labeled(model, x_fp=x_fp, x_desc=x_desc, y=y)
            loss_ce, info_ce = loss_supervised_ce(model, x_fp=x_fp, x_desc=x_desc, y=y)

            loss = loss_elbo + alpha_ce * loss_ce

            info_unl = {}
            if has_unl and unl_iter is not None:
                b_unl = next(unl_iter)
                xu_fp = b_unl["x_fp"].to(device)
                xu_desc = b_unl["x_desc"].to(device)
                loss_unl, info_unl = loss_unlabeled(model, x_fp=xu_fp, x_desc=xu_desc)
                loss = loss + lambda_unl * loss_unl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            if logger:
                logger.log_metrics(
                    {
                        "epoch": epoch,
                        "loss": float(loss.detach().cpu()),
                        **info_elbo,
                        **info_ce,
                        **info_unl,
                    }
                )

        # epoch eval
        val = evaluate_classifier(model, val_labeled, device)
        val_f1 = float(val.get("f1_macro", float("nan")))

        if logger:
            logger.log(f"epoch {epoch}: val={val}")
            logger.log_metrics({"epoch": epoch, "val_acc": val["acc"], "val_bal_acc": val["bal_acc"], "val_f1_macro": val["f1_macro"], "val_f1_weighted": val["f1_weighted"]})

        # checkpointing
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model=model, optimizer=opt, epoch=epoch, extra={"val": val})

        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = epoch
            bad = 0
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, optimizer=opt, epoch=epoch, extra={"val": val})
        else:
            bad += 1
            if bad >= patience:
                if logger:
                    logger.log(f"early stop at epoch {epoch} (best epoch {best_epoch}, best f1 {best_val:.4f})")
                break

    return TrainResult(best_epoch=best_epoch, best_val_f1=best_val, last_epoch=epoch)
