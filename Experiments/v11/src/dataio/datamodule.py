# src/dataio/datamodule.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import FeatureTensors, MoleculeDataset


def _indices_for_split(split: np.ndarray, split_name: str) -> np.ndarray:
    return np.where(split.astype(str) == split_name)[0]


def make_feature_tensors(features: Dict[str, np.ndarray], device: str = "cpu") -> FeatureTensors:
    X_fp = torch.tensor(features["X_fp"], dtype=torch.float32, device=device)
    X_desc = torch.tensor(features["X_desc"], dtype=torch.float32, device=device)
    y = torch.tensor(features["y"], dtype=torch.int64, device=device)
    split = features["split"]
    inchi_key = features["inchi_key"]
    return FeatureTensors(X_fp=X_fp, X_desc=X_desc, y=y, split=split, inchi_key=inchi_key)


@dataclass(frozen=True)
class DataLoaders:
    labeled_train: DataLoader
    unlabeled_train: DataLoader
    val_labeled: DataLoader
    test_labeled: DataLoader


def _drop_last_for(n: int, batch_size: int) -> bool:
    # Avoid empty DataLoader when n < batch_size.
    return bool(n >= batch_size and batch_size > 0)


def build_dataloaders(
    tensors: FeatureTensors,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoaders:
    split = tensors.split

    tr = _indices_for_split(split, "train")
    va = _indices_for_split(split, "val")
    te = _indices_for_split(split, "test")

    y_np = tensors.y.detach().cpu().numpy()

    tr_lab = tr[y_np[tr] >= 0]
    tr_unlab = tr[y_np[tr] < 0]

    va_lab = va[y_np[va] >= 0]
    te_lab = te[y_np[te] >= 0]

    labeled_train_ds = MoleculeDataset(tensors, tr_lab)
    unlabeled_train_ds = MoleculeDataset(tensors, tr_unlab)
    val_labeled_ds = MoleculeDataset(tensors, va_lab)
    test_labeled_ds = MoleculeDataset(tensors, te_lab)

    labeled_train = DataLoader(
        labeled_train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=_drop_last_for(len(labeled_train_ds), batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    unlabeled_train = DataLoader(
        unlabeled_train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=_drop_last_for(len(unlabeled_train_ds), batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_labeled = DataLoader(
        val_labeled_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_labeled = DataLoader(
        test_labeled_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataLoaders(
        labeled_train=labeled_train,
        unlabeled_train=unlabeled_train,
        val_labeled=val_labeled,
        test_labeled=test_labeled,
    )
