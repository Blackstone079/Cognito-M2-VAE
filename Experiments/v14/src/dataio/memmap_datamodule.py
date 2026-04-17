# src/dataio/memmap_datamodule.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class MemmapFeatures:
    X_fp: np.ndarray  # memmap-like, (N, D_fp) float32
    X_desc: np.ndarray  # memmap-like, (N, D_desc) float32
    y: np.ndarray  # memmap-like, (N,) int64
    split_code: np.ndarray  # memmap-like, (N,) uint8 (0=train,1=val,2=test)
    inchi_key: np.ndarray  # memmap-like, (N,) 'S27'


def open_memmap_features(dir_path: Path) -> MemmapFeatures:
    dir_path = Path(dir_path)
    X_fp = np.load(dir_path / "X_fp.npy", mmap_mode="r")
    X_desc = np.load(dir_path / "X_desc.npy", mmap_mode="r")
    y = np.load(dir_path / "y.npy", mmap_mode="r")
    split_code = np.load(dir_path / "split_code.npy", mmap_mode="r")
    inchi_key = np.load(dir_path / "inchi_key.npy", mmap_mode="r")
    return MemmapFeatures(X_fp=X_fp, X_desc=X_desc, y=y, split_code=split_code, inchi_key=inchi_key)


class MemmapMoleculeDataset(Dataset):
    def __init__(self, feats: MemmapFeatures, indices: np.ndarray):
        self.f = feats
        self.idx = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.idx))

    def __getitem__(self, i: int) -> Dict:
        j = int(self.idx[i])
        x_fp = torch.from_numpy(self.f.X_fp[j])  # float32 view
        x_desc = torch.from_numpy(self.f.X_desc[j])  # float32 view
        y = torch.from_numpy(self.f.y[j : j + 1]).squeeze(0)
        # decode only if you need it; keep bytes to avoid overhead
        ik = self.f.inchi_key[j]
        return {"x_fp": x_fp, "x_desc": x_desc, "y": y, "inchi_key": ik}


@dataclass(frozen=True)
class DataLoaders:
    labeled_train: DataLoader
    unlabeled_train: DataLoader
    val_labeled: DataLoader
    test_labeled: DataLoader


def _drop_last_for(n: int, batch_size: int) -> bool:
    return bool(n >= batch_size and batch_size > 0)


def build_dataloaders_memmap(
    feats: MemmapFeatures,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoaders:
    sc = np.asarray(feats.split_code)
    tr = np.where(sc == 0)[0]
    va = np.where(sc == 1)[0]
    te = np.where(sc == 2)[0]

    # y is small enough to snapshot for indexing.
    y_np = np.asarray(feats.y)

    tr_lab = tr[y_np[tr] >= 0]
    tr_unlab = tr[y_np[tr] < 0]
    va_lab = va[y_np[va] >= 0]
    te_lab = te[y_np[te] >= 0]

    labeled_train_ds = MemmapMoleculeDataset(feats, tr_lab)
    unlabeled_train_ds = MemmapMoleculeDataset(feats, tr_unlab)
    val_labeled_ds = MemmapMoleculeDataset(feats, va_lab)
    test_labeled_ds = MemmapMoleculeDataset(feats, te_lab)

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
