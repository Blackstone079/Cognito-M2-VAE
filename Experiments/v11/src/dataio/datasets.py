# src/dataio/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FeatureTensors:
    X_fp: torch.Tensor     # (N, D_fp) float32 in {0,1}
    X_desc: torch.Tensor   # (N, D_desc) float32 (scaled)
    y: torch.Tensor        # (N,) int64 with -1 for unlabeled
    split: np.ndarray      # (N,) object array: train/val/test
    inchi_key: np.ndarray  # (N,) object array


class MoleculeDataset(Dataset):
    def __init__(
        self,
        tensors: FeatureTensors,
        indices: np.ndarray,
    ) -> None:
        self.t = tensors
        self.idx = indices.astype(int)

    def __len__(self) -> int:
        return int(len(self.idx))

    def __getitem__(self, i: int) -> Dict:
        j = int(self.idx[i])
        return {
            "x_fp": self.t.X_fp[j],
            "x_desc": self.t.X_desc[j],
            "y": self.t.y[j],
            "inchi_key": self.t.inchi_key[j],
        }
