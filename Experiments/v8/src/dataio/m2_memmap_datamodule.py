from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from src.labels.task_registry import PROTOX_TASK, meta_path_from_features_dir, task_alias


@dataclass(frozen=True)
class StructuredMemmapFeatures:
    X_fp: np.ndarray
    X_desc: np.ndarray
    y_protox: np.ndarray
    label_arrays: Dict[str, np.ndarray]
    split_code: np.ndarray
    inchi_key: np.ndarray
    binary_tasks: tuple[str, ...]
    binary_task_aliases: Dict[str, str]
    binary_task_files: Dict[str, str]

    @property
    def y_resp(self) -> np.ndarray:
        return self.label_arrays['respiratory_toxicity']

    @property
    def y_ames(self) -> np.ndarray:
        return self.label_arrays['ames_mutagenic']

    def get_label_array(self, task: str) -> np.ndarray:
        return self.label_arrays[task]


def open_structured_memmap_features(dir_path: Path) -> StructuredMemmapFeatures:
    dir_path = Path(dir_path)
    meta_path = meta_path_from_features_dir(dir_path)
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    binary_tasks = tuple(meta.get('binary_tasks') or ['respiratory_toxicity', 'ames_mutagenic'])
    binary_task_aliases = dict(meta.get('binary_task_aliases') or {task: task_alias(task) for task in binary_tasks})
    binary_task_files = dict(meta.get('binary_task_files') or {task: f"y_{binary_task_aliases[task]}.npy" for task in binary_tasks})
    label_arrays = {task: np.load(dir_path / binary_task_files[task], mmap_mode='r') for task in binary_tasks}
    return StructuredMemmapFeatures(
        X_fp=np.load(dir_path / 'X_fp.npy', mmap_mode='r'),
        X_desc=np.load(dir_path / 'X_desc.npy', mmap_mode='r'),
        y_protox=np.load(dir_path / 'y_protox.npy', mmap_mode='r'),
        label_arrays=label_arrays,
        split_code=np.load(dir_path / 'split_code.npy', mmap_mode='r'),
        inchi_key=np.load(dir_path / 'inchi_key.npy', mmap_mode='r'),
        binary_tasks=binary_tasks,
        binary_task_aliases=binary_task_aliases,
        binary_task_files=binary_task_files,
    )


class StructuredMemmapDataset(Dataset):
    def __init__(self, feats: StructuredMemmapFeatures, indices: np.ndarray):
        self.f = feats
        self.idx = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.idx))

    def __getitem__(self, i: int) -> Dict:
        j = int(self.idx[i])
        x_fp = torch.from_numpy(np.array(self.f.X_fp[j], copy=True))
        x_desc = torch.from_numpy(np.array(self.f.X_desc[j], copy=True))
        y_protox = int(self.f.y_protox[j])
        item = {
            'x_fp': x_fp.float(),
            'x_desc': x_desc.float(),
            'y_protox': torch.tensor(y_protox, dtype=torch.long),
            'mask_protox': torch.tensor(1 if y_protox >= 0 else 0, dtype=torch.float32),
            'inchi_key': self.f.inchi_key[j],
        }
        for task in self.f.binary_tasks:
            alias = self.f.binary_task_aliases[task]
            yv = int(self.f.label_arrays[task][j])
            item[f'y_{alias}'] = torch.tensor(yv, dtype=torch.long)
            item[f'mask_{alias}'] = torch.tensor(1 if yv >= 0 else 0, dtype=torch.float32)
        return item


def _resolve_batch_mix_counts(batch_size: int, train_mix: dict | None, available_aliases: tuple[str, ...] | list[str] | None = None) -> tuple[int, dict[str, int]]:
    mix = train_mix or {}
    aliases = tuple(available_aliases or ())
    targeted: dict[str, int] = {}

    if isinstance(mix.get('n_per_task'), dict):
        targeted = {str(k): int(v) for k, v in mix.get('n_per_task', {}).items() if int(v) > 0}
    elif aliases:
        for alias in aliases:
            key = f'n_{alias}'
            if key in mix and int(mix.get(key, 0)) > 0:
                targeted[alias] = int(mix.get(key, 0))

    if aliases:
        unknown = sorted(set(targeted) - set(aliases))
        if unknown:
            raise ValueError(f'batch mix requested unknown task aliases: {unknown}')

    if 'n_random' in mix:
        n_random = int(mix.get('n_random', 0))
    else:
        n_random = int(batch_size) - int(sum(targeted.values()))

    if n_random < 0:
        raise ValueError('batch mix overfills batch_size')
    if n_random + int(sum(targeted.values())) != int(batch_size):
        raise ValueError('batch mix must sum exactly to batch_size')
    return n_random, targeted



def _draw_without_replacement_cycle(
    rng: np.random.RandomState,
    arr: np.ndarray,
    size: int,
    state: dict[str, object],
) -> np.ndarray:
    if size <= 0 or len(arr) == 0:
        return np.array([], dtype=np.int64)
    perm = state.get('perm')
    cursor = int(state.get('cursor', 0))
    if perm is None or len(perm) != len(arr):
        perm = rng.permutation(arr)
        cursor = 0
    out: list[np.ndarray] = []
    need = int(size)
    while need > 0:
        remain = len(arr) - cursor
        take = min(need, remain)
        if take > 0:
            out.append(np.asarray(perm[cursor:cursor + take], dtype=np.int64))
            cursor += take
            need -= take
        if cursor >= len(arr) and need > 0:
            perm = rng.permutation(arr)
            cursor = 0
    state['perm'] = perm
    state['cursor'] = cursor
    return np.concatenate(out).astype(np.int64) if out else np.array([], dtype=np.int64)


class MixedTaskBatchSampler(BatchSampler):
    def __init__(
        self,
        indices_all: np.ndarray,
        task_indices: Dict[str, np.ndarray],
        *,
        batch_size: int,
        train_mix: dict | None,
        seed: int = 0,
    ) -> None:
        self.indices_all = np.asarray(indices_all, dtype=np.int64)
        self.task_indices = {str(k): np.asarray(v, dtype=np.int64) for k, v in task_indices.items()}
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._epoch = 0
        self.n_random, self.n_per_task = _resolve_batch_mix_counts(self.batch_size, train_mix, tuple(self.task_indices.keys()))
        self.n_batches = max(1, int(math.ceil(len(self.indices_all) / float(self.batch_size))))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.RandomState(self.seed + self._epoch)
        self._epoch += 1

        shuffled_all = rng.permutation(self.indices_all)
        cursor = 0
        task_states: dict[str, dict[str, object]] = {alias: {} for alias in self.n_per_task}
        for _ in range(self.n_batches):
            batch: list[int] = []
            if self.n_random > 0:
                if cursor + self.n_random <= len(shuffled_all):
                    batch.extend(shuffled_all[cursor:cursor + self.n_random].tolist())
                    cursor += self.n_random
                else:
                    remain = shuffled_all[cursor:]
                    batch.extend(remain.tolist())
                    refill_n = self.n_random - len(remain)
                    shuffled_all = rng.permutation(self.indices_all)
                    cursor = 0
                    if refill_n > 0:
                        batch.extend(shuffled_all[cursor:cursor + refill_n].tolist())
                        cursor += refill_n
            for alias, n_take in self.n_per_task.items():
                batch.extend(_draw_without_replacement_cycle(rng, self.task_indices[alias], n_take, task_states[alias]).tolist())
            rng.shuffle(batch)
            yield batch


@dataclass(frozen=True)
class StructuredDataLoaders:
    train_all: DataLoader
    train_pretrain_labeled: DataLoader
    train_eval_all: DataLoader
    val_all: DataLoader
    test_all: DataLoader


def build_structured_dataloaders_memmap(
    feats: StructuredMemmapFeatures,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_mix: dict | None = None,
    seed: int = 0,
) -> StructuredDataLoaders:
    sc = np.asarray(feats.split_code)
    tr = np.where(sc == 0)[0]
    va = np.where(sc == 1)[0]
    te = np.where(sc == 2)[0]

    train_ds = StructuredMemmapDataset(feats, tr)
    val_ds = StructuredMemmapDataset(feats, va)
    test_ds = StructuredMemmapDataset(feats, te)

    train_known_mask = np.zeros(len(tr), dtype=bool)
    y_protox_tr = np.asarray(feats.y_protox)[tr]
    train_known_mask |= (y_protox_tr >= 0)
    for task in feats.binary_tasks:
        y_tr = np.asarray(feats.get_label_array(task))[tr]
        train_known_mask |= (y_tr >= 0)
    tr_labeled_local = np.where(train_known_mask)[0].astype(np.int64)

    if train_mix and bool(train_mix.get('enabled', False)):
        local_all = np.arange(len(tr), dtype=np.int64)
        task_indices = {}
        for task in feats.binary_tasks:
            alias = feats.binary_task_aliases[task]
            y_tr = np.asarray(feats.get_label_array(task))[tr]
            task_indices[alias] = np.where(y_tr >= 0)[0].astype(np.int64)
        batch_sampler = MixedTaskBatchSampler(
            local_all,
            task_indices,
            batch_size=batch_size,
            train_mix=train_mix,
            seed=seed,
        )
        train_all = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_all = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

    train_pretrain_labeled = DataLoader(
        StructuredMemmapDataset(feats, tr[tr_labeled_local]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_eval_all = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
    val_all = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
    test_all = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
    return StructuredDataLoaders(train_all=train_all, train_pretrain_labeled=train_pretrain_labeled, train_eval_all=train_eval_all, val_all=val_all, test_all=test_all)
