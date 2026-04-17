from __future__ import annotations

from pathlib import Path
from typing import Iterable

PROTOX_TASK = 'protox_toxclass'

SPECIAL_ALIASES = {
    'respiratory_toxicity': 'resp',
    'ames_mutagenic': 'ames',
    'dili_classification': 'dili',
}


def task_alias(task: str) -> str:
    task = str(task)
    if task in SPECIAL_ALIASES:
        return SPECIAL_ALIASES[task]
    out = ''.join(ch if ch.isalnum() else '_' for ch in task.lower()).strip('_')
    return out or 'task'


def label_filename(task: str) -> str:
    return f"y_{task_alias(task)}.npy"


def active_binary_tasks_from_cfg(cfg: dict, available_fields: Iterable[str] | None = None) -> list[str]:
    tasks_cfg = cfg.get('tasks', {}) or {}
    allowed = set(available_fields) if available_fields is not None else None
    out = []
    for task, enabled in tasks_cfg.items():
        if not bool(enabled):
            continue
        if task == PROTOX_TASK:
            continue
        if allowed is not None and task not in allowed:
            continue
        out.append(str(task))
    return out


def meta_path_from_features_dir(dir_path: Path) -> Path:
    dir_path = Path(dir_path)
    name = dir_path.name
    if name.endswith('_memmap'):
        meta_name = f'{name}_meta.json'
    else:
        meta_name = f'{name}_meta.json'
    return dir_path.parent / meta_name
