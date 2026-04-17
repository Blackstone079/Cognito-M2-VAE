"""Step 2: build multitask memmap features for M2 and downstream baselines."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.features.featurize_rdkit import DescScaler, descriptor_names, mol_descriptors, mol_from_smiles, morgan_fp  # noqa: E402
from src.labels.task_registry import PROTOX_TASK, active_binary_tasks_from_cfg, label_filename, task_alias  # noqa: E402
from src.utils.logging import get_pipeline_run_dir, get_stage_results_dir, write_manifest  # noqa: E402

SPLIT_TO_CODE = {'train': 0, 'val': 1, 'test': 2}
UNKNOWN = -1


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _to_int_or_unknown(v) -> int:
    if v is None:
        return UNKNOWN
    try:
        if isinstance(v, str) and not v.strip():
            return UNKNOWN
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return UNKNOWN


def _welford_update(n: int, mean: np.ndarray, M2: np.ndarray, x: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n1 = n + 1
    delta = x - mean
    mean = mean + delta / n1
    delta2 = x - mean
    M2 = M2 + delta * delta2
    return n1, mean, M2


def _train_binary_prior(counter: Counter) -> float:
    pos = int(counter.get(1, 0))
    neg = int(counter.get(0, 0))
    den = pos + neg
    return float(pos / den) if den > 0 else 0.5


def _train_binary_counts(counter: Counter) -> dict[str, int]:
    return {'neg': int(counter.get(0, 0)), 'pos': int(counter.get(1, 0))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='pipelines/m2/config.yaml')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)
    paths = Paths()
    feature_cfg = cfg.get('featurizer', {}) or {}
    feature_id = str(feature_cfg.get('feature_id', 'm2_v1'))
    run_dir = get_pipeline_run_dir(paths.RESULTS, feature_id, prefix=str(cfg.get('run', {}).get('prefix', 'm2')))
    stage_dir = get_stage_results_dir(run_dir, '02_build_features')

    tasks_cfg = cfg.get('tasks', {}) or {}
    include_protox = bool(tasks_cfg.get(PROTOX_TASK, False))

    in_csv = paths.INTERIM / 'drug_table_multitask.csv'
    split_path = paths.SPLITS / f'split_scaffold_{feature_id}.csv'
    if not in_csv.exists():
        raise FileNotFoundError(f'Missing {in_csv}. Run pipelines/m2/extract.py first.')
    if not split_path.exists():
        raise FileNotFoundError(f'Missing {split_path}. Run pipelines/m2/split.py first.')

    with in_csv.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
    binary_tasks = active_binary_tasks_from_cfg(cfg, fieldnames)
    binary_aliases = {task: task_alias(task) for task in binary_tasks}
    binary_files = {task: label_filename(task) for task in binary_tasks}

    desc_panel = str(feature_cfg.get('desc_panel', 'desc8'))
    desc_names = list(descriptor_names(desc_panel))
    d_desc = int(len(desc_names))
    fp_bits = int(feature_cfg['fp_bits'])
    fp_radius = int(feature_cfg['fp_radius'])
    fp_mode = 'count' if bool(feature_cfg.get('fp_use_counts', False)) else 'bit'
    fp_count_simulation = bool(feature_cfg.get('fp_count_simulation', False))
    fp_include_chirality = bool(feature_cfg.get('fp_include_chirality', False))
    fp_count_transform = str(feature_cfg.get('fp_count_transform', 'none')).strip().lower()

    split_map: dict[str, int] = {}
    scaffold_meta: dict[str, dict[str, str]] = {}
    with split_path.open('r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            ik = (row.get('inchi_key') or '').strip()
            sp = (row.get('split') or '').strip().lower()
            if ik and sp in SPLIT_TO_CODE:
                split_map[ik] = int(SPLIT_TO_CODE[sp])
                scaffold_meta[ik] = {
                    'scaffold_smiles': row.get('scaffold_smiles') or '',
                    'scaffold_hash': row.get('scaffold_hash') or '',
                }

    N = len(split_map)
    if N == 0:
        raise RuntimeError('Split file contained 0 valid rows.')

    out_dir = paths.FEATURES / f'features_{feature_id}_memmap'
    out_dir.mkdir(parents=True, exist_ok=True)

    X_fp = np.lib.format.open_memmap(out_dir / 'X_fp.npy', mode='w+', dtype=np.float32, shape=(N, fp_bits))
    X_desc = np.lib.format.open_memmap(out_dir / 'X_desc.npy', mode='w+', dtype=np.float32, shape=(N, d_desc))
    y_protox = np.lib.format.open_memmap(out_dir / 'y_protox.npy', mode='w+', dtype=np.int64, shape=(N,))
    label_arrays = {task: np.lib.format.open_memmap(out_dir / binary_files[task], mode='w+', dtype=np.int64, shape=(N,)) for task in binary_tasks}
    split_mm = np.lib.format.open_memmap(out_dir / 'split_code.npy', mode='w+', dtype=np.uint8, shape=(N,))
    ik_mm = np.lib.format.open_memmap(out_dir / 'inchi_key.npy', mode='w+', dtype='S27', shape=(N,))

    n_train = 0
    mean = np.zeros((d_desc,), dtype=np.float64)
    M2 = np.zeros((d_desc,), dtype=np.float64)
    counts = {'protox': defaultdict(Counter)}
    for task in binary_tasks:
        counts[task] = defaultdict(Counter)

    idx = 0
    with in_csv.open('r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            ik = (row.get('inchi_key') or '').strip()
            code = split_map.get(ik)
            if code is None:
                continue
            mol = mol_from_smiles(row.get('smiles'))
            if mol is None:
                continue
            d = mol_descriptors(mol, panel=desc_panel).astype(np.float32)
            X_desc[idx, :] = d
            yp = _to_int_or_unknown(row.get(PROTOX_TASK))
            if yp not in {-1, 1, 2, 3, 4, 5, 6}:
                yp = UNKNOWN
            y_protox[idx] = int(yp)
            for task in binary_tasks:
                yv = _to_int_or_unknown(row.get(task))
                if yv not in {-1, 0, 1}:
                    yv = UNKNOWN
                label_arrays[task][idx] = int(yv)
            split_mm[idx] = int(code)
            ik_mm[idx] = ik.encode('ascii', errors='ignore')[:27]
            sp_name = [k for k, v in SPLIT_TO_CODE.items() if v == int(code)][0]
            counts['protox'][sp_name][int(yp)] += 1
            for task in binary_tasks:
                counts[task][sp_name][int(label_arrays[task][idx])] += 1
            if int(code) == 0:
                n_train, mean, M2 = _welford_update(n_train, mean, M2, d.astype(np.float64))
            idx += 1

    if idx != N:
        raise RuntimeError(f'Expected N={N} valid rows from split file, but wrote idx={idx}.')
    if n_train <= 0:
        raise RuntimeError('No TRAIN rows found; cannot fit descriptor scaler.')

    var = (M2 / max(n_train, 1)).astype(np.float32)
    sd = (np.sqrt(var) + 1e-8).astype(np.float32)
    mu = mean.astype(np.float32)
    scaler = DescScaler(mean=mu, std=sd)

    chunk = 100_000
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        X_desc[s:e, :] = ((X_desc[s:e, :] - scaler.mean) / scaler.std).astype(np.float32)

    idx = 0
    with in_csv.open('r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            ik = (row.get('inchi_key') or '').strip()
            if ik not in split_map:
                continue
            mol = mol_from_smiles(row.get('smiles'))
            if mol is None:
                continue
            X_fp[idx, :] = morgan_fp(
                mol,
                n_bits=fp_bits,
                radius=fp_radius,
                mode=fp_mode,
                count_simulation=fp_count_simulation,
                include_chirality=fp_include_chirality,
                count_transform=fp_count_transform,
            )
            idx += 1
    if idx != N:
        raise RuntimeError(f'Fingerprint pass mismatch: expected N={N}, got idx={idx}.')

    protox_counts_train = counts['protox']['train']
    protox_prior = []
    protox_total = sum(int(protox_counts_train.get(k, 0)) for k in (1, 2, 3, 4, 5, 6))
    for k in (1, 2, 3, 4, 5, 6):
        val = int(protox_counts_train.get(k, 0))
        protox_prior.append(float(val / protox_total) if protox_total > 0 else float(1.0 / 6.0))

    priors_by_task = {task: _train_binary_prior(counts[task]['train']) for task in binary_tasks}
    class_counts_by_task = {task: _train_binary_counts(counts[task]['train']) for task in binary_tasks}

    active_tasks = []
    if include_protox:
        active_tasks.append(PROTOX_TASK)
    active_tasks.extend(binary_tasks)

    legacy_priors = {'protox': protox_prior}
    legacy_train_class_counts = {}
    for task in binary_tasks:
        alias = binary_aliases[task]
        legacy_priors[alias] = priors_by_task[task]
        legacy_train_class_counts[alias] = class_counts_by_task[task]

    meta = {
        'num_rows': N,
        'd_fp': fp_bits,
        'd_desc': d_desc,
        'desc_panel': desc_panel,
        'desc_names': desc_names,
        'fp_radius': fp_radius,
        'fp_mode': fp_mode,
        'fp_count_simulation': fp_count_simulation,
        'fp_include_chirality': fp_include_chirality,
        'fp_count_transform': fp_count_transform,
        'feature_id': feature_id,
        'include_protox': include_protox,
        'protox_classes': [1, 2, 3, 4, 5, 6],
        'tasks': [PROTOX_TASK] + binary_tasks,
        'active_tasks': active_tasks,
        'binary_tasks': binary_tasks,
        'binary_task_aliases': binary_aliases,
        'binary_task_files': binary_files,
        'priors': legacy_priors,
        'priors_by_task': priors_by_task,
        'train_class_counts': legacy_train_class_counts,
        'train_class_counts_by_task': class_counts_by_task,
    }
    meta_path = paths.FEATURES / f'features_{feature_id}_memmap_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    scaler_path = paths.FEATURES / f'desc_scaler_{feature_id}.json'
    scaler_path.write_text(json.dumps({'mean': scaler.mean.tolist(), 'std': scaler.std.tolist()}, indent=2), encoding='utf-8')
    shutil.copy2(meta_path, run_dir / meta_path.name)
    shutil.copy2(scaler_path, run_dir / scaler_path.name)

    split_summary_csv = paths.FEATURES / f'split_summary_{feature_id}.csv'
    with split_summary_csv.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['split', 'total', 'protox_labeled']
        for task in binary_tasks:
            alias = binary_aliases[task]
            fieldnames.extend([f'{alias}_labeled', f'{alias}_neg', f'{alias}_pos'])
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sp in ('train', 'val', 'test'):
            row = {'split': sp, 'total': int(sum(counts['protox'][sp].values())), 'protox_labeled': int(sum(v for k, v in counts['protox'][sp].items() if int(k) != UNKNOWN))}
            for task in binary_tasks:
                alias = binary_aliases[task]
                row[f'{alias}_labeled'] = int(counts[task][sp].get(0, 0) + counts[task][sp].get(1, 0))
                row[f'{alias}_neg'] = int(counts[task][sp].get(0, 0))
                row[f'{alias}_pos'] = int(counts[task][sp].get(1, 0))
            w.writerow(row)

    split_label_counts_csv = paths.FEATURES / f'split_label_counts_{feature_id}.csv'
    with split_label_counts_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'task', 'task_alias', 'label', 'count'])
        w.writeheader()
        for sp in ('train', 'val', 'test'):
            for label, count in counts['protox'][sp].items():
                w.writerow({'split': sp, 'task': PROTOX_TASK, 'task_alias': 'protox', 'label': int(label), 'count': int(count)})
            for task in binary_tasks:
                alias = binary_aliases[task]
                for label, count in counts[task][sp].items():
                    w.writerow({'split': sp, 'task': task, 'task_alias': alias, 'label': int(label), 'count': int(count)})

    scaffold_meta_csv = paths.FEATURES / f'scaffold_meta_{feature_id}.csv'
    with scaffold_meta_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['inchi_key', 'scaffold_smiles', 'scaffold_hash'])
        w.writeheader()
        for ik, meta_row in sorted(scaffold_meta.items()):
            w.writerow({'inchi_key': ik, **meta_row})

    output_paths = [out_dir / 'X_fp.npy', out_dir / 'X_desc.npy', out_dir / 'y_protox.npy', out_dir / 'split_code.npy', out_dir / 'inchi_key.npy', meta_path, scaler_path, split_summary_csv, split_label_counts_csv, scaffold_meta_csv]
    output_paths.extend(out_dir / binary_files[task] for task in binary_tasks)
    for src in [split_summary_csv, split_label_counts_csv, scaffold_meta_csv]:
        shutil.copy2(src, run_dir / src.name)

    write_manifest(stage_dir / 'manifest.json', stage_name='02_build_features', config_path=cfg_path, inputs=[in_csv, split_path], outputs=output_paths, extra={'feature_id': feature_id, 'num_rows': N, 'binary_tasks': binary_tasks, 'binary_task_files': binary_files, 'desc_panel': desc_panel, 'fp_mode': fp_mode, 'fp_count_simulation': fp_count_simulation, 'fp_include_chirality': fp_include_chirality, 'fp_count_transform': fp_count_transform})
    print(f'[ok] features -> {out_dir}')
    print(f'[ok] meta -> {meta_path}')


if __name__ == '__main__':
    main()
