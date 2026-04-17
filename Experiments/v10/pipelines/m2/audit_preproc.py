"""Audit preprocessing artifacts before training."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.dataio.m2_memmap_datamodule import _resolve_batch_mix_counts  # noqa: E402
from src.labels.task_registry import active_binary_tasks_from_cfg, task_alias  # noqa: E402
from src.utils.logging import get_pipeline_run_dir, get_stage_results_dir, write_json, write_manifest  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _to_int_or_none(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def _safe_rate(num: float, den: float):
    return None if den <= 0 else float(num / den)


def main() -> None:
    ap = argparse.ArgumentParser(description='Audit preprocessing outputs for M2.')
    ap.add_argument('--config', type=str, default='pipelines/m2/config.yaml')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)
    paths = Paths()
    feature_id = str(cfg['featurizer'].get('feature_id', 'm2_v1'))
    run_dir = get_pipeline_run_dir(paths.RESULTS, feature_id, prefix=str(cfg.get('run', {}).get('prefix', 'm2')))
    stage_dir = get_stage_results_dir(run_dir, '03_audit_preproc')
    audit_dir = stage_dir
    audit_dir.mkdir(parents=True, exist_ok=True)

    interim_csv = paths.INTERIM / 'drug_table_multitask.csv'
    extract_summary_path = paths.INTERIM / 'extract_multitask_summary.json'
    invalid_smiles_path = paths.INTERIM / f'invalid_smiles_{feature_id}.csv'
    split_path = paths.SPLITS / f'split_scaffold_{feature_id}.csv'
    split_summary_path = paths.SPLITS / f'split_summary_{feature_id}.csv'
    split_label_counts_path = paths.SPLITS / f'split_label_counts_{feature_id}.csv'
    split_optimizer_path = paths.SPLITS / f'split_optimizer_summary_{feature_id}.json'
    features_meta_path = paths.FEATURES / f'features_{feature_id}_memmap_meta.json'

    for req in [interim_csv, split_path, split_summary_path, split_label_counts_path, features_meta_path]:
        if not req.exists():
            raise FileNotFoundError(f'Missing required preprocessing artifact: {req}')

    with interim_csv.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        tasks = active_binary_tasks_from_cfg(cfg, fieldnames)
        rows = list(reader)

    extract_summary = json.loads(extract_summary_path.read_text(encoding='utf-8')) if extract_summary_path.exists() else {}
    features_meta = json.loads(features_meta_path.read_text(encoding='utf-8'))
    optimizer_summary = json.loads(split_optimizer_path.read_text(encoding='utf-8')) if split_optimizer_path.exists() else {}

    row_labels = {}
    global_counts = {task: Counter() for task in tasks}
    for row in rows:
        ik = (row.get('inchi_key') or '').strip()
        if not ik:
            continue
        row_labels[ik] = {}
        for task in tasks:
            value = _to_int_or_none(row.get(task))
            row_labels[ik][task] = value
            if value in {0, 1}:
                global_counts[task][value] += 1

    split_counts = Counter()
    scaffold_sizes = Counter()
    task_split_counts = {task: defaultdict(Counter) for task in tasks}
    labeled_scaffolds = {task: defaultdict(set) for task in tasks}
    task_scaffold_label_counts = {task: defaultdict(Counter) for task in tasks}
    split_inchis = defaultdict(set)
    split_scaffolds = defaultdict(set)
    with split_path.open('r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            ik = (row.get('inchi_key') or '').strip()
            sp = (row.get('split') or '').strip()
            sh = (row.get('scaffold_hash') or '').strip()
            split_counts[sp] += 1
            scaffold_sizes[(sp, sh)] += 1
            split_inchis[sp].add(ik)
            split_scaffolds[sp].add(sh)
            labels = row_labels.get(ik, {})
            for task in tasks:
                value = labels.get(task)
                if value in {0, 1}:
                    task_split_counts[task][sp][value] += 1
                    labeled_scaffolds[task][sp].add(sh)
                    task_scaffold_label_counts[task][sp][sh] += 1

    leakage_checks = []
    findings = []
    for a, b in combinations(sorted(split_counts.keys()), 2):
        inchi_overlap = split_inchis[a] & split_inchis[b]
        scaffold_overlap = split_scaffolds[a] & split_scaffolds[b]
        leakage_checks.append({
            'split_a': a,
            'split_b': b,
            'inchi_overlap_count': int(len(inchi_overlap)),
            'scaffold_overlap_count': int(len(scaffold_overlap)),
        })
        if inchi_overlap:
            findings.append(f'Potential leakage: {a}/{b} share {len(inchi_overlap)} inchi_keys.')
        if scaffold_overlap:
            findings.append(f'Scaffold leakage: {a}/{b} share {len(scaffold_overlap)} scaffold hashes.')

    unique_scaffolds_per_split = {sp: len({sh for split_name, sh in scaffold_sizes if split_name == sp}) for sp in split_counts}
    largest_scaffolds = [
        {'split': sp, 'scaffold_hash': sh, 'size': int(size)}
        for (sp, sh), size in scaffold_sizes.most_common(25)
    ]

    batch_size = int(cfg['training']['batch_size'])
    train_total = int(split_counts.get('train', 0))
    n_batches = int(math.ceil(train_total / max(batch_size, 1))) if train_total > 0 else 0
    batch_mix = cfg.get('training', {}).get('batch_mix', {}) or {}
    task_aliases = {task: task_alias(task) for task in tasks}
    random_per_batch, targeted_per_batch = _resolve_batch_mix_counts(batch_size, batch_mix, tuple(task_aliases.values()))

    exposure = {}
    for task in tasks:
        alias = task_aliases[task]
        labeled = int(task_split_counts[task]['train'][0] + task_split_counts[task]['train'][1])
        pos = int(task_split_counts[task]['train'][1])
        neg = int(task_split_counts[task]['train'][0])
        labeled_frac = float(labeled / train_total) if train_total > 0 else 0.0
        random_labeled_per_batch = float(random_per_batch * labeled_frac)
        targeted = int(targeted_per_batch.get(alias, 0))
        supervised_draws_per_epoch = float(n_batches * (random_labeled_per_batch + targeted))
        avg_reuse = None if labeled <= 0 else float(supervised_draws_per_epoch / labeled)
        exposure[task] = {
            'task_alias': alias,
            'train_total': train_total,
            'train_labeled': labeled,
            'train_pos': pos,
            'train_neg': neg,
            'n_batches_per_epoch': n_batches,
            'random_labeled_per_batch_expected': random_labeled_per_batch,
            'targeted_per_batch': targeted,
            'supervised_draws_per_epoch_expected': supervised_draws_per_epoch,
            'avg_labeled_reuse_per_epoch_expected': avg_reuse,
            'targeted_sampling': 'without_replacement_cycle' if targeted > 0 else 'none',
        }
        if avg_reuse is not None and avg_reuse > 10:
            findings.append(f'{task}: expected labelled reuse per epoch is high ({avg_reuse:.1f}).')
        if pos < 50:
            findings.append(f'{task}: train positives are very limited ({pos}).')

    top_task_scaffolds = []
    for task in tasks:
        alias = task_aliases[task]
        for sp in ('train', 'val', 'test'):
            total_labeled = int(task_split_counts[task][sp][0] + task_split_counts[task][sp][1])
            if total_labeled <= 0:
                continue
            ranked = task_scaffold_label_counts[task][sp].most_common(5)
            for rank, (sh, count) in enumerate(ranked, start=1):
                top_task_scaffolds.append({
                    'task': task,
                    'task_alias': alias,
                    'split': sp,
                    'rank': rank,
                    'scaffold_hash': sh,
                    'labeled_count': int(count),
                    'share_of_task_labeled': float(count / total_labeled),
                })
                if sp in {'val', 'test'} and rank == 1 and (count / total_labeled) > 0.50:
                    findings.append(f'{task}: {sp} labels are dominated by one scaffold ({count / total_labeled:.1%}).')

    prevalence_rows = []
    for task in tasks:
        alias = task_aliases[task]
        global_pos = int(global_counts[task][1])
        global_neg = int(global_counts[task][0])
        global_prev = _safe_rate(global_pos, global_pos + global_neg)
        for sp in ('train', 'val', 'test'):
            pos = int(task_split_counts[task][sp][1])
            neg = int(task_split_counts[task][sp][0])
            prev = _safe_rate(pos, pos + neg)
            drift = None if global_prev is None or prev is None else float(prev - global_prev)
            prevalence_rows.append({
                'task': task,
                'task_alias': alias,
                'split': sp,
                'pos': pos,
                'neg': neg,
                'labeled': pos + neg,
                'prevalence': prev,
                'global_prevalence': global_prev,
                'drift_vs_global': drift,
                'labeled_scaffolds': int(len(labeled_scaffolds[task][sp])),
            })
            if sp in {'val', 'test'} and abs(drift or 0.0) > 0.10:
                findings.append(f'{task}: {sp} prevalence drift is large ({drift:+.3f}).')

    summary = {
        'feature_id': feature_id,
        'extract_summary': extract_summary,
        'binary_tasks': tasks,
        'features_meta_num_rows': int(features_meta.get('num_rows', 0)),
        'split_counts': {k: int(v) for k, v in split_counts.items()},
        'unique_scaffolds_per_split': {k: int(v) for k, v in unique_scaffolds_per_split.items()},
        'largest_scaffolds': largest_scaffolds,
        'prevalence_rows': prevalence_rows,
        'top_task_scaffolds': top_task_scaffolds,
        'exposure': exposure,
        'leakage_checks': leakage_checks,
        'optimizer_summary': optimizer_summary,
        'findings': findings,
    }

    summary_json = audit_dir / 'preproc_audit_summary.json'
    write_json(summary_json, summary)
    prevalence_csv = audit_dir / 'preproc_audit_prevalence.csv'
    with prevalence_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(prevalence_rows[0].keys()) if prevalence_rows else ['task', 'split'])
        w.writeheader()
        if prevalence_rows:
            w.writerows(prevalence_rows)

    top_scaffolds_csv = audit_dir / 'preproc_audit_top_task_scaffolds.csv'
    with top_scaffolds_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(top_task_scaffolds[0].keys()) if top_task_scaffolds else ['task', 'split', 'rank', 'scaffold_hash', 'labeled_count', 'share_of_task_labeled'])
        w.writeheader()
        if top_task_scaffolds:
            w.writerows(top_task_scaffolds)

    leakage_csv = audit_dir / 'preproc_audit_leakage_checks.csv'
    with leakage_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(leakage_checks[0].keys()) if leakage_checks else ['split_a', 'split_b', 'inchi_overlap_count', 'scaffold_overlap_count'])
        w.writeheader()
        if leakage_checks:
            w.writerows(leakage_checks)

    write_manifest(
        stage_dir / 'manifest.json',
        stage_name='03_audit_preproc',
        config_path=cfg_path,
        inputs=[interim_csv, split_path, split_summary_path, split_label_counts_path, split_optimizer_path, features_meta_path, invalid_smiles_path],
        outputs=[summary_json, prevalence_csv, top_scaffolds_csv, leakage_csv],
        extra={'feature_id': feature_id, 'n_findings': int(len(findings)), 'findings': findings},
    )
    print(f'[ok] wrote {summary_json}')
    print(f'[ok] wrote {prevalence_csv}')
    print(f'[ok] wrote {top_scaffolds_csv}')
    print(f'[ok] wrote {leakage_csv}')


if __name__ == '__main__':
    main()
