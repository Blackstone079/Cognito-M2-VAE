"""Step 1: balanced scaffold split with Chemprop-style oversized-bin handling and task-aware stratification."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.features.featurize_rdkit import mol_from_smiles, murcko_scaffold_smiles  # noqa: E402
from src.utils.logging import get_pipeline_run_dir, get_stage_results_dir, write_json, write_manifest  # noqa: E402

SPLITS = ('train', 'val', 'test')


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _hash64(s: str) -> np.uint64:
    h = hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest()
    return np.frombuffer(h, dtype=np.uint64)[0]


def _to_int_or_none(v: str | None) -> int | None:
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


def _slug(task: str) -> str:
    special = {'respiratory_toxicity': 'resp', 'ames_mutagenic': 'ames', 'dili_classification': 'dili'}
    if task in special:
        return special[task]
    s = re.sub(r'[^a-z0-9]+', '_', str(task).lower()).strip('_')
    return s or 'task'


def _get_active_binary_tasks(cfg: dict, fieldnames: list[str]) -> list[str]:
    tasks_cfg = cfg.get('tasks', {}) or {}
    active = []
    for task, enabled in tasks_cfg.items():
        if not bool(enabled):
            continue
        if task == 'protox_toxclass':
            continue
        if task not in fieldnames:
            continue
        active.append(task)
    return active


def _empty_counts(active_binary_tasks: list[str]) -> dict:
    counts = {'total': 0}
    for task in active_binary_tasks:
        alias = _slug(task)
        counts[f'{alias}_labeled'] = 0
        counts[f'{alias}_neg'] = 0
        counts[f'{alias}_pos'] = 0
    return counts


def _empty_scaffold_counts(active_binary_tasks: list[str]) -> dict:
    return {_slug(task): 0 for task in active_binary_tasks}


def _group_presence(group_counts: dict, active_binary_tasks: list[str]) -> dict:
    return {_slug(task): int(group_counts.get(f'{_slug(task)}_labeled', 0) > 0) for task in active_binary_tasks}


def _accumulate_counts(dst: dict, src: dict, sign: int = 1) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + sign * int(v)


def _accumulate_scaffold_counts(dst: dict, src: dict, sign: int = 1) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + sign * int(v)


def _scaled_targets(total_counts: dict, frac_train: float, frac_val: float) -> dict[str, dict[str, int]]:
    frac_test = 1.0 - frac_train - frac_val
    fracs = {'train': frac_train, 'val': frac_val, 'test': frac_test}
    targets = {sp: {} for sp in SPLITS}
    for key, value in total_counts.items():
        if key == 'total':
            base_train = int(frac_train * value)
            base_val = int(frac_val * value)
            targets['train'][key] = base_train
            targets['val'][key] = base_val
            targets['test'][key] = int(value - base_train - base_val)
            continue
        so_far = 0
        for i, sp in enumerate(SPLITS):
            if i < 2:
                tv = int(round(fracs[sp] * value))
                targets[sp][key] = tv
                so_far += tv
            else:
                targets[sp][key] = int(value - so_far)
    return targets


def _build_minimums(cfg: dict, active_binary_tasks: list[str]) -> dict[str, dict[str, int]]:
    split_cfg = cfg.get('split', {}) or {}
    minimums = {sp: {} for sp in SPLITS}
    min_labeled = {'val': int(split_cfg.get('min_val_labeled_per_task', 0)), 'test': int(split_cfg.get('min_test_labeled_per_task', 0))}
    min_pos = {'val': int(split_cfg.get('min_val_pos_per_task', 0)), 'test': int(split_cfg.get('min_test_pos_per_task', 0))}
    min_neg = {'val': int(split_cfg.get('min_val_neg_per_task', 0)), 'test': int(split_cfg.get('min_test_neg_per_task', 0))}
    for sp in ('val', 'test'):
        for task in active_binary_tasks:
            alias = _slug(task)
            minimums[sp][f'{alias}_labeled'] = min_labeled[sp]
            minimums[sp][f'{alias}_pos'] = min_pos[sp]
            minimums[sp][f'{alias}_neg'] = min_neg[sp]
    return minimums


def _build_scaffold_targets(total_scaffold_counts: dict, frac_train: float, frac_val: float) -> dict[str, dict[str, int]]:
    frac_test = 1.0 - frac_train - frac_val
    fracs = {'train': frac_train, 'val': frac_val, 'test': frac_test}
    targets = {sp: {} for sp in SPLITS}
    for alias, value in total_scaffold_counts.items():
        so_far = 0
        for i, sp in enumerate(SPLITS):
            if i < 2:
                tv = int(round(fracs[sp] * value))
                targets[sp][alias] = tv
                so_far += tv
            else:
                targets[sp][alias] = int(value - so_far)
    return targets


def _build_scaffold_minimums(cfg: dict, active_binary_tasks: list[str], total_scaffold_counts: dict) -> dict[str, dict[str, int]]:
    split_cfg = cfg.get('split', {}) or {}
    min_val = int(split_cfg.get('min_val_labeled_scaffolds_per_task', 0))
    min_test = int(split_cfg.get('min_test_labeled_scaffolds_per_task', 0))
    mins = {sp: {} for sp in SPLITS}
    for task in active_binary_tasks:
        alias = _slug(task)
        available = int(total_scaffold_counts.get(alias, 0))
        mins['val'][alias] = min(min_val, available)
        mins['test'][alias] = min(min_test, available)
    return mins


def _global_prevalence(total_counts: dict, active_binary_tasks: list[str]) -> dict[str, float | None]:
    prev = {}
    for task in active_binary_tasks:
        alias = _slug(task)
        pos = float(total_counts.get(f'{alias}_pos', 0))
        neg = float(total_counts.get(f'{alias}_neg', 0))
        den = pos + neg
        prev[task] = None if den <= 0 else float(pos / den)
    return prev


def _group_priority(group_counts: dict, active_binary_tasks: list[str], rng: np.random.RandomState) -> tuple:
    labeled = sum(int(group_counts.get(f'{_slug(task)}_labeled', 0)) for task in active_binary_tasks)
    pos = sum(int(group_counts.get(f'{_slug(task)}_pos', 0)) for task in active_binary_tasks)
    total = int(group_counts.get('total', 0))
    return (-labeled, -pos, -total, float(rng.rand()))


def _threshold_count(frac: float, target: int) -> int:
    if target <= 0:
        return 0
    return max(1, int(math.floor(float(frac) * float(target))))


def _forced_train_reasons(group_counts: dict, *, active_binary_tasks: list[str], targets: dict[str, dict[str, int]], cfg: dict) -> list[str]:
    split_cfg = cfg.get('split', {}) or {}
    reasons = []

    eval_target_total = int(min(targets['val'].get('total', 0), targets['test'].get('total', 0)))
    total_frac = float(split_cfg.get('large_scaffold_frac_of_eval_total', 0.5))
    total_threshold = _threshold_count(total_frac, eval_target_total)
    if total_threshold > 0 and int(group_counts.get('total', 0)) > total_threshold:
        reasons.append(f'total>{total_threshold}')

    default_frac = float(split_cfg.get('large_task_frac_of_eval_target', 0.5))
    suffix_fracs = {
        'labeled': float(split_cfg.get('large_task_labeled_frac_of_eval_target', default_frac)),
        'pos': float(split_cfg.get('large_task_pos_frac_of_eval_target', default_frac)),
        'neg': float(split_cfg.get('large_task_neg_frac_of_eval_target', default_frac)),
    }
    for task in active_binary_tasks:
        alias = _slug(task)
        for suffix, frac in suffix_fracs.items():
            key = f'{alias}_{suffix}'
            eval_target = int(min(targets['val'].get(key, 0), targets['test'].get(key, 0)))
            threshold = _threshold_count(frac, eval_target)
            if threshold > 0 and int(group_counts.get(key, 0)) > threshold:
                reasons.append(f'{key}>{threshold}')
    return reasons


def _objective_terms(
    counts_by_split: dict[str, dict[str, int]],
    scaffold_counts_by_split: dict[str, dict[str, int]],
    *,
    targets: dict[str, dict[str, int]],
    minimums: dict[str, dict[str, int]],
    scaffold_targets: dict[str, dict[str, int]],
    scaffold_minimums: dict[str, dict[str, int]],
    active_binary_tasks: list[str],
    prevalence: dict[str, float | None],
    cfg: dict,
) -> dict[str, float]:
    split_cfg = cfg.get('split', {}) or {}
    size_weight = float(split_cfg.get('size_weight', 2.0))
    labeled_weight = float(split_cfg.get('labeled_weight', 6.0))
    class_weight = float(split_cfg.get('class_weight', 8.0))
    prevalence_weight = float(split_cfg.get('prevalence_weight', 2.0))
    hard_min_weight = float(split_cfg.get('hard_min_weight', 50.0))
    overflow_weight = float(split_cfg.get('overflow_weight', 20.0))
    scaffold_weight = float(split_cfg.get('scaffold_weight', 5.0))
    scaffold_min_weight = float(split_cfg.get('scaffold_min_weight', 20.0))
    max_eval_frac = float(split_cfg.get('max_eval_frac_over_target', 1.10))

    terms = {
        'size_loss': 0.0,
        'labeled_loss': 0.0,
        'class_loss': 0.0,
        'prevalence_loss': 0.0,
        'hard_min_deficit': 0.0,
        'overflow_loss': 0.0,
        'scaffold_loss': 0.0,
        'scaffold_min_deficit': 0.0,
    }

    for sp in SPLITS:
        cur_total = float(counts_by_split[sp].get('total', 0))
        tgt_total = float(max(targets[sp].get('total', 1), 1))
        terms['size_loss'] += ((cur_total - tgt_total) / tgt_total) ** 2
        if sp in {'val', 'test'}:
            allowed = max_eval_frac * tgt_total
            if cur_total > allowed:
                terms['overflow_loss'] += ((cur_total - allowed) / tgt_total) ** 2

        for task in active_binary_tasks:
            alias = _slug(task)
            cur_lab = float(counts_by_split[sp].get(f'{alias}_labeled', 0))
            tgt_lab = float(max(targets[sp].get(f'{alias}_labeled', 1), 1))
            terms['labeled_loss'] += ((cur_lab - tgt_lab) / tgt_lab) ** 2

            for suffix in ('pos', 'neg'):
                cur = float(counts_by_split[sp].get(f'{alias}_{suffix}', 0))
                tgt = float(max(targets[sp].get(f'{alias}_{suffix}', 1), 1))
                terms['class_loss'] += ((cur - tgt) / tgt) ** 2

            global_prev = prevalence.get(task)
            cur_pos = float(counts_by_split[sp].get(f'{alias}_pos', 0))
            cur_neg = float(counts_by_split[sp].get(f'{alias}_neg', 0))
            den = cur_pos + cur_neg
            if global_prev is not None and den > 0:
                local_prev = cur_pos / den
                terms['prevalence_loss'] += (local_prev - global_prev) ** 2

            cur_scaf = float(scaffold_counts_by_split[sp].get(alias, 0))
            tgt_scaf = float(max(scaffold_targets[sp].get(alias, 1), 1))
            terms['scaffold_loss'] += ((cur_scaf - tgt_scaf) / tgt_scaf) ** 2

            if sp in {'val', 'test'}:
                for key, min_val in minimums[sp].items():
                    cur = float(counts_by_split[sp].get(key, 0))
                    if cur < float(min_val):
                        terms['hard_min_deficit'] += ((float(min_val) - cur) / max(float(min_val), 1.0)) ** 2
                min_scaf = float(scaffold_minimums[sp].get(alias, 0))
                if cur_scaf < min_scaf:
                    terms['scaffold_min_deficit'] += ((min_scaf - cur_scaf) / max(min_scaf, 1.0)) ** 2

    terms['total'] = (
        size_weight * terms['size_loss']
        + labeled_weight * terms['labeled_loss']
        + class_weight * terms['class_loss']
        + prevalence_weight * terms['prevalence_loss']
        + hard_min_weight * terms['hard_min_deficit']
        + overflow_weight * terms['overflow_loss']
        + scaffold_weight * terms['scaffold_loss']
        + scaffold_min_weight * terms['scaffold_min_deficit']
    )
    return terms


def _score_with_move(
    counts_by_split: dict[str, dict[str, int]],
    scaffold_counts_by_split: dict[str, dict[str, int]],
    *,
    dst: str,
    group_counts: dict,
    group_presence: dict,
    targets: dict[str, dict[str, int]],
    minimums: dict[str, dict[str, int]],
    scaffold_targets: dict[str, dict[str, int]],
    scaffold_minimums: dict[str, dict[str, int]],
    active_binary_tasks: list[str],
    prevalence: dict[str, float | None],
    cfg: dict,
    src: str | None = None,
) -> tuple[float, dict[str, float]]:
    if src is not None:
        _accumulate_counts(counts_by_split[src], group_counts, sign=-1)
        _accumulate_scaffold_counts(scaffold_counts_by_split[src], group_presence, sign=-1)
    _accumulate_counts(counts_by_split[dst], group_counts, sign=1)
    _accumulate_scaffold_counts(scaffold_counts_by_split[dst], group_presence, sign=1)
    terms = _objective_terms(
        counts_by_split,
        scaffold_counts_by_split,
        targets=targets,
        minimums=minimums,
        scaffold_targets=scaffold_targets,
        scaffold_minimums=scaffold_minimums,
        active_binary_tasks=active_binary_tasks,
        prevalence=prevalence,
        cfg=cfg,
    )
    _accumulate_counts(counts_by_split[dst], group_counts, sign=-1)
    _accumulate_scaffold_counts(scaffold_counts_by_split[dst], group_presence, sign=-1)
    if src is not None:
        _accumulate_counts(counts_by_split[src], group_counts, sign=1)
        _accumulate_scaffold_counts(scaffold_counts_by_split[src], group_presence, sign=1)
    return float(terms['total']), terms


def _copy_counts(counts_by_split: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {sp: dict(vals) for sp, vals in counts_by_split.items()}


def _copy_scaffold_counts(counts_by_split: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {sp: dict(vals) for sp, vals in counts_by_split.items()}


def _build_summary_rows(counts_by_split: dict[str, dict[str, int]], active_binary_tasks: list[str]) -> tuple[list[dict], list[dict]]:
    split_summary = []
    split_label_counts = []
    for sp in SPLITS:
        c = counts_by_split[sp]
        summary_row = {'split': sp, 'total': int(c.get('total', 0))}
        for task in active_binary_tasks:
            alias = _slug(task)
            summary_row[f'{alias}_labeled'] = int(c.get(f'{alias}_labeled', 0))
            summary_row[f'{alias}_neg'] = int(c.get(f'{alias}_neg', 0))
            summary_row[f'{alias}_pos'] = int(c.get(f'{alias}_pos', 0))
            split_label_counts.extend([
                {'split': sp, 'task': task, 'label': '0', 'count': int(c.get(f'{alias}_neg', 0))},
                {'split': sp, 'task': task, 'label': '1', 'count': int(c.get(f'{alias}_pos', 0))},
                {'split': sp, 'task': task, 'label': 'known', 'count': int(c.get(f'{alias}_labeled', 0))},
            ])
        split_summary.append(summary_row)
    return split_summary, split_label_counts


def _build_scaffold_summary_rows(scaffold_counts_by_split: dict[str, dict[str, int]], active_binary_tasks: list[str]) -> list[dict]:
    rows = []
    for sp in SPLITS:
        row = {'split': sp}
        for task in active_binary_tasks:
            alias = _slug(task)
            row[f'{alias}_labeled_scaffolds'] = int(scaffold_counts_by_split[sp].get(alias, 0))
        rows.append(row)
    return rows


def _write_split_outputs(
    paths: Paths,
    feature_id: str,
    *,
    split_rows: list[dict],
    split_summary: list[dict],
    split_label_counts: list[dict],
    scaffold_summary: list[dict],
    n_invalid: int,
    optimizer_summary: dict,
) -> list[Path]:
    out_split = paths.SPLITS / f'split_scaffold_{feature_id}.csv'
    with out_split.open('w', newline='', encoding='utf-8') as f_out:
        w = csv.DictWriter(f_out, fieldnames=['inchi_key', 'split', 'scaffold_smiles', 'scaffold_hash'])
        w.writeheader()
        w.writerows(split_rows)

    valid_keys_path = paths.SPLITS / f'valid_inchi_keys_{feature_id}.csv'
    valid_keys_path.write_text('\n'.join(row['inchi_key'] for row in split_rows) + '\n', encoding='utf-8')

    split_summary_csv = paths.SPLITS / f'split_summary_{feature_id}.csv'
    with split_summary_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(split_summary[0].keys()))
        w.writeheader()
        w.writerows(split_summary)

    split_label_counts_csv = paths.SPLITS / f'split_label_counts_{feature_id}.csv'
    with split_label_counts_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'task', 'label', 'count'])
        w.writeheader()
        w.writerows(split_label_counts)

    scaffold_summary_csv = paths.SPLITS / f'split_scaffold_counts_{feature_id}.csv'
    with scaffold_summary_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(scaffold_summary[0].keys()))
        w.writeheader()
        w.writerows(scaffold_summary)

    split_summary_json = paths.SPLITS / f'split_summary_{feature_id}.json'
    split_summary_json.write_text(json.dumps({'n_invalid_smiles': int(n_invalid), 'splits': split_summary}, indent=2), encoding='utf-8')

    optimizer_summary_json = paths.SPLITS / f'split_optimizer_summary_{feature_id}.json'
    write_json(optimizer_summary_json, optimizer_summary)

    print(f'[ok] wrote {out_split}')
    print(f'[ok] wrote {valid_keys_path}')
    print(f'[ok] wrote {split_summary_csv}')
    print(f'[ok] wrote {optimizer_summary_json}')
    return [out_split, valid_keys_path, split_summary_csv, split_label_counts_csv, scaffold_summary_csv, split_summary_json, optimizer_summary_json]


def _top_scaffold_rows(groups: dict[str, dict], assignments: dict[str, str], active_binary_tasks: list[str], top_k: int = 25) -> list[dict]:
    rows = []
    ranked = sorted(groups.items(), key=lambda kv: (-kv[1]['counts'].get('total', 0), kv[0]))[:top_k]
    for sh, group in ranked:
        row = {
            'scaffold_hash': sh,
            'split': assignments.get(sh, ''),
            'scaffold_smiles': group.get('scaffold_smiles', ''),
            'size': int(group['counts'].get('total', 0)),
        }
        for task in active_binary_tasks:
            alias = _slug(task)
            row[f'{alias}_labeled'] = int(group['counts'].get(f'{alias}_labeled', 0))
            row[f'{alias}_pos'] = int(group['counts'].get(f'{alias}_pos', 0))
            row[f'{alias}_neg'] = int(group['counts'].get(f'{alias}_neg', 0))
        rows.append(row)
    return rows


def _top_forced_rows(groups: dict[str, dict], forced_train: dict[str, list[str]], active_binary_tasks: list[str], top_k: int = 25) -> list[dict]:
    ranked = sorted(forced_train.keys(), key=lambda sh: (-groups[sh]['counts'].get('total', 0), sh))[:top_k]
    rows = []
    for sh in ranked:
        group = groups[sh]
        row = {
            'scaffold_hash': sh,
            'scaffold_smiles': group.get('scaffold_smiles', ''),
            'size': int(group['counts'].get('total', 0)),
            'reasons': ';'.join(forced_train[sh]),
        }
        for task in active_binary_tasks:
            alias = _slug(task)
            row[f'{alias}_labeled'] = int(group['counts'].get(f'{alias}_labeled', 0))
            row[f'{alias}_pos'] = int(group['counts'].get(f'{alias}_pos', 0))
            row[f'{alias}_neg'] = int(group['counts'].get(f'{alias}_neg', 0))
        rows.append(row)
    return rows


def _refine_assignments(
    groups: dict[str, dict],
    assignments: dict[str, str],
    counts_by_split: dict[str, dict[str, int]],
    scaffold_counts_by_split: dict[str, dict[str, int]],
    *,
    targets: dict[str, dict[str, int]],
    minimums: dict[str, dict[str, int]],
    scaffold_targets: dict[str, dict[str, int]],
    scaffold_minimums: dict[str, dict[str, int]],
    active_binary_tasks: list[str],
    prevalence: dict[str, float | None],
    cfg: dict,
    refine_order: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, float]]:
    split_cfg = cfg.get('split', {}) or {}
    refine_passes = int(split_cfg.get('refine_passes', 0))
    if refine_passes <= 0:
        terms = _objective_terms(
            counts_by_split,
            scaffold_counts_by_split,
            targets=targets,
            minimums=minimums,
            scaffold_targets=scaffold_targets,
            scaffold_minimums=scaffold_minimums,
            active_binary_tasks=active_binary_tasks,
            prevalence=prevalence,
            cfg=cfg,
        )
        return assignments, counts_by_split, scaffold_counts_by_split, terms

    best_terms = _objective_terms(
        counts_by_split,
        scaffold_counts_by_split,
        targets=targets,
        minimums=minimums,
        scaffold_targets=scaffold_targets,
        scaffold_minimums=scaffold_minimums,
        active_binary_tasks=active_binary_tasks,
        prevalence=prevalence,
        cfg=cfg,
    )
    best_score = float(best_terms['total'])

    for _ in range(refine_passes):
        improved = False
        for sh in refine_order:
            src = assignments[sh]
            group_counts = groups[sh]['counts']
            group_presence = groups[sh]['presence']
            chosen = src
            chosen_score = best_score
            chosen_terms = best_terms
            for dst in SPLITS:
                if dst == src:
                    continue
                score, terms = _score_with_move(
                    counts_by_split,
                    scaffold_counts_by_split,
                    dst=dst,
                    src=src,
                    group_counts=group_counts,
                    group_presence=group_presence,
                    targets=targets,
                    minimums=minimums,
                    scaffold_targets=scaffold_targets,
                    scaffold_minimums=scaffold_minimums,
                    active_binary_tasks=active_binary_tasks,
                    prevalence=prevalence,
                    cfg=cfg,
                )
                if score + 1e-12 < chosen_score:
                    chosen = dst
                    chosen_score = score
                    chosen_terms = terms
            if chosen != src:
                _accumulate_counts(counts_by_split[src], group_counts, sign=-1)
                _accumulate_scaffold_counts(scaffold_counts_by_split[src], group_presence, sign=-1)
                _accumulate_counts(counts_by_split[chosen], group_counts, sign=1)
                _accumulate_scaffold_counts(scaffold_counts_by_split[chosen], group_presence, sign=1)
                assignments[sh] = chosen
                best_score = chosen_score
                best_terms = chosen_terms
                improved = True
        if not improved:
            break
    return assignments, counts_by_split, scaffold_counts_by_split, best_terms


def main() -> None:
    ap = argparse.ArgumentParser(description='Balanced scaffold split for M2.')
    ap.add_argument('--config', type=str, default='pipelines/m2/config.yaml')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)
    paths = Paths()
    feature_id = str(cfg['featurizer'].get('feature_id', 'm2_v1'))
    run_dir = get_pipeline_run_dir(paths.RESULTS, feature_id, prefix=str(cfg.get('run', {}).get('prefix', 'm2')))
    stage_dir = get_stage_results_dir(run_dir, '01_split')

    in_csv = paths.INTERIM / 'drug_table_multitask.csv'
    if not in_csv.exists():
        raise FileNotFoundError(f'Missing {in_csv}. Run pipelines/m2/extract.py first.')

    invalid_path = paths.INTERIM / f'invalid_smiles_{feature_id}.csv'
    invalid_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    groups: dict[str, dict] = {}
    n_invalid = 0

    with in_csv.open('r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        active_binary_tasks = _get_active_binary_tasks(cfg, fieldnames)
        if not active_binary_tasks:
            raise RuntimeError('No active binary tasks found in config/CSV.')
        with invalid_path.open('w', newline='', encoding='utf-8') as f_bad:
            w_bad = csv.DictWriter(f_bad, fieldnames=['inchi_key', 'smiles'])
            w_bad.writeheader()

            for row in reader:
                ik = (row.get('inchi_key') or '').strip()
                smi = row.get('smiles')
                mol = mol_from_smiles(smi)
                if mol is None:
                    w_bad.writerow({'inchi_key': ik, 'smiles': smi if smi is not None else ''})
                    n_invalid += 1
                    continue

                scaff = murcko_scaffold_smiles(mol)
                sh = str(int(_hash64(scaff)))
                rec = {'inchi_key': ik, 'scaffold_smiles': scaff, 'scaffold_hash': sh}
                for task in active_binary_tasks:
                    rec[task] = _to_int_or_none(row.get(task))
                rows.append(rec)
                if sh not in groups:
                    groups[sh] = {'scaffold_smiles': scaff, 'rows': [], 'counts': _empty_counts(active_binary_tasks)}
                groups[sh]['rows'].append(rec)
                groups[sh]['counts']['total'] += 1
                for task in active_binary_tasks:
                    value = rec[task]
                    alias = _slug(task)
                    if value in {0, 1}:
                        groups[sh]['counts'][f'{alias}_labeled'] += 1
                        groups[sh]['counts'][f'{alias}_{"pos" if value == 1 else "neg"}'] += 1

    if not rows:
        raise RuntimeError('No valid SMILES found.')

    total_counts = _empty_counts(active_binary_tasks)
    total_scaffold_counts = _empty_scaffold_counts(active_binary_tasks)
    for group in groups.values():
        group['presence'] = _group_presence(group['counts'], active_binary_tasks)
        _accumulate_counts(total_counts, group['counts'])
        _accumulate_scaffold_counts(total_scaffold_counts, group['presence'])

    frac_train = float(cfg['split']['frac_train'])
    frac_val = float(cfg['split']['frac_val'])
    targets = _scaled_targets(total_counts, frac_train=frac_train, frac_val=frac_val)
    minimums = _build_minimums(cfg, active_binary_tasks)
    scaffold_targets = _build_scaffold_targets(total_scaffold_counts, frac_train=frac_train, frac_val=frac_val)
    scaffold_minimums = _build_scaffold_minimums(cfg, active_binary_tasks, total_scaffold_counts)

    for sp in ('val', 'test'):
        for task in active_binary_tasks:
            alias = _slug(task)
            targets[sp][f'{alias}_labeled'] = max(targets[sp].get(f'{alias}_labeled', 0), minimums[sp][f'{alias}_labeled'])
            targets[sp][f'{alias}_pos'] = max(targets[sp].get(f'{alias}_pos', 0), minimums[sp][f'{alias}_pos'])
            targets[sp][f'{alias}_neg'] = max(targets[sp].get(f'{alias}_neg', 0), minimums[sp][f'{alias}_neg'])
            scaffold_targets[sp][alias] = max(scaffold_targets[sp].get(alias, 0), scaffold_minimums[sp][alias])

    prevalence = _global_prevalence(total_counts, active_binary_tasks)
    split_seed = int(cfg['split'].get('seed', 0))
    n_restarts = int(cfg['split'].get('n_restarts', 8))

    forced_train: dict[str, list[str]] = {}
    base_counts_by_split = {sp: _empty_counts(active_binary_tasks) for sp in SPLITS}
    base_scaffold_counts = {sp: _empty_scaffold_counts(active_binary_tasks) for sp in SPLITS}
    base_assignments = {}
    remaining_keys = []

    for sh, group in groups.items():
        reasons = _forced_train_reasons(group['counts'], active_binary_tasks=active_binary_tasks, targets=targets, cfg=cfg)
        if reasons:
            forced_train[sh] = reasons
            base_assignments[sh] = 'train'
            _accumulate_counts(base_counts_by_split['train'], group['counts'], sign=1)
            _accumulate_scaffold_counts(base_scaffold_counts['train'], group['presence'], sign=1)
        else:
            remaining_keys.append(sh)

    candidate_summaries = []
    best = None
    for restart in range(max(n_restarts, 1)):
        rng = np.random.RandomState(split_seed + restart)
        order = sorted(remaining_keys, key=lambda sh: _group_priority(groups[sh]['counts'], active_binary_tasks, rng))
        counts_by_split = _copy_counts(base_counts_by_split)
        scaffold_counts_by_split = _copy_scaffold_counts(base_scaffold_counts)
        assignments = dict(base_assignments)

        for sh in order:
            group_counts = groups[sh]['counts']
            group_presence = groups[sh]['presence']
            best_sp = None
            best_score = None
            best_terms = None
            for sp in SPLITS:
                score, terms = _score_with_move(
                    counts_by_split,
                    scaffold_counts_by_split,
                    dst=sp,
                    group_counts=group_counts,
                    group_presence=group_presence,
                    targets=targets,
                    minimums=minimums,
                    scaffold_targets=scaffold_targets,
                    scaffold_minimums=scaffold_minimums,
                    active_binary_tasks=active_binary_tasks,
                    prevalence=prevalence,
                    cfg=cfg,
                )
                if best_score is None or score < best_score - 1e-12:
                    best_sp = sp
                    best_score = score
                    best_terms = terms
            assert best_sp is not None
            assignments[sh] = best_sp
            _accumulate_counts(counts_by_split[best_sp], group_counts, sign=1)
            _accumulate_scaffold_counts(scaffold_counts_by_split[best_sp], group_presence, sign=1)

        assignments, counts_by_split, scaffold_counts_by_split, final_terms = _refine_assignments(
            groups,
            assignments,
            counts_by_split,
            scaffold_counts_by_split,
            targets=targets,
            minimums=minimums,
            scaffold_targets=scaffold_targets,
            scaffold_minimums=scaffold_minimums,
            active_binary_tasks=active_binary_tasks,
            prevalence=prevalence,
            cfg=cfg,
            refine_order=order,
        )

        split_summary, _ = _build_summary_rows(counts_by_split, active_binary_tasks)
        scaffold_summary = _build_scaffold_summary_rows(scaffold_counts_by_split, active_binary_tasks)
        cand = {
            'restart': restart,
            'score': float(final_terms['total']),
            'terms': {k: float(v) for k, v in final_terms.items()},
            'split_summary': split_summary,
            'scaffold_summary': scaffold_summary,
        }
        candidate_summaries.append(cand)
        if best is None or cand['score'] < best['score']:
            best = {
                'restart': restart,
                'score': float(cand['score']),
                'terms': {k: float(v) for k, v in final_terms.items()},
                'assignments': dict(assignments),
                'counts_by_split': _copy_counts(counts_by_split),
                'scaffold_counts_by_split': _copy_scaffold_counts(scaffold_counts_by_split),
            }

    assert best is not None
    assignments = best['assignments']
    counts_by_split = best['counts_by_split']
    scaffold_counts_by_split = best['scaffold_counts_by_split']
    split_summary, split_label_counts = _build_summary_rows(counts_by_split, active_binary_tasks)
    scaffold_summary = _build_scaffold_summary_rows(scaffold_counts_by_split, active_binary_tasks)

    split_rows = []
    for row in rows:
        sp = assignments[row['scaffold_hash']]
        split_rows.append({
            'inchi_key': row['inchi_key'],
            'split': sp,
            'scaffold_smiles': row['scaffold_smiles'],
            'scaffold_hash': row['scaffold_hash'],
        })

    candidate_summaries = sorted(candidate_summaries, key=lambda d: d['score'])
    optimizer_summary = {
        'feature_id': feature_id,
        'seed': split_seed,
        'n_restarts': int(max(n_restarts, 1)),
        'active_binary_tasks': active_binary_tasks,
        'targets': targets,
        'minimums': minimums,
        'scaffold_targets': scaffold_targets,
        'scaffold_minimums': scaffold_minimums,
        'global_prevalence': prevalence,
        'forced_train_count': int(len(forced_train)),
        'forced_train_examples': int(sum(groups[sh]['counts'].get('total', 0) for sh in forced_train)),
        'forced_train_top_scaffolds': _top_forced_rows(groups, forced_train, active_binary_tasks, top_k=25),
        'best_restart': int(best['restart']),
        'best_score': float(best['score']),
        'best_terms': {k: float(v) for k, v in best['terms'].items()},
        'split_summary': split_summary,
        'scaffold_summary': scaffold_summary,
        'top_candidates': candidate_summaries[:10],
        'top_scaffolds': _top_scaffold_rows(groups, assignments, active_binary_tasks, top_k=25),
        'n_invalid_smiles': int(n_invalid),
    }

    outputs = _write_split_outputs(
        paths,
        feature_id,
        split_rows=split_rows,
        split_summary=split_summary,
        split_label_counts=split_label_counts,
        scaffold_summary=scaffold_summary,
        n_invalid=n_invalid,
        optimizer_summary=optimizer_summary,
    )
    for out in outputs + [invalid_path]:
        if out.exists():
            shutil.copy2(out, stage_dir / out.name)

    write_manifest(
        stage_dir / 'manifest.json',
        stage_name='01_split',
        config_path=cfg_path,
        inputs=[in_csv, invalid_path],
        outputs=outputs,
        extra={
            'feature_id': feature_id,
            'active_binary_tasks': active_binary_tasks,
            'forced_train_count': int(len(forced_train)),
            'best_restart': int(best['restart']),
            'best_score': float(best['score']),
            'best_terms': {k: float(v) for k, v in best['terms'].items()},
        },
    )


if __name__ == '__main__':
    main()
