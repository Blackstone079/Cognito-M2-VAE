"""Train strong supervised baselines on the current split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.dataio.m2_memmap_datamodule import open_structured_memmap_features  # noqa: E402
from src.training.m2_metrics import binary_metrics, select_binary_threshold  # noqa: E402
from src.utils.logging import write_json, write_manifest, get_pipeline_run_dir, get_stage_results_dir  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _labeled_split_arrays(feats, task_name: str):
    y_all = np.asarray(feats.get_label_array(task_name))
    split = np.asarray(feats.split_code)

    def take(split_code: int):
        idx = np.where((split == split_code) & (y_all >= 0))[0].astype(np.int64)
        x_fp = np.asarray(feats.X_fp[idx], dtype=np.float32)
        x_desc = np.asarray(feats.X_desc[idx], dtype=np.float32)
        X = np.concatenate([x_fp, x_desc], axis=1).astype(np.float32, copy=False)
        return X, y_all[idx].astype(int), idx

    return {'train': take(0), 'val': take(1), 'test': take(2)}


def _prob_pos(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    raise RuntimeError('Model has neither predict_proba nor decision_function')


def _cm_counts(y_true: np.ndarray, probs_pos: np.ndarray, threshold: float) -> dict:
    pred = (probs_pos >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def _decode_inchi_key(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode('ascii', errors='ignore')
    if hasattr(value, 'tobytes'):
        return value.tobytes().decode('ascii', errors='ignore')
    return str(value)


def _load_scaffold_lookup(scaffold_meta_path: Path) -> dict[str, dict[str, object]]:
    if not scaffold_meta_path.exists():
        return {}
    df = pd.read_csv(scaffold_meta_path, usecols=['inchi_key', 'scaffold_smiles', 'scaffold_hash'])
    lookup = {}
    for row in df.to_dict(orient='records'):
        lookup[str(row['inchi_key'])] = {
            'scaffold_smiles': row.get('scaffold_smiles'),
            'scaffold_hash': row.get('scaffold_hash'),
        }
    return lookup


def _append_prediction_rows(rows: list[dict], *, feats, idx_arr: np.ndarray, y_true: np.ndarray, probs_pos: np.ndarray, threshold: float, task_name: str, task_alias: str, model_name: str, split_name: str, scaffold_lookup: dict[str, dict[str, object]]) -> None:
    pred = (probs_pos >= threshold).astype(int)
    for i, idx in enumerate(idx_arr.tolist()):
        ik = _decode_inchi_key(feats.inchi_key[idx])
        scaffold = scaffold_lookup.get(ik, {})
        rows.append({
            'task': task_alias,
            'task_name': task_name,
            'model': model_name,
            'split': split_name,
            'inchi_key': ik,
            'scaffold_smiles': scaffold.get('scaffold_smiles'),
            'scaffold_hash': scaffold.get('scaffold_hash'),
            'y_true': int(y_true[i]),
            'p_pos': float(probs_pos[i]),
            'pred': int(pred[i]),
            'threshold': float(threshold),
            'correct': int(pred[i] == y_true[i]),
        })


def main() -> None:
    ap = argparse.ArgumentParser(description='Train supervised baselines on M2 features.')
    ap.add_argument('--config', type=str, default='pipelines/m2/config.yaml')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)
    paths = Paths()
    feature_id = str(cfg['featurizer'].get('feature_id', 'm2_v1'))
    feats_dir = paths.FEATURES / f'features_{feature_id}_memmap'
    meta_path = paths.FEATURES / f'features_{feature_id}_memmap_meta.json'
    scaffold_meta_path = paths.FEATURES / f'scaffold_meta_{feature_id}.csv'
    if not feats_dir.exists():
        raise FileNotFoundError(f'Missing {feats_dir}. Run pipelines/m2/build_features.py first.')

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    feats = open_structured_memmap_features(feats_dir)
    scaffold_lookup = _load_scaffold_lookup(scaffold_meta_path)
    binary_tasks = list(meta.get('binary_tasks') or list(feats.binary_tasks))
    binary_task_aliases = dict(meta.get('binary_task_aliases') or feats.binary_task_aliases)

    run_dir = get_pipeline_run_dir(paths.RESULTS, feature_id, prefix=str(cfg.get('run', {}).get('prefix', 'm2')))
    run_id = run_dir.name
    stage_dir = get_stage_results_dir(run_dir, '11_train_baselines')
    (run_dir / 'run_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    (run_dir / 'features_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    threshold_metric = str(cfg.get('training', {}).get('threshold_metric', 'bal_acc'))
    seed = int(cfg.get('training', {}).get('seed', 0))

    model_builders = {
        'logreg': lambda: LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced', random_state=seed),
        'rf': lambda: RandomForestClassifier(n_estimators=400, class_weight='balanced_subsample', n_jobs=-1, random_state=seed),
        'mlp': lambda: MLPClassifier(hidden_layer_sizes=(512, 256), early_stopping=True, max_iter=150, random_state=seed),
    }

    results = []
    confusion_payload: dict[str, dict] = {}
    outputs = [run_dir / 'run_config.yaml', run_dir / 'features_meta.json']

    for task_name in binary_tasks:
        task_alias = binary_task_aliases[task_name]
        splits = _labeled_split_arrays(feats, task_name)
        X_train, y_train, train_idx = splits['train']
        X_val, y_val, val_idx = splits['val']
        X_test, y_test, test_idx = splits['test']

        if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
            continue

        pred_rows = []
        confusion_payload[task_alias] = {'task_name': task_name}
        for model_name, builder in model_builders.items():
            model = builder()
            model.fit(X_train, y_train)
            p_val = _prob_pos(model, X_val)
            thr = float(select_binary_threshold(y_val, p_val, metric=threshold_metric))
            p_test = _prob_pos(model, X_test)
            p_train = _prob_pos(model, X_train)

            train_metrics = binary_metrics(y_train, p_train, threshold=thr)
            val_metrics = binary_metrics(y_val, p_val, threshold=thr)
            test_metrics = binary_metrics(y_test, p_test, threshold=thr)

            train_cm = _cm_counts(y_train, p_train, thr)
            val_cm = _cm_counts(y_val, p_val, thr)
            test_cm = _cm_counts(y_test, p_test, thr)

            results.append({
                'task': task_alias,
                'task_name': task_name,
                'model': model_name,
                'threshold_metric': threshold_metric,
                'threshold': thr,
                'train_n': int(train_metrics['n']),
                'train_acc': float(train_metrics['acc']),
                'train_bal_acc': float(train_metrics['bal_acc']),
                'train_f1': float(train_metrics['f1']),
                'train_precision': float(train_metrics['precision']),
                'train_recall': float(train_metrics['recall']),
                'train_specificity': float(train_metrics['specificity']),
                'train_auroc': float(train_metrics['auroc']),
                'train_auprc': float(train_metrics['auprc']),
                'train_tn': train_cm['tn'],
                'train_fp': train_cm['fp'],
                'train_fn': train_cm['fn'],
                'train_tp': train_cm['tp'],
                'val_n': int(val_metrics['n']),
                'val_acc': float(val_metrics['acc']),
                'val_bal_acc': float(val_metrics['bal_acc']),
                'val_f1': float(val_metrics['f1']),
                'val_precision': float(val_metrics['precision']),
                'val_recall': float(val_metrics['recall']),
                'val_specificity': float(val_metrics['specificity']),
                'val_auroc': float(val_metrics['auroc']),
                'val_auprc': float(val_metrics['auprc']),
                'val_tn': val_cm['tn'],
                'val_fp': val_cm['fp'],
                'val_fn': val_cm['fn'],
                'val_tp': val_cm['tp'],
                'test_n': int(test_metrics['n']),
                'test_acc': float(test_metrics['acc']),
                'test_bal_acc': float(test_metrics['bal_acc']),
                'test_f1': float(test_metrics['f1']),
                'test_precision': float(test_metrics['precision']),
                'test_recall': float(test_metrics['recall']),
                'test_specificity': float(test_metrics['specificity']),
                'test_auroc': float(test_metrics['auroc']),
                'test_auprc': float(test_metrics['auprc']),
                'test_tn': test_cm['tn'],
                'test_fp': test_cm['fp'],
                'test_fn': test_cm['fn'],
                'test_tp': test_cm['tp'],
            })

            confusion_payload[task_alias][model_name] = {
                'threshold_metric': threshold_metric,
                'threshold': thr,
                'train': train_cm,
                'val': val_cm,
                'test': test_cm,
            }

            _append_prediction_rows(pred_rows, feats=feats, idx_arr=val_idx, y_true=y_val, probs_pos=p_val, threshold=thr, task_name=task_name, task_alias=task_alias, model_name=model_name, split_name='val', scaffold_lookup=scaffold_lookup)
            _append_prediction_rows(pred_rows, feats=feats, idx_arr=test_idx, y_true=y_test, probs_pos=p_test, threshold=thr, task_name=task_name, task_alias=task_alias, model_name=model_name, split_name='test', scaffold_lookup=scaffold_lookup)

        if pred_rows:
            pred_path = run_dir / f'baseline_predictions_{task_alias}.csv'
            pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
            outputs.append(pred_path)

    results_df = pd.DataFrame(results)
    results_csv = run_dir / 'baseline_results.csv'
    results_json = run_dir / 'baseline_results.json'
    confusion_json = run_dir / 'baseline_confusion_matrices.json'
    stage_summary_json = stage_dir / 'summary.json'

    results_df.to_csv(results_csv, index=False)
    results_json.write_text(results_df.to_json(orient='records', indent=2), encoding='utf-8')
    write_json(confusion_json, confusion_payload)
    write_json(stage_summary_json, {
        'run_id': run_id,
        'feature_id': feature_id,
        'threshold_metric': threshold_metric,
        'tasks': sorted(results_df['task'].unique().tolist()) if not results_df.empty else [],
        'task_names': sorted(results_df['task_name'].unique().tolist()) if not results_df.empty else [],
        'models': sorted(results_df['model'].unique().tolist()) if not results_df.empty else [],
        'n_result_rows': int(len(results_df)),
        'artifacts': [str(results_csv.name), str(results_json.name), str(confusion_json.name)],
    })
    outputs.extend([results_csv, results_json, confusion_json, stage_summary_json])

    label_inputs = [feats_dir / feats.binary_task_files[task] for task in binary_tasks]
    write_manifest(
        stage_dir / 'manifest.json',
        stage_name='11_train_baselines',
        config_path=cfg_path,
        inputs=[meta_path, feats_dir / 'X_fp.npy', feats_dir / 'X_desc.npy', *label_inputs, scaffold_meta_path],
        outputs=outputs,
        extra={
            'run_id': run_id,
            'feature_id': feature_id,
            'n_rows': int(len(results_df)),
            'tasks': sorted(results_df['task'].unique().tolist()) if not results_df.empty else [],
            'task_names': sorted(results_df['task_name'].unique().tolist()) if not results_df.empty else [],
            'models': sorted(results_df['model'].unique().tolist()) if not results_df.empty else [],
        },
    )
    print(f'[ok] run_dir: {run_dir}')


if __name__ == '__main__':
    main()
