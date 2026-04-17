"""Train structured multitask M2 VAE on memmap features."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.dataio.m2_memmap_datamodule import build_structured_dataloaders_memmap, open_structured_memmap_features  # noqa: E402
from src.models.m2 import StructuredM2Dims, StructuredM2VAE  # noqa: E402
from src.training.m2_loops import StructuredPriors, evaluate_structured, train_structured_m2, _score_from_metrics  # noqa: E402
from src.training.m2_metrics import select_binary_threshold  # noqa: E402
from src.training.checkpoints import load_checkpoint  # noqa: E402
from src.utils.fingerprint import build_data_fingerprint  # noqa: E402
from src.utils.logging import RunLogger, write_manifest, get_pipeline_run_dir, get_stage_results_dir, write_json  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))



def _save_confusion(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr is None or getattr(arr, 'size', 0) == 0:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for row in arr.tolist():
            w.writerow(row)



def _balanced_pos_weight(counts: dict) -> float | None:
    neg = float(counts.get('neg', 0))
    pos = float(counts.get('pos', 0))
    if neg <= 0 or pos <= 0:
        return None
    return float(neg / pos)



def _choose_thresholds(val_out, task_aliases: list[str], metric: str) -> dict[str, float]:
    thresholds = {}
    for task in task_aliases:
        arr = val_out.arrays.get(task, {})
        y = arr.get('y_true')
        p = arr.get('probs')
        if y is not None and p is not None and len(y) > 0:
            thresholds[task] = float(select_binary_threshold(y, p, metric=metric))
        else:
            thresholds[task] = 0.5
    return thresholds



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



def _to_labeled_long_rows(rows: list[dict], *, split_name: str, thresholds: dict[str, float], scaffold_lookup: dict[str, dict[str, object]], include_latents: bool, task_aliases: list[str]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        ik = str(row.get('inchi_key', ''))
        scaffold = scaffold_lookup.get(ik, {})
        latent_items = {k: v for k, v in row.items() if k.startswith('z_mu_')} if include_latents else {}
        for task in task_aliases:
            if int(row.get(f'{task}_known', 0)) != 1:
                continue
            p = float(row[f'p_{task}'])
            pred = int(row[f'{task}_pred'])
            y_true = int(row[f'y_{task}_true'])
            out_row = {
                'split': split_name,
                'task': task,
                'inchi_key': ik,
                'scaffold_smiles': scaffold.get('scaffold_smiles'),
                'scaffold_hash': scaffold.get('scaffold_hash'),
                'y_true': y_true,
                'p_pos': p,
                'pred': pred,
                'threshold': float(thresholds.get(task, 0.5)),
                'correct': int(pred == y_true),
            }
            if include_latents:
                out_row.update(latent_items)
            out.append(out_row)
    return out



def _save_prediction_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=['split', 'task', 'inchi_key', 'scaffold_smiles', 'scaffold_hash', 'y_true', 'p_pos', 'pred', 'threshold', 'correct']).to_csv(path, index=False)



def _prediction_summary_for_rows(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    if not rows:
        return out
    df = pd.DataFrame(rows)
    for split_name in sorted(df['split'].dropna().unique().tolist()):
        sdf = df[df['split'] == split_name]
        out[str(split_name)] = {}
        for task in sorted(sdf['task'].dropna().unique().tolist()):
            tdf = sdf[sdf['task'] == task]
            probs = tdf['p_pos'].to_numpy(dtype=float)
            y = tdf['y_true'].to_numpy(dtype=int)
            pos_probs = probs[y == 1]
            neg_probs = probs[y == 0]
            out[str(split_name)][str(task)] = {
                'n': int(len(tdf)),
                'n_pos': int((y == 1).sum()),
                'n_neg': int((y == 0).sum()),
                'accuracy': float(tdf['correct'].mean()) if len(tdf) else float('nan'),
                'prob_mean': float(np.mean(probs)) if len(probs) else float('nan'),
                'prob_std': float(np.std(probs)) if len(probs) else float('nan'),
                'prob_quantiles': {q: float(np.quantile(probs, qv)) for q, qv in [('q01', 0.01), ('q05', 0.05), ('q25', 0.25), ('q50', 0.50), ('q75', 0.75), ('q95', 0.95), ('q99', 0.99)]} if len(probs) else {},
                'prob_mean_pos': float(np.mean(pos_probs)) if len(pos_probs) else float('nan'),
                'prob_mean_neg': float(np.mean(neg_probs)) if len(neg_probs) else float('nan'),
            }
    return out



def _latent_summary(pred_rows: list[dict]) -> dict:
    if not pred_rows:
        return {}
    latent_cols = [k for k in pred_rows[0].keys() if k.startswith('z_mu_')]
    if not latent_cols:
        return {}
    out = {}
    df = pd.DataFrame(pred_rows)
    known_cols = [c for c in df.columns if c.endswith('_known')]
    split_masks = {'all_rows': np.ones(len(df), dtype=bool)}
    for col in known_cols:
        split_masks[f'{col[:-6]}_labeled_rows'] = df[col].to_numpy(dtype=int) == 1
    Z = df[latent_cols].to_numpy(dtype=float)
    for name, mask in split_masks.items():
        Zm = Z[mask]
        if Zm.size == 0:
            out[name] = {'n_rows': 0}
            continue
        mean_per_dim = Zm.mean(axis=0)
        var_per_dim = Zm.var(axis=0)
        out[name] = {
            'n_rows': int(Zm.shape[0]),
            'mean_abs_mean': float(np.mean(np.abs(mean_per_dim))),
            'mean_var': float(np.mean(var_per_dim)),
            'median_var': float(np.median(var_per_dim)),
            'n_active_var_gt_1e-2': int(np.sum(var_per_dim > 1e-2)),
            'n_active_var_gt_1e-3': int(np.sum(var_per_dim > 1e-3)),
            'per_dim_mean': [float(x) for x in mean_per_dim.tolist()],
            'per_dim_var': [float(x) for x in var_per_dim.tolist()],
        }
    return out



def _save_probability_arrays(path: Path, *, task_aliases: list[str], train_out, val_out, test_out) -> None:
    arrays = {}
    for split_name, out in [('train', train_out), ('val', val_out), ('test', test_out)]:
        for task in task_aliases:
            arr = out.arrays.get(task, {})
            y = arr.get('y_true')
            p = arr.get('probs')
            if y is not None and p is not None:
                arrays[f'{split_name}_{task}_y_true'] = np.asarray(y)
                arrays[f'{split_name}_{task}_probs'] = np.asarray(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)



def main() -> None:
    ap = argparse.ArgumentParser(description='Train structured multitask M2 VAE.')
    ap.add_argument('--config', type=str, default='pipelines/m2/config.yaml')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)
    paths = Paths()

    feature_id = str(cfg['featurizer'].get('feature_id', 'm2_v1'))
    feats_dir = paths.FEATURES / f'features_{feature_id}_memmap'
    meta_path = paths.FEATURES / f'features_{feature_id}_memmap_meta.json'
    split_summary_path = paths.FEATURES / f'split_summary_{feature_id}.csv'
    split_label_counts_path = paths.FEATURES / f'split_label_counts_{feature_id}.csv'
    scaffold_meta_path = paths.FEATURES / f'scaffold_meta_{feature_id}.csv'
    split_file_path = paths.SPLITS / f'split_scaffold_{feature_id}.csv'
    split_optimizer_path = paths.SPLITS / f'split_optimizer_summary_{feature_id}.json'

    if not feats_dir.exists():
        raise FileNotFoundError(f'Missing {feats_dir}. Run pipelines/m2/build_features.py first.')

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    include_protox = bool(meta.get('include_protox', False))
    binary_tasks = list(meta.get('binary_tasks') or [])
    binary_aliases = dict(meta.get('binary_task_aliases') or {})
    task_aliases = [binary_aliases[t] for t in binary_tasks]

    run_dir = get_pipeline_run_dir(paths.RESULTS, feature_id, prefix=str(cfg.get('run', {}).get('prefix', 'm2')))
    run_id = run_dir.name
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir, stage_name='10_train_gpu')
    stage_dir = get_stage_results_dir(run_dir, '10_train_gpu')

    (run_dir / 'run_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    (run_dir / 'features_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    copied_inputs = [meta_path]
    for src in [split_summary_path, split_label_counts_path, scaffold_meta_path, split_file_path, split_optimizer_path]:
        if src.exists():
            shutil.copy2(src, run_dir / src.name)
            copied_inputs.append(src)

    db_path = Path(cfg['db']['path'])
    if not db_path.is_absolute():
        db_path = paths.ROOT / db_path
    fp = build_data_fingerprint(db_path=db_path, featurizer_cfg=cfg.get('featurizer', {}), label_cfg=cfg.get('tasks', {}), split_cfg=cfg.get('split', {}))
    data_fingerprint_path = run_dir / 'data_fingerprint.json'
    data_fingerprint_path.write_text(json.dumps(fp, indent=2), encoding='utf-8')

    seed = int(cfg['training'].get('seed', 0))
    set_global_seed(seed)

    device = str(cfg['training'].get('device', 'cpu'))
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    logger.log(f'device={device}')

    feats = open_structured_memmap_features(feats_dir)
    training_cfg = cfg.get('training', {}) or {}
    train_label_fraction_by_task = training_cfg.get('train_label_fraction_by_task', {}) or {}
    dls = build_structured_dataloaders_memmap(
        feats,
        batch_size=int(training_cfg['batch_size']),
        num_workers=0,
        pin_memory=False,
        train_mix=training_cfg.get('batch_mix', {}),
        seed=seed,
        train_label_fraction=float(training_cfg.get('train_label_fraction', 1.0)),
        train_label_fraction_by_task=train_label_fraction_by_task,
        train_label_mask_seed=int(training_cfg.get('train_label_mask_seed', seed)),
        train_label_fraction_protox=float(training_cfg.get('train_label_fraction_protox', training_cfg.get('train_label_fraction', 1.0))),
    )

    priors = StructuredPriors(
        binary_pos={binary_aliases[t]: float(meta['priors_by_task'][t]) for t in binary_tasks},
        protox=torch.tensor(meta['priors']['protox'], dtype=torch.float32) if include_protox else None,
    )

    dims = StructuredM2Dims(
        d_fp=int(meta['d_fp']),
        d_desc=int(meta['d_desc']),
        z_dim=int(cfg['model']['z_dim']),
        binary_tasks=tuple(task_aliases),
        include_protox=include_protox,
    )
    n_unl = len(dls.train_unlabeled) if dls.train_unlabeled is not None else 0
    logger.log(f'train loader sizes: all={len(dls.train_all)} pretrain_labeled={len(dls.train_pretrain_labeled)} joint_labeled={len(dls.train_joint_labeled)} unlabeled={n_unl} val={len(dls.val_all)} test={len(dls.test_all)}')

    model = StructuredM2VAE(
        dims,
        clf_hidden=tuple(int(x) for x in cfg['model']['clf_hidden']),
        enc_hidden=tuple(int(x) for x in cfg['model']['enc_hidden']),
        dec_hidden=tuple(int(x) for x in cfg['model']['dec_hidden']),
        dropout=float(cfg['model'].get('dropout', 0.1)),
        ifm_num_frequencies=int(cfg['model'].get('ifm_num_frequencies', 0)),
        ifm_learnable=bool(cfg['model'].get('ifm_learnable', True)),
        ifm_include_raw=bool(cfg['model'].get('ifm_include_raw', False)),
        ifm_init_std=float(cfg['model'].get('ifm_init_std', 6.0)),
        ifm_apply_to_encoder=bool(cfg['model'].get('ifm_apply_to_encoder', False)),
        fp_recon_distribution=str(cfg['model'].get('fp_recon_distribution', 'bernoulli')),
    )

    pos_weights = {}
    if str(training_cfg.get('pos_weight_mode', 'none')).lower() == 'balanced':
        counts_by_task = meta.get('train_class_counts_by_task', {}) or {}
        for task in binary_tasks:
            pos_weights[binary_aliases[task]] = _balanced_pos_weight(counts_by_task.get(task, {}))

    alpha_binary_default = float(training_cfg.get('alpha_binary_default', 1.0))
    alpha_binary = {alias: float(training_cfg.get(f'alpha_{alias}', alpha_binary_default)) for alias in task_aliases}
    score_weights = {alias: float(training_cfg.get(f'score_weight_{alias}', 1.0)) for alias in task_aliases}

    training_regime = str(training_cfg.get('regime', 'joint')).lower().strip()
    train_beta_kl = float(training_cfg.get('beta_kl', 1.0))
    train_gen_weight = float(training_cfg.get('gen_weight', 1.0))
    train_joint_unlabeled_weight = float(training_cfg.get('joint_unlabeled_weight', 0.25))
    train_joint_unlabeled_batches_per_step = int(training_cfg.get('joint_unlabeled_batches_per_step', 1))
    train_min_joint_unlabeled_passes_before_early_stop = float(training_cfg.get('min_joint_unlabeled_passes_before_early_stop', 0.0))
    if training_regime == 'supcontrol':
        train_beta_kl = 0.0
        train_gen_weight = 0.0
        train_joint_unlabeled_weight = 0.0
        train_joint_unlabeled_batches_per_step = 0
        train_min_joint_unlabeled_passes_before_early_stop = 0.0
    elif training_regime != 'joint':
        raise ValueError(f'unknown training regime: {training_regime}')
    logger.log(
        'training_regime='
        f'{training_regime} train_label_fraction={float(training_cfg.get("train_label_fraction", 1.0)):.3f} '
        f'train_label_mask_seed={int(training_cfg.get("train_label_mask_seed", seed))}'
    )

    res = train_structured_m2(
        model=model,
        train_all=dls.train_all,
        train_pretrain_labeled=dls.train_pretrain_labeled,
        train_joint_labeled=dls.train_joint_labeled,
        train_unlabeled=dls.train_unlabeled,
        val_all=dls.val_all,
        device=device,
        priors=priors,
        run_dir=run_dir,
        epochs=int(cfg['training']['epochs']),
        lr=float(cfg['training']['lr']),
        weight_decay=float(cfg['training']['weight_decay']),
        beta_kl=train_beta_kl,
        lambda_recon=float(cfg['training'].get('lambda_recon', 1.0)),
        gen_weight=train_gen_weight,
        alpha_protox=float(cfg['training'].get('alpha_protox', 0.0)),
        alpha_binary=alpha_binary,
        score_weights=score_weights,
        grad_clip=float(cfg['training']['grad_clip']),
        patience=int(cfg['training']['patience']),
        normalize_recon=bool(cfg['training'].get('normalize_recon', False)),
        pos_weights=pos_weights,
        pretrain_supervised_epochs=int(cfg['training'].get('pretrain_supervised_epochs', 0)),
        pretrain_supervised_epochs_min=int(cfg['training'].get('pretrain_supervised_epochs_min', 0)),
        pretrain_supervised_epochs_max=int(cfg['training'].get('pretrain_supervised_epochs_max', 0)),
        pretrain_transition_patience=int(cfg['training'].get('pretrain_transition_patience', 2)),
        kl_warmup_epochs=int(cfg['training'].get('kl_warmup_epochs', 0)),
        gen_warmup_epochs=int(cfg['training'].get('gen_warmup_epochs', 0)),
        free_bits_per_dim=float(cfg['training'].get('free_bits_per_dim', 0.0)),
        joint_unlabeled_weight=train_joint_unlabeled_weight,
        joint_steps_mode=str(cfg['training'].get('joint_steps_mode', 'labeled')),
        joint_unlabeled_batches_per_step=train_joint_unlabeled_batches_per_step,
        min_joint_epochs_before_early_stop=int(cfg['training'].get('min_joint_epochs_before_early_stop', 0)),
        min_joint_unlabeled_passes_before_early_stop=train_min_joint_unlabeled_passes_before_early_stop,
        beta_schedule=str(cfg['training'].get('beta_schedule', 'cyclical')),
        beta_cycle_length=int(cfg['training'].get('beta_cycle_length', 12)),
        beta_cycle_ramp_ratio=float(cfg['training'].get('beta_cycle_ramp_ratio', 0.5)),
        sanity_guard_epoch=int(cfg['training'].get('sanity_guard_epoch', 2)),
        sanity_abort_on_constant_preds=bool(cfg['training'].get('sanity_abort_on_constant_preds', True)),
        logger=logger,
    )

    ckpt_path = run_dir / 'checkpoints' / res.checkpoint_used
    load_checkpoint(ckpt_path, model=model)
    model.to(device)

    val_raw = evaluate_structured(model, dls.val_all, device, priors)
    thresholds = _choose_thresholds(val_raw, task_aliases, metric=str(training_cfg.get('threshold_metric', 'bal_acc')))
    train_out = evaluate_structured(model, dls.train_eval_all, device, priors, thresholds=thresholds)
    val_out = evaluate_structured(model, dls.val_all, device, priors, thresholds=thresholds)
    test_out = evaluate_structured(model, dls.test_all, device, priors, thresholds=thresholds)

    checkpoint_eval = {}
    checkpoint_latent = {}
    for ckpt_name in ['best_pretrain.pt', 'best_joint.pt', 'best_equalweight.pt', 'best_overall.pt', 'last.pt']:
        path = run_dir / 'checkpoints' / ckpt_name
        if not path.exists():
            continue
        load_checkpoint(path, model=model)
        model.to(device)
        val_raw_ckpt = evaluate_structured(model, dls.val_all, device, priors)
        thresholds_ckpt = _choose_thresholds(val_raw_ckpt, task_aliases, metric=str(training_cfg.get('threshold_metric', 'bal_acc')))
        train_ckpt = evaluate_structured(model, dls.train_eval_all, device, priors, thresholds=thresholds_ckpt)
        val_ckpt = evaluate_structured(model, dls.val_all, device, priors, thresholds=thresholds_ckpt)
        test_ckpt = evaluate_structured(model, dls.test_all, device, priors, thresholds=thresholds_ckpt)
        checkpoint_eval[ckpt_name] = {
            'thresholds': thresholds_ckpt,
            'val_score_weighted': _score_from_metrics(val_ckpt.metrics, binary_tasks=tuple(task_aliases), score_weights=score_weights),
            'val_score_equalweight': _score_from_metrics(val_ckpt.metrics, binary_tasks=tuple(task_aliases), score_weights=None),
            'train': train_ckpt.metrics,
            'val': val_ckpt.metrics,
            'test': test_ckpt.metrics,
        }
        checkpoint_latent[ckpt_name] = {
            'train': _latent_summary(train_ckpt.predictions),
            'val': _latent_summary(val_ckpt.predictions),
            'test': _latent_summary(test_ckpt.predictions),
        }
    load_checkpoint(ckpt_path, model=model)
    model.to(device)

    log_cfg = cfg.get('logging', {}) or {}
    labeled_only = bool(log_cfg.get('predictions_labeled_only', True))
    include_latents = bool(log_cfg.get('include_prediction_latents', False))
    scaffold_lookup = _load_scaffold_lookup(scaffold_meta_path)

    outputs = [run_dir / 'run_config.yaml', run_dir / 'features_meta.json', data_fingerprint_path, ckpt_path]
    if bool(log_cfg.get('save_train_predictions', False)):
        train_pred_path = run_dir / 'predictions_train.csv'
        train_rows = _to_labeled_long_rows(train_out.predictions, split_name='train', thresholds=thresholds, scaffold_lookup=scaffold_lookup, include_latents=include_latents, task_aliases=task_aliases) if labeled_only else train_out.predictions
        _save_prediction_csv(train_pred_path, train_rows) if labeled_only else pd.DataFrame(train_rows).to_csv(train_pred_path, index=False)
        outputs.append(train_pred_path)
    else:
        train_rows = []
    if bool(log_cfg.get('save_val_predictions', True)):
        val_pred_path = run_dir / 'predictions_val.csv'
        val_rows = _to_labeled_long_rows(val_out.predictions, split_name='val', thresholds=thresholds, scaffold_lookup=scaffold_lookup, include_latents=include_latents, task_aliases=task_aliases) if labeled_only else val_out.predictions
        _save_prediction_csv(val_pred_path, val_rows) if labeled_only else pd.DataFrame(val_rows).to_csv(val_pred_path, index=False)
        outputs.append(val_pred_path)
    else:
        val_rows = []
    if bool(log_cfg.get('save_test_predictions', True)):
        test_pred_path = run_dir / 'predictions_test.csv'
        test_rows = _to_labeled_long_rows(test_out.predictions, split_name='test', thresholds=thresholds, scaffold_lookup=scaffold_lookup, include_latents=include_latents, task_aliases=task_aliases) if labeled_only else test_out.predictions
        _save_prediction_csv(test_pred_path, test_rows) if labeled_only else pd.DataFrame(test_rows).to_csv(test_pred_path, index=False)
        outputs.append(test_pred_path)
    else:
        test_rows = []

    if 'protox' in train_out.confusion:
        ptrain = run_dir / 'confusion_protox_train.csv'
        pval = run_dir / 'confusion_protox_val.csv'
        ptest = run_dir / 'confusion_protox_test.csv'
        _save_confusion(ptrain, train_out.confusion['protox'])
        _save_confusion(pval, val_out.confusion.get('protox'))
        _save_confusion(ptest, test_out.confusion.get('protox'))
        outputs.extend([ptrain, pval, ptest])

    confusion_payload = {
        'threshold_metric': str(cfg['training'].get('threshold_metric', 'bal_acc')),
        'thresholds': thresholds,
    }
    for task in task_aliases:
        train_path = run_dir / f'confusion_{task}_train.csv'
        val_path = run_dir / f'confusion_{task}_val.csv'
        test_path = run_dir / f'confusion_{task}_test.csv'
        _save_confusion(train_path, train_out.confusion[task])
        _save_confusion(val_path, val_out.confusion[task])
        _save_confusion(test_path, test_out.confusion[task])
        outputs.extend([train_path, val_path, test_path])
        confusion_payload[task] = {
            'train': train_out.confusion[task].tolist(),
            'val': val_out.confusion[task].tolist(),
            'test': test_out.confusion[task].tolist(),
        }

    if bool(log_cfg.get('save_probability_arrays', True)):
        prob_arrays_path = stage_dir / 'probability_arrays.npz'
        _save_probability_arrays(prob_arrays_path, task_aliases=task_aliases, train_out=train_out, val_out=val_out, test_out=test_out)
        outputs.append(prob_arrays_path)

    if bool(log_cfg.get('save_prediction_summary', True)):
        prediction_summary = {}
        if bool(log_cfg.get('save_train_predictions', False)):
            prediction_summary['train'] = _prediction_summary_for_rows(train_rows if labeled_only else train_out.predictions)
        if bool(log_cfg.get('save_val_predictions', True)):
            prediction_summary['val'] = _prediction_summary_for_rows(val_rows if labeled_only else val_out.predictions)
        if bool(log_cfg.get('save_test_predictions', True)):
            prediction_summary['test'] = _prediction_summary_for_rows(test_rows if labeled_only else test_out.predictions)
        prediction_summary_path = stage_dir / 'prediction_summary.json'
        write_json(prediction_summary_path, prediction_summary)
        outputs.append(prediction_summary_path)

    if bool(log_cfg.get('save_latent_summary', True)):
        latent_summary_path = stage_dir / 'latent_summary.json'
        write_json(latent_summary_path, {
            'train': _latent_summary(train_out.predictions),
            'val': _latent_summary(val_out.predictions),
            'test': _latent_summary(test_out.predictions),
        })
        outputs.append(latent_summary_path)
        latent_by_ckpt_path = stage_dir / 'latent_summary_by_checkpoint.json'
        write_json(latent_by_ckpt_path, checkpoint_latent)
        outputs.append(latent_by_ckpt_path)

    confusion_json = run_dir / 'm2_confusion_matrices.json'
    write_json(confusion_json, confusion_payload)
    outputs.append(confusion_json)

    eval_json = {
        'run_id': run_id,
        'best_epoch': int(res.best_epoch),
        'best_score': float(res.best_score),
        'best_pretrain_epoch': int(res.best_pretrain_epoch),
        'best_pretrain_score': float(res.best_pretrain_score) if not np.isnan(res.best_pretrain_score) else None,
        'best_joint_epoch': int(res.best_joint_epoch),
        'best_joint_score': float(res.best_joint_score) if not np.isnan(res.best_joint_score) else None,
        'best_equalweight_epoch': int(res.best_equalweight_epoch),
        'best_equalweight_score': float(res.best_equalweight_score) if not np.isnan(res.best_equalweight_score) else None,
        'pretrain_epochs_used': int(res.pretrain_epochs_used),
        'joint_unlabeled_batches_seen': int(res.joint_unlabeled_batches_seen),
        'checkpoint_used': res.checkpoint_used,
        'training_regime': training_regime,
        'train_label_fraction': float(training_cfg.get('train_label_fraction', 1.0)),
        'train_label_fraction_by_task': train_label_fraction_by_task,
        'train_label_mask_seed': int(training_cfg.get('train_label_mask_seed', seed)),
        'threshold_metric': str(training_cfg.get('threshold_metric', 'bal_acc')),
        'thresholds': thresholds,
        'train': train_out.metrics,
        'val': val_out.metrics,
        'test': test_out.metrics,
    }
    eval_path = run_dir / 'eval.json'
    eval_path.write_text(json.dumps(eval_json, indent=2), encoding='utf-8')
    outputs.append(eval_path)

    checkpoint_eval_path = run_dir / 'eval_by_checkpoint.json'
    checkpoint_eval_path.write_text(json.dumps(checkpoint_eval, indent=2), encoding='utf-8')
    outputs.append(checkpoint_eval_path)

    summary_json = stage_dir / 'summary.json'
    write_json(summary_json, {
        'run_id': run_id,
        'feature_id': feature_id,
        'best_epoch': int(res.best_epoch),
        'best_score': float(res.best_score),
        'best_pretrain_epoch': int(res.best_pretrain_epoch),
        'best_pretrain_score': float(res.best_pretrain_score) if not np.isnan(res.best_pretrain_score) else None,
        'best_joint_epoch': int(res.best_joint_epoch),
        'best_joint_score': float(res.best_joint_score) if not np.isnan(res.best_joint_score) else None,
        'best_equalweight_epoch': int(res.best_equalweight_epoch),
        'best_equalweight_score': float(res.best_equalweight_score) if not np.isnan(res.best_equalweight_score) else None,
        'pretrain_epochs_used': int(res.pretrain_epochs_used),
        'joint_unlabeled_batches_seen': int(res.joint_unlabeled_batches_seen),
        'checkpoint_used': res.checkpoint_used,
        'joint_steps_mode': str(training_cfg.get('joint_steps_mode', 'labeled')),
        'joint_unlabeled_batches_per_step': int(training_cfg.get('joint_unlabeled_batches_per_step', 1)),
        'training_regime': training_regime,
        'train_label_fraction': float(training_cfg.get('train_label_fraction', 1.0)),
        'train_label_fraction_by_task': train_label_fraction_by_task,
        'train_label_mask_seed': int(training_cfg.get('train_label_mask_seed', seed)),
        'threshold_metric': str(training_cfg.get('threshold_metric', 'bal_acc')),
        'thresholds': thresholds,
        'predictions_labeled_only': labeled_only,
        'include_prediction_latents': include_latents,
        'task_aliases': task_aliases,
        'evaluated_checkpoints': sorted(checkpoint_eval.keys()),
    })
    outputs.append(summary_json)

    input_paths = [meta_path, feats_dir / 'X_fp.npy', feats_dir / 'X_desc.npy', split_file_path, split_optimizer_path]
    input_paths.extend(feats_dir / meta['binary_task_files'][task] for task in binary_tasks)
    write_manifest(
        stage_dir / 'manifest.json',
        stage_name='10_train_gpu',
        config_path=cfg_path,
        inputs=input_paths,
        outputs=outputs,
        extra={'run_id': run_id, 'feature_id': feature_id, 'best_epoch': int(res.best_epoch), 'best_score': float(res.best_score), 'best_pretrain_epoch': int(res.best_pretrain_epoch), 'best_joint_epoch': int(res.best_joint_epoch), 'pretrain_epochs_used': int(res.pretrain_epochs_used), 'thresholds': thresholds, 'task_aliases': task_aliases, 'evaluated_checkpoints': sorted(checkpoint_eval.keys()), 'training_regime': training_regime, 'train_label_fraction': float(training_cfg.get('train_label_fraction', 1.0)), 'train_label_mask_seed': int(training_cfg.get('train_label_mask_seed', seed))},
    )
    logger.log(f'done: {eval_json}')
    print(f'[ok] run_dir: {run_dir}')


if __name__ == '__main__':
    main()
