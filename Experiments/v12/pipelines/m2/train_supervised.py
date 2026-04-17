"""Train a supervised-only multitask MLP on frozen features."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.dataio.m2_memmap_datamodule import open_structured_memmap_features  # noqa: E402
from src.training.checkpoints import save_checkpoint  # noqa: E402
from src.training.m2_metrics import binary_metrics, confusion_from_probs, select_binary_threshold  # noqa: E402
from src.utils.fingerprint import build_data_fingerprint  # noqa: E402
from src.utils.logging import RunLogger, get_pipeline_run_dir, get_stage_results_dir, write_json, write_manifest  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402


class SupervisedMultitaskDataset(Dataset):
    def __init__(self, feats, indices: np.ndarray, task_entries: list[dict[str, str]]):
        self.feats = feats
        self.indices = np.asarray(indices, dtype=np.int64)
        self.task_entries = list(task_entries)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> dict:
        j = int(self.indices[i])
        x_fp = torch.from_numpy(np.array(self.feats.X_fp[j], copy=True)).float()
        x_desc = torch.from_numpy(np.array(self.feats.X_desc[j], copy=True)).float()
        item = {'x_fp': x_fp, 'x_desc': x_desc, 'inchi_key': self.feats.inchi_key[j]}
        for entry in self.task_entries:
            alias = entry['alias']
            yv = int(self.feats.get_label_array(entry['name'])[j])
            item[f'y_{alias}'] = torch.tensor(yv, dtype=torch.long)
            item[f'mask_{alias}'] = torch.tensor(1 if yv >= 0 else 0, dtype=torch.float32)
        return item


class MultitaskMLP(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...], dropout: float, task_aliases: list[str]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(d_in)
        for h in hidden:
            layers.extend([nn.Linear(prev, int(h)), nn.ReLU(), nn.Dropout(float(dropout))])
            prev = int(h)
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.heads = nn.ModuleDict({alias: nn.Linear(prev, 1) for alias in task_aliases})

    def forward(self, x_fp: torch.Tensor, x_desc: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([x_fp, x_desc], dim=1)
        h = self.backbone(x)
        out = {alias: head(h).squeeze(1) for alias, head in self.heads.items()}
        out['h'] = h
        return out


@dataclass
class SupervisedEval:
    metrics: dict[str, dict[str, float]]
    predictions: list[dict]
    confusion: dict[str, np.ndarray]
    arrays: dict[str, dict[str, np.ndarray]]


@dataclass
class SupervisedTrainResult:
    best_epoch: int
    best_score: float
    checkpoint_used: str



def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))



def _balanced_pos_weight(counts: dict) -> float | None:
    neg = float(counts.get('neg', 0))
    pos = float(counts.get('pos', 0))
    if neg <= 0 or pos <= 0:
        return None
    return float(neg / pos)



def _decode_inchi_key(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode('ascii', errors='ignore')
    if hasattr(value, 'tobytes'):
        return value.tobytes().decode('ascii', errors='ignore')
    return str(value)



def _save_confusion(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr is None or getattr(arr, 'size', 0) == 0:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for row in arr.tolist():
            w.writerow(row)



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



def _build_supervised_indices(feats, split_code: int, task_entries: list[dict[str, str]]) -> np.ndarray:
    split = np.asarray(feats.split_code)
    known_mask = np.zeros_like(split, dtype=bool)
    for entry in task_entries:
        known_mask |= np.asarray(feats.get_label_array(entry['name'])) >= 0
    mask = (split == split_code) & known_mask
    return np.where(mask)[0].astype(np.int64)



def _make_loader(feats, indices: np.ndarray, *, batch_size: int, shuffle: bool, task_entries: list[dict[str, str]]) -> DataLoader:
    ds = SupervisedMultitaskDataset(feats, indices, task_entries)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0, pin_memory=False)



def _loss_for_batch(logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, task_entries: list[dict[str, str]], pos_weights: dict[str, float | None], alphas: dict[str, float], device: str) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.tensor(0.0, device=device)
    info = {'loss': 0.0}
    for entry in task_entries:
        alias = entry['alias']
        mask = batch[f'mask_{alias}'].to(device) > 0.5
        y = batch[f'y_{alias}'].to(device)
        if mask.any():
            pos_weight = pos_weights.get(alias)
            pw = None if pos_weight is None else torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
            task_loss = F.binary_cross_entropy_with_logits(logits[alias][mask], y[mask].float(), pos_weight=pw)
        else:
            task_loss = torch.tensor(0.0, device=device)
        total = total + float(alphas.get(alias, 1.0)) * task_loss
        info[f'loss_{alias}'] = float(task_loss.detach().cpu())
    info['loss'] = float(total.detach().cpu())
    return total, info



def _score_from_metrics(metrics: dict[str, dict[str, float]], task_aliases: list[str]) -> float:
    vals = []
    for alias in task_aliases:
        m = metrics.get(alias, {})
        if not np.isnan(m.get('auprc', np.nan)):
            vals.append(m['auprc'])
        elif not np.isnan(m.get('bal_acc', np.nan)):
            vals.append(m['bal_acc'])
    return float(np.mean(vals)) if vals else float('nan')



def _evaluate(model: MultitaskMLP, dl: DataLoader, device: str, *, task_entries: list[dict[str, str]], thresholds: dict[str, float] | None = None) -> SupervisedEval:
    thresholds = thresholds or {}
    model.eval()

    true_store: dict[str, list[np.ndarray]] = {entry['alias']: [] for entry in task_entries}
    prob_store: dict[str, list[np.ndarray]] = {entry['alias']: [] for entry in task_entries}
    preds_rows: list[dict] = []

    with torch.no_grad():
        for batch in dl:
            x_fp = batch['x_fp'].to(device)
            x_desc = batch['x_desc'].to(device)
            out = model(x_fp, x_desc)
            h_cpu = out['h'].detach().cpu()

            prob_map = {entry['alias']: torch.sigmoid(out[entry['alias']]).detach().cpu() for entry in task_entries}
            pred_map = {entry['alias']: (prob_map[entry['alias']] >= float(thresholds.get(entry['alias'], 0.5))).long() for entry in task_entries}
            y_map = {entry['alias']: batch[f'y_{entry["alias"]}'].cpu() for entry in task_entries}
            mask_map = {entry['alias']: batch[f'mask_{entry["alias"]}'].cpu() > 0.5 for entry in task_entries}

            for i in range(x_fp.shape[0]):
                row = {'inchi_key': _decode_inchi_key(batch['inchi_key'][i])}
                for entry in task_entries:
                    alias = entry['alias']
                    row[f'y_{alias}_true'] = int(y_map[alias][i].item())
                    row[f'{alias}_known'] = int(mask_map[alias][i].item())
                    row[f'{alias}_pred'] = int(pred_map[alias][i].item())
                    row[f'p_{alias}'] = float(prob_map[alias][i].item())
                row.update({f'h_{k}': float(h_cpu[i, k].item()) for k in range(h_cpu.shape[1])})
                preds_rows.append(row)

            for entry in task_entries:
                alias = entry['alias']
                if mask_map[alias].any():
                    true_store[alias].append(y_map[alias][mask_map[alias]].numpy())
                    prob_store[alias].append(prob_map[alias][mask_map[alias]].numpy())

    metrics: dict[str, dict[str, float]] = {}
    confusion: dict[str, np.ndarray] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for entry in task_entries:
        alias = entry['alias']
        thr = float(thresholds.get(alias, 0.5))
        if true_store[alias]:
            y = np.concatenate(true_store[alias])
            p = np.concatenate(prob_store[alias])
            arrays[alias] = {'y_true': y, 'probs': p}
            metrics[alias] = binary_metrics(y, p, threshold=thr)
            confusion[alias] = confusion_from_probs(y, p, task=alias, threshold=thr)
        else:
            arrays[alias] = {'y_true': np.array([]), 'probs': np.array([])}
            metrics[alias] = binary_metrics(np.array([]), np.array([]), threshold=thr)
            confusion[alias] = np.zeros((0, 0), dtype=int)

    return SupervisedEval(metrics=metrics, predictions=preds_rows, confusion=confusion, arrays=arrays)



def _choose_thresholds(val_out: SupervisedEval, task_entries: list[dict[str, str]], metric: str) -> dict[str, float]:
    thresholds = {}
    for entry in task_entries:
        alias = entry['alias']
        arr = val_out.arrays.get(alias, {})
        y = arr.get('y_true')
        p = arr.get('probs')
        if y is not None and p is not None and len(y) > 0:
            thresholds[alias] = float(select_binary_threshold(y, p, metric=metric))
        else:
            thresholds[alias] = 0.5
    return thresholds



def _to_labeled_long_rows(rows: list[dict], *, split_name: str, task_entries: list[dict[str, str]], thresholds: dict[str, float], scaffold_lookup: dict[str, dict[str, object]], include_hidden: bool) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        ik = str(row.get('inchi_key', ''))
        scaffold = scaffold_lookup.get(ik, {})
        hidden_items = {k: v for k, v in row.items() if k.startswith('h_')} if include_hidden else {}
        for entry in task_entries:
            alias = entry['alias']
            if int(row.get(f'{alias}_known', 0)) != 1:
                continue
            p = float(row[f'p_{alias}'])
            pred = int(row[f'{alias}_pred'])
            y_true = int(row[f'y_{alias}_true'])
            out_row = {
                'split': split_name,
                'task': alias,
                'task_name': entry['name'],
                'inchi_key': ik,
                'scaffold_smiles': scaffold.get('scaffold_smiles'),
                'scaffold_hash': scaffold.get('scaffold_hash'),
                'y_true': y_true,
                'p_pos': p,
                'pred': pred,
                'threshold': float(thresholds.get(alias, 0.5)),
                'correct': int(pred == y_true),
            }
            if include_hidden:
                out_row.update(hidden_items)
            out.append(out_row)
    return out



def _save_prediction_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=['split', 'task', 'task_name', 'inchi_key', 'scaffold_smiles', 'scaffold_hash', 'y_true', 'p_pos', 'pred', 'threshold', 'correct']).to_csv(path, index=False)



def main() -> None:
    ap = argparse.ArgumentParser(description='Train supervised-only multitask MLP.')
    ap.add_argument('--config', type=str, default='pipelines/m2/config_gpu.yaml')
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
    feats = open_structured_memmap_features(feats_dir)
    scaffold_lookup = _load_scaffold_lookup(scaffold_meta_path)
    binary_tasks = list(meta.get('binary_tasks') or list(feats.binary_tasks))
    binary_task_aliases = dict(meta.get('binary_task_aliases') or feats.binary_task_aliases)
    task_entries = [{'name': task, 'alias': binary_task_aliases[task]} for task in binary_tasks]
    if not task_entries:
        raise RuntimeError('No active binary tasks were found for supervised training.')

    run_dir = get_pipeline_run_dir(paths.RESULTS, feature_id, prefix=str(cfg.get('run', {}).get('prefix', 'm2')))
    run_id = run_dir.name
    ckpt_dir = run_dir / 'checkpoints_supervised'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir, stage_name='12_train_supervised_gpu')
    stage_dir = get_stage_results_dir(run_dir, '12_train_supervised_gpu')

    (run_dir / 'run_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    (run_dir / 'features_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    for src in [split_summary_path, split_label_counts_path, scaffold_meta_path, split_file_path, split_optimizer_path]:
        if src.exists() and not (run_dir / src.name).exists():
            shutil.copy2(src, run_dir / src.name)

    db_path = Path(cfg['db']['path'])
    if not db_path.is_absolute():
        db_path = paths.ROOT / db_path
    fp = build_data_fingerprint(db_path=db_path, featurizer_cfg=cfg.get('featurizer', {}), label_cfg=cfg.get('tasks', {}), split_cfg=cfg.get('split', {}))
    data_fingerprint_path = run_dir / 'data_fingerprint.json'
    if not data_fingerprint_path.exists():
        data_fingerprint_path.write_text(json.dumps(fp, indent=2), encoding='utf-8')

    seed = int(cfg['training'].get('seed', 0))
    set_global_seed(seed)
    device = str(cfg['training'].get('device', 'cpu'))
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    logger.log(f'device={device}')

    batch_size = int(cfg['training']['batch_size'])
    train_idx = _build_supervised_indices(feats, 0, task_entries)
    val_idx = _build_supervised_indices(feats, 1, task_entries)
    test_idx = _build_supervised_indices(feats, 2, task_entries)
    train_dl = _make_loader(feats, train_idx, batch_size=batch_size, shuffle=True, task_entries=task_entries)
    train_eval_dl = _make_loader(feats, train_idx, batch_size=batch_size, shuffle=False, task_entries=task_entries)
    val_dl = _make_loader(feats, val_idx, batch_size=batch_size, shuffle=False, task_entries=task_entries)
    test_dl = _make_loader(feats, test_idx, batch_size=batch_size, shuffle=False, task_entries=task_entries)

    d_in = int(meta['d_fp']) + int(meta['d_desc'])
    hidden = tuple(int(x) for x in cfg['model']['clf_hidden'])
    dropout = float(cfg['model'].get('dropout', 0.1))
    task_aliases = [entry['alias'] for entry in task_entries]
    model = MultitaskMLP(d_in=d_in, hidden=hidden, dropout=dropout, task_aliases=task_aliases).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['weight_decay']))

    pos_weights = {}
    class_counts_by_task = meta.get('train_class_counts_by_task') or {}
    if str(cfg['training'].get('pos_weight_mode', 'none')).lower() == 'balanced':
        for entry in task_entries:
            pos_weights[entry['alias']] = _balanced_pos_weight(class_counts_by_task.get(entry['name'], {}))
    else:
        pos_weights = {entry['alias']: None for entry in task_entries}

    default_alpha = float(cfg['training'].get('alpha_binary_default', 5.0))
    alphas = {entry['alias']: float(cfg['training'].get(f'alpha_{entry["alias"]}', default_alpha)) for entry in task_entries}

    best_score = -1.0
    best_epoch = -1
    bad = 0
    epochs = int(cfg['training']['epochs'])
    patience = int(cfg['training']['patience'])
    grad_clip = float(cfg['training'].get('grad_clip', 5.0))
    threshold_metric = str(cfg['training'].get('threshold_metric', 'bal_acc'))

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_dl:
            x_fp = batch['x_fp'].to(device)
            x_desc = batch['x_desc'].to(device)
            logits = model(x_fp, x_desc)
            loss, info = _loss_for_batch(logits, batch, task_entries=task_entries, pos_weights=pos_weights, alphas=alphas, device=device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            logger.log_metrics({'epoch': epoch, **info})

        val_raw = _evaluate(model, val_dl, device, task_entries=task_entries)
        val_thresholds = _choose_thresholds(val_raw, task_entries, threshold_metric)
        val_out = _evaluate(model, val_dl, device, task_entries=task_entries, thresholds=val_thresholds)
        score = _score_from_metrics(val_out.metrics, task_aliases)
        logger.log(f'epoch {epoch}: val={val_out.metrics}')
        metric_row = {'epoch': epoch, 'val_score': score}
        for alias in task_aliases:
            metric_row[f'val_{alias}_auprc'] = val_out.metrics[alias].get('auprc')
            metric_row[f'val_{alias}_bal_acc'] = val_out.metrics[alias].get('bal_acc')
        logger.log_metrics(metric_row)
        save_checkpoint(ckpt_dir / 'last.pt', model=model, optimizer=opt, epoch=epoch, extra={'val': val_out.metrics, 'score': score, 'thresholds': val_thresholds})
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            save_checkpoint(ckpt_dir / 'best.pt', model=model, optimizer=opt, epoch=epoch, extra={'val': val_out.metrics, 'score': score, 'thresholds': val_thresholds})
        else:
            bad += 1
            if bad >= patience:
                logger.log(f'early stop at epoch {epoch} (best epoch {best_epoch}, best score {best_score:.4f})')
                break

    ckpt_name = 'best.pt' if (ckpt_dir / 'best.pt').exists() else 'last.pt'
    payload = torch.load(ckpt_dir / ckpt_name, map_location='cpu')
    model.load_state_dict(payload['model_state'])
    model.to(device)

    val_raw = _evaluate(model, val_dl, device, task_entries=task_entries)
    thresholds = _choose_thresholds(val_raw, task_entries, threshold_metric)
    train_out = _evaluate(model, train_eval_dl, device, task_entries=task_entries, thresholds=thresholds)
    val_out = _evaluate(model, val_dl, device, task_entries=task_entries, thresholds=thresholds)
    test_out = _evaluate(model, test_dl, device, task_entries=task_entries, thresholds=thresholds)

    log_cfg = cfg.get('logging', {}) or {}
    include_hidden = bool(log_cfg.get('include_prediction_latents', False))
    outputs = [run_dir / 'run_config.yaml', run_dir / 'features_meta.json', data_fingerprint_path, ckpt_dir / ckpt_name]

    if bool(log_cfg.get('save_train_predictions', False)):
        train_pred_path = run_dir / 'supervised_predictions_train.csv'
        _save_prediction_csv(train_pred_path, _to_labeled_long_rows(train_out.predictions, split_name='train', task_entries=task_entries, thresholds=thresholds, scaffold_lookup=scaffold_lookup, include_hidden=include_hidden))
        outputs.append(train_pred_path)
    if bool(log_cfg.get('save_val_predictions', True)):
        val_pred_path = run_dir / 'supervised_predictions_val.csv'
        _save_prediction_csv(val_pred_path, _to_labeled_long_rows(val_out.predictions, split_name='val', task_entries=task_entries, thresholds=thresholds, scaffold_lookup=scaffold_lookup, include_hidden=include_hidden))
        outputs.append(val_pred_path)
    if bool(log_cfg.get('save_test_predictions', True)):
        test_pred_path = run_dir / 'supervised_predictions_test.csv'
        _save_prediction_csv(test_pred_path, _to_labeled_long_rows(test_out.predictions, split_name='test', task_entries=task_entries, thresholds=thresholds, scaffold_lookup=scaffold_lookup, include_hidden=include_hidden))
        outputs.append(test_pred_path)

    confusion_payload = {'threshold_metric': threshold_metric, 'thresholds': thresholds}
    for entry in task_entries:
        alias = entry['alias']
        for split_name, out in [('train', train_out), ('val', val_out), ('test', test_out)]:
            path = run_dir / f'supervised_confusion_{alias}_{split_name}.csv'
            _save_confusion(path, out.confusion[alias])
            outputs.append(path)
        confusion_payload[alias] = {
            'task_name': entry['name'],
            'train': train_out.confusion[alias].tolist(),
            'val': val_out.confusion[alias].tolist(),
            'test': test_out.confusion[alias].tolist(),
        }

    confusion_json = run_dir / 'supervised_confusion_matrices.json'
    write_json(confusion_json, confusion_payload)
    outputs.append(confusion_json)

    eval_json = {
        'run_id': run_id,
        'best_epoch': int(best_epoch),
        'best_score': float(best_score),
        'checkpoint_used': ckpt_name,
        'threshold_metric': threshold_metric,
        'thresholds': thresholds,
        'train': train_out.metrics,
        'val': val_out.metrics,
        'test': test_out.metrics,
        'task_aliases': task_aliases,
        'task_names': {entry['alias']: entry['name'] for entry in task_entries},
    }
    eval_path = run_dir / 'supervised_eval.json'
    eval_path.write_text(json.dumps(eval_json, indent=2), encoding='utf-8')
    outputs.append(eval_path)

    summary_json = stage_dir / 'summary.json'
    write_json(summary_json, {
        'run_id': run_id,
        'feature_id': feature_id,
        'best_epoch': int(best_epoch),
        'best_score': float(best_score),
        'checkpoint_used': ckpt_name,
        'threshold_metric': threshold_metric,
        'thresholds': thresholds,
        'task_aliases': task_aliases,
        'task_names': {entry['alias']: entry['name'] for entry in task_entries},
        'include_prediction_latents': include_hidden,
    })
    outputs.append(summary_json)

    label_inputs = [feats_dir / feats.binary_task_files[task] for task in binary_tasks]
    write_manifest(
        stage_dir / 'manifest.json',
        stage_name='12_train_supervised_gpu',
        config_path=cfg_path,
        inputs=[meta_path, feats_dir / 'X_fp.npy', feats_dir / 'X_desc.npy', *label_inputs, scaffold_meta_path, split_file_path],
        outputs=outputs,
        extra={
            'run_id': run_id,
            'feature_id': feature_id,
            'best_epoch': int(best_epoch),
            'best_score': float(best_score),
            'thresholds': thresholds,
            'task_aliases': task_aliases,
        },
    )
    logger.log(f'done: {eval_json}')
    print(f'[ok] run_dir: {run_dir}')


if __name__ == '__main__':
    main()
