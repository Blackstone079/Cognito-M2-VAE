from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.labels.task_registry import PROTOX_TASK

BINARY_TASKS = [
    'respiratory_toxicity',
    'ames_mutagenic',
    'dili_classification',
]

MULTITASK_FIELDS = [
    'inchi_key',
    'smiles',
    'mol_weight',
    'chem_formula',
    'drugbank_id',
    'chembl_id',
    'n_source_records',
    'protox_num_observations',
    'respiratory_num_observations',
    'ames_num_observations',
    'dili_num_observations',
    'protox_values_observed',
    'respiratory_values_observed',
    'ames_values_observed',
    'dili_values_observed',
    'protox_resolution',
    'respiratory_resolution',
    'ames_resolution',
    'dili_resolution',
    PROTOX_TASK,
    'respiratory_toxicity',
    'ames_mutagenic',
    'dili_classification',
]


def _safe_json_loads(s: Optional[str]) -> dict[str, Any]:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _serialize(v: Any) -> Any:
    return '' if v is None else v


def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        if isinstance(v, str) and not v.strip():
            return None
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _to_bool01_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        if int(v) in (0, 1):
            return int(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {'1', 'true', 't', 'yes', 'y', 'positive', 'pos'}:
            return 1
        if s in {'0', 'false', 'f', 'no', 'n', 'negative', 'neg'}:
            return 0
    return None


def _csv_join(values) -> str:
    return '|'.join(str(v) for v in sorted(values))


def _empty_rec(inchi_key: str, smi: Optional[str], mw, cf, dbid, chembl) -> dict[str, Any]:
    return {
        'inchi_key': inchi_key,
        'smiles': smi,
        'mol_weight': mw,
        'chem_formula': cf,
        'drugbank_id': dbid,
        'chembl_id': chembl,
        'n_source_records': 0,
        'protox_counts': Counter(),
        'respiratory_toxicity_counts': Counter(),
        'ames_mutagenic_counts': Counter(),
        'dili_classification_counts': Counter(),
    }


def _resolve_field(counts: Counter[int]) -> tuple[Optional[int], str]:
    total = int(sum(counts.values()))
    if total <= 0:
        return None, 'missing'
    if len(counts) == 1:
        return int(next(iter(counts.keys()))), 'agreement'
    top_val, top_count = max(sorted(counts.items()), key=lambda kv: kv[1])
    if int(top_count) > total / 2.0:
        return int(top_val), 'majority'
    return None, 'conflict'


def _finalize_row(rec: dict[str, Any]) -> dict[str, Any]:
    protox_value, protox_resolution = _resolve_field(rec['protox_counts'])
    out = {
        'inchi_key': rec['inchi_key'],
        'smiles': rec['smiles'],
        'mol_weight': rec['mol_weight'],
        'chem_formula': rec['chem_formula'],
        'drugbank_id': rec['drugbank_id'],
        'chembl_id': rec['chembl_id'],
        'n_source_records': int(rec['n_source_records']),
        'protox_num_observations': int(sum(rec['protox_counts'].values())),
        'protox_values_observed': _csv_join(rec['protox_counts'].keys()),
        'protox_resolution': protox_resolution,
        PROTOX_TASK: protox_value,
    }
    for task in BINARY_TASKS:
        prefix = task.split('_')[0] if task != 'dili_classification' else 'dili'
        counts = rec[f'{task}_counts']
        value, resolution = _resolve_field(counts)
        out[f'{prefix}_num_observations'] = int(sum(counts.values()))
        out[f'{prefix}_values_observed'] = _csv_join(counts.keys())
        out[f'{prefix}_resolution'] = resolution
        out[task] = value
    return out


def _empty_summary() -> dict[str, int]:
    out = {
        'n_rows': 0,
        'n_smiles_nonnull': 0,
        'n_protox_labeled': 0,
        'n_protox_conflicts': 0,
        'n_protox_majority': 0,
    }
    for task in BINARY_TASKS:
        prefix = task.split('_')[0] if task != 'dili_classification' else 'dili'
        out[f'n_{prefix}_labeled'] = 0
        out[f'n_{prefix}_conflicts'] = 0
        out[f'n_{prefix}_majority'] = 0
    return out


def _update_summary_from_row(summary: dict[str, int], row: dict[str, Any]) -> None:
    summary['n_rows'] += 1
    if row.get('smiles'):
        summary['n_smiles_nonnull'] += 1
    if row.get(PROTOX_TASK) is not None:
        summary['n_protox_labeled'] += 1
    if row.get('protox_resolution') == 'conflict':
        summary['n_protox_conflicts'] += 1
    if row.get('protox_resolution') == 'majority':
        summary['n_protox_majority'] += 1
    for task in BINARY_TASKS:
        prefix = task.split('_')[0] if task != 'dili_classification' else 'dili'
        if row.get(task) is not None:
            summary[f'n_{prefix}_labeled'] += 1
        if row.get(f'{prefix}_resolution') == 'conflict':
            summary[f'n_{prefix}_conflicts'] += 1
        if row.get(f'{prefix}_resolution') == 'majority':
            summary[f'n_{prefix}_majority'] += 1


def extract_multitask_table_stream(
    db_path: str | Path,
    out_csv: str | Path,
    *,
    fetchmany: int = 100_000,
) -> dict[str, int]:
    db_path = str(db_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT
          d.inchi_key, d.smiles, d.mol_weight, d.chem_formula, d.drugbank_id, d.chembl_id,
          s.name AS source_name, r.data_json
        FROM drugs d
        JOIN source_records r ON r.drug_inchi_key = d.inchi_key
        JOIN sources s ON s.id = r.source_id
        ORDER BY d.inchi_key
        """
    )

    summary = _empty_summary()

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=MULTITASK_FIELDS)
        w.writeheader()

        cur_key: Optional[str] = None
        rec: Optional[dict[str, Any]] = None

        while True:
            batch = cur.fetchmany(fetchmany)
            if not batch:
                break
            for inchi_key, smi, mw, cf, dbid, chembl, source_name, data_json in batch:
                if cur_key is None:
                    cur_key = inchi_key
                    rec = _empty_rec(inchi_key, smi, mw, cf, dbid, chembl)

                if inchi_key != cur_key:
                    assert rec is not None
                    row = _finalize_row(rec)
                    _update_summary_from_row(summary, row)
                    w.writerow({k: _serialize(row.get(k)) for k in MULTITASK_FIELDS})
                    cur_key = inchi_key
                    rec = _empty_rec(inchi_key, smi, mw, cf, dbid, chembl)

                assert rec is not None
                rec['n_source_records'] += 1
                data = _safe_json_loads(data_json)

                protox = _to_int_or_none(data.get(PROTOX_TASK)) if PROTOX_TASK in data else None
                if protox in {1, 2, 3, 4, 5, 6}:
                    rec['protox_counts'][int(protox)] += 1

                for task in BINARY_TASKS:
                    value = _to_bool01_or_none(data.get(task)) if task in data else None
                    if value in {0, 1}:
                        rec[f'{task}_counts'][int(value)] += 1

        if rec is not None:
            row = _finalize_row(rec)
            _update_summary_from_row(summary, row)
            w.writerow({k: _serialize(row.get(k)) for k in MULTITASK_FIELDS})

    cur.close()
    con.close()
    return summary
