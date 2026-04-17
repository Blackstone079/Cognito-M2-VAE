"""Step 0: extract M2 multitask toxicity table from SQLite DB."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.db.extract_m2_stream import extract_multitask_table_stream  # noqa: E402
from src.utils.logging import get_pipeline_run_dir, get_stage_results_dir, write_manifest  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def main() -> None:
    ap = argparse.ArgumentParser(description='Extract M2 multitask toxicity table from SQLite DB.')
    ap.add_argument('--config', type=str, default='pipelines/m2/config.yaml')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)
    paths = Paths()
    feature_id = str(cfg['featurizer'].get('feature_id', 'm2_v1'))
    run_dir = get_pipeline_run_dir(paths.RESULTS, feature_id, prefix=str(cfg.get('run', {}).get('prefix', 'm2')))
    stage_dir = get_stage_results_dir(run_dir, '00_extract')

    db_path = Path(cfg['db']['path'])
    if not db_path.is_absolute():
        db_path = paths.ROOT / db_path

    out_csv = paths.INTERIM / 'drug_table_multitask.csv'
    summary_path = paths.INTERIM / 'extract_multitask_summary.json'
    summary = extract_multitask_table_stream(db_path, out_csv)
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    shutil.copy2(summary_path, stage_dir / summary_path.name)

    write_manifest(
        stage_dir / 'manifest.json',
        stage_name='00_extract',
        config_path=cfg_path,
        inputs=[db_path],
        outputs=[out_csv, summary_path],
        extra={'feature_id': feature_id, 'n_rows': int(summary.get('n_rows', 0))},
    )
    print(f'[ok] wrote {out_csv}')
    print(f'[ok] wrote {summary_path}')


if __name__ == '__main__':
    main()
