from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from src.utils.fingerprint import sha256_file


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sanitize_stage_name(stage_name: str) -> str:
    return str(stage_name).strip().replace(' ', '_').replace('/', '_')


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(text.rstrip() + '\n')


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=_json_default, sort_keys=True) + '\n')


def write_json(path: Path, payload: Dict[str, Any] | list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default, sort_keys=False), encoding='utf-8')


def file_record(path: Path) -> Dict[str, Any]:
    path = Path(path)
    rec: Dict[str, Any] = {'path': str(path), 'exists': bool(path.exists())}
    if path.exists() and path.is_file():
        rec['size_bytes'] = int(path.stat().st_size)
        rec['sha256'] = sha256_file(path)
    return rec


def write_manifest(
    manifest_path: Path,
    *,
    stage_name: str,
    config_path: Path | None = None,
    inputs: Iterable[Path] | None = None,
    outputs: Iterable[Path] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        'stage_name': stage_name,
        'created_utc': utc_now_iso(),
        'config_path': str(config_path) if config_path is not None else None,
        'inputs': [file_record(Path(p)) for p in (inputs or [])],
        'outputs': [file_record(Path(p)) for p in (outputs or [])],
    }
    if extra:
        payload['extra'] = extra
    write_json(manifest_path, payload)
    return payload


def make_timestamped_run_id(prefix: str, feature_id: str) -> str:
    ts = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    return f'{ts}__{prefix}__{feature_id}'


def _pipeline_state_dir(results_root: Path) -> Path:
    return ensure_dir(results_root / '.pipeline_state')


def _latest_run_file(results_root: Path, feature_id: str) -> Path:
    return _pipeline_state_dir(results_root) / f'{feature_id}.latest'


def get_pipeline_run_dir(results_root: Path, feature_id: str, prefix: str = 'm2') -> Path:
    env_run_id = os.environ.get('PIPELINE_RUN_ID', '').strip()
    latest_file = _latest_run_file(results_root, feature_id)
    if env_run_id:
        run_id = env_run_id
    elif latest_file.exists():
        run_id = latest_file.read_text(encoding='utf-8').strip()
    else:
        run_id = make_timestamped_run_id(prefix=prefix, feature_id=feature_id)
    latest_file.write_text(run_id, encoding='utf-8')
    return ensure_dir(results_root / 'runs' / run_id)


def get_stage_results_dir(run_dir: Path, stage_name: str) -> Path:
    return ensure_dir(run_dir / 'stages' / sanitize_stage_name(stage_name))


@dataclass
class RunLogger:
    run_dir: Path
    stage_name: str = '10_train_gpu'

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir = get_stage_results_dir(self.run_dir, self.stage_name)
        self.log_path = self.stage_dir / 'run.log'
        self.metrics_path = self.stage_dir / 'metrics.jsonl'
        self.events_path = self.stage_dir / 'events.jsonl'

    def log(self, msg: str) -> None:
        append_text(self.log_path, msg)

    def log_metrics(self, d: Dict[str, Any]) -> None:
        append_jsonl(self.metrics_path, d)

    def log_event(self, name: str, **kwargs: Any) -> None:
        rec = {'event': name, 'created_utc': utc_now_iso(), **kwargs}
        append_jsonl(self.events_path, rec)
