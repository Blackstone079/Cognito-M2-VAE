"""Step 3 (memmap): train the M2 model.

This mirrors `experiments/10_train_m2.py`, but loads features from memmap arrays.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import torch
import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.dataio.memmap_datamodule import build_dataloaders_memmap, open_memmap_features  # noqa: E402
from src.models.legacy_single_target_m2 import M2Dims, M2VAE  # noqa: E402
from src.training.loops import evaluate_classifier, train_m2  # noqa: E402
from src.utils.fingerprint import build_data_fingerprint  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_run_id(prefix: str = "m2", feature_id: str = "v1") -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    return f"{ts}__{prefix}__{feature_id}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the M2 model (memmap features).")
    ap.add_argument("--config", type=str, default="pipelines/legacy_single_target/config.yaml", help="Path to YAML config.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)
    paths = Paths()

    feature_id = str(cfg["featurizer"].get("feature_id", "v1"))

    feat_dir = paths.FEATURES / f"features_{feature_id}_memmap"
    meta_path = paths.FEATURES / f"features_{feature_id}_memmap_meta.json"
    label_map_path = paths.FEATURES / f"label_map_{feature_id}.json"

    if not feat_dir.exists():
        raise FileNotFoundError(f"Missing {feat_dir}. Run pipelines/legacy_single_target/build_features.py first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run pipelines/legacy_single_target/build_features.py first.")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing {label_map_path}. Run pipelines/legacy_single_target/build_features.py first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

    K = int(label_map["num_classes"])
    d_fp = int(meta["d_fp"])
    d_desc = int(meta["d_desc"])

    run_id = make_run_id(prefix="m2", feature_id=feature_id)
    run_dir = paths.RUNS / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    logger = RunLogger(run_dir)
    logger.log(f"run_id={run_id}")

    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (run_dir / "features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # fingerprint
    db_path = Path(cfg["db"]["path"])
    if not db_path.is_absolute():
        db_path = paths.ROOT / db_path

    fp = build_data_fingerprint(
        db_path=db_path,
        featurizer_cfg=cfg.get("featurizer", {}),
        label_cfg=cfg.get("label", {}),
        split_cfg=cfg.get("split", {}),
    )
    (run_dir / "data_fingerprint.json").write_text(json.dumps(fp, indent=2), encoding="utf-8")

    # seed + device
    seed = int(cfg["training"]["seed"])
    set_global_seed(seed)

    device = str(cfg["training"].get("device", "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    logger.log(f"device={device}")

    feats = open_memmap_features(feat_dir)

    dls = build_dataloaders_memmap(
        feats,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=bool(cfg["training"].get("pin_memory", False)),
    )

    if len(dls.labeled_train.dataset) == 0:
        raise RuntimeError("No labeled samples in TRAIN split. Adjust split or label policy.")

    dims = M2Dims(d_fp=d_fp, d_desc=d_desc, K=K, z_dim=int(cfg["model"]["z_dim"]))
    model = M2VAE(
        dims,
        clf_hidden=tuple(int(x) for x in cfg["model"]["clf_hidden"]),
        enc_hidden=tuple(int(x) for x in cfg["model"]["enc_hidden"]),
        dec_hidden=tuple(int(x) for x in cfg["model"]["dec_hidden"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )

    res = train_m2(
        model=model,
        labeled_train=dls.labeled_train,
        unlabeled_train=dls.unlabeled_train,
        val_labeled=dls.val_labeled,
        device=device,
        run_dir=run_dir,
        epochs=int(cfg["training"]["epochs"]),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        alpha_ce=float(cfg["training"]["alpha_ce"]),
        lambda_unl=float(cfg["training"]["lambda_unl"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
        patience=int(cfg["training"]["patience"]),
        logger=logger,
    )

    # final evaluation using best checkpoint
    best_ckpt = run_dir / "checkpoints" / "best.pt"
    payload = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(payload["model_state"])

    val_metrics = evaluate_classifier(model, dls.val_labeled, device)
    test_metrics = evaluate_classifier(model, dls.test_labeled, device)

    out = {
        "run_id": run_id,
        "best_epoch": int(res.best_epoch),
        "best_val_f1_macro": float(res.best_val_f1),
        "val": val_metrics,
        "test": test_metrics,
        "n_labeled_train": int(len(dls.labeled_train.dataset)),
        "n_unlabeled_train": int(len(dls.unlabeled_train.dataset)),
        "n_labeled_val": int(len(dls.val_labeled.dataset)),
        "n_labeled_test": int(len(dls.test_labeled.dataset)),
    }

    (run_dir / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.log(f"done: {out}")
    print(f"[ok] run_dir: {run_dir}")


if __name__ == "__main__":
    main()
