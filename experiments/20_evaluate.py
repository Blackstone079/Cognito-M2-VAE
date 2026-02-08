# experiments/20_evaluate.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.cache import load_features_npz
from src.dataio.datamodule import make_feature_tensors, build_dataloaders
from src.models.m2 import M2VAE, M2Dims
from src.training.metrics import confusion, classification_metrics


def plot_cm(cm: np.ndarray, class_names: list[str], out_png: Path, title: str) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


@torch.no_grad()
def predict_labels(model: M2VAE, dl, device: str):
    model.eval()
    ys, yp = [], []
    for b in dl:
        x_fp = b["x_fp"].to(device)
        x_desc = b["x_desc"].to(device)
        y = b["y"].to(device)
        logits = model.q_y_logits(x_fp, x_desc)
        pred = torch.argmax(logits, dim=1)
        ys.append(y.cpu().numpy())
        yp.append(pred.cpu().numpy())
    if len(ys) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(ys), np.concatenate(yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    feat_meta = json.loads((run_dir / "features_meta.json").read_text(encoding="utf-8"))
    label_map = json.loads((run_dir / "label_map.json").read_text(encoding="utf-8"))
    class_names = [c["name"] for c in label_map["classes"]]

    feature_id = str(feat_meta.get("feature_id", "v1"))
    repo_root = run_dir.parents[2]
    features_npz = repo_root / "data" / "processed" / "features" / f"features_{feature_id}.npz"

    feats = load_features_npz(features_npz)
    tensors = make_feature_tensors(feats, device="cpu")
    dls = build_dataloaders(tensors, batch_size=512)

    K = int(label_map["num_classes"])
    dims = M2Dims(
        d_fp=int(feat_meta["d_fp"]),
        d_desc=int(feat_meta["d_desc"]),
        K=K,
        z_dim=int(cfg["model"]["z_dim"]),
    )

    model = M2VAE(
        dims,
        clf_hidden=tuple(int(x) for x in cfg["model"]["clf_hidden"]),
        enc_hidden=tuple(int(x) for x in cfg["model"]["enc_hidden"]),
        dec_hidden=tuple(int(x) for x in cfg["model"]["dec_hidden"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )

    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)

    yv, pv = predict_labels(model, dls.val_labeled, device)
    yt, pt = predict_labels(model, dls.test_labeled, device)

    val = classification_metrics(yv, pv) if len(yv) else {}
    test = classification_metrics(yt, pt) if len(yt) else {}

    out = {"val": val, "test": test}
    (run_dir / "eval_refreshed.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    if len(yv):
        cmv = confusion(yv, pv)
        plot_cm(cmv, class_names, run_dir / "plots" / "cm_val.png", "Confusion matrix (val)")
    if len(yt):
        cmt = confusion(yt, pt)
        plot_cm(cmt, class_names, run_dir / "plots" / "cm_test.png", "Confusion matrix (test)")

    print(f"[ok] wrote {run_dir / 'eval_refreshed.json'}")


if __name__ == "__main__":
    main()
