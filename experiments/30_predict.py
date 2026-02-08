# experiments/30_predict.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import yaml

import numpy as np
import pandas as pd
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.featurize_rdkit import featurize_df
from src.features.scaling import scaler_from_json, apply_scaler
from src.models.m2 import M2VAE, M2Dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--smiles_csv", type=str, required=True, help="CSV with a 'smiles' column. Optionally include an 'id' column.")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    feat_meta = json.loads((run_dir / "features_meta.json").read_text(encoding="utf-8"))
    label_map = json.loads((run_dir / "label_map.json").read_text(encoding="utf-8"))
    class_names = [c["name"] for c in label_map["classes"]]
    K = int(label_map["num_classes"])

    feature_id = str(feat_meta.get("feature_id", "v1"))
    repo_root = run_dir.parents[2]
    scaler_path = repo_root / "data" / "processed" / "features" / f"desc_scaler_{feature_id}.json"
    scaler = scaler_from_json(json.loads(scaler_path.read_text(encoding="utf-8")))

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
    model.eval()

    df_in = pd.read_csv(args.smiles_csv)
    if "smiles" not in df_in.columns:
        raise KeyError("Input CSV must contain a 'smiles' column.")
    if "id" not in df_in.columns:
        df_in["id"] = [f"row_{i}" for i in range(len(df_in))]
    df_in["id"] = df_in["id"].astype(str)

    # build temporary df for featurizer
    tmp = pd.DataFrame({"inchi_key": df_in["id"].astype(str), "smiles": df_in["smiles"].astype(str)})

    df_f, X_fp, X_desc, _, invalid_df = featurize_df(
        tmp,
        n_bits=int(cfg["featurizer"]["fp_bits"]),
        radius=int(cfg["featurizer"]["fp_radius"]),
    )

    X_desc_s = apply_scaler(X_desc, scaler)

    x_fp = torch.tensor(X_fp, dtype=torch.float32, device=device)
    x_desc = torch.tensor(X_desc_s, dtype=torch.float32, device=device)

    with torch.no_grad():
        probs = torch.softmax(model.q_y_logits(x_fp, x_desc), dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)

    out = df_in[["id", "smiles"]].copy()
    out["is_valid_smiles"] = True
    out["pred_class"] = None

    for k in range(K):
        out[f"p_{class_names[k]}"] = np.nan

    # fill valid rows
    valid_ids = df_f["inchi_key"].astype(str).tolist()
    id_to_pos = {rid: i for i, rid in enumerate(valid_ids)}

    for i, rid in enumerate(out["id"].astype(str).tolist()):
        pos = id_to_pos.get(rid, None)
        if pos is None:
            continue
        out.at[i, "pred_class"] = class_names[int(pred[pos])]
        for k in range(K):
            out.at[i, f"p_{class_names[k]}"] = float(probs[pos, k])

    # mark invalids
    if len(invalid_df) > 0:
        invalid_ids = set(invalid_df["inchi_key"].astype(str).tolist())
        out.loc[out["id"].astype(str).isin(invalid_ids), "is_valid_smiles"] = False

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
