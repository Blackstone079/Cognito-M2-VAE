# Cognito M2 (Semi-supervised VAE) – Toxicity Prediction

This repo implements a **semi-supervised VAE (M2)** for predicting **multi-class toxicity** from molecular structure.

**Data model (current):**
- Input: SMILES (from `epshteins_list.db`)
- Target: `protox_toxclass` (categorical; optionally merged/filtered via `LabelPolicy`)
- Features: Morgan fingerprint (bit vector) + 8 RDKit descriptors (continuous)

## Quick start

1) Put your SQLite DB at:
- `data/raw/epshteins_list.db`

2) Install dependencies (recommended: conda, because RDKit is easiest there):

```bash
conda create -n cognito-m2 python=3.11 -y
conda activate cognito-m2
conda install -c conda-forge rdkit pytorch scikit-learn pandas pyyaml -y
pip install -r requirements.txt
```

3) Run the pipeline (from repo root):

```bash
python experiments/00_extract_dataset.py
python experiments/01_make_split.py
python experiments/02_build_features.py
python experiments/10_train_m2.py --config experiments/config_m2.yaml
python experiments/20_evaluate.py --run_dir results/runs/<RUN_ID>
```

4) Predict on new molecules:

```bash
python experiments/30_predict.py --run_dir results/runs/<RUN_ID> --smiles_csv <your.csv> --out_csv preds.csv
```

## Notes

- The pipeline is designed so **future DB updates** or **more labels** do not require a refactor:
  you re-run extract / features / train, and only the label policy / class count changes.
- Run artifacts are stored in `results/runs/<RUN_ID>/` (config, label_map, checkpoints, metrics, plots).
