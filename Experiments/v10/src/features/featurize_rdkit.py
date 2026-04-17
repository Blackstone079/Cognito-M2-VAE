# src/features/featurize_rdkit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")  # hide SMILES Parse Error spam


@dataclass(frozen=True)
class DescScaler:
    mean: np.ndarray
    std: np.ndarray


def mol_from_smiles(smiles: Optional[str]) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None
    return Chem.MolFromSmiles(s)


def morgan_fp(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def desc8(mol: Chem.Mol) -> np.ndarray:
    return np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
        ],
        dtype=np.float32,
    )


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def fit_desc_scaler(X_desc: np.ndarray) -> DescScaler:
    mu = X_desc.mean(axis=0)
    sd = X_desc.std(axis=0) + 1e-8
    return DescScaler(mean=mu, std=sd)


def transform_desc(X_desc: np.ndarray, scaler: DescScaler) -> np.ndarray:
    return (X_desc - scaler.mean) / scaler.std


def featurize_df(
    df: pd.DataFrame,
    n_bits: int = 2048,
    radius: int = 2,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Filter invalid SMILES and return features.

    Returns:
      - df_filt (reset index)
      - X_fp: (N, n_bits) float32 in {0,1}
      - X_desc: (N, 8) float32 (NOT scaled)
      - scaffolds: (N,) object array
      - invalid_df: rows with invalid smiles
    """
    mols = []
    keep_rows = []
    invalid_rows = []

    for i, (inchi_key, smi) in enumerate(zip(df["inchi_key"].tolist(), df["smiles"].tolist())):
        mol = mol_from_smiles(smi)
        if mol is None:
            invalid_rows.append({"inchi_key": inchi_key, "smiles": smi})
            continue
        mols.append(mol)
        keep_rows.append(i)

    df_f = df.iloc[keep_rows].copy().reset_index(drop=True)

    X_fp = np.stack([morgan_fp(m, n_bits=n_bits, radius=radius) for m in mols], axis=0)
    X_desc = np.stack([desc8(m) for m in mols], axis=0)
    scaff = np.array([murcko_scaffold_smiles(m) for m in mols], dtype=object)

    invalid_df = pd.DataFrame(invalid_rows)

    return df_f, X_fp, X_desc, scaff, invalid_df
