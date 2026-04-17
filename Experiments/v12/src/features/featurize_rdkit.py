from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

RDLogger.DisableLog('rdApp.error')


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


_DESCRIPTOR_FUNCS: Dict[str, Callable[[Chem.Mol], float]] = {
    'MolWt': Descriptors.MolWt,
    'MolLogP': Crippen.MolLogP,
    'TPSA': rdMolDescriptors.CalcTPSA,
    'NumHBD': rdMolDescriptors.CalcNumHBD,
    'NumHBA': rdMolDescriptors.CalcNumHBA,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'NumRings': rdMolDescriptors.CalcNumRings,
    'FractionCSP3': rdMolDescriptors.CalcFractionCSP3,
    'HeavyAtomCount': rdMolDescriptors.CalcNumHeavyAtoms,
    'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms,
    'NumValenceElectrons': Descriptors.NumValenceElectrons,
    'LabuteASA': rdMolDescriptors.CalcLabuteASA,
    'MolMR': Crippen.MolMR,
    'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings,
    'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings,
    'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings,
    'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles,
    'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles,
    'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles,
    'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles,
    'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles,
    'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles,
    'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms,
    'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms,
}

_DESCRIPTOR_PANELS: Dict[str, Tuple[str, ...]] = {
    'desc8': (
        'MolWt', 'MolLogP', 'TPSA', 'NumHBD', 'NumHBA', 'NumRotatableBonds', 'NumRings', 'FractionCSP3',
    ),
    'tox24': (
        'MolWt', 'MolLogP', 'TPSA', 'NumHBD', 'NumHBA', 'NumRotatableBonds', 'NumRings', 'FractionCSP3',
        'HeavyAtomCount', 'NumHeteroatoms', 'NumValenceElectrons', 'LabuteASA', 'MolMR',
        'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings',
        'NumAromaticHeterocycles', 'NumAromaticCarbocycles',
        'NumAliphaticHeterocycles', 'NumAliphaticCarbocycles',
        'NumSaturatedHeterocycles', 'NumSaturatedCarbocycles',
        'NumBridgeheadAtoms', 'NumSpiroAtoms',
    ),
}


def descriptor_names(panel: str = 'desc8') -> Tuple[str, ...]:
    key = str(panel).strip().lower()
    if key not in _DESCRIPTOR_PANELS:
        raise ValueError(f'unknown descriptor panel: {panel}')
    return _DESCRIPTOR_PANELS[key]


def mol_descriptors(mol: Chem.Mol, panel: str = 'desc8') -> np.ndarray:
    names = descriptor_names(panel)
    vals = []
    for name in names:
        vals.append(float(_DESCRIPTOR_FUNCS[name](mol)))
    return np.asarray(vals, dtype=np.float32)


def desc8(mol: Chem.Mol) -> np.ndarray:
    return mol_descriptors(mol, panel='desc8')


def morgan_fp(
    mol: Chem.Mol,
    n_bits: int = 2048,
    radius: int = 2,
    *,
    count_simulation: bool = False,
    include_chirality: bool = False,
) -> np.ndarray:
    gen = GetMorganGenerator(
        radius=radius,
        countSimulation=bool(count_simulation),
        includeChirality=bool(include_chirality),
        fpSize=n_bits,
    )
    return np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ''


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
    *,
    count_simulation: bool = False,
    include_chirality: bool = False,
    desc_panel: str = 'desc8',
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    mols = []
    keep_rows = []
    invalid_rows = []
    for i, (inchi_key, smi) in enumerate(zip(df['inchi_key'].tolist(), df['smiles'].tolist())):
        mol = mol_from_smiles(smi)
        if mol is None:
            invalid_rows.append({'inchi_key': inchi_key, 'smiles': smi})
            continue
        mols.append(mol)
        keep_rows.append(i)
    df_f = df.iloc[keep_rows].copy().reset_index(drop=True)
    X_fp = np.stack([
        morgan_fp(m, n_bits=n_bits, radius=radius, count_simulation=count_simulation, include_chirality=include_chirality)
        for m in mols
    ], axis=0)
    X_desc = np.stack([mol_descriptors(m, panel=desc_panel) for m in mols], axis=0)
    scaff = np.array([murcko_scaffold_smiles(m) for m in mols], dtype=object)
    invalid_df = pd.DataFrame(invalid_rows)
    return df_f, X_fp, X_desc, scaff, invalid_df
