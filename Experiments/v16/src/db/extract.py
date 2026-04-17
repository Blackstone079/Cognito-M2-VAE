# src/db/extract.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


def _norm_source_name(name: str) -> str:
    return (name or "").strip().lower()


def _safe_json_loads(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@dataclass
class ExtractConfig:
    include_drugbank_text: bool = False  # avoid leakage by default


def extract_drug_table(db_path: str, cfg: ExtractConfig | None = None) -> pd.DataFrame:
    """Return one row per drug (inchi_key), aggregating `source_records.data_json` per source.

    Core columns:
      - inchi_key, smiles, mol_weight, chem_formula, drugbank_id, chembl_id
      - db_groups (list[str] or None)
      - protox_toxclass (float/None), ld50 (float/None), toxicity_types (list[str] or None)
      - is_withdrawn_source (0/1), is_approved (0/1/None), is_withdrawn_drugbank (0/1/None)

    The DB can evolve: the extractor is robust to missing JSON keys.
    """
    cfg = cfg or ExtractConfig()

    con = sqlite3.connect(db_path)
    q = """
    SELECT
        d.inchi_key, d.smiles, d.mol_weight, d.chem_formula, d.drugbank_id, d.chembl_id,
        s.name AS source_name, r.data_json
    FROM drugs d
    JOIN source_records r ON r.drug_inchi_key = d.inchi_key
    JOIN sources s ON s.id = r.source_id
    """
    rows = pd.read_sql(q, con)

    per: dict[str, dict[str, Any]] = {}

    for inchi_key, smi, mw, cf, dbid, chembl, source_name, data_json in rows.itertuples(index=False):
        rec = per.setdefault(
            inchi_key,
            dict(
                inchi_key=inchi_key,
                smiles=smi,
                mol_weight=mw,
                chem_formula=cf,
                drugbank_id=dbid,
                chembl_id=chembl,
                db_groups=None,
                is_approved=None,
                is_withdrawn_drugbank=None,
                is_withdrawn_source=0,
                toxicity_types=None,
                protox_toxclass=None,
                ld50=None,
                first_approval_year=None,
                first_withdrawn_year=None,
                last_withdrawn_year=None,
                # optional text/cats:
                db_categories=None,
                db_atc_codes=None,
                db_toxicity_text=None,
                db_moa_text=None,
                db_description=None,
            ),
        )

        src = _norm_source_name(source_name)
        data = _safe_json_loads(data_json)

        if src == "drugbank":
            groups = data.get("groups", None)
            if isinstance(groups, list):
                gl = [str(g).strip().lower() for g in groups if str(g).strip()]
                rec["db_groups"] = gl
                rec["is_approved"] = 1 if "approved" in gl else 0
                rec["is_withdrawn_drugbank"] = 1 if "withdrawn" in gl else 0

            if cfg.include_drugbank_text:
                for k, out in [
                    ("categories", "db_categories"),
                    ("atc_codes", "db_atc_codes"),
                    ("toxicity", "db_toxicity_text"),
                    ("moa", "db_moa_text"),
                    ("description", "db_description"),
                ]:
                    v = data.get(k, None)
                    if v is not None:
                        rec[out] = v

        elif src == "withdrawn":
            rec["is_withdrawn_source"] = 1

            tt = data.get("toxicity_types", None)
            if isinstance(tt, list):
                rec["toxicity_types"] = [str(x).strip().lower() for x in tt if str(x).strip()]

            rec["protox_toxclass"] = data.get("protox_toxclass", rec["protox_toxclass"])
            rec["ld50"] = data.get("ld50", rec["ld50"])
            rec["first_approval_year"] = data.get("first_approval_year", rec["first_approval_year"])
            rec["first_withdrawn_year"] = data.get("first_withdrawn_year", rec["first_withdrawn_year"])
            rec["last_withdrawn_year"] = data.get("last_withdrawn_year", rec["last_withdrawn_year"])

    df = pd.DataFrame(list(per.values()))
    return df
