# src/db/extract_stream.py
from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Tuple


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


CORE_FIELDS = [
    "inchi_key",
    "smiles",
    "mol_weight",
    "chem_formula",
    "drugbank_id",
    "chembl_id",
    "db_groups",
    "is_approved",
    "is_withdrawn_drugbank",
    "is_withdrawn_source",
    "toxicity_types",
    "protox_toxclass",
    "ld50",
    "first_approval_year",
    "first_withdrawn_year",
    "last_withdrawn_year",
    # optional drugbank text fields
    "db_categories",
    "db_atc_codes",
    "db_toxicity_text",
    "db_moa_text",
    "db_description",
]


def _empty_rec(inchi_key: str, smi: Optional[str], mw, cf, dbid, chembl) -> Dict[str, Any]:
    return dict(
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
        db_categories=None,
        db_atc_codes=None,
        db_toxicity_text=None,
        db_moa_text=None,
        db_description=None,
    )


def _serialize(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return v


def extract_drug_table_stream(
    db_path: str | Path,
    out_csv: str | Path,
    *,
    cfg: ExtractConfig | None = None,
    fetchmany: int = 100_000,
) -> Dict[str, int]:
    """Stream the (drugs x sources) join and write one row per drug.

    This avoids `pd.read_sql` over the full join, which can explode RAM for large DBs.

    Returns a small summary dict.
    """

    cfg = cfg or ExtractConfig()
    db_path = str(db_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Order by inchi_key so we can aggregate per drug without keeping everything in RAM.
    q = """
    SELECT
      d.inchi_key, d.smiles, d.mol_weight, d.chem_formula, d.drugbank_id, d.chembl_id,
      s.name AS source_name, r.data_json
    FROM drugs d
    JOIN source_records r ON r.drug_inchi_key = d.inchi_key
    JOIN sources s ON s.id = r.source_id
    ORDER BY d.inchi_key
    """

    cur.execute(q)

    n_rows = 0
    n_smiles_nonnull = 0
    n_protox_labeled = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CORE_FIELDS)
        w.writeheader()

        cur_key: Optional[str] = None
        rec: Optional[Dict[str, Any]] = None

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
                    # finalize + write previous
                    n_rows += 1
                    if rec.get("smiles"):
                        n_smiles_nonnull += 1
                    if rec.get("protox_toxclass") is not None:
                        n_protox_labeled += 1
                    w.writerow({k: _serialize(rec.get(k)) for k in CORE_FIELDS})

                    # start new
                    cur_key = inchi_key
                    rec = _empty_rec(inchi_key, smi, mw, cf, dbid, chembl)

                assert rec is not None
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
                    if "protox_toxclass" in data:
                        rec["protox_toxclass"] = data.get("protox_toxclass", rec["protox_toxclass"])
                    if "ld50" in data:
                        rec["ld50"] = data.get("ld50", rec["ld50"])
                    rec["first_approval_year"] = data.get("first_approval_year", rec["first_approval_year"])
                    rec["first_withdrawn_year"] = data.get("first_withdrawn_year", rec["first_withdrawn_year"])
                    rec["last_withdrawn_year"] = data.get("last_withdrawn_year", rec["last_withdrawn_year"])

        # write last record
        if rec is not None:
            n_rows += 1
            if rec.get("smiles"):
                n_smiles_nonnull += 1
            if rec.get("protox_toxclass") is not None:
                n_protox_labeled += 1
            w.writerow({k: _serialize(rec.get(k)) for k in CORE_FIELDS})

    cur.close()
    con.close()

    return {
        "n_rows": int(n_rows),
        "n_smiles_nonnull": int(n_smiles_nonnull),
        "n_protox_labeled": int(n_protox_labeled),
    }
