# src/labels/schemas.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class LabelPolicy:
    """How to convert raw labels into model classes."""

    name: str  # e.g. "strict_6_class", "merge_I_II", "drop_min_count"
    min_count: int = 0  # used when name == "drop_min_count"


@dataclass(frozen=True)
class LabelSpec:
    column: str  # e.g. "protox_toxclass"
    policy: LabelPolicy
    unknown_value: int = -1  # used for unlabeled / dropped
    # if you later add other targets, keep them separate from M2 y
