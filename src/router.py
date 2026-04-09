from __future__ import annotations

from typing import Any

import pandas as pd


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def detect_stage(record: dict[str, Any]) -> str:
    """
    Production-friendly routing:
    - A: new-to-bank (tenure_months <= 0)
    - C: meaningful delinquency / mature risk
    - B: main decision population

    Notes:
    - Do NOT send every max_dpd > 0 case to C.
    - Mild delinquency (1, 3, 5, 7, 15 DPD) can still stay in B.
    """
    tenure = _to_float(record.get("tenure_months"), 0.0)
    max_dpd = _to_float(record.get("max_dpd"), 0.0)
    recent_ontime_ratio = _to_float(record.get("recent_ontime_ratio"), 1.0)

    if tenure <= 0:
        return "A"

    # Route to C only for clearly riskier delinquency / mature-risk cases
    if max_dpd >= 30:
        return "C"

    # Optional mature-risk rule: longer tenure and clearly weak recent behavior
    if tenure > 12 and recent_ontime_ratio < 0.65:
        return "C"

    return "B"


def detect_stage_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stage"] = out.apply(lambda r: detect_stage(r.to_dict()), axis=1)
    return out
