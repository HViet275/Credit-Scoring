from __future__ import annotations

import io
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DISPLAY_NAME_MAP = {
    "approve_preferred": "Approve Preferred",
    "approve_standard": "Approve Standard",
    "manual_review": "Manual Review",
    "reject_or_intensive_review": "Decline / Intensive Review",
    "review_or_decline": "Review or Decline",
    "starter_loan_small": "Starter Loan Small",
    "starter_loan_standard": "Starter Loan Standard",
    "intensive_collection_priority": "Intensive Collection Priority",
    "high_risk_review_priority": "High Risk Review Priority",
    "monitor_or_standard_queue": "Monitor / Standard Queue",
    "approve": "Approve",
    "reject": "Decline",
    "decline": "Decline",
    "data_warning": "Data Warning",
}

STATUS_COLOR_MAP = {
    "Approve Preferred": "green",
    "Approve Standard": "green",
    "Approve": "green",
    "Starter Loan Standard": "green",
    "Starter Loan Small": "blue",
    "Manual Review": "orange",
    "Review or Decline": "orange",
    "Decline / Intensive Review": "red",
    "Decline": "red",
    "Intensive Collection Priority": "red",
    "High Risk Review Priority": "orange",
    "Monitor / Standard Queue": "blue",
    "Data Warning": "gray",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def humanize_token(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    mapped = DISPLAY_NAME_MAP.get(text)
    if mapped:
        return mapped
    return text.replace("_", " ").replace("-", " ").title()


def status_color(value: str | None) -> str:
    return STATUS_COLOR_MAP.get(humanize_token(value), "gray")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(float(value))
    except Exception:
        return default


def compact_json(data: Any) -> str:
    if is_dataclass(data):
        data = asdict(data)
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def dataframe_to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")


def dataframe_to_download_bytes(df: pd.DataFrame) -> bytes:
    return dataframe_to_csv_bytes(df)


def coerce_none_if_empty(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def load_sample_payload(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"customer_id": "DEMO_001", "tenure_months": 8, "max_dpd": 0, "main_income": 15000000}


def dataframe_from_single_payload(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([payload])


def validate_required_columns(df: pd.DataFrame, schema: pd.DataFrame) -> list[str]:
    if schema is None or schema.empty or "field_name" not in schema.columns:
        return []
    required_col = "required_for_model" if "required_for_model" in schema.columns else ("required" if "required" in schema.columns else None)
    if required_col is None:
        return []
    required_fields = schema.loc[schema[required_col].fillna(False).astype(bool), "field_name"].astype(str).tolist()
    return [c for c in required_fields if c not in df.columns]


def coerce_input_types(df: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if schema is None or schema.empty:
        return out
    if "field_name" not in schema.columns or "type" not in schema.columns:
        return out
    for _, row in schema.iterrows():
        col = row["field_name"]
        typ = str(row["type"]).lower()
        if col not in out.columns:
            continue
        if typ in {"number", "numeric", "float", "int", "integer"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = out[col].astype("string")
    return out
