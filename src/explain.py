from __future__ import annotations

from typing import Any

import pandas as pd


TECHNICAL_STATUS_MAP = {
    "model_inference_complete": "The exported stage pipeline scored this record directly.",
    "model_inference_complete_with_imputed_inputs": "The exported stage pipeline scored this record directly, with some missing stage features imputed inside the sklearn pipeline.",
    "fallback_proxy_scoring_used": "The app had to use a proxy fallback score because model inference could not be completed.",
    "validation_failed": "The record could not be scored because one or more required inputs were invalid.",
}


def _safe_get(row: pd.Series | dict[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        value = row.get(key, default)
        return default if pd.isna(value) else value
    return row[key] if key in row and pd.notna(row[key]) else default


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text



def normalize_reason_records(record: dict[str, Any]) -> list[dict[str, str]]:
    stage = _normalize_text(record.get("stage", ""))
    max_dpd = record.get("max_dpd", None)
    tenure = record.get("tenure_months", None)
    income = record.get("main_income", None)
    missing_model_features = record.get("missing_model_features", []) or []

    reasons: list[dict[str, str]] = []

    if max_dpd is not None:
        try:
            if float(max_dpd) <= 0:
                reasons.append({"reason_label": "Recent delinquency", "reason_text": "No recent delinquency signal observed"})
            elif float(max_dpd) <= 7:
                reasons.append({"reason_label": "Recent delinquency", "reason_text": "Minor recent delinquency signal detected"})
            else:
                reasons.append({"reason_label": "Recent delinquency", "reason_text": "Elevated recent delinquency signal increases risk"})
        except Exception:
            pass

    if tenure is not None:
        try:
            if float(tenure) == 0:
                reasons.append({"reason_label": "Tenure profile", "reason_text": "New-to-bank profile with limited internal repayment history"})
            elif float(tenure) <= 12:
                reasons.append({"reason_label": "Tenure profile", "reason_text": "Early repayment history available for evaluation"})
            else:
                reasons.append({"reason_label": "Tenure profile", "reason_text": "Longer repayment history supports behavioral assessment"})
        except Exception:
            pass

    if income is not None:
        try:
            if float(income) >= 15000000:
                reasons.append({"reason_label": "Income capacity", "reason_text": "Income level supports repayment capacity"})
            elif float(income) > 0:
                reasons.append({"reason_label": "Income capacity", "reason_text": "Income information is available for affordability assessment"})
        except Exception:
            pass

    if missing_model_features:
        reasons.append({
            "reason_label": "Model input coverage",
            "reason_text": f"Some stage-specific model inputs were not provided and were imputed by the exported pipeline ({len(missing_model_features)} fields)",
        })

    if stage:
        reasons.append({"reason_label": "Stage routing", "reason_text": f"Customer was routed to Stage {stage} based on current profile"})

    while len(reasons) < 3:
        reasons.append({"reason_label": "Model signal", "reason_text": "Additional model features were used in the final decision"})

    return reasons[:3]



def describe_scoring_status(status: str | None) -> str:
    text = _normalize_text(status)
    if text.startswith("artifact_inference_failed"):
        return text
    return TECHNICAL_STATUS_MAP.get(text, text.replace("_", " ").capitalize() if text else "")



def build_business_explanation(record: dict[str, Any] | pd.Series) -> str:
    stage = _normalize_text(_safe_get(record, "stage", "Unknown"))
    score = _safe_get(record, "score_300_900", "")
    action = _normalize_text(_safe_get(record, "policy_action_label", _safe_get(record, "policy_action", "")))
    zone = _normalize_text(_safe_get(record, "decision_zone_label", _safe_get(record, "decision_zone", "")))

    reason_1 = _normalize_text(_safe_get(record, "reason_text_1", ""))
    reason_2 = _normalize_text(_safe_get(record, "reason_text_2", ""))
    reason_3 = _normalize_text(_safe_get(record, "reason_text_3", ""))

    parts: list[str] = []
    main_line = f"The customer was routed to Stage {stage}"
    if score != "":
        main_line += f" with a score of {score}"
    if action:
        main_line += f", leading to the decision {action}"
    if zone:
        main_line += f" in the {zone} zone"
    parts.append(main_line + ".")

    supporting = [x for x in [reason_1, reason_2, reason_3] if x]
    if supporting:
        parts.append("Key drivers: " + "; ".join(supporting) + ".")
    return " ".join(parts).strip()



def build_explanation_block(df: pd.DataFrame, artifacts=None) -> pd.DataFrame:
    work_df = df.copy()
    for col in ["reason_text_1", "reason_text_2", "reason_text_3"]:
        if col not in work_df.columns:
            work_df[col] = ""

    for idx, row in work_df.iterrows():
        reasons = normalize_reason_records(row.to_dict())
        for n in range(1, 4):
            if not _normalize_text(row.get(f"reason_text_{n}", "")):
                work_df.at[idx, f"reason_text_{n}"] = reasons[n - 1]["reason_text"]
                work_df.at[idx, f"reason_label_{n}"] = reasons[n - 1]["reason_label"]
        work_df.at[idx, "business_explanation"] = build_business_explanation(work_df.loc[idx])
        work_df.at[idx, "technical_status_note"] = describe_scoring_status(work_df.loc[idx].get("scoring_status"))
    return work_df



def reason_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["reason_text", "count"])
    reason_cols = [c for c in ["reason_text_1", "reason_text_2", "reason_text_3"] if c in df.columns]
    if not reason_cols:
        return pd.DataFrame(columns=["reason_text", "count"])
    vals = []
    for col in reason_cols:
        vals.extend([str(v).strip() for v in df[col].dropna().tolist() if str(v).strip()])
    if not vals:
        return pd.DataFrame(columns=["reason_text", "count"])
    out = pd.Series(vals).value_counts().reset_index()
    out.columns = ["reason_text", "count"]
    return out
