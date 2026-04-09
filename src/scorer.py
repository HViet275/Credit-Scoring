from __future__ import annotations

from typing import Any

import pandas as pd

from .explain import build_business_explanation, normalize_reason_records
from .inference import run_stage_inference
from .policy import apply_policy_to_score
from .router import detect_stage
from .utils import humanize_token, utc_now_iso
from .validation import validate_batch_dataframe, validate_single_record


BASE_STAGE_SCORE = {"A": 320.0, "B": 350.0, "C": 260.0}


def _fallback_proxy_score(record: dict[str, Any], stage: str) -> tuple[float, str]:
    score = BASE_STAGE_SCORE.get(stage, 320.0)
    tenure = float(record.get("tenure_months") or 0.0)
    max_dpd = float(record.get("max_dpd") or 0.0)
    income = float(record.get("main_income") or 0.0)
    recent_ontime = record.get("recent_ontime_ratio")
    score += min(60.0, tenure * 2.5)
    score -= min(220.0, max_dpd * 6.0)
    score += min(80.0, income / 1_000_000 * 4.0)
    if recent_ontime is not None:
        score += (float(recent_ontime or 0.0) - 0.5) * 120
    clipped = float(max(0.0, min(900.0, score)))
    return clipped, "fallback_proxy_scoring_used"


def score_single_record(record: dict[str, Any], artifact_bundle: Any) -> dict[str, Any]:
    validation = validate_single_record(record)
    if not validation.is_valid:
        return {
            "customer_id": record.get("customer_id"),
            "stage": None,
            "score_300_900": None,
            "decision_zone": None,
            "policy_action": None,
            "scoring_status": "validation_failed",
            "validation_messages": [issue.message for issue in validation.issues],
            "completeness_ratio": validation.completeness_ratio,
            "warnings": [issue.message for issue in validation.issues if issue.level == "warning"],
        }

    normalized = validation.normalized_record
    stage = detect_stage(normalized)
    policy_rules = getattr(artifact_bundle, "policy_rules", getattr(getattr(artifact_bundle, "bundle", None), "policy_rules", {}))

    inference = run_stage_inference(normalized, stage, artifact_bundle)
    warnings = [issue.message for issue in validation.issues if issue.level == "warning"]

    if inference.score_300_900 is None:
        score, scoring_status = _fallback_proxy_score(normalized, stage)
        warnings.append(f"Model inference failed for stage {stage}; proxy fallback was used.")
        if inference.fallback_reason:
            warnings.append(str(inference.fallback_reason))
        risk_proba = None
        model_features = inference.model_features
        missing_model_features = inference.missing_model_features
        model_input = inference.model_input
    else:
        score = inference.score_300_900
        scoring_status = inference.scoring_status
        risk_proba = inference.risk_proba
        model_features = inference.model_features
        missing_model_features = inference.missing_model_features
        model_input = inference.model_input
        if missing_model_features:
            warnings.append(
                "Some stage-specific model fields were missing and were passed as null into the exported sklearn pipeline's imputer: "
                + ", ".join(missing_model_features)
            )

    policy_result = apply_policy_to_score(stage, score, policy_rules)
    reasons = normalize_reason_records({**normalized, **model_input, "stage": stage})

    result = {
        "customer_id": normalized.get("customer_id"),
        "stage": stage,
        "score_300_900": round(float(score), 6),
        "risk_proba": None if risk_proba is None else round(float(risk_proba), 8),
        "decision_zone": policy_result["zone"],
        "policy_action": policy_result["decision"],
        "policy_action_label": policy_result["decision_label"],
        "decision_zone_label": policy_result["zone_label"],
        "scoring_status": scoring_status,
        "completeness_ratio": round(validation.completeness_ratio, 4),
        "missing_fields": validation.missing_fields,
        "band_min": policy_result.get("band_min"),
        "band_max": policy_result.get("band_max"),
        "review_threshold": policy_result.get("review_threshold"),
        "reject_threshold": policy_result.get("reject_threshold"),
        "policy_status": policy_result.get("policy_status"),
        "scored_at": utc_now_iso(),
        "warnings": warnings,
        "model_features_used": model_features,
        "missing_model_features": missing_model_features,
        "model_input_payload": model_input,
    }
    for idx, reason in enumerate(reasons, start=1):
        result[f"reason_label_{idx}"] = reason["reason_label"]
        result[f"reason_text_{idx}"] = reason["reason_text"]
    result["business_explanation"] = build_business_explanation(result)
    result["policy_action_display"] = humanize_token(result["policy_action"])
    return result


def score_batch_dataframe(df: pd.DataFrame, artifact_bundle: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid_df, invalid_df, validation_summary = validate_batch_dataframe(df)
    scored_rows = [score_single_record(row.to_dict(), artifact_bundle) for _, row in valid_df.iterrows()]
    scored_df = pd.DataFrame(scored_rows)
    if scored_df.empty:
        return scored_df, invalid_df, {**validation_summary, "scored_rows": 0, "avg_score": None, "top_action": None, "top_stage": None}
    batch_summary = {
        **validation_summary,
        "scored_rows": int(len(scored_df)),
        "avg_score": round(float(scored_df["score_300_900"].mean()), 2),
        "top_action": humanize_token(scored_df["policy_action"].mode().iloc[0]) if not scored_df["policy_action"].mode().empty else None,
        "top_stage": str(scored_df["stage"].mode().iloc[0]) if not scored_df["stage"].mode().empty else None,
    }
    return scored_df, invalid_df, batch_summary


def score_dataframe(df: pd.DataFrame, artifacts: Any) -> pd.DataFrame:
    out_rows = [score_single_record(row.to_dict(), artifacts) for _, row in df.iterrows()]
    return pd.DataFrame(out_rows)
