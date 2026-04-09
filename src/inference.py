from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import safe_float


@dataclass
class InferenceResult:
    stage: str
    risk_proba: float | None
    score_300_900: float | None
    scoring_status: str
    model_features: list[str]
    model_input: dict[str, Any]
    missing_model_features: list[str]
    used_fallback: bool = False
    fallback_reason: str | None = None


def _extract_stage_models(artifacts: Any) -> dict[str, Any]:
    bundle = getattr(artifacts, "bundle", artifacts)
    if isinstance(getattr(bundle, "inference_service_pack", None), dict):
        stage_models = (
            bundle.inference_service_pack.get("inference_pack", {})
            .get("stage_models", {})
        )
        if stage_models:
            return stage_models
    if isinstance(getattr(bundle, "unified_stage_pack", None), dict):
        stage_models = bundle.unified_stage_pack.get("stage_models", {})
        if stage_models:
            return stage_models
    if isinstance(getattr(bundle, "registry_pack", None), dict):
        stage_models = (
            bundle.registry_pack.get("inference_pack", {})
            .get("stage_models", {})
        )
        if stage_models:
            return stage_models
    return {}


def get_stage_model_bundle(artifacts: Any, stage: str) -> dict[str, Any] | None:
    stage_models = _extract_stage_models(artifacts)
    model_bundle = stage_models.get(stage)
    return model_bundle if isinstance(model_bundle, dict) else None


def get_stage_feature_map(artifacts: Any) -> dict[str, list[str]]:
    stage_models = _extract_stage_models(artifacts)
    out: dict[str, list[str]] = {}
    for stage, cfg in stage_models.items():
        if isinstance(cfg, dict):
            out[str(stage)] = [str(x) for x in cfg.get("features_kept", [])]
    return out


def derive_feature_value(record: dict[str, Any], feature: str, stage: str) -> float | None:
    if feature in record and record.get(feature) not in (None, ""):
        return safe_float(record.get(feature), default=None)

    # Shared / router-derived helpers
    max_dpd = safe_float(record.get("max_dpd"), default=None)
    recent_ontime_ratio = safe_float(record.get("recent_ontime_ratio"), default=None)
    installment_ratio = safe_float(record.get("installment_paid_before_due_ratio"), default=None)
    main_income = safe_float(record.get("main_income"), default=None)
    ext_risk = safe_float(record.get("external_risk_score"), default=None)
    open_loans = safe_float(record.get("num_open_loans"), default=None)

    if feature == "a_has_external_credit_exposure":
        if open_loans is not None:
            return float(open_loans > 0)
        if ext_risk is not None:
            return 1.0
        return None
    if feature == "a_tax_amount_4527230_max":
        return main_income

    if feature == "b_has_recent_dpd":
        return float(max_dpd is not None and max_dpd > 0)
    if feature == "b_has_paid_before_due_signal":
        if installment_ratio is not None:
            return float(installment_ratio > 0)
        if recent_ontime_ratio is not None:
            return float(recent_ontime_ratio >= 0.85)
        return None
    if feature == "b_annuity_max":
        return safe_float(record.get("b_annuity_max"), default=main_income)
    if feature == "b_amtinstpaidbefdue_max":
        return safe_float(record.get("b_amtinstpaidbefdue_max"), default=None)
    if feature == "b_paid_before_due_to_annuity_ratio":
        numerator = safe_float(record.get("b_amtinstpaidbefdue_max"), default=None)
        denominator = safe_float(record.get("b_annuity_max"), default=None)
        if numerator is not None and denominator not in (None, 0):
            return numerator / denominator
        if installment_ratio is not None:
            return installment_ratio
        if recent_ontime_ratio is not None:
            return recent_ontime_ratio
        return None
    if feature == "b_avgmaxdpdlast9m_max":
        return max_dpd

    if feature == "c_actualdpd_max":
        return max_dpd
    if feature == "c_recent_40dpd_flag":
        return float(max_dpd is not None and max_dpd >= 40)
    if feature == "c_avgdbddpdlast24m_max":
        return max_dpd
    if feature == "c_avgdpdtolclosure24_max":
        return max_dpd
    if feature == "c_recent_vs_long_dpd_ratio":
        short = safe_float(record.get("c_actualdpd_max"), default=max_dpd)
        long = safe_float(record.get("c_avgdbddpdlast24m_max"), default=max_dpd)
        if short is not None and long not in (None, 0):
            return short / long
        return 1.0 if short not in (None, 0) else None
    if feature == "c_days_since_last_40dpd":
        return safe_float(record.get("c_days_since_last_40dpd"), default=None)

    return None


def build_stage_feature_frame(record: dict[str, Any], stage: str, artifacts: Any) -> tuple[pd.DataFrame | None, list[str], dict[str, Any]]:
    model_bundle = get_stage_model_bundle(artifacts, stage)
    if not model_bundle:
        return None, [], {}

    features = [str(x) for x in model_bundle.get("features_kept", [])]
    row: dict[str, Any] = {}
    missing: list[str] = []
    for feature in features:
        value = derive_feature_value(record, feature, stage)
        row[feature] = np.nan if value is None else value
        if value is None:
            missing.append(feature)
    return pd.DataFrame([row], columns=features), missing, row


def run_stage_inference(record: dict[str, Any], stage: str, artifacts: Any) -> InferenceResult:
    model_bundle = get_stage_model_bundle(artifacts, stage)
    if not model_bundle:
        return InferenceResult(
            stage=stage,
            risk_proba=None,
            score_300_900=None,
            scoring_status="artifact_stage_model_missing",
            model_features=[],
            model_input={},
            missing_model_features=[],
            used_fallback=True,
            fallback_reason="stage_model_missing",
        )

    pipeline = model_bundle.get("pipeline")
    features = [str(x) for x in model_bundle.get("features_kept", [])]
    if pipeline is None or not hasattr(pipeline, "predict_proba"):
        return InferenceResult(
            stage=stage,
            risk_proba=None,
            score_300_900=None,
            scoring_status="artifact_pipeline_missing_predict_proba",
            model_features=features,
            model_input={},
            missing_model_features=features,
            used_fallback=True,
            fallback_reason="predict_proba_missing",
        )

    X, missing, model_input = build_stage_feature_frame(record, stage, artifacts)
    if X is None:
        return InferenceResult(
            stage=stage,
            risk_proba=None,
            score_300_900=None,
            scoring_status="artifact_stage_feature_frame_unavailable",
            model_features=features,
            model_input={},
            missing_model_features=features,
            used_fallback=True,
            fallback_reason="feature_frame_unavailable",
        )

    try:
        probabilities = pipeline.predict_proba(X)
        risk_proba = float(probabilities[0, -1])
        score = float(300.0 + 600.0 * risk_proba)
        status = "model_inference_complete"
        if missing:
            status = "model_inference_complete_with_imputed_inputs"
        return InferenceResult(
            stage=stage,
            risk_proba=risk_proba,
            score_300_900=score,
            scoring_status=status,
            model_features=features,
            model_input=model_input,
            missing_model_features=missing,
            used_fallback=False,
        )
    except Exception as exc:
        return InferenceResult(
            stage=stage,
            risk_proba=None,
            score_300_900=None,
            scoring_status=f"artifact_inference_failed: {type(exc).__name__}",
            model_features=features,
            model_input=model_input,
            missing_model_features=missing,
            used_fallback=True,
            fallback_reason=str(exc),
        )
