from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import joblib
except Exception:
    joblib = None


@dataclass
class ArtifactBundle:
    root_dir: Path
    manifest: dict[str, Any]
    policy_rules: dict[str, Any]
    product_manifest: dict[str, Any]
    input_schema: pd.DataFrame
    business_glossary: str
    unified_stage_pack: Any | None
    registry_pack: Any | None
    inference_service_pack: Any | None
    loaded_files: dict[str, bool]


DEFAULT_MANIFEST = {
    "app_title": "Credit Decision Console",
    "champion_stage": "B",
    "model_version": "demo-v1",
    "policy_version": "policy-v1",
}

DEFAULT_POLICY_RULES = {
    "stages": {
        "A": {"label": "Starter Offer", "bands": [
            {"min_score": 0, "max_score": 260, "decision": "review_or_decline", "zone": "review_or_decline"},
            {"min_score": 260, "max_score": 340, "decision": "starter_loan_small", "zone": "starter_loan_small"},
            {"min_score": 340, "max_score": 901, "decision": "starter_loan_standard", "zone": "starter_loan_standard"},
        ]},
        "B": {"label": "Champion Decision", "bands": [
            {"min_score": 0, "max_score": 260, "decision": "reject_or_intensive_review", "zone": "reject_or_intensive_review"},
            {"min_score": 260, "max_score": 330, "decision": "manual_review", "zone": "manual_review"},
            {"min_score": 330, "max_score": 390, "decision": "approve_standard", "zone": "approve_standard"},
            {"min_score": 390, "max_score": 901, "decision": "approve_preferred", "zone": "approve_preferred"},
        ]},
        "C": {"label": "Collection Guardrail", "bands": [
            {"min_score": 0, "max_score": 220, "decision": "intensive_collection_priority", "zone": "intensive_collection_priority"},
            {"min_score": 220, "max_score": 320, "decision": "high_risk_review_priority", "zone": "high_risk_review_priority"},
            {"min_score": 320, "max_score": 901, "decision": "monitor_or_standard_queue", "zone": "monitor_or_standard_queue"},
        ]},
    }
}

DEFAULT_PRODUCT_MANIFEST = {
    "supported_tabs": ["single_customer", "batch_scoring", "decision_policy"],
    "primary_stage": "B",
    "notes": ["Fallback proxy scoring is allowed when artifact service is unavailable."],
}

DEFAULT_SCHEMA = pd.DataFrame([
    {"field_name": "customer_id", "required": True, "required_for_model": True, "type": "string", "description": "Customer identifier"},
    {"field_name": "tenure_months", "required": True, "required_for_model": True, "type": "number", "description": "Months since first loan"},
    {"field_name": "max_dpd", "required": True, "required_for_model": True, "type": "number", "description": "Maximum days past due"},
    {"field_name": "main_income", "required": True, "required_for_model": True, "type": "number", "description": "Monthly primary income"},
    {"field_name": "recent_ontime_ratio", "required": False, "required_for_model": False, "type": "number", "description": "Recent on-time repayment ratio"},
])


def _read_json_if_exists(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _read_text_if_exists(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else default


def _read_csv_if_exists(path: Path, default: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else default.copy()


def _load_joblib_if_exists(path: Path) -> Any | None:
    if not path.exists() or joblib is None:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_artifact_bundle(root_dir: str | Path = "artifacts") -> ArtifactBundle:
    root = Path(root_dir)
    manifest_path = root / "model_registry_manifest.json"
    policy_path = root / "policy_rules.json"
    product_manifest_path = root / "product_pack_manifest.json"
    glossary_path = root / "business_glossary.md"
    schema_path = root / "streamlit_input_schema.csv"
    unified_stage_path = root / "unified_stage_inference_pack.joblib"
    registry_pack_path = root / "model_registry_pack.joblib"
    inference_pack_path = root / "unified_inference_service.joblib"

    loaded_files = {
        "model_registry_manifest.json": manifest_path.exists(),
        "policy_rules.json": policy_path.exists(),
        "product_pack_manifest.json": product_manifest_path.exists(),
        "business_glossary.md": glossary_path.exists(),
        "streamlit_input_schema.csv": schema_path.exists(),
        "unified_stage_inference_pack.joblib": unified_stage_path.exists(),
        "model_registry_pack.joblib": registry_pack_path.exists(),
        "unified_inference_service.joblib": inference_pack_path.exists(),
    }

    return ArtifactBundle(
        root_dir=root,
        manifest=_read_json_if_exists(manifest_path, DEFAULT_MANIFEST),
        policy_rules=_read_json_if_exists(policy_path, DEFAULT_POLICY_RULES),
        product_manifest=_read_json_if_exists(product_manifest_path, DEFAULT_PRODUCT_MANIFEST),
        input_schema=_read_csv_if_exists(schema_path, DEFAULT_SCHEMA),
        business_glossary=_read_text_if_exists(glossary_path, "Glossary file not found."),
        unified_stage_pack=_load_joblib_if_exists(unified_stage_path),
        registry_pack=_load_joblib_if_exists(registry_pack_path),
        inference_service_pack=_load_joblib_if_exists(inference_pack_path),
        loaded_files=loaded_files,
    )


@dataclass
class AppArtifacts:
    champion_stage: str
    demo_mode: bool
    loaded_files: list[str]
    missing_files: list[str]
    schema: pd.DataFrame
    policy_rules: dict[str, Any]
    registry_manifest: dict[str, Any]
    product_manifest: dict[str, Any]
    inference_service: Any
    bundle: ArtifactBundle


def load_artifacts(root_dir: str | Path = "artifacts") -> AppArtifacts:
    bundle = load_artifact_bundle(root_dir)
    loaded_files = [k for k, v in bundle.loaded_files.items() if v]
    missing_files = [k for k, v in bundle.loaded_files.items() if not v]
    schema = bundle.input_schema.copy()
    if "required_for_model" not in schema.columns and "required" in schema.columns:
        schema["required_for_model"] = schema["required"]
    return AppArtifacts(
        champion_stage=str(bundle.manifest.get("champion_stage", "B")),
        demo_mode=not any(bundle.loaded_files.values()),
        loaded_files=loaded_files,
        missing_files=missing_files,
        schema=schema,
        policy_rules=bundle.policy_rules,
        registry_manifest=bundle.manifest,
        product_manifest=bundle.product_manifest,
        inference_service=bundle.inference_service_pack or {},
        bundle=bundle,
    )
