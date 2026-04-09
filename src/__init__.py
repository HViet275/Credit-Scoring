"""Production-like credit scoring app service layer."""

from .loader import ArtifactBundle, AppArtifacts, load_artifact_bundle, load_artifacts
from .router import detect_stage, detect_stage_dataframe
from .scorer import score_single_record, score_batch_dataframe, score_dataframe
from .policy import apply_policy_to_score, apply_policy_to_dataframe, build_policy_threshold_table
from .explain import build_business_explanation, normalize_reason_records, build_explanation_block

__all__ = [
    "ArtifactBundle",
    "AppArtifacts",
    "load_artifact_bundle",
    "load_artifacts",
    "detect_stage",
    "detect_stage_dataframe",
    "score_single_record",
    "score_batch_dataframe",
    "score_dataframe",
    "apply_policy_to_score",
    "apply_policy_to_dataframe",
    "build_policy_threshold_table",
    "build_business_explanation",
    "normalize_reason_records",
    "build_explanation_block",
]
