from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .utils import safe_float


ROUTER_REQUIRED_FIELDS = ["tenure_months", "max_dpd"]
BASE_OPTIONAL_FIELDS = [
    "customer_id",
    "main_income",
    "num_open_loans",
    "recent_ontime_ratio",
    "utilization_ratio",
    "external_risk_score",
    "installment_paid_before_due_ratio",
    "a_has_external_credit_exposure",
    "a_tax_amount_4527230_max",
    "b_has_paid_before_due_signal",
    "b_has_recent_dpd",
    "b_annuity_max",
    "b_amtinstpaidbefdue_max",
    "b_paid_before_due_to_annuity_ratio",
    "b_avgmaxdpdlast9m_max",
    "c_actualdpd_max",
    "c_recent_40dpd_flag",
    "c_avgdbddpdlast24m_max",
    "c_avgdpdtolclosure24_max",
    "c_recent_vs_long_dpd_ratio",
    "c_days_since_last_40dpd",
]
ALL_FIELDS = ROUTER_REQUIRED_FIELDS + BASE_OPTIONAL_FIELDS


@dataclass
class ValidationIssue:
    field: str
    level: str
    message: str


@dataclass
class ValidationSummary:
    is_valid: bool
    issues: list[ValidationIssue]
    normalized_record: dict[str, Any]
    completeness_ratio: float
    missing_fields: list[str]



def _convert_record_types(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    numeric_fields = [c for c in ALL_FIELDS if c != "customer_id"]
    for col in numeric_fields:
        if col in normalized and normalized[col] not in (None, ""):
            normalized[col] = safe_float(normalized[col], default=None)
    if "customer_id" in normalized and normalized["customer_id"] is not None:
        normalized["customer_id"] = str(normalized["customer_id"]).strip()
    return normalized



def validate_single_record(record: dict[str, Any]) -> ValidationSummary:
    normalized = _convert_record_types(record)
    issues: list[ValidationIssue] = []

    missing_fields = [field for field in ROUTER_REQUIRED_FIELDS if normalized.get(field) in (None, "")]
    for field in missing_fields:
        issues.append(ValidationIssue(field=field, level="error", message="Routing field is missing."))

    if normalized.get("tenure_months") is not None and normalized["tenure_months"] < 0:
        issues.append(ValidationIssue("tenure_months", "error", "Tenure months cannot be negative."))

    if normalized.get("max_dpd") is not None and normalized["max_dpd"] < 0:
        issues.append(ValidationIssue("max_dpd", "error", "Max DPD cannot be negative."))

    if normalized.get("main_income") is not None and normalized["main_income"] <= 0:
        issues.append(ValidationIssue("main_income", "warning", "Main income should usually be greater than 0 if provided."))

    if normalized.get("tenure_months") is not None and normalized["tenure_months"] > 240:
        issues.append(ValidationIssue("tenure_months", "warning", "Tenure months looks unusually high. Please verify input."))

    if normalized.get("max_dpd") is not None and normalized["max_dpd"] > 365:
        issues.append(ValidationIssue("max_dpd", "warning", "Max DPD is unusually high. Please verify input."))

    available_count = sum(1 for field in ALL_FIELDS if normalized.get(field) not in (None, ""))
    completeness_ratio = available_count / len(ALL_FIELDS)
    is_valid = not any(issue.level == "error" for issue in issues)

    return ValidationSummary(
        is_valid=is_valid,
        issues=issues,
        normalized_record=normalized,
        completeness_ratio=completeness_ratio,
        missing_fields=[field for field in ALL_FIELDS if normalized.get(field) in (None, "")],
    )



def validate_batch_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    working = df.copy()
    for col in ALL_FIELDS:
        if col not in working.columns:
            working[col] = None

    valid_rows = []
    invalid_rows = []
    warning_count = 0

    for _, row in working.iterrows():
        summary = validate_single_record(row.to_dict())
        payload = summary.normalized_record.copy()
        payload["completeness_ratio"] = round(summary.completeness_ratio, 4)
        payload["validation_issue_count"] = len(summary.issues)
        payload["validation_messages"] = " | ".join(issue.message for issue in summary.issues)
        if any(issue.level == "warning" for issue in summary.issues):
            warning_count += 1
        if summary.is_valid:
            valid_rows.append(payload)
        else:
            invalid_rows.append(payload)

    valid_df = pd.DataFrame(valid_rows)
    invalid_df = pd.DataFrame(invalid_rows)
    summary = {
        "total_rows": int(len(working)),
        "valid_rows": int(len(valid_df)),
        "invalid_rows": int(len(invalid_df)),
        "warning_rows": int(warning_count),
        "required_fields": ROUTER_REQUIRED_FIELDS,
        "optional_fields": BASE_OPTIONAL_FIELDS,
    }
    return valid_df, invalid_df, summary
