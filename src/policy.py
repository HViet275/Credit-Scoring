from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import humanize_token


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _extract_thresholds(stage_rules: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    if not isinstance(stage_rules, dict):
        return None, None, None

    recommended = stage_rules.get("recommended_thresholds", {}) if isinstance(stage_rules.get("recommended_thresholds"), dict) else {}
    review_threshold = _to_float(stage_rules.get("review_threshold_score_300_900"))
    reject_threshold = _to_float(stage_rules.get("reject_threshold_score_300_900"))
    small_offer_threshold = _to_float(stage_rules.get("small_offer_threshold_score_300_900"))

    if review_threshold is None:
        review_threshold = _to_float(recommended.get("review_threshold_score_300_900"))
    if reject_threshold is None:
        reject_threshold = _to_float(recommended.get("reject_threshold_score_300_900"))
    if small_offer_threshold is None:
        small_offer_threshold = _to_float(recommended.get("small_offer_threshold_score_300_900"))

    return review_threshold, reject_threshold, small_offer_threshold


def apply_policy_to_score(stage: str, score: float, policy_rules: dict[str, Any]) -> dict[str, Any]:
    stage_rules = policy_rules.get("stages", {}).get(stage, {}) if isinstance(policy_rules, dict) else {}
    review_threshold, reject_threshold, small_offer_threshold = _extract_thresholds(stage_rules)

    if review_threshold is not None and reject_threshold is not None:
        review_threshold, reject_threshold = min(review_threshold, reject_threshold), max(review_threshold, reject_threshold)

        if stage == "A":
            if score >= reject_threshold:
                decision, zone = "review_or_decline", "review_or_decline"
                band_min, band_max = reject_threshold, 900.0
            elif small_offer_threshold is not None and score >= small_offer_threshold:
                decision, zone = "starter_loan_small", "starter_loan_small"
                band_min, band_max = small_offer_threshold, reject_threshold
            else:
                decision, zone = "starter_loan_standard", "starter_loan_standard"
                band_min, band_max = 300.0, small_offer_threshold if small_offer_threshold is not None else review_threshold
        elif stage == "B":
            preferred_threshold = small_offer_threshold
            if score >= reject_threshold:
                decision, zone = "reject_or_intensive_review", "reject_or_intensive_review"
                band_min, band_max = reject_threshold, 900.0
            elif score >= review_threshold:
                decision, zone = "manual_review", "manual_review"
                band_min, band_max = review_threshold, reject_threshold
            elif preferred_threshold is not None and score < preferred_threshold:
                decision, zone = "approve_preferred", "approve_preferred"
                band_min, band_max = 300.0, preferred_threshold
            else:
                decision, zone = "approve_standard", "approve_standard"
                band_min, band_max = preferred_threshold if preferred_threshold is not None else 300.0, review_threshold
        elif stage == "C":
            if score >= reject_threshold:
                decision, zone = "intensive_collection_priority", "intensive_collection_priority"
                band_min, band_max = reject_threshold, 900.0
            elif score >= review_threshold:
                decision, zone = "high_risk_review_priority", "high_risk_review_priority"
                band_min, band_max = review_threshold, reject_threshold
            else:
                decision, zone = "monitor_or_standard_queue", "monitor_or_standard_queue"
                band_min, band_max = 300.0, review_threshold
        else:
            if score >= reject_threshold:
                decision, zone = "decline", "reject"
                band_min, band_max = reject_threshold, 900.0
            elif score >= review_threshold:
                decision, zone = "manual_review", "review"
                band_min, band_max = review_threshold, reject_threshold
            else:
                decision, zone = "approve", "approve"
                band_min, band_max = 300.0, review_threshold

        return {
            "stage": stage,
            "decision": decision,
            "decision_label": humanize_token(decision),
            "zone": zone,
            "zone_label": humanize_token(zone),
            "band_min": band_min,
            "band_max": band_max,
            "review_threshold": review_threshold,
            "reject_threshold": reject_threshold,
            "policy_status": "threshold_applied",
        }

    return {
        "stage": stage,
        "decision": "manual_review",
        "decision_label": humanize_token("manual_review"),
        "zone": "manual_review",
        "zone_label": humanize_token("manual_review"),
        "band_min": None,
        "band_max": None,
        "review_threshold": review_threshold,
        "reject_threshold": reject_threshold,
        "policy_status": "threshold_missing_fallback_review",
    }


def apply_policy_to_dataframe(df: pd.DataFrame, artifacts: Any) -> pd.DataFrame:
    out = df.copy()
    policy_rules = getattr(artifacts, "policy_rules", {})
    rows = []

    for _, row in out.iterrows():
        p = apply_policy_to_score(
            str(row.get("stage", "")),
            float(row.get("score_300_900", 0) or 0),
            policy_rules,
        )
        rows.append(p)

    pol = pd.DataFrame(rows)
    out["decision_zone"] = pol["zone"]
    out["policy_action"] = pol["decision"]
    out["policy_action_label"] = pol["decision_label"]
    out["decision_zone_label"] = pol["zone_label"]
    out["band_min"] = pol["band_min"]
    out["band_max"] = pol["band_max"]
    out["review_threshold"] = pol["review_threshold"]
    out["reject_threshold"] = pol["reject_threshold"]
    out["policy_status"] = pol["policy_status"]
    return out


def build_policy_threshold_table(policy_rules: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for stage, info in policy_rules.get("stages", {}).items():
        review_threshold, reject_threshold, small_offer_threshold = _extract_thresholds(info)
        if review_threshold is None or reject_threshold is None:
            continue

        if stage == "A":
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": 300.0, "max_score": small_offer_threshold, "decision": humanize_token("starter_loan_standard"), "zone": humanize_token("starter_loan_standard"), "score_range": f"300 - {small_offer_threshold:.2f}"})
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": small_offer_threshold, "max_score": reject_threshold, "decision": humanize_token("starter_loan_small"), "zone": humanize_token("starter_loan_small"), "score_range": f"{small_offer_threshold:.2f} - {reject_threshold:.2f}"})
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": reject_threshold, "max_score": 900.0, "decision": humanize_token("review_or_decline"), "zone": humanize_token("review_or_decline"), "score_range": f"{reject_threshold:.2f} - 900"})
        elif stage == "B":
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": 300.0, "max_score": small_offer_threshold, "decision": humanize_token("approve_preferred"), "zone": humanize_token("approve_preferred"), "score_range": f"300 - {small_offer_threshold:.2f}"})
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": small_offer_threshold, "max_score": review_threshold, "decision": humanize_token("approve_standard"), "zone": humanize_token("approve_standard"), "score_range": f"{small_offer_threshold:.2f} - {review_threshold:.2f}"})
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": review_threshold, "max_score": reject_threshold, "decision": humanize_token("manual_review"), "zone": humanize_token("manual_review"), "score_range": f"{review_threshold:.2f} - {reject_threshold:.2f}"})
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": reject_threshold, "max_score": 900.0, "decision": humanize_token("reject_or_intensive_review"), "zone": humanize_token("reject_or_intensive_review"), "score_range": f"{reject_threshold:.2f} - 900"})
        elif stage == "C":
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": 300.0, "max_score": review_threshold, "decision": humanize_token("monitor_or_standard_queue"), "zone": humanize_token("monitor_or_standard_queue"), "score_range": f"300 - {review_threshold:.2f}"})
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": review_threshold, "max_score": reject_threshold, "decision": humanize_token("high_risk_review_priority"), "zone": humanize_token("high_risk_review_priority"), "score_range": f"{review_threshold:.2f} - {reject_threshold:.2f}"})
            rows.append({"stage": stage, "stage_label": info.get("recommended_role", stage), "min_score": reject_threshold, "max_score": 900.0, "decision": humanize_token("intensive_collection_priority"), "zone": humanize_token("intensive_collection_priority"), "score_range": f"{reject_threshold:.2f} - 900"})
    return pd.DataFrame(rows)
