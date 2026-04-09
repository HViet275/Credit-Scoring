from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .explain import reason_frequency_table
from .utils import humanize_token


COLOR_MAP = {
    "Decline / Intensive Review": "#d92d20",
    "Decline": "#d92d20",
    "Manual Review": "#f59e0b",
    "Review Or Decline": "#f59e0b",
    "Approve Standard": "#0f766e",
    "Approve Preferred": "#0b8f7a",
    "Starter Loan Small": "#2563eb",
    "Starter Loan Standard": "#10b981",
    "Monitor / Standard Queue": "#2563eb",
    "Intensive Collection Priority": "#d92d20",
    "High Risk Review Priority": "#f59e0b",
}

GAUGE_COLORS = ["#0f766e", "#22c55e", "#f59e0b", "#f97316", "#d92d20"]
GAUGE_LABELS = [
    (300, 420, "Prime"),
    (420, 540, "Strong"),
    (540, 660, "Watch"),
    (660, 780, "Review"),
    (780, 900, "High Risk"),
]


def _empty_figure(height: int = 160, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30 if title else 10, b=10), title=title)
    return fig


def _normalize_policy_thresholds(policy_thresholds: pd.DataFrame | None) -> pd.DataFrame:
    expected_cols = ["stage", "min_score", "max_score", "decision", "score_range"]
    if policy_thresholds is None or not isinstance(policy_thresholds, pd.DataFrame) or policy_thresholds.empty:
        return pd.DataFrame(columns=expected_cols)

    df = policy_thresholds.copy()

    rename_map = {}
    for col in df.columns:
        lc = str(col).strip().lower()
        if lc == "stage":
            rename_map[col] = "stage"
        elif lc in {"min_score", "band_min", "lower", "lower_bound"}:
            rename_map[col] = "min_score"
        elif lc in {"max_score", "band_max", "upper", "upper_bound"}:
            rename_map[col] = "max_score"
        elif lc in {"decision", "action", "policy_action", "decision_label"}:
            rename_map[col] = "decision"
        elif lc in {"score_range", "range"}:
            rename_map[col] = "score_range"

    if rename_map:
        df = df.rename(columns=rename_map)

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df["stage"] = df["stage"].astype(str).str.strip()
    df["min_score"] = pd.to_numeric(df["min_score"], errors="coerce")
    df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")
    df["decision"] = df["decision"].fillna("").map(humanize_token)

    missing_range = df["score_range"].isna() | (df["score_range"].astype(str).str.strip() == "")
    df.loc[missing_range, "score_range"] = df.loc[missing_range].apply(
        lambda r: f"{int(r['min_score']) if pd.notna(r['min_score']) else '?'} - {int(r['max_score']) if pd.notna(r['max_score']) else '?'}",
        axis=1,
    )

    return df[["stage", "min_score", "max_score", "decision", "score_range"]]


def plot_credit_meter(score: float, stage: str | None = None):
    score = float(score or 0)
    stage_text = f"Stage {str(stage).strip()}" if stage else "Credit Score"

    steps = []
    for (lo, hi, label), color in zip(GAUGE_LABELS, GAUGE_COLORS):
        steps.append({"range": [lo, hi], "color": color})

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"size": 42, "color": "#0b1f3a"}},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{stage_text}</b><br><span style='font-size:14px;color:#64748b;'>Model score on 300–900 risk scale</span>"},
            gauge={
                "axis": {"range": [300, 900], "tickwidth": 1, "tickcolor": "#64748b", "tickvals": [300, 420, 540, 660, 780, 900]},
                "bar": {"color": "#111827", "thickness": 0.22},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": steps,
                "threshold": {"line": {"color": "#111827", "width": 5}, "thickness": 0.85, "value": score},
            },
        )
    )

    annotations = []
    for lo, hi, label in GAUGE_LABELS:
        annotations.append(
            dict(
                x=(lo + hi) / 2,
                y=0,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=12, color="#334155"),
            )
        )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=85, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#0b1f3a"),
        annotations=[],
    )
    return fig


def plot_single_threshold_bar(score: float, stage: str, policy_thresholds: pd.DataFrame | None):
    df = _normalize_policy_thresholds(policy_thresholds)
    if df.empty or "stage" not in df.columns:
        return _empty_figure(140, "Threshold Bands Unavailable")

    stage = str(stage or "").strip()
    stage_df = df[df["stage"] == stage].copy()
    if stage_df.empty:
        return _empty_figure(140, f"No Threshold Bands for Stage {stage or 'Unknown'}")

    stage_df = stage_df.dropna(subset=["min_score", "max_score"]).sort_values(["min_score", "max_score"])
    if stage_df.empty:
        return _empty_figure(140, f"No Valid Threshold Bands for Stage {stage or 'Unknown'}")

    fig = go.Figure()
    for _, row in stage_df.iterrows():
        width = max(float(row["max_score"]) - float(row["min_score"]), 0)
        fig.add_trace(
            go.Bar(
                x=[width],
                y=["Score Band"],
                base=[float(row["min_score"])],
                orientation="h",
                name=str(row["decision"]),
                marker_color=COLOR_MAP.get(str(row["decision"]), "#94a3b8"),
                hovertemplate=f"{row['decision']}<br>{row['score_range']}<extra></extra>",
            )
        )
    fig.add_vline(x=float(score), line_width=3, line_dash="dash", line_color="#111827")
    fig.update_layout(
        barmode="stack",
        height=180,
        xaxis_title="Score",
        yaxis_title="",
        showlegend=True,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def plot_batch_score_histogram(df: pd.DataFrame):
    if df is None or df.empty or "score_300_900" not in df.columns:
        return _empty_figure(title="Score Distribution")
    ser = pd.to_numeric(df["score_300_900"], errors="coerce").dropna()
    if ser.empty:
        return _empty_figure(title="Score Distribution")
    return px.histogram(pd.DataFrame({"score_300_900": ser}), x="score_300_900", nbins=20, title="Score Distribution")


def plot_action_distribution(df: pd.DataFrame):
    if df is None or df.empty or "policy_action" not in df.columns:
        return _empty_figure(title="Decision Distribution")
    plot_df = (
        df.assign(action_label=df["policy_action"].map(humanize_token))
        .groupby("action_label", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    return px.bar(plot_df, x="action_label", y="count", title="Decision Distribution")


def plot_stage_distribution(df: pd.DataFrame):
    if df is None or df.empty or "stage" not in df.columns:
        return _empty_figure(title="Stage Distribution")
    plot_df = df.groupby("stage", as_index=False).size().rename(columns={"size": "count"})
    return px.pie(plot_df, names="stage", values="count", title="Stage Distribution", hole=0.45)


def plot_reason_frequency(df: pd.DataFrame):
    plot_df = reason_frequency_table(df)
    if plot_df is None or plot_df.empty:
        return _empty_figure(title="Top Driver Frequency")
    top_df = plot_df.head(10).sort_values("count", ascending=True)
    return px.bar(top_df, x="count", y="reason_text", orientation="h", title="Top Driver Frequency")


def plot_data_quality_summary(scored_df: pd.DataFrame, invalid_df: pd.DataFrame):
    scored_df = scored_df if isinstance(scored_df, pd.DataFrame) else pd.DataFrame()
    invalid_df = invalid_df if isinstance(invalid_df, pd.DataFrame) else pd.DataFrame()
    missing_count = 0
    if not scored_df.empty and "completeness_ratio" in scored_df.columns:
        missing_count = int((pd.to_numeric(scored_df["completeness_ratio"], errors="coerce") < 1).sum())
    plot_df = pd.DataFrame(
        {
            "category": ["Scored", "Rejected", "Has Missing Fields"],
            "count": [len(scored_df), len(invalid_df), missing_count],
        }
    )
    return px.bar(plot_df, x="category", y="count", title="Data Quality Summary")


def plot_policy_threshold_map(policy_thresholds: pd.DataFrame | None):
    df = _normalize_policy_thresholds(policy_thresholds)
    if df.empty:
        return _empty_figure(title="Policy Threshold Map")
    df = df.dropna(subset=["min_score", "max_score"]).copy()
    if df.empty:
        return _empty_figure(title="Policy Threshold Map")
    df["band_width"] = (df["max_score"] - df["min_score"]).clip(lower=0)
    fig = px.bar(
        df,
        x="band_width",
        y="stage",
        base="min_score",
        color="decision",
        orientation="h",
        title="Policy Threshold Map",
        hover_data={"score_range": True, "band_width": False, "min_score": True, "max_score": True},
    )
    fig.update_layout(xaxis_title="Score", yaxis_title="Stage")
    return fig
