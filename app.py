from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.explain import build_explanation_block
from src.inference import get_stage_feature_map
from src.loader import AppArtifacts, load_artifacts
from src.policy import apply_policy_to_dataframe, build_policy_threshold_table
from src.router import detect_stage, detect_stage_dataframe
from src.scorer import score_batch_dataframe, score_dataframe
from src.utils import (
    coerce_input_types,
    dataframe_from_single_payload,
    dataframe_to_csv_bytes,
    dataframe_to_download_bytes,
    humanize_token,
    load_sample_payload,
    validate_required_columns,
)
from src.validation import validate_batch_dataframe, validate_single_record
from src.visuals import (
    plot_action_distribution,
    plot_batch_score_histogram,
    plot_credit_meter,
    plot_data_quality_summary,
    plot_policy_threshold_map,
    plot_reason_frequency,
    plot_single_threshold_bar,
    plot_stage_distribution,
)

APP_TITLE = "Credit Decision Console"
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
SAMPLE_DIR = BASE_DIR / "sample_data"
SAMPLE_BATCH_PATH = SAMPLE_DIR / "test_batch.csv"


@st.cache_resource(show_spinner=False)
def get_artifacts() -> AppArtifacts:
    return load_artifacts(ARTIFACT_DIR)


@st.cache_data(show_spinner=False)
def load_sample_batch_bytes() -> bytes:
    return SAMPLE_BATCH_PATH.read_bytes()


@st.cache_data(show_spinner=False)
def load_sample_batch_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_BATCH_PATH)


def run_scoring_pipeline(df: pd.DataFrame, artifacts: AppArtifacts) -> pd.DataFrame:
    work_df = coerce_input_types(df.copy(), artifacts.schema)
    work_df = detect_stage_dataframe(work_df)
    work_df = score_dataframe(work_df, artifacts)
    work_df = apply_policy_to_dataframe(work_df, artifacts)
    work_df = build_explanation_block(work_df, artifacts)
    return work_df


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #f4f7fb;
            --navy-950: #081a33;
            --navy-900: #0b1f3a;
            --navy-800: #16345e;
            --navy-700: #204e8b;
            --slate-700: #42526b;
            --slate-600: #52637d;
            --slate-500: #64748b;
            --line: #d8e1ee;
            --line-soft: #e8eef7;
            --card-bg: rgba(255,255,255,0.96);
            --teal: #0ea5a4;
            --cyan: #2563eb;
        }

        html {scroll-behavior: smooth;}
        .stApp {
            background: radial-gradient(circle at top left, #eef4ff 0%, #f6f9fc 40%, #eef3f8 100%);
            color: var(--navy-900);
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .main .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.4rem;
            max-width: 1340px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1f3a 0%, #132b4f 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }
        [data-testid="stSidebar"] * {color: #ecf4ff !important;}

        .sidebar-section-title {
            font-size: 1.02rem;
            font-weight: 800;
            margin: 0 0 0.7rem 0;
            color: #ffffff;
        }
        .sidebar-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.95rem 0.95rem 0.8rem 0.95rem;
            margin-bottom: 1rem;
        }
        .sidebar-copy {
            color: rgba(236,244,255,0.76);
            line-height: 1.65;
            font-size: 0.93rem;
        }
        .sidebar-nav {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            margin-top: 0.45rem;
            margin-bottom: 0.9rem;
        }
        .sidebar-nav a {
            display: block;
            text-decoration: none;
            color: rgba(255,255,255,0.84);
            padding: 0.72rem 0.82rem;
            border-radius: 12px;
            font-size: 0.96rem;
            font-weight: 700;
            border: 1px solid transparent;
            transition: 0.2s ease;
        }
        .sidebar-nav a:hover {
            background: rgba(255,255,255,0.07);
            border-color: rgba(255,255,255,0.08);
            color: #ffffff;
        }
        .sidebar-nav a.active {
            background: linear-gradient(135deg, rgba(14,165,164,0.18) 0%, rgba(37,99,235,0.24) 100%);
            border-color: rgba(147,197,253,0.28);
            color: #ffffff;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
        }

        .hero-shell {
            padding: 26px 28px;
            border-radius: 28px;
            background: linear-gradient(135deg, #0b1f3a 0%, #16345e 58%, #204e8b 100%);
            box-shadow: 0 18px 40px rgba(11,31,58,0.18);
            margin-bottom: 1rem;
            color: white;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .hero-kicker {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: rgba(231,242,255,0.82);
            margin-bottom: 0.4rem;
            font-weight: 700;
        }
        .hero-title {
            font-size: 2.25rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            color: white;
            line-height: 1.15;
        }
        .hero-sub {
            color: rgba(231,242,255,0.9);
            margin-bottom: 1rem;
            font-size: 1.02rem;
            line-height: 1.7;
            max-width: 980px;
        }
        .hero-meta {
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            margin-top: 0.4rem;
            margin-bottom: 1rem;
        }
        .hero-pill {
            padding: 0.46rem 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.12);
            font-size: 0.85rem;
            color: #f8fbff;
            font-weight: 700;
        }
        .stage-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
        }
        .stage-card {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 20px;
            padding: 1rem;
            min-height: 156px;
        }
        .stage-label {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.84;
            margin-bottom: 0.35rem;
        }
        .stage-title {
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .stage-text {
            font-size: 0.94rem;
            line-height: 1.62;
            color: rgba(255,255,255,0.9);
        }

        .section-wrap {
            padding-top: 0.4rem;
            padding-bottom: 1.6rem;
            border-bottom: 1px solid var(--line-soft);
            margin-bottom: 0.55rem;
        }
        .section-title-anchor {
            scroll-margin-top: 72px;
        }
        .section-title-xl {
            font-size: 1.95rem;
            font-weight: 800;
            color: var(--navy-900);
            margin-bottom: 0.28rem;
            line-height: 1.15;
        }
        .section-subtitle {
            color: var(--slate-600);
            margin-bottom: 1rem;
            font-size: 1rem;
            line-height: 1.7;
        }
        .section-shell {
            border: 1px solid var(--line-soft);
            border-radius: 22px;
            padding: 1.2rem;
            background: rgba(255,255,255,0.95);
            box-shadow: 0 10px 28px rgba(11,31,58,0.05);
        }

        .decision-card, .reason-card, .info-card {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 18px 20px;
            background: var(--card-bg);
            backdrop-filter: blur(8px);
            box-shadow: 0 12px 32px rgba(11,31,58,0.08);
            margin-bottom: 14px;
        }
        .decision-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(240,246,255,0.96) 100%);
        }
        .card-label {
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--slate-500);
            margin-bottom: 6px;
            font-weight: 700;
        }
        .card-value {font-size: 1.24rem; font-weight: 800; color: var(--navy-900); line-height: 1.3;}
        .section-title {
            font-size: 1.02rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            color: var(--navy-900);
            margin: 0.7rem 0 0.8rem 0;
        }
        .stMetric {
            background: rgba(255,255,255,0.95);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.7rem 0.8rem;
            box-shadow: 0 8px 22px rgba(11,31,58,0.05);
        }
        label, .stTextInput label, .stNumberInput label, .stSlider label {
            color: var(--navy-800) !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(15,118,110,0.18);
            font-weight: 700;
            padding: 0.55rem 1.1rem;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
            color: white;
            border: none;
            box-shadow: 0 12px 28px rgba(37,99,235,0.22);
        }

        .footer-shell {
            margin-top: 1.2rem;
            padding: 1.5rem 1.5rem 1rem 1.5rem;
            border-radius: 24px 24px 0 0;
            background: linear-gradient(180deg, #0b1f3a 0%, #10284a 100%);
            color: rgba(255,255,255,0.9);
        }
        .footer-grid {
            display: grid;
            grid-template-columns: 1.3fr 1fr 1fr;
            gap: 1.2rem;
        }
        .footer-title {
            font-size: 0.96rem;
            font-weight: 800;
            margin-bottom: 0.7rem;
            color: #ffffff;
        }
        .footer-copy {
            font-size: 0.92rem;
            line-height: 1.65;
            color: rgba(255,255,255,0.76);
        }
        .footer-link {
            display: block;
            color: rgba(255,255,255,0.84) !important;
            text-decoration: none;
            font-size: 0.92rem;
            margin-bottom: 0.42rem;
        }
        .footer-link:hover {
            color: #ffffff !important;
            text-decoration: underline;
        }
        .footer-bottom {
            margin-top: 1rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(255,255,255,0.12);
            font-size: 0.88rem;
            color: rgba(255,255,255,0.68);
        }

        @media (max-width: 1100px) {
            .stage-grid, .footer-grid {
                grid-template-columns: 1fr;
            }
            .hero-title {
                font-size: 1.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_scroll_observer() -> None:
    components.html(
        """
        <script>
        const setupSpaNav = () => {
          const rootDoc = window.parent.document;
          const navLinks = Array.from(rootDoc.querySelectorAll('#sidebar-spa-nav a'));
          const sectionIds = ['single-decision', 'batch-decision', 'policy'];
          const sections = sectionIds
            .map(id => rootDoc.getElementById(`${id}-title`) || rootDoc.getElementById(id))
            .filter(Boolean);

          if (!navLinks.length || !sections.length) return;

          if (window.parent.__spaNavCleanup) {
            try { window.parent.__spaNavCleanup(); } catch (e) {}
          }

          const updateActive = (id) => {
            navLinks.forEach(link => link.classList.remove('active'));
            const active = navLinks.find(link => link.getAttribute('href') === `#${id}`);
            if (active) active.classList.add('active');
          };

          const scrollToSection = (sectionId) => {
            const target = rootDoc.getElementById(`${sectionId}-title`) || rootDoc.getElementById(sectionId);
            if (!target) return;
            updateActive(sectionId);
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            setTimeout(() => {
              try { window.parent.scrollBy({ top: -12, behavior: 'instant' }); } catch (e) {}
            }, 250);
          };

          const clickHandlers = [];
          navLinks.forEach(link => {
            const handler = (e) => {
              const href = link.getAttribute('href');
              if (!href || !href.startsWith('#')) return;
              e.preventDefault();
              scrollToSection(href.slice(1));
            };
            link.addEventListener('click', handler);
            clickHandlers.push([link, handler]);
          });

          const getClosestSection = () => {
            const viewportOffset = 120;
            let bestId = sectionIds[0];
            let bestDistance = Infinity;
            sections.forEach((section, idx) => {
              const distance = Math.abs(section.getBoundingClientRect().top - viewportOffset);
              if (distance < bestDistance) {
                bestDistance = distance;
                bestId = sectionIds[idx];
              }
            });
            return bestId;
          };

          const scrollHost = rootDoc.querySelector('section[data-testid="stMain"]') || window.parent;
          let ticking = false;
          const onScroll = () => {
            if (ticking) return;
            ticking = true;
            window.parent.requestAnimationFrame(() => {
              updateActive(getClosestSection());
              ticking = false;
            });
          };

          scrollHost.addEventListener('scroll', onScroll, { passive: true });
          window.parent.addEventListener('scroll', onScroll, { passive: true });
          onScroll();

          window.parent.__spaNavCleanup = () => {
            clickHandlers.forEach(([link, handler]) => link.removeEventListener('click', handler));
            scrollHost.removeEventListener('scroll', onScroll);
            window.parent.removeEventListener('scroll', onScroll);
          };
        };
        setTimeout(setupSpaNav, 500);
        </script>
        """,
        height=0,
    )


def compute_risk_band(score: float) -> str:
    if score >= 390:
        return "Low Risk"
    if score >= 330:
        return "Moderate Risk"
    if score >= 260:
        return "Elevated Risk"
    return "High Risk"


def compute_data_quality_label(completeness_ratio: float) -> str:
    if completeness_ratio >= 0.95:
        return "High"
    if completeness_ratio >= 0.75:
        return "Medium"
    return "Low"


def render_sidebar(artifacts: AppArtifacts) -> None:
    manifest = artifacts.registry_manifest or {}

    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">System Snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-copy"><b style="color:#fff;">Current champion stage:</b> {artifacts.champion_stage}</div>
                <div style="height:8px;"></div>
                <div class="sidebar-copy"><b style="color:#fff;">App mode:</b> {'Demo fallback' if artifacts.demo_mode else 'Artifact-backed'}</div>
                <div style="height:8px;"></div>
                <div class="sidebar-copy"><b style="color:#fff;">Model version:</b> {manifest.get('model_version', 'n/a')}</div>
                <div style="height:8px;"></div>
                <div class="sidebar-copy"><b style="color:#fff;">Policy version:</b> {manifest.get('policy_version', 'n/a')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Menu</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-nav" id="sidebar-spa-nav">
                <a href="#single-decision" class="active">Single Decision</a>
                <a href="#batch-decision">Batch Decision</a>
                <a href="#policy">Policy</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Credit Scoring System</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-copy">
                A one-page credit decisioning workspace for single-case review, batch processing,
                and policy-led deployment governance.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero(artifacts: AppArtifacts) -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Retail Banking • Decisioning • Payment Risk</div>
            <div class="hero-title">Credit Scoring Workspace</div>
            <div class="hero-sub">
                A comprehensive credit scoring workspace for application routing, champion-model inference,
                batch review operations, and policy decisions across the full customer lifecycle.
                Current champion routing is configured to Stage <b>{artifacts.champion_stage}</b>.
            </div>
            <div class="hero-meta">
                <div class="hero-pill">Single-case decisioning</div>
                <div class="hero-pill">Batch decision processing</div>
                <div class="hero-pill">Policy-led governance</div>
                <div class="hero-pill">Stage-aware A / B / C routing</div>
            </div>
            <div class="stage-grid">
                <div class="stage-card">
                    <div class="stage-label">Stage A</div>
                    <div class="stage-title">Application / New-to-Bank</div>
                    <div class="stage-text">
                        Evaluate first-time or thin-file applicants using application attributes,
                        affordability signals, and external credit context for onboarding decisions.
                    </div>
                </div>
                <div class="stage-card">
                    <div class="stage-label">Stage B</div>
                    <div class="stage-title">Early Behavior / Champion Stage</div>
                    <div class="stage-text">
                        Score customers with early repayment behavior and initial portfolio signals.
                        This stage is optimized for champion inference and operational routing.
                    </div>
                </div>
                <div class="stage-card">
                    <div class="stage-label">Stage C</div>
                    <div class="stage-title">Mature Portfolio / Risk Control</div>
                    <div class="stage-text">
                        Assess customers with richer repayment history to support line management,
                        policy control, and portfolio-level risk monitoring.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Loaded artifact files", expanded=False):
        st.write("Model, policy, schema, and manifest files are loaded from the artifacts folder.")

    with st.expander("Expected input fields", expanded=False):
        st.write("Check the required fields for scoring input.")


def render_decision_summary(row: dict, policy_thresholds: pd.DataFrame) -> None:
    decision = humanize_token(row.get("policy_action"))
    zone = humanize_token(row.get("decision_zone"))
    score = float(row.get("score_300_900") or 0)
    risk_band = compute_risk_band(score)
    data_quality = compute_data_quality_label(float(row.get("completeness_ratio") or 0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Decision", decision)
    c2.metric("Stage", str(row.get("stage", "")))
    c3.metric("Credit Score", f"{score:.1f}")
    c4.metric("Risk Band", risk_band)
    c5.metric("Data Quality", data_quality)

    st.markdown("<div class='section-title'>Decision Summary</div>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1.35])
    with left:
        st.plotly_chart(plot_credit_meter(score, str(row.get("stage", ""))), use_container_width=True)
    with right:
        st.markdown(
            f"<div class='decision-card'><div class='card-label'>Decision Overview</div><div class='card-value'>{decision}</div>"
            f"<div style='margin-top:10px;color:#4b5563; line-height:1.6;'>"
            f"<b>Policy zone:</b> {zone}<br>"
            f"<b>Risk probability:</b> {float(row.get('risk_proba') or 0):.4f}<br>"
            f"<b>Business note:</b> {row.get('business_explanation', '')}</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_single_threshold_bar(score, str(row.get("stage", "")), policy_thresholds), use_container_width=True)


def render_reasons(row: dict) -> None:
    st.markdown("<div class='section-title'>Top Reasons</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, col in enumerate(cols, start=1):
        label = row.get(f"reason_label_{idx}", f"Reason {idx}")
        text = row.get(f"reason_text_{idx}", "")
        col.markdown(
            f"<div class='reason-card'><div class='card-label'>{label}</div><div style='font-weight:600;color:#111827;'>{text}</div></div>",
            unsafe_allow_html=True,
        )


def render_policy_explanation(row: dict) -> None:
    st.markdown("<div class='section-title'>Policy Explanation</div>", unsafe_allow_html=True)
    decision = humanize_token(row.get("policy_action"))
    zone = humanize_token(row.get("decision_zone"))
    band_min = row.get("band_min")
    band_max = row.get("band_max")
    band_text = f"{band_min} to {band_max}" if band_min is not None and band_max is not None else "n/a"
    st.markdown(
        f"<div class='info-card'>The model score estimates relative risk. The policy layer converts that score into an operational decision. "
        f"For this customer, the current score falls in the <b>{zone}</b> band, resulting in the decision <b>{decision}</b>. "
        f"Current score range used for this band: <b>{band_text}</b>.</div>",
        unsafe_allow_html=True,
    )


def render_warning_block(row: dict) -> None:
    warnings = row.get("warnings", []) or []
    technical_status_note = row.get("technical_status_note", "")
    if warnings:
        for msg in warnings:
            st.warning(msg)
    if technical_status_note:
        st.info(technical_status_note)


FRIENDLY_FIELD_LABELS = {
    "customer_id": "Customer ID",
    "main_income": "Main Monthly Income ($)",
    "num_open_loans": "Number of Open Loans",
    "recent_ontime_ratio": "Recent On-time Payment Ratio",
    "utilization_ratio": "Credit Utilization Ratio",
    "external_risk_score": "External Credit Risk Score",
    "installment_paid_before_due_ratio": "Paid Before Due Ratio",
    "a_has_external_credit_exposure": "Has External Credit Exposure",
    "a_tax_amount_4527230_max": "Maximum Recorded Tax-related Amount",
    "b_has_paid_before_due_signal": "Has Early Payment Signal",
    "b_has_recent_dpd": "Has Recent Late Payment",
    "b_annuity_max": "Maximum Installment Amount",
    "b_amtinstpaidbefdue_max": "Maximum Amount Paid Before Due",
    "b_paid_before_due_to_annuity_ratio": "Early Payment to Installment Ratio",
    "b_avgmaxdpdlast9m_max": "Recent Delinquency Severity (Last 9 Months)",
    "c_actualdpd_max": "Maximum Actual Days Past Due",
    "c_recent_40dpd_flag": "Recent 40+ DPD Flag",
    "c_avgdbddpdlast24m_max": "Average Late Payment Severity (24 Months)",
    "c_avgdpdtolclosure24_max": "Average DPD to Closure (24 Months)",
    "c_recent_vs_long_dpd_ratio": "Recent vs Long-term Delinquency Ratio",
    "c_days_since_last_40dpd": "Days Since Last 40+ DPD",
}


def build_single_form(sample_payload: dict, artifacts: AppArtifacts) -> dict:
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='section-title'>Customer Info</div>", unsafe_allow_html=True)
        customer_id = st.text_input("Customer ID", value=str(sample_payload.get("customer_id", "DEMO_001")))
        tenure_months = st.number_input(
            "Tenure Months",
            min_value=0.0,
            value=float(sample_payload.get("tenure_months", 0) or 0),
            step=1.0,
        )
        main_income = st.number_input(
            "Main Income ($)",
            min_value=0.0,
            value=float(sample_payload.get("main_income", 0) or 0),
            step=1000000.0,
        )
        num_open_loans = st.number_input(
            "Num Open Loans",
            min_value=0.0,
            value=float(sample_payload.get("num_open_loans", 0) or 0),
            step=1.0,
        )
        external_risk_score = st.number_input(
            "External Risk Score",
            min_value=0.0,
            value=float(sample_payload.get("external_risk_score", 0) or 0),
            step=1.0,
        )
    with right:
        st.markdown("<div class='section-title'>Repayment Behavior</div>", unsafe_allow_html=True)
        max_dpd = st.number_input(
            "Max DPD",
            min_value=0.0,
            value=float(sample_payload.get("max_dpd", 0) or 0),
            step=1.0,
        )
        recent_ontime_ratio = st.slider(
            "Recent On-time Ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(sample_payload.get("recent_ontime_ratio", 0.8) or 0.8),
            step=0.01,
        )
        installment_paid_before_due_ratio = st.slider(
            "Installment Paid Before Due Ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(sample_payload.get("installment_paid_before_due_ratio", 0.0) or 0.0),
            step=0.01,
        )

    payload = {
        "customer_id": customer_id,
        "tenure_months": tenure_months,
        "max_dpd": max_dpd,
        "main_income": main_income,
        "recent_ontime_ratio": recent_ontime_ratio,
        "num_open_loans": num_open_loans,
        "external_risk_score": external_risk_score,
        "installment_paid_before_due_ratio": installment_paid_before_due_ratio,
    }

    stage_preview = detect_stage(payload)
    stage_feature_map = get_stage_feature_map(artifacts)
    stage_features = stage_feature_map.get(stage_preview, [])

    st.markdown("<div class='section-title'>Credit Risk Input Details</div>", unsafe_allow_html=True)
    st.caption(f"Current routing preview: Stage {stage_preview}. These inputs feed the exported sklearn pipeline for that stage.")

    feature_cols = st.columns(2)
    for idx, feature in enumerate(stage_features):
        if feature in payload:
            continue

        default_val = sample_payload.get(feature)
        label = FRIENDLY_FIELD_LABELS.get(feature, feature.replace("_", " ").title())

        with feature_cols[idx % 2]:
            payload[feature] = st.text_input(
                label,
                value="" if default_val in (None, "") else str(default_val),
                help=feature,
            )

    with st.expander("Advanced JSON Input", expanded=False):
        json_text = st.text_area(
            "Optional JSON override",
            value=json.dumps(payload, indent=2, ensure_ascii=False),
            height=320,
        )
        try:
            payload = json.loads(json_text)
        except Exception:
            st.caption("JSON is invalid. The form values above will be used.")

    return payload


def render_single_case(artifacts: AppArtifacts) -> None:
    st.markdown("<div class='section-title'>Single-case Review</div>", unsafe_allow_html=True)
    st.caption("Use the guided form below to score one customer and review the decision summary.")
    sample_payload = load_sample_payload(SAMPLE_DIR / "sample_request_payload.json")
    payload = build_single_form(sample_payload, artifacts)

    summary = validate_single_record(payload)
    c1, c2, c3 = st.columns(3)
    c1.metric("Routing fields present", f"{len([f for f in ['tenure_months','max_dpd'] if summary.normalized_record.get(f) not in (None, '')])}/2")
    c2.metric("Input completeness", f"{summary.completeness_ratio:.0%}")
    c3.metric("Validation status", "Ready" if summary.is_valid else "Needs Fix")

    if summary.issues:
        for issue in summary.issues:
            if issue.level == "error":
                st.error(f"{issue.field}: {issue.message}")
            else:
                st.warning(f"{issue.field}: {issue.message}")

    if st.button("Run Decision", type="primary", key="run_single_decision_btn"):
        if not summary.is_valid:
            st.error("Please fix the required input issues before scoring.")
            return

        input_df = dataframe_from_single_payload(summary.normalized_record)
        result_df = run_scoring_pipeline(input_df, artifacts)
        if result_df.empty:
            st.error("No result was produced.")
            return
        row = result_df.iloc[0].to_dict()
        policy_thresholds = build_policy_threshold_table(artifacts.policy_rules)
        if not isinstance(policy_thresholds, pd.DataFrame):
            policy_thresholds = pd.DataFrame()

        render_warning_block(row)
        render_decision_summary(row, policy_thresholds)
        render_reasons(row)
        render_policy_explanation(row)

        with st.expander("Technical Notes", expanded=False):
            tech_view = {
                "Scoring status": row.get("scoring_status", ""),
                "Technical status note": row.get("technical_status_note", ""),
                "Completeness ratio": row.get("completeness_ratio", ""),
                "Missing fields": row.get("missing_fields", []),
                "Risk probability": row.get("risk_proba", None),
                "Model features used": row.get("model_features_used", []),
                "Missing model features": row.get("missing_model_features", []),
                "Model input payload": row.get("model_input_payload", {}),
                "Scored at": row.get("scored_at", ""),
            }
            st.json(tech_view)

        with st.expander("Raw Result", expanded=False):
            st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download decision result",
            data=dataframe_to_download_bytes(result_df),
            file_name="single_decision_result.csv",
            mime="text/csv",
        )


def render_batch_case(artifacts: AppArtifacts) -> None:
    st.markdown("<div class='section-title'>Batch Processing</div>", unsafe_allow_html=True)
    st.caption("Upload a CSV file, validate rows, run scoring, and review the batch decision dashboard.")

    if "batch_input_df" not in st.session_state:
        st.session_state.batch_input_df = None

    upload_col, demo_col = st.columns([1, 1])

    with upload_col:
        uploaded = st.file_uploader("Upload input CSV", type=["csv"], key="batch_uploader")

    with demo_col:
        if SAMPLE_BATCH_PATH.exists():
            st.download_button(
                label="Download sample CSV",
                data=load_sample_batch_bytes(),
                file_name="sample_batch.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("Use demo batch", use_container_width=True, key="use_demo_batch_btn"):
                st.session_state.batch_input_df = load_sample_batch_df()
                st.success("Loaded demo batch from sample_data/test_batch.csv")
        else:
            st.caption("No sample CSV found in sample_data/test_batch.csv")

    if uploaded is not None:
        st.session_state.batch_input_df = pd.read_csv(uploaded)

    input_df = st.session_state.batch_input_df
    if input_df is None:
        return

    st.markdown("<div class='section-title'>Upload Preview</div>", unsafe_allow_html=True)
    st.dataframe(input_df.head(20), use_container_width=True, hide_index=True)

    missing = validate_required_columns(input_df, artifacts.schema)
    _, invalid_df, validation_summary = validate_batch_dataframe(input_df)
    duplicates = int(input_df["customer_id"].duplicated().sum()) if "customer_id" in input_df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total rows", validation_summary["total_rows"])
    c2.metric("Valid rows", validation_summary["valid_rows"])
    c3.metric("Invalid rows", validation_summary["invalid_rows"])
    c4.metric("Warning rows", validation_summary["warning_rows"])
    c5.metric("Duplicate IDs", duplicates)

    if missing:
        st.warning("Missing required model fields: " + ", ".join(missing))
    if not invalid_df.empty:
        st.info("Some rows cannot be scored until input issues are fixed. You can still score the valid rows.")

    if st.button("Run Batch Scoring", type="primary", key="run_batch_scoring_btn"):
        scored_df, invalid_df2, batch_summary = score_batch_dataframe(input_df, artifacts)
        if scored_df.empty and invalid_df2.empty:
            st.error("No output was produced.")
            return

        st.session_state.batch_scored_df = scored_df
        st.session_state.batch_invalid_df = invalid_df2
        st.session_state.batch_summary = batch_summary

    if "batch_scored_df" not in st.session_state or "batch_summary" not in st.session_state:
        return

    scored_df = st.session_state.batch_scored_df
    invalid_df2 = st.session_state.get("batch_invalid_df", pd.DataFrame())
    batch_summary = st.session_state.batch_summary

    st.markdown("<div class='section-title'>Batch Result Summary</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Scored customers", batch_summary.get("scored_rows", 0))
    s2.metric("Average score", batch_summary.get("avg_score", "n/a"))
    s3.metric("Most common decision", batch_summary.get("top_action", "n/a"))
    s4.metric("Most common stage", batch_summary.get("top_stage", "n/a"))

    if not scored_df.empty:
        scored_df = apply_policy_to_dataframe(scored_df, artifacts)
        scored_df = build_explanation_block(scored_df, artifacts)

        col1, col2 = st.columns(2)
        col1.plotly_chart(plot_batch_score_histogram(scored_df), use_container_width=True)
        col2.plotly_chart(plot_action_distribution(scored_df), use_container_width=True)
        col3, col4 = st.columns(2)
        col3.plotly_chart(plot_stage_distribution(scored_df), use_container_width=True)
        col4.plotly_chart(plot_data_quality_summary(scored_df, invalid_df2), use_container_width=True)
        st.plotly_chart(plot_reason_frequency(scored_df), use_container_width=True)

    with st.expander("Scored Output Preview", expanded=True):
        st.dataframe(scored_df.head(100), use_container_width=True, hide_index=True)

    if not invalid_df2.empty:
        with st.expander("Rows Requiring Fix", expanded=False):
            st.dataframe(invalid_df2.head(100), use_container_width=True, hide_index=True)

    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "Download scored output",
        data=dataframe_to_csv_bytes(scored_df),
        file_name="batch_scored_output.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_col2.download_button(
        "Download rows requiring fix",
        data=dataframe_to_csv_bytes(invalid_df2),
        file_name="batch_invalid_rows.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button("Clear batch session", use_container_width=True, key="clear_batch_session_btn"):
        st.session_state.batch_input_df = None
        st.session_state.pop("batch_scored_df", None)
        st.session_state.pop("batch_invalid_df", None)
        st.session_state.pop("batch_summary", None)
        st.rerun()


def render_policy_view(artifacts: AppArtifacts) -> None:
    st.markdown("<div class='section-title'>Policy Governance</div>", unsafe_allow_html=True)
    st.caption("Review how model scores are translated into operational decisions and stage-level thresholds.")
    st.markdown(
        "The model score estimates relative risk. Policy rules then convert the score into an action, and each stage can use different score thresholds."
    )

    policy_thresholds = build_policy_threshold_table(artifacts.policy_rules)
    if not isinstance(policy_thresholds, pd.DataFrame):
        policy_thresholds = pd.DataFrame()
    if not policy_thresholds.empty:
        display_df = policy_thresholds[["stage", "stage_label", "score_range", "decision", "zone"]].rename(
            columns={
                "stage": "Stage",
                "stage_label": "Stage Description",
                "score_range": "Score Range",
                "decision": "Decision",
                "zone": "Policy Zone",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.plotly_chart(plot_policy_threshold_map(policy_thresholds), use_container_width=True)

    with st.expander("Registry Manifest", expanded=False):
        st.json(artifacts.registry_manifest)
    with st.expander("Product Manifest", expanded=False):
        st.json(artifacts.product_manifest)
    if isinstance(artifacts.inference_service, dict) and artifacts.inference_service:
        with st.expander("Inference Service Reference", expanded=False):
            st.json(artifacts.inference_service)


def render_section_start(section_id: str, title: str, subtitle: str, border: bool = True) -> None:
    border_style = "" if border else " style='border-bottom:none;'"
    st.markdown(
        f"""
        <div id="{section_id}" class="section-wrap"{border_style}>
            <div id="{section_id}-title" class="section-title-anchor section-title-xl">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
            <div class="section-shell">
        """,
        unsafe_allow_html=True,
    )


def render_section_end() -> None:
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_footer() -> None:
    current_year = datetime.now().year
    st.markdown(
        f"""
        <div class="footer-shell">
            <div class="footer-grid">
                <div>
                    <div class="footer-title">Credit Scoring System</div>
                    <div class="footer-copy">
                        A professional decisioning platform for single-case review, batch scoring operations,
                        and policy-driven risk governance across retail banking workflows.
                    </div>
                </div>
                <div>
                    <div class="footer-title">Support / Help</div>
                    <a class="footer-link" href="#">Documentation</a>
                    <a class="footer-link" href="#">API Reference</a>
                    <a class="footer-link" href="#">Contact Support</a>
                </div>
                <div>
                    <div class="footer-title">Legal</div>
                    <a class="footer-link" href="#">Privacy Policy</a>
                    <a class="footer-link" href="#">Terms of Service</a>
                    <a class="footer-link" href="#">Security</a>
                </div>
            </div>
            <div class="footer-bottom">© {current_year} Credit Scoring System. All rights reserved.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💳", layout="wide", initial_sidebar_state="expanded")
    artifacts = get_artifacts()
    inject_css()
    render_sidebar(artifacts)
    render_hero(artifacts)
    inject_scroll_observer()

    render_section_start(
        "single-decision",
        "Single Decision",
        "Review one customer at a time, run stage-aware scoring, and inspect the final decision with score, band, and explanation.",
    )
    render_single_case(artifacts)
    render_section_end()

    render_section_start(
        "batch-decision",
        "Batch Decision",
        "Upload a structured file, validate the schema, process multiple records, and export production-style batch decisions.",
    )
    render_batch_case(artifacts)
    render_section_end()

    render_section_start(
        "policy",
        "Policy",
        "View stage thresholds, understand decision zones, and review the policy layer that translates model scores into business actions.",
        border=False,
    )
    render_policy_view(artifacts)
    render_section_end()

    render_footer()


if __name__ == "__main__":
    main()
