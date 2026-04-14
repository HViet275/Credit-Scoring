# Home Credit Default Risk - Proxy Modeling Pipeline

This repository contains an end-to-end machine learning and policy pipeline designed to predict credit default risk. Instead of a monolithic model, this project implements a **Stage-Aware Routing System** that segments customers based on their internal history and delinquency signals, applying tailored models and business rules to each segment.

## 🏗️ Architecture & Stages

The pipeline dynamically routes applicants into one of three operational stages:

* **Stage A (Application / New-to-Bank):** Customers with no internal history. The model output is used for starter-offer recommendations rather than strict underwriting.
* **Stage B (Behavior / Thin-file):** Customers with internal history but no severe collection signals. This acts as the **Champion Decision Model** for standard approvals.
* **Stage C (Collection / Mature Risk):** Customers with active delinquency or collection signals. This model is used as a guardrail for intensive review and collection prioritization.

## 📓 Pipeline Overview

The workflow is broken down into a series of sequential Jupyter Notebooks:

### 1. EDA Part 1: Inventory & Proxy Mapping 
* Scans raw CSV headers to build a lightweight data dictionary.
* Maps raw variables to core business proxy concepts (e.g., `capacity_to_repay`, `delinquency_risk`).
* **Key Outputs:** `feature_catalog.parquet`, `proxy_candidate_catalog.parquet`

### 2. EDA Part 2: Stage Definition & Shortlisting
* Defines business logic for Stages A, B, and C using internal anchor dates.
* Validates collection signals and runs a lightweight profiler to filter out high-leakage or highly-missing features.
* **Key Outputs:** `stage_assignment_proxy.parquet`, `feature_plan.parquet`

### 3. Feature Mart Construction 
* Utilizes **Polars** for high-performance data aggregation.
* Rolls up raw event-level data to the `case_id` level.
* Splits the aggregated data into stage-specific feature blocks ready for training.
* **Key Outputs:** `proxy_feature_mart.parquet`, `proxy_features_A/B/C.parquet`

### 4. Model Training 
* Trains isolated models for Stages A, B, and C.
* **Key Outputs:** `unified_stage_inference_pack.joblib`, validation metrics.

### 5. Benchmark & Registry
* Evaluates model performance (AUC, KS, Brier score).
* Establishes threshold cutoffs (Review, Reject, Small Offer) tailored to the risk profile of each stage.
* **Key Outputs:** `model_registry_manifest.json`, `model_registry_pack.joblib`

### 6. Product Pack & Policy Simulator
* Packages the models and business rules into a deployable artifact.
* Simulates the workload impact (Review vs. Auto-Approve rates) of the chosen thresholds.
* Includes operational overrides to prevent manual review bottlenecks.
* **Key Outputs:** `unified_inference_service.joblib`, `policy_rules.json`, `streamlit_input_schema.csv`

## 🚀 How to Use the Inference Service

The final artifact, `unified_inference_service.joblib`, can be loaded directly into a Streamlit app or FastAPI service. 

It handles:
1. Identifying the correct Stage (A/B/C) based on payload inputs (e.g., `tenure_months`, `max_dpd`).
2. Scoring the payload using the respective stage model.
3. Translating the probability into a standard `300-900` risk score (Higher = Riskier).
4. Emitting a definitive `policy_action` (e.g., `approve_preferred`, `reject_or_intensive_review`) based on `policy_rules.json`.

## 📌 Business Glossary
Refer to `business_glossary.md` (generated in Notebook 6) for exact definitions of decision zones, scoring statuses, and reason codes used in the final inference outputs.
