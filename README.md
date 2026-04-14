# Home Credit Default Risk - Proxy Modeling & Policy Pipeline

This repository contains an end-to-end machine learning and policy pipeline designed to predict credit default risk. Instead of a monolithic model, this project implements a **Stage-Aware Routing System** that segments customers based on their internal history and delinquency signals, applying tailored models and business rules to each segment.

## 🌐 Streamlit Web Application

The offline machine learning pipeline has been fully integrated and deployed as an interactive web application using Streamlit. This translates the complex, multi-stage models into a seamless tool for loan officers and risk analysts.

**Live Demo:** [Credit Scoring System App](https://credit-scoring-sys.streamlit.app/)

### 🛠️ Key App Features
Powered by the product pack generated in Notebook 6, the web app acts as a real-time inference engine and policy simulator:

* **Smart Stage Routing (A/B/C):** The app evaluates basic customer context (like `tenure_months` and `max_dpd`) to automatically assign the applicant to the correct risk stage (New-to-Bank, Thin-File, or Mature Risk).
* **Dynamic Input Schema:** Driven by the `streamlit_input_schema.csv` artifact, the UI dynamically adapts. It only asks the user to input the specific data points required by the model for that exact customer stage, keeping the interface clean and efficient.
* **Interpretable Risk Scoring:** It translates raw machine learning probabilities (`score_proba`) into an intuitive **300–900 Risk Score**, where higher numbers indicate higher risk.
* **Actionable Business Decisions:** The app doesn't just output a score. It applies the threshold logic from `policy_rules.json` to immediately recommend a **Decision Zone** (e.g., *Starter Loan Small*, *Approve Preferred*, *High-Risk Review Priority*). 
* **Loan Structuring & Insights:** Alongside the decision, it provides recommended credit limits, suggested loan tenors (months), and top **Reason Codes** (e.g., `low_income_signal`, `positive_dpd_signal`) to explain *why* the decision was made.

---

## Architecture & Stages

The pipeline dynamically routes applicants into one of three operational stages:

* **Stage A (Application / New-to-Bank):** Customers with no internal history. The model output is used for starter-offer recommendations rather than strict underwriting.
* **Stage B (Behavior / Thin-file):** Customers with internal history but no severe collection signals. This acts as the **Champion Decision Model** for standard approvals.
* **Stage C (Collection / Mature Risk):** Customers with active delinquency or collection signals. This model is used as a guardrail for intensive review and collection prioritization.

---

## 📓 Pipeline Overview

The workflow is broken down into a series of sequential Jupyter Notebooks:

### 1. EDA Part 1: Inventory & Proxy Mapping (`homecredit-eda-part1.ipynb`)
* Scans raw CSV headers to build a lightweight data dictionary without loading massive datasets into memory.
* Maps raw variables to core business proxy concepts (e.g., `capacity_to_repay`, `delinquency_risk`).
* **Key Outputs:** `feature_catalog.parquet`, `proxy_candidate_catalog.parquet`.

### 2. EDA Part 2: Stage Definition & Shortlisting (`homecredit-eda-part2.ipynb`)
* Defines business logic for Stages A, B, and C using internal anchor dates.
* Validates collection signals and runs a lightweight profiler to filter out high-leakage or highly-missing features.
* **Key Outputs:** `stage_assignment_proxy.parquet`, `feature_plan.parquet`.

### 3. Feature Mart Construction (`notebook-3-latest.ipynb`)
* Utilizes **Polars** for high-performance data aggregation.
* Rolls up raw event-level data to the `case_id` level.
* Splits the aggregated data into stage-specific feature blocks ready for training.
* **Key Outputs:** `proxy_feature_mart.parquet`, `proxy_features_A.parquet`, `proxy_features_B.parquet`, `proxy_features_C.parquet`.

### 4. Model Training (Notebook 4)
* Trains isolated models for Stages A, B, and C (implied step based on pipeline architecture).
* **Key Outputs:** `unified_stage_inference_pack.joblib`, validation metrics.

### 5. Benchmark & Registry (`notebook-5.ipynb`)
* Evaluates model performance (AUC, KS, Brier score).
* Establishes threshold cutoffs (Review, Reject, Small Offer) tailored to the risk profile of each stage.
* **Key Outputs:** `model_registry_manifest.json`, `model_registry_pack.joblib`.

### 6. Product Pack & Policy Simulator (`notebook-6.ipynb`)
* Packages the models and business rules into a deployable artifact.
* Simulates the workload impact (Review vs. Auto-Approve rates) of the chosen thresholds.
* Includes operational overrides to prevent manual review bottlenecks (especially for Stage A).
* **Key Outputs:** `unified_inference_service.joblib`, `policy_rules.json`, `streamlit_input_schema.csv`.

---

## 💻 Running the App Locally

If you have cloned the repository and want to run the web app locally on your machine:

1. Ensure you have the `requirements.txt` installed (`pip install -r requirements.txt`).
2. Verify that the Notebook 6 output artifacts (specifically `unified_inference_service.joblib`, `streamlit_input_schema.csv`, and `policy_rules.json`) are in your app's root directory.
3. Run the following command in your terminal:
   ```bash
   streamlit run app.py
