
# credit_scoring_app

Stage-aware credit scoring prototype wired to your **real artifacts** from notebook 4–6.

## What changed in this version

This repo is no longer a generic fallback skeleton. It now reads the real exported files:

- `unified_stage_inference_pack.joblib`
- `unified_inference_service.joblib`
- `model_registry_manifest.json`
- `product_pack_manifest.json`
- `policy_rules.json`
- `streamlit_input_schema.csv`

The scoring flow is aligned to the structures found inside your artifacts:

- stage models are loaded from `stage_models`
- stage routing uses `tenure_months` and `max_dpd`
- scoring uses each stage pipeline's `predict_proba`
- `score_300_900 = 300 + 600 * risk_proba`
- policy actions are mapped from the real thresholds in `policy_rules.json`

## Business framing

- **Stage A**: starter-offer / onboarding recommendation
- **Stage B**: champion decision model
- **Stage C**: collection / monitoring guardrail

Higher `score_300_900` means **higher risk**.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important note

The uploaded artifacts only contained the final scoring packs and policy/json files.  
This app is therefore wired directly to those exported packs and does **not** need the older per-stage joblib files to run.

## Repo structure

```text
credit_scoring_app/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── artifacts/
├── src/
└── sample_data/
```


## Version note

The exported sklearn pipelines were serialized from **scikit-learn 1.6.1**. The repo pins that version to avoid model-loading or transform errors.


## Input note

The current UI now exposes the stage-router fields (`tenure_months`, `max_dpd`) and also lets you supply stage-specific model inputs. If some stage features are left blank, the exported sklearn pipeline still runs and imputes those missing values internally.
