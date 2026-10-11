# FraudLens

End-to-end fraud-risk decision system: behavioral transaction simulator → PostgreSQL →
point-in-time features (with leakage tests) → Rules / Logistic Regression / XGBoost under
temporal validation → probability calibration → cost-based decision policy with analyst
capacity limits → Isolation Forest unseen-fraud experiment → SHAP → FastAPI + Streamlit →
monitoring.

**Status: work in progress.** Parts are being built one at a time per `prompts/`; see
`docs/progress/`. Synthetic data only — this is a portfolio/research system, not a
production banking system.