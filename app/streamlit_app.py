"""FraudLens analyst console — Streamlit entry point (Part 14).

Run with:  streamlit run app/streamlit_app.py
Offline scores come from the Part 13 API, so start it first:
    python scripts/run_api.py --profile dev
This file contains no feature, scoring or cost logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
from components import api_client, context  # noqa: E402

st.set_page_config(page_title="FraudLens", layout="wide")
context.sidebar()

art = context.artifacts()
st.title("FraudLens analyst console")
st.caption(context.BANNER)

windows = art.manifest["windows"]
cols = st.columns(4)
cols[0].metric("Profile", art.manifest["profile"])
cols[1].metric("Model", art.manifest["model_version"])
cols[2].metric("Policy", art.manifest["policy_version"] or "none")
cols[3].metric("Sample rows", f"{art.manifest['n_scored_sample']:,}")

st.markdown(
    f"""
**Pages**

1. **Transaction Analyzer** — score one transaction and read its explanation.
2. **Review Queue** — move the cost/capacity sliders and watch the queue change.
3. **Model Performance** — validation curves, calibration, drift and the frozen
   test-window results.

All precomputed views cover the **validation window only**
(`{windows['t_train_end']}` → `{windows['t_val_end']}`); the test window stays
frozen and its numbers are read from `results/`.

Offline panels read `{art.directory}`. Live scoring calls the API; start it with
`{api_client.API_COMMAND}`.
"""
)
