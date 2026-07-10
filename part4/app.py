"""
Streamlit frontend for the Capstone Project (Part 4).

Two pages:
  1. Home           - project overview + links to EDA/ML plots from Parts 1-3
  2. Predict & Explain - enter diamond features, get a price-class prediction
                         from best_model.pkl, then an LLM-generated explanation
                         (reusing the exact guardrail + prompt logic from part_4.py)

Run with:
    streamlit run app.py
"""

import os
import glob
import json

import joblib
import pandas as pd
import streamlit as st

# --- reuse everything from your existing Part 4 pipeline, no duplication ---
from part_4 import (
    encode_record,
    guarded_call_llm,
    build_user_prompt,
    validate_explanation,
    has_pii,
    SYSTEM_PROMPT,
    CUT_ORDER,
    CLARITY_ORDER,
    COLOR_LEVELS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best_model.pkl")

st.set_page_config(page_title="Diamond Price Intelligence", page_icon="💎", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def find_plots():
    """Collect plot images from part1/plots and part2/plots (sibling folders)."""
    plot_dirs = [
        os.path.join(HERE, "..", "part1", "plots"),
        os.path.join(HERE, "..", "part2", "plots"),
    ]
    found = []
    for d in plot_dirs:
        if os.path.isdir(d):
            found.extend(sorted(glob.glob(os.path.join(d, "*.png"))))
    return found


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔮 Predict & Explain"])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Capstone Project — EDA (Part 1) → ML Model (Part 2/3) → "
    "LLM-Powered Explanation System (Part 4)"
)

# ----------------------------------------------------------------------------
# PAGE 1: Home
# ----------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("💎 Diamond Price Intelligence")
    st.markdown(
        """
This app demonstrates the full pipeline built across the capstone:

1. **EDA** — cleaning, null handling, outlier detection, correlation analysis
2. **ML Models** — Linear/Ridge/Logistic Regression, Decision Trees, Random Forest,
   Gradient Boosting, tuned via `GridSearchCV`
3. **LLM Integration** — an explainability layer that turns a raw model prediction
   into a human-readable explanation, with a PII guardrail and JSON-schema validation
4. **This frontend** — a usable interface on top of all of that
        """
    )

    st.subheader("Exploratory plots")
    plots = find_plots()
    if plots:
        cols = st.columns(3)
        for i, p in enumerate(plots):
            with cols[i % 3]:
                st.image(p, caption=os.path.basename(p), use_container_width=True)
    else:
        st.info(
            "No plot images found. Make sure `part1/plots` and `part2/plots` "
            "exist relative to this app, or run this app from the `part4/` folder."
        )

# ----------------------------------------------------------------------------
# PAGE 2: Predict & Explain
# ----------------------------------------------------------------------------
else:
    st.title("🔮 Predict & Explain a Diamond's Price Class")
    st.caption(
        "Model predicts whether price is above or below the training-set median. "
        "The LLM then explains the prediction in plain language."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(f"best_model.pkl not found at {MODEL_PATH}. Run part_3.py first.")
        st.stop()

    model = load_model()

    with st.form("diamond_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            carat = st.number_input("Carat", min_value=0.01, max_value=6.0, value=1.0, step=0.01)
            cut = st.selectbox("Cut", list(CUT_ORDER.keys()), index=4)
        with c2:
            color = st.selectbox("Color", COLOR_LEVELS, index=3)
            clarity = st.selectbox("Clarity", list(CLARITY_ORDER.keys()), index=4)
        with c3:
            table = st.number_input("Table", min_value=40.0, max_value=80.0, value=57.0, step=0.5)
            x = st.number_input("x (mm)", min_value=0.1, max_value=15.0, value=6.5, step=0.1)

        c4, c5 = st.columns(2)
        with c4:
            y = st.number_input("y (mm)", min_value=0.1, max_value=15.0, value=6.5, step=0.1)
        with c5:
            z = st.number_input("z (mm)", min_value=0.1, max_value=15.0, value=4.0, step=0.1)

        temperature = st.slider("LLM temperature", 0.0, 1.0, 0.0, 0.1)
        analyst_note = st.text_input(
            "Optional analyst note (checked for PII before being logged)", value=""
        )

        submitted = st.form_submit_button("Predict & Explain")

    if submitted:
        features = {
            "carat": carat, "cut": cut, "color": color, "clarity": clarity,
            "table": table, "x": x, "y": y, "z": z,
        }

        if analyst_note and has_pii(analyst_note):
            st.warning("Your analyst note looks like it contains PII (email/phone) and was not logged.")

        encoded = encode_record(features)
        pred_class = int(model.predict(encoded)[0])
        pred_prob = float(model.predict_proba(encoded)[0, pred_class])
        label = "Above median price" if pred_class == 1 else "Below median price"

        m1, m2 = st.columns(2)
        m1.metric("Prediction", label)
        m2.metric("Confidence", f"{pred_prob:.1%}")

        with st.spinner("Asking the LLM to explain this prediction..."):
            user_prompt = build_user_prompt(features, pred_class, pred_prob)
            raw = guarded_call_llm(SYSTEM_PROMPT, user_prompt, temperature=temperature, max_tokens=300)
            parsed, status = validate_explanation(raw)

        st.subheader("LLM Explanation")
        if status != "pass":
            st.error(f"Explanation failed validation ({status}). Showing raw response instead.")
            st.code(raw)
        else:
            st.success(f"Confidence level (LLM's own read): **{parsed['confidence_level']}**")
            st.write(f"**Top reason:** {parsed['top_reason']}")
            st.write(f"**Second reason:** {parsed['second_reason']}")
            st.write(f"**Recommended next step:** {parsed['next_step']}")

        with st.expander("Raw JSON response"):
            st.code(json.dumps(parsed, indent=2))