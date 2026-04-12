# ============================================================
#  🍝 Spaghetti Code Detector — app.py
#  Angry Senior Dev AI powered by NASA JM1 defect dataset
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Spaghetti Code Detector",
    page_icon="🍝",
    layout="centered",
)

# ── Load model artifacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load and cache all model files so they're only read once."""
    model         = joblib.load("model.pkl")       # RandomForest
    lr_model      = joblib.load("lr_model.pkl")    # LogisticRegression
    scaler        = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, lr_model, scaler, feature_names

try:
    rf_model, lr_model, scaler, FEATURE_NAMES = load_artifacts()
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    missing_file = str(e)

# ── Angry Senior Dev response bank ────────────────────────────────────────
RESPONSES = {
    False: {
        "headline": "✅  LGTM. Approved.",
        "messages": [
            "I'm genuinely surprised. Don't touch it again. Ship it.",
            "Not bad. I mean, it's still your code, but not bad.",
            "Looks clean. I had my coffee so I'm being generous today.",
            "Fine. Merge it. But write a test next time.",
            "My blood pressure is fine for once. Approved.",
        ],
        "color": "success",
    },
    True: {
        "headline": "❌  REJECTED. Do Not Merge.",
        "messages": [
            "Dear god. Did you write this with your elbows? Do NOT push to main.",
            "This is pure spaghetti. I can smell the bugs through the screen.",
            "I've seen better code written by interns on their first day. Rewrite it.",
            "What is this? A function? A crime scene? Hard to tell.",
            "NO. Step away from the keyboard. Slowly.",
        ],
        "color": "error",
    },
}

# ── Meter helper ──────────────────────────────────────────────────────────
def spaghetti_meter(probability: float):
    """Render a colour-coded progress bar styled as a spaghetti-o-meter."""
    pct = int(probability * 100)
    color = (
        "#4CAF50" if pct < 30
        else "#FF9800" if pct < 60
        else "#f44336"
    )
    label = (
        "Clean pasta 🌿" if pct < 30
        else "Getting tangled 🍜" if pct < 60
        else "Full spaghetti 🍝"
    )
    st.markdown(
        f"""
        <div style="margin: 8px 0 4px; font-size: 13px; color: #888;">
            Spaghetti-o-meter
        </div>
        <div style="
            background: #e0e0e0;
            border-radius: 999px;
            height: 18px;
            overflow: hidden;
            margin-bottom: 4px;
        ">
            <div style="
                width: {pct}%;
                height: 100%;
                background: {color};
                border-radius: 999px;
                transition: width 0.5s ease;
            "></div>
        </div>
        <div style="font-size: 12px; color: #888; margin-bottom: 16px;">
            {pct}% defect probability — {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Feature importance chart ──────────────────────────────────────────────
def show_feature_importance(model, feature_names, top_n=10):
    """Horizontal bar chart of the top N feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    bars = ax.barh(top.index, top.values, color="#1976D2", edgecolor="none", height=0.6)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top {top_n} features driving prediction", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Build the input dict from sliders (fills unused features with median) ─
def build_input_array(slider_values: dict, feature_names: list) -> np.ndarray:
    """
    Map slider values onto the full feature vector.
    Features not covered by sliders get value 0 (scaler will handle it).
    """
    row = {f: 0.0 for f in feature_names}
    row.update(slider_values)
    return np.array([[row[f] for f in feature_names]])


# ═══════════════════════════════════════════════════════════════════════════
#  UI LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

st.title("🍝 Spaghetti Code Detector")
st.caption(
    "Submit your PR metrics. The Senior Dev will review it. Brace yourself."
)
st.divider()

# ── Guard: show error if pkl files are missing ────────────────────────────
if not models_loaded:
    st.error(
        f"Could not load model files: `{missing_file}`\n\n"
        "Make sure `model.pkl`, `lr_model.pkl`, `scaler.pkl`, and "
        "`feature_names.pkl` are in the same folder as `app.py`.\n\n"
        "Run the training notebook first to generate these files."
    )
    st.stop()

# ── Sidebar: settings ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Review Settings")

    chosen_model_name = st.radio(
        "Model",
        ["Random Forest", "Logistic Regression"],
        help="Random Forest generally has higher recall for bugs.",
    )
    chosen_model = rf_model if chosen_model_name == "Random Forest" else lr_model

    show_importance = st.checkbox("Show feature importance chart", value=True)
    show_raw        = st.checkbox("Show raw probability value",    value=False)
    senior_mood     = st.select_slider(
        "Senior Dev's mood today",
        options=["Hungover", "On deadline", "Post-lunch", "Just had coffee"],
        value="On deadline",
    )
    mood_prefix = {
        "Hungover":        "Ugh...",
        "On deadline":     "I don't have time for this.",
        "Post-lunch":      "Okay, let's see...",
        "Just had coffee": "Alright, bring it on.",
    }[senior_mood]

    st.divider()
    st.caption("NASA JM1 Software Defect Dataset")
    st.caption(f"Model: {chosen_model_name}")
    st.caption(f"Features: {len(FEATURE_NAMES)}")

# ── Main: input sliders ───────────────────────────────────────────────────
st.subheader("Pull Request Metrics")
st.caption(
    "Adjust the sliders to match your module's code metrics. "
    "These are the same metrics the model was trained on."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Size & Complexity**")
    loc  = st.slider("Lines of code (loc)",           10,  5000, 150, step=10)
    vg   = st.slider("Cyclomatic complexity v(g)",     1,   100,   8)
    evg  = st.slider("Essential complexity ev(g)",     1,    50,   3)
    ivg  = st.slider("Design complexity iv(g)",        1,    50,   4)

with col2:
    st.markdown("**Halstead Metrics**")
    n_val = st.slider("Halstead length N",            10, 10000, 400, step=10)
    v_val = st.slider("Halstead volume V",            10,  5000, 200, step=10)
    l_val = st.slider("Halstead level L",           0.0,   1.0, 0.1, step=0.01)
    b_val = st.slider("Halstead bugs estimate b",   0.0,   5.0, 0.1, step=0.01)

st.divider()

# ── Predict button ────────────────────────────────────────────────────────
if st.button("🔍  Submit for Code Review", use_container_width=True, type="primary"):

    slider_values = {
        "loc":   loc,
        "v(g)":  vg,
        "ev(g)": evg,
        "iv(g)": ivg,
        "n":     n_val,
        "v":     v_val,
        "l":     l_val,
        "b":     b_val,
    }

    input_array  = build_input_array(slider_values, FEATURE_NAMES)
    input_scaled = scaler.transform(input_array)

    prediction   = bool(chosen_model.predict(input_scaled)[0])
    probability  = float(chosen_model.predict_proba(input_scaled)[0][1])

    # Pick a random message from the response bank
    import random
    r        = RESPONSES[prediction]
    message  = random.choice(r["messages"])
    full_msg = f"{mood_prefix} {message}"

    st.subheader(r["headline"])
    spaghetti_meter(probability)

    if show_raw:
        st.caption(f"Raw defect probability: {probability:.4f}")

    # Display the Senior Dev's verdict
    if prediction:
        st.error(f"**Senior Dev says:** {full_msg}")
    else:
        st.success(f"**Senior Dev says:** {full_msg}")

    # Metric summary cards
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lines of code",    loc)
    m2.metric("Complexity v(g)",  vg)
    m3.metric("Halstead N",       n_val)
    m4.metric("Defect prob",      f"{probability:.1%}")

    # Feature importance chart
    if show_importance and chosen_model_name == "Random Forest":
        st.divider()
        st.subheader("Why did the model decide this?")
        show_feature_importance(chosen_model, FEATURE_NAMES)
        st.caption(
            "Bar length = how much each metric contributed to the model's "
            "overall decision across the training data. Higher = more influential."
        )

    # Advice section
    st.divider()
    st.subheader("What to fix")
    advice_given = False
    if vg > 20:
        st.warning(f"**v(g) = {vg}** — Cyclomatic complexity above 20 means too many execution paths. Break this function up.")
        advice_given = True
    if evg > 10:
        st.warning(f"**ev(g) = {evg}** — High essential complexity means the logic can't be simplified further. Redesign the algorithm.")
        advice_given = True
    if loc > 500:
        st.warning(f"**loc = {loc}** — Modules over 500 lines are hard to test and maintain. Extract smaller functions.")
        advice_given = True
    if b_val > 1.0:
        st.warning(f"**b = {b_val:.2f}** — Halstead bug estimate above 1 predicts at least one fault. Review your logic carefully.")
        advice_given = True
    if not advice_given:
        st.info("No specific red flags in the individual metrics. The model is reacting to the combination of values.")

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Trained on the NASA MDP JM1 software defect dataset · "
    "McCabe & Halstead metrics · RandomForest + LogisticRegression"
)
