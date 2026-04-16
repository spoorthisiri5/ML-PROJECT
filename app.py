# app.py — Spaghetti Code Detector v3
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import math
import random
import shap

st.set_page_config(
    page_title="Spaghetti Code Detector",
    page_icon="🍝",
    layout="centered"
)

# ── Load all artifacts ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf_model       = joblib.load("model.pkl")
    lr_model       = joblib.load("lr_model.pkl")
    gb_model       = joblib.load("gb_model.pkl")
    ensemble_model = joblib.load("ensemble_model.pkl")
    scaler         = joblib.load("scaler.pkl")
    features       = joblib.load("feature_names.pkl")
    medians        = joblib.load("feature_medians.pkl")
    return rf_model, lr_model, gb_model, ensemble_model, scaler, features, medians

rf_model, lr_model, gb_model, ensemble_model, \
    scaler, FEATURES, MEDIANS = load_artifacts()

# ── SHAP explainer (cached — slow to build) ────────────────
@st.cache_resource
def get_shap_explainer(_model):
    # TreeExplainer works for RF and GB
    # use a small background sample for speed
    return shap.TreeExplainer(_model)

# ── Halstead calculator ────────────────────────────────────
def calculate_halstead(n1, n2, N1, N2):
    n = n1 + n2
    N = N1 + N2
    if n == 0 or n1 == 0 or n2 == 0:
        return 0, 0, 0, 0
    V = N * math.log2(n)
    D = (n1 / 2) * (N2 / n2)
    L = 1 / D if D != 0 else 0
    b = V / 3000
    return round(N), round(V, 2), round(L, 4), round(b, 4)

# ── Build full input using MEDIANS for hidden features ──────
# This is the key fix for Problem 1
def build_input(slider_vals):
    """
    Fill all 21 features with dataset medians,
    then override with whatever the user set via sliders.
    This prevents hidden features dragging prediction to Clean.
    """
    full = {f: MEDIANS.get(f, 0) for f in FEATURES}
    full.update(slider_vals)  # override with user values
    return np.array([[full[f] for f in FEATURES]])

# ── Responses ──────────────────────────────────────────────
RESPONSES = {
    False: {
        "status": "✅ APPROVED — No Defects Predicted",
        "lines": [
            "LGTM. Ship it before I change my mind.",
            "This is... acceptable. Don't tell anyone I said that.",
            "Fine. It works. Go home.",
        ]
    },
    True: {
        "status": "❌ REJECTED — High Defect Probability",
        "lines": [
            "Did you write this with your elbows? Do NOT push to main.",
            "I've seen spaghetti with more structure. Rewrite all of it.",
            "This will crash in production. I guarantee it.",
        ]
    }
}

# ── Fix engine ─────────────────────────────────────────────
def generate_fixes(loc, vg, evg, ivg, N, V, L, b):
    fixes = []
    if loc > 500:
        fixes.append({
            "metric": "Lines of Code", "value": f"{loc} lines", "severity": "high",
            "problem": "Module is too large. No single function should exceed 300 lines.",
            "fix": [
                "Split into smaller modules — one responsibility per file.",
                "Extract repeated logic into helper functions.",
                f"Target: break into at least {loc // 150} separate functions."
            ]
        })
    if vg > 50:
        fixes.append({
            "metric": "Cyclomatic Complexity v(g)", "value": f"{vg} paths", "severity": "high",
            "problem": f"v(g)={vg} means {vg} independent paths. You need {vg} test cases to cover all branches — nobody writes that many.",
            "fix": [
                "Replace nested if-else chains with early returns (guard clauses).",
                "Use a dictionary instead of long if-elif chains.",
                f"Target: split into at least {vg // 8} smaller functions, each with v(g) ≤ 10."
            ]
        })
    elif vg > 15:
        fixes.append({
            "metric": "Cyclomatic Complexity v(g)", "value": f"{vg} paths", "severity": "medium",
            "problem": "Above the safe threshold of 10. Testing all paths is difficult.",
            "fix": [
                "Extract the most complex if-else block into its own function.",
                "Replace for-loops with embedded conditions using list comprehensions."
            ]
        })
    if evg > 20:
        fixes.append({
            "metric": "Essential Complexity ev(g)", "value": f"{evg}", "severity": "high",
            "problem": f"ev(g)={evg} means fundamentally unstructured code. Cannot be simplified without a full rewrite.",
            "fix": [
                "Eliminate all break statements inside nested loops.",
                "Remove continue statements — restructure loop conditions instead.",
                "ev(g)=1 is the goal. This module likely needs a complete rewrite."
            ]
        })
    if ivg > 30:
        fixes.append({
            "metric": "Design Complexity iv(g)", "value": f"{ivg}", "severity": "high",
            "problem": f"iv(g)={ivg} means tightly coupled to {ivg} other modules. Changing anything here will break other things.",
            "fix": [
                "Apply dependency injection — pass dependencies as parameters.",
                "Introduce an interface between this module and what it calls.",
                f"Target: iv(g) ≤ 10. Split into {ivg // 10} loosely coupled modules."
            ]
        })
    if V > 10000:
        fixes.append({
            "metric": "Halstead Volume (V)", "value": f"{V}", "severity": "high",
            "problem": f"Volume={V}. Halstead's theory says anything above 8,000 is almost guaranteed to have bugs.",
            "fix": [
                "Reduce unique operators — replace complex expressions with named variables.",
                "Split the module — high volume almost always means it's doing two jobs."
            ]
        })
    if b > 2:
        fixes.append({
            "metric": "Halstead Bug Estimate (b)", "value": f"{b:.2f} bugs", "severity": "high",
            "problem": f"Halstead's formula estimates {b:.1f} bugs purely from size and complexity.",
            "fix": [
                "Write unit tests for every function immediately.",
                "Run pylint or flake8 on this file before committing.",
                f"Bug estimate drops below 1 when loc < 200 and v(g) < 10."
            ]
        })
    return fixes

# ══════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════
st.title("🍝 Spaghetti Code Detector")
st.caption("Submit your PR metrics. The Senior Dev will review it.")
st.divider()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Model",
        ["Voting Ensemble (Best)", "Random Forest",
         "Gradient Boosting", "Logistic Regression"],
        help="Voting Ensemble combines all three models for the most reliable prediction"
    )
    model_map = {
        "Random Forest"          : rf_model,
        "Logistic Regression"    : lr_model,
        "Gradient Boosting"      : gb_model,
        "Voting Ensemble (Best)" : ensemble_model
    }
    active_model = model_map[model_choice]

    st.divider()

    # ── Threshold slider — key fix for Problem 1 ──────────
    st.subheader("Decision Threshold")
    threshold = st.slider(
        "Defect threshold",
        min_value=0.10, max_value=0.60,
        value=0.30, step=0.05,
        help="Lower = more sensitive to defects. "
             "Default 0.30 is recommended for defect detection "
             "because missing a bug is worse than a false alarm."
    )
    if threshold <= 0.25:
        st.warning("Very sensitive — many false alarms expected.")
    elif threshold >= 0.50:
        st.warning("Standard threshold — may miss real defects.")
    else:
        st.success(f"Recommended zone for defect detection.")

    st.divider()
    st.caption("Dataset: NASA JM1 Software Defect")
    st.caption("~10,000 C-language modules | ~19% defect rate")
    st.caption("Hidden features filled with dataset medians")

# ── Step 1: McCabe sliders ─────────────────────────────────
st.subheader("Step 1 — Code Structure Metrics")
col1, col2 = st.columns(2)
with col1:
    loc = st.slider("Lines of code (loc)",          1, 5000, 80)
    vg  = st.slider("Cyclomatic complexity v(g)",   1, 100,  5)
with col2:
    evg = st.slider("Essential complexity ev(g)",   1, 60,   2)
    ivg = st.slider("Design complexity iv(g)",      1, 60,   3)

# ── Step 2: Halstead calculator ────────────────────────────
st.divider()
st.subheader("Step 2 — Halstead Metrics")
st.caption("Enter raw counts — values are calculated automatically.")

with st.expander("What are operators and operands? ↓"):
    st.markdown("""
**Operators:** `if`, `for`, `while`, `=`, `+`, `-`, `*`, `/`, `return`, `def`
**Operands:** variable names, numbers, strings

**Quick estimate:**
- 50-line function → n1≈10, n2≈15, N1≈80, N2≈100
- 200-line function → n1≈15, n2≈40, N1≈400, N2≈500
    """)

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Distinct counts**")
    n1 = st.number_input("n1 — distinct operators", 1, 200, 10)
    n2 = st.number_input("n2 — distinct operands",  1, 500, 15)
with c4:
    st.markdown("**Total counts**")
    N1 = st.number_input("N1 — total operators",    1, 20000, 80)
    N2 = st.number_input("N2 — total operands",     1, 20000, 100)

N, V, L, b = calculate_halstead(n1, n2, N1, N2)
hc1, hc2, hc3, hc4 = st.columns(4)
hc1.metric("N (length)", N)
hc2.metric("V (volume)", V)
hc3.metric("L (level)",  L)
hc4.metric("b (bugs)",   b)

# ── Submit ─────────────────────────────────────────────────
st.divider()
if st.button("🔍 Submit for Code Review", use_container_width=True):

    slider_vals = {
        "loc": loc, "v(g)": vg, "ev(g)": evg, "iv(g)": ivg,
        "n": N, "b": b
    }

    # ── FIX 1: use medians for hidden features ─────────────
    input_array = build_input(slider_vals)
    scaled = scaler.transform(input_array)

    # Raw probability
    proba = float(active_model.predict_proba(scaled)[0][1])

    # ── FIX 2: use custom threshold instead of 0.5 ─────────
    pred = proba >= threshold

    response = RESPONSES[pred]
    st.divider()
    st.subheader(response["status"])

    # Show probability with threshold marker
    st.metric("Defect probability", f"{proba:.1%}",
              delta=f"Threshold: {threshold:.0%}",
              delta_color="off")
    st.progress(proba)

    if proba >= threshold:
        st.error(f'**Senior Dev says:** "{random.choice(response["lines"])}"')
    else:
        st.success(f'**Senior Dev says:** "{random.choice(response["lines"])}"')

    # ── Fixes ──────────────────────────────────────────────
    fixes = generate_fixes(loc, vg, evg, ivg, N, V, L, b)
    if fixes:
        st.divider()
        st.subheader("🔧 What is wrong and how to fix it")
        for fix in fixes:
            icon = "🔴" if fix["severity"] == "high" else "🟡"
            with st.expander(
                f"{icon}  {fix['metric']} = {fix['value']}  "
                f"({'Critical' if fix['severity'] == 'high' else 'Warning'})"
            ):
                if fix["severity"] == "high":
                    st.error(f"**Problem:** {fix['problem']}")
                else:
                    st.warning(f"**Problem:** {fix['problem']}")
                st.markdown("**How to fix it:**")
                for i, step in enumerate(fix["fix"], 1):
                    st.markdown(f"{i}. {step}")
    elif not pred:
        st.success("✅ No significant issues. Keep functions small and v(g) under 10.")

    # ── FIX 3: SHAP values instead of global importance ────
    # This chart CHANGES per prediction — shows why THIS input got THIS verdict
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        with st.expander("Why did the model decide THIS? (SHAP) ↓"):
            st.caption(
                "SHAP values show the contribution of EACH feature "
                "to THIS specific prediction. "
                "Red = pushed toward Defective. Blue = pushed toward Clean. "
                "This chart changes as you move the sliders."
            )
            try:
                tree_model = rf_model if model_choice == "Random Forest" \
                             else gb_model
                explainer  = get_shap_explainer(tree_model)
                shap_vals  = explainer.shap_values(scaled)

                # For binary classification, shap_values returns [class0, class1]
                # We want class 1 (defective)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0]

                feature_names = FEATURES
                shap_df = pd.DataFrame({
                    "Feature"   : feature_names,
                    "SHAP value": sv,
                    "Input value": [
                        round(build_input(slider_vals)[0][i], 2)
                        for i in range(len(feature_names))
                    ]
                }).sort_values("SHAP value", key=abs, ascending=False).head(12)

                colors = [
                    "#f44336" if v > 0 else "#2196F3"
                    for v in shap_df["SHAP value"]
                ]
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.barh(
                    shap_df["Feature"],
                    shap_df["SHAP value"],
                    color=colors,
                    edgecolor="none"
                )
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP value  →  Red pushes toward Defective, Blue toward Clean")
                ax.set_title(
                    f"Per-prediction explanation  |  Defect prob: {proba:.1%}"
                )
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig)

                st.dataframe(
                    shap_df[["Feature", "Input value", "SHAP value"]].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.warning(f"SHAP explanation unavailable for this model: {e}")
    # ── Model comparison ───────────────────────────────────
    with st.expander("Compare all models ↓"):
        rows = []
        for name, m in model_map.items():
            p = float(m.predict_proba(scaled)[0][1])
            rows.append({
                "Model"             : name,
                "Defect probability": f"{p:.1%}",
                f"Verdict (t={threshold:.2f})": "Defective" if p >= threshold else "Clean"
            })
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True



   
