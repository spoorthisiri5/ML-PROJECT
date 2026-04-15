# app.py — Spaghetti Code Detector (v2 — Teacher's Edition)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import math
import random

st.set_page_config(
    page_title="Spaghetti Code Detector",
    page_icon="🍝",
    layout="centered"
)

# ── Load artifacts ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("model.pkl")
    lr_model = joblib.load("lr_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    features = joblib.load("feature_names.pkl")
    return model, lr_model, scaler, features

rf_model, lr_model, scaler, FEATURES = load_artifacts()

# ── Halstead calculator ────────────────────────────────────
def calculate_halstead(n1, n2, N1, N2):
    """
    n1 = number of distinct operators
    n2 = number of distinct operands
    N1 = total operators
    N2 = total operands
    """
    n  = n1 + n2          # vocabulary
    N  = N1 + N2          # length
    if n == 0 or n1 == 0 or n2 == 0:
        return 0, 0, 0, 0
    V  = N * math.log2(n)                    # volume
    D  = (n1 / 2) * (N2 / n2)               # difficulty
    L  = 1 / D if D != 0 else 0             # level
    b  = V / 3000                            # bugs estimate
    return round(N), round(V, 2), round(L, 4), round(b, 4)

# ── Fix engine ─────────────────────────────────────────────
def generate_fixes(loc, vg, evg, ivg, N, V, L, b):
    """
    Returns a list of specific, actionable fixes
    based on which metrics are in the danger zone.
    """
    fixes = []

    # LOC fixes
    if loc > 500:
        fixes.append({
            "metric"  : "Lines of Code",
            "value"   : f"{loc} lines",
            "severity": "high",
            "problem" : "This module is too large. No single function should exceed 200–300 lines.",
            "fix"     : [
                "Split this file into smaller modules, one responsibility per file.",
                "Extract repeated logic into helper functions.",
                "Apply the Single Responsibility Principle — one function, one job.",
                f"Target: break this into at least {loc // 150} separate functions."
            ]
        })
    elif loc > 200:
        fixes.append({
            "metric"  : "Lines of Code",
            "value"   : f"{loc} lines",
            "severity": "medium",
            "problem" : "Module is getting large. Consider splitting soon.",
            "fix"     : [
                "Look for sections of code that could be extracted into a separate function.",
                "Any block of code with a comment above it is a candidate for extraction."
            ]
        })

    # Cyclomatic complexity fixes
    if vg > 50:
        fixes.append({
            "metric"  : "Cyclomatic Complexity v(g)",
            "value"   : f"{vg} paths",
            "severity": "high",
            "problem" : f"v(g) = {vg} means there are {vg} independent paths through this code. You would need {vg} test cases just to cover every branch — nobody writes that many.",
            "fix"     : [
                "Replace deeply nested if-else chains with early returns (guard clauses).",
                "Move complex conditions into clearly named boolean variables.",
                "Use a dictionary/map instead of long if-elif-elif chains.",
                "Each function should have v(g) ≤ 10. Aim to split this into at least "
                f"{vg // 8} smaller functions.",
                "Example: instead of 'if x and y and z and w', create "
                "'is_valid = x and y' and 'is_ready = z and w'."
            ]
        })
    elif vg > 15:
        fixes.append({
            "metric"  : "Cyclomatic Complexity v(g)",
            "value"   : f"{vg} paths",
            "severity": "medium",
            "problem" : f"v(g) = {vg} is above the safe threshold of 10. Testing all paths is difficult.",
            "fix"     : [
                "Extract the most complex if-else block into its own function.",
                "Replace any for-loop with embedded conditions with list comprehensions or filter().",
                "Consider a state machine if this code handles many states/modes."
            ]
        })

    # Essential complexity fixes
    if evg > 20:
        fixes.append({
            "metric"  : "Essential Complexity ev(g)",
            "value"   : f"{evg}",
            "severity": "high",
            "problem" : f"ev(g) = {evg} means this code is fundamentally unstructured — it cannot be simplified without a full rewrite. This is the classic sign of spaghetti code.",
            "fix"     : [
                "Eliminate all break statements inside nested loops — use a function with a return instead.",
                "Remove any continue statements — restructure the loop condition instead.",
                "Never use flags (done = True / found = False) to control loop flow — use while conditions.",
                "Ensure every if block has a matching else or uses an early return.",
                "ev(g) = 1 is the goal. This module likely needs to be completely rewritten."
            ]
        })
    elif evg > 5:
        fixes.append({
            "metric"  : "Essential Complexity ev(g)",
            "value"   : f"{evg}",
            "severity": "medium",
            "problem" : "Some unstructured control flow detected — breaks, continues, or flag variables.",
            "fix"     : [
                "Find every 'break' inside a loop and replace with a function that returns early.",
                "Replace boolean flag variables with cleaner loop conditions."
            ]
        })

    # Design complexity fixes
    if ivg > 30:
        fixes.append({
            "metric"  : "Design Complexity iv(g)",
            "value"   : f"{ivg}",
            "severity": "high",
            "problem" : f"iv(g) = {ivg} means this module is tightly coupled to {ivg} other modules. Changing anything here will break things elsewhere — this is the definition of 'fragile' code.",
            "fix"     : [
                "Apply dependency injection — pass dependencies as parameters instead of importing them directly.",
                "Introduce an interface or abstract class between this module and what it calls.",
                "Move shared logic into a utility module that both can call independently.",
                "Draw a dependency diagram — any module that touches more than 5 others needs to be split.",
                "Target: iv(g) ≤ 10. Consider breaking this into "
                f"{ivg // 10} separate, loosely coupled modules."
            ]
        })
    elif ivg > 10:
        fixes.append({
            "metric"  : "Design Complexity iv(g)",
            "value"   : f"{ivg}",
            "severity": "medium",
            "problem" : "This module calls too many other modules. It is becoming a 'god function'.",
            "fix"     : [
                "Identify which external calls could be moved to a separate coordinator function.",
                "Group related external calls together and extract them into a helper."
            ]
        })

    # Halstead volume fixes
    if V > 10000:
        fixes.append({
            "metric"  : "Halstead Volume (V)",
            "value"   : f"{V}",
            "severity": "high",
            "problem" : f"Volume = {V} means an enormous amount of logic is packed into this module. Halstead's theory says anything above 8,000 is almost guaranteed to have bugs.",
            "fix"     : [
                "Reduce the number of unique operators — replace complex expressions with named variables.",
                "Eliminate redundant variable names — reuse variables where semantically appropriate.",
                "Split the module in half — high volume almost always means the module is doing two jobs."
            ]
        })

    # Halstead level fixes
    if L < 0.05:
        fixes.append({
            "metric"  : "Halstead Level (L)",
            "value"   : f"{L}",
            "severity": "high",
            "problem" : f"Level = {L} (very close to 0) means this code is extremely difficult to write correctly. Halstead's theory says errors are nearly inevitable at this complexity level.",
            "fix"     : [
                "Simplify operator usage — fewer chained operations per line.",
                "Break complex one-liners into multiple named steps.",
                "Target L > 0.3 — this usually means reducing both volume and difficulty simultaneously."
            ]
        })

    # Bugs estimate
    if b > 2:
        fixes.append({
            "metric"  : "Halstead Bug Estimate (b)",
            "value"   : f"{b:.2f} estimated bugs",
            "severity": "high",
            "problem" : f"Halstead's formula estimates {b:.1f} bugs exist in this module purely based on its size and complexity — before even running it.",
            "fix"     : [
                "Write unit tests for every function immediately — don't ship without them.",
                "Do a line-by-line code review with a senior developer.",
                "Run a static analyser (pylint, flake8) on this file before committing.",
                f"The estimated bug count drops to under 1 when you reduce loc to under 200 "
                "and v(g) to under 10."
            ]
        })

    return fixes

# ── Senior Dev responses ───────────────────────────────────
RESPONSES = {
    False: {
        "status": "✅ APPROVED — No Defects Predicted",
        "lines": [
            "LGTM. Ship it before I change my mind.",
            "This is... acceptable. Don't tell anyone I said that.",
            "Fine. It works. Go home.",
            "I've reviewed worse. Barely. Ship it."
        ]
    },
    True: {
        "status": "❌ REJECTED — High Defect Probability",
        "lines": [
            "Did you write this with your elbows? Do NOT push to main.",
            "I've seen spaghetti with more structure. Rewrite. All of it.",
            "This will crash in production. I guarantee it.",
            "My eyes are bleeding. Fix every issue below before you even "
            "think about opening a pull request."
        ]
    }
}

# ══════════════════════════════════════════════════════════
#   UI STARTS HERE
# ══════════════════════════════════════════════════════════

st.title("🍝 Spaghetti Code Detector")
st.caption("Submit your PR metrics. The Senior Dev will review it.")
st.divider()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Model",
        ["Random Forest", "Logistic Regression"]
    )
    active_model = rf_model if model_choice == "Random Forest" \
                  else lr_model
    st.divider()
    st.caption("Dataset: NASA JM1 Software Defect")
    st.caption("~10,000 C-language modules")
    st.caption("~19% defect rate (class-imbalanced)")

# ══════════════════════════════════════════════════════════
#   SECTION 1 — Basic Metrics
# ══════════════════════════════════════════════════════════
st.subheader("Step 1 — Enter your code metrics")
st.caption("These describe the structure and size of your code module.")

col1, col2 = st.columns(2)
with col1:
    loc = st.slider("Lines of code (loc)",              1, 5000, 80)
    vg  = st.slider("Cyclomatic complexity v(g)",       1, 100,  5)
with col2:
    evg = st.slider("Essential complexity ev(g)",       1, 60,   2)
    ivg = st.slider("Design complexity iv(g)",          1, 60,   3)

# ══════════════════════════════════════════════════════════
#   SECTION 2 — Halstead Calculator
# ══════════════════════════════════════════════════════════
st.divider()
st.subheader("Step 2 — Calculate Halstead metrics")
st.caption(
    "Enter the raw counts from your code — "
    "the Halstead values will be calculated automatically."
)

with st.expander("What are operators and operands? ↓"):
    st.markdown("""
**Operators** are the action symbols in your code:
`if`, `for`, `while`, `=`, `+`, `-`, `*`, `/`, `and`, `or`, `not`, `return`, `def`, `(`, `)`, `,`

**Operands** are the data in your code:
variable names (`price`, `user`, `total`), numbers (`0`, `42`, `3.14`), strings (`"hello"`)

**Distinct** means unique — count each one only once for n1/n2.
**Total** means every occurrence — count every time it appears for N1/N2.

**Quick estimate if you don't want to count manually:**
- For a 50-line function: n1≈10, n2≈15, N1≈80, N2≈100
- For a 200-line function: n1≈15, n2≈40, N1≈400, N2≈500
- For a 1000-line file: n1≈20, n2≈150, N1≈2000, N2≈2500
    """)

col3, col4 = st.columns(2)
with col3:
    st.markdown("**Distinct counts (unique only)**")
    n1 = st.number_input(
        "n1 — distinct operators",
        min_value=1, max_value=200, value=10,
        help="How many unique operator types exist in your code?"
    )
    n2 = st.number_input(
        "n2 — distinct operands",
        min_value=1, max_value=500, value=15,
        help="How many unique variable names, numbers, strings?"
    )
with col4:
    st.markdown("**Total counts (every occurrence)**")
    N1 = st.number_input(
        "N1 — total operators",
        min_value=1, max_value=20000, value=80,
        help="Total number of operator usages across the whole file"
    )
    N2 = st.number_input(
        "N2 — total operands",
        min_value=1, max_value=20000, value=100,
        help="Total number of operand usages across the whole file"
    )

# Calculate Halstead values live
N, V, L, b = calculate_halstead(n1, n2, N1, N2)

# Show computed values
st.markdown("**Computed Halstead values:**")
h_col1, h_col2, h_col3, h_col4 = st.columns(4)
h_col1.metric("N (length)",    N)
h_col2.metric("V (volume)",    V)
h_col3.metric("L (level)",     L)
h_col4.metric("b (bugs est.)", b)

# ══════════════════════════════════════════════════════════
#   SECTION 3 — Submit
# ══════════════════════════════════════════════════════════
st.divider()

if st.button("🔍 Submit for Code Review", use_container_width=True):

    # Build model input
    slider_vals = {
        "loc"  : loc,
        "v(g)" : vg,
        "ev(g)": evg,
        "iv(g)": ivg,
        "n"    : N,
        "b"    : b
    }
    full_input  = [slider_vals.get(f, 0) for f in FEATURES]
    scaled      = scaler.transform(np.array([full_input]))
    pred        = bool(active_model.predict(scaled)[0])
    proba       = float(active_model.predict_proba(scaled)[0][1])
    response    = RESPONSES[pred]

    # ── Verdict header ─────────────────────────────────────
    st.divider()
    st.subheader(response["status"])
    st.metric("Defect probability", f"{proba:.1%}")
    st.progress(proba)

    msg = random.choice(response["lines"])
    if pred:
        st.error(f'**Senior Dev says:** "{msg}"')
    else:
        st.success(f'**Senior Dev says:** "{msg}"')

    # ── Detailed fixes (the new section) ──────────────────
    fixes = generate_fixes(loc, vg, evg, ivg, N, V, L, b)

    if fixes:
        st.divider()
        st.subheader("🔧 What is wrong and how to fix it")
        st.caption(
            "Each issue is listed in order of severity. "
            "Fix the HIGH severity ones first."
        )

        for fix in fixes:
            severity = fix["severity"]
            icon     = "🔴" if severity == "high" else "🟡"
            color    = "error" if severity == "high" else "warning"

            with st.expander(
                f"{icon}  {fix['metric']} = {fix['value']}  "
                f"({'Critical' if severity == 'high' else 'Warning'})"
            ):
                if color == "error":
                    st.error(f"**Problem:** {fix['problem']}")
                else:
                    st.warning(f"**Problem:** {fix['problem']}")

                st.markdown("**How to fix it:**")
                for i, step in enumerate(fix["fix"], 1):
                    st.markdown(f"{i}. {step}")

    elif not pred:
        st.divider()
        st.success(
            "✅ No significant issues detected. "
            "This module is within acceptable complexity bounds. "
            "Keep functions small, keep v(g) under 10, "
            "and you'll stay in the green zone."
        )

    # ── Feature importance chart ───────────────────────────
    if model_choice == "Random Forest":
        with st.expander("Why did the model decide this? ↓"):
            importances = pd.Series(
                rf_model.feature_importances_,
                index=FEATURES
            ).sort_values(ascending=True).tail(10)

            colors = [
                "#f44336" if f in slider_vals else "#90A4AE"
                for f in importances.index
            ]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.barh(
                importances.index,
                importances.values,
                color=colors,
                edgecolor="none"
            )
            ax.set_xlabel("Importance score")
            ax.set_title("Top 10 features  (red = your input)")
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            st.pyplot(fig)

    # ── Model comparison ───────────────────────────────────
    with st.expander("Compare both models ↓"):
        rf_prob = float(rf_model.predict_proba(scaled)[0][1])
        lr_prob = float(lr_model.predict_proba(scaled)[0][1])
        st.dataframe(pd.DataFrame({
            "Model"             : ["Random Forest", "Logistic Regression"],
            "Defect probability": [f"{rf_prob:.1%}", f"{lr_prob:.1%}"],
            "Verdict"           : [
                "Defective" if rf_prob > 0.5 else "Clean",
                "Defective" if lr_prob > 0.5 else "Clean"
            ]
        }), hide_index=True, use_container_width=True)


