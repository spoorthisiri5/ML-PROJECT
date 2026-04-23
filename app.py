# ── SHAP explainer — NO cache_resource, built fresh each time ──
# cache_resource fails silently on sklearn objects in many Streamlit versions
def get_shap_explainer(model):
    import shap
    return shap.TreeExplainer(model)

# ── In your Submit button block, replace the SHAP expander with: ──

if model_choice in ["Random Forest", "Gradient Boosting"]:
    with st.expander("🔍 Why did the model decide THIS? (SHAP) ↓"):
        st.caption(
            "Red bars pushed this prediction toward Defective. "
            "Blue bars pushed it toward Clean. "
            "This chart changes every time you move a slider."
        )
        try:
            import shap
            import matplotlib.pyplot as plt
            import pandas as pd

            # Select the right base model (not the ensemble)
            tree_model = rf_model if model_choice == "Random Forest" \
                         else gb_model

            # Build explainer
            explainer = shap.TreeExplainer(tree_model)

            # Get SHAP values
            shap_vals = explainer.shap_values(scaled)

            # ── Handle BOTH old and new SHAP output formats ───────
            # Old SHAP (<0.40): returns list [class0_array, class1_array]
            # New SHAP (>=0.40): returns single array OR Explanation object

            if isinstance(shap_vals, list):
                # Old format — take class 1 (Defective), first row
                sv = shap_vals[1][0]

            elif hasattr(shap_vals, "values"):
                # Explanation object (newest SHAP)
                vals = shap_vals.values
                if vals.ndim == 3:
                    # shape: (n_samples, n_features, n_classes)
                    sv = vals[0, :, 1]
                elif vals.ndim == 2:
                    # shape: (n_samples, n_features) — already class 1
                    sv = vals[0, :]
                else:
                    sv = vals[0]

            else:
                # Numpy array fallback
                arr = shap_vals
                if arr.ndim == 3:
                    sv = arr[0, :, 1]
                elif arr.ndim == 2:
                    sv = arr[0, :]
                else:
                    sv = arr[0]

            # ── Build readable dataframe ──────────────────────────
            input_values = input_array[0].tolist()

            shap_df = pd.DataFrame({
                "Feature"    : FEATURES,
                "Your input" : [round(v, 3) for v in input_values],
                "SHAP value" : sv,
            })
            shap_df["abs_shap"] = shap_df["SHAP value"].abs()
            shap_df = (
                shap_df
                .sort_values("abs_shap", ascending=False)
                .head(12)
                .reset_index(drop=True)
            )

            # ── Check we actually have non-zero values ─────────────
            if shap_df["SHAP value"].abs().sum() < 1e-10:
                st.warning(
                    "SHAP values are all near zero for this input. "
                    "This can happen with very extreme or very average inputs. "
                    "Try adjusting the sliders."
                )
            else:
                # ── Plot ───────────────────────────────────────────
                colors = [
                    "#f44336" if v > 0 else "#2196F3"
                    for v in shap_df["SHAP value"]
                ]

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(
                    shap_df["Feature"],
                    shap_df["SHAP value"],
                    color=colors,
                    edgecolor="none"
                )
                ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.set_xlabel(
                    "← Pushes toward Clean          "
                    "Pushes toward Defective →"
                )
                ax.set_title(
                    f"SHAP — why this input got {proba:.1%} defect probability"
                )
                # Make background transparent so it matches Streamlit's theme
                fig.patch.set_facecolor("none")
                ax.set_facecolor("none")
                ax.tick_params(colors="gray")
                ax.xaxis.label.set_color("gray")
                ax.title.set_color("gray")
                for spine in ax.spines.values():
                    spine.set_edgecolor("gray")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)   # ← prevents memory leak in Streamlit

                # ── Table ──────────────────────────────────────────
                display_df = shap_df[["Feature","Your input","SHAP value"]].copy()
                display_df["SHAP value"] = display_df["SHAP value"].round(4)
                display_df["Direction"]  = display_df["SHAP value"].apply(
                    lambda x: "→ Defective" if x > 0 else "→ Clean"
                )
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ── Plain English summary ──────────────────────────
                top     = shap_df.iloc[0]
                top_dir = "Defective" if top["SHAP value"] > 0 else "Clean"
                st.info(
                    f"**Strongest driver:** `{top['Feature']}` = "
                    f"`{top['Your input']}` "
                    f"pushed this prediction toward **{top_dir}** "
                    f"with SHAP = `{top['SHAP value']:+.4f}`"
                )

        except Exception as e:
            import traceback
            st.error(f"SHAP failed: {e}")
            st.code(traceback.format_exc())
            st.caption(
                "Most common fix: run `pip install shap --upgrade` "
                "and add `shap` to requirements.txt"
            )
