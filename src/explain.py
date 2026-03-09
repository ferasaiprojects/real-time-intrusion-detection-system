"""
Explainability module using SHAP
"""

import os
import shap
import matplotlib.pyplot as plt


def explain_model(model, X_test, feature_names):

    os.makedirs("results", exist_ok=True)

    # Guard: don't sample more rows than available
    n = min(500, len(X_test))
    sample = X_test.sample(n, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # For binary classification, LightGBM returns [class_0_vals, class_1_vals]
    # We want class 1 = ATTACK
    if isinstance(shap_values, list):
        attack_shap_values = shap_values[1]
    else:
        attack_shap_values = shap_values  # single output fallback

    # SHAP Summary Plot (feature importance by impact)
    plt.figure()
    shap.summary_plot(
        attack_shap_values,
        sample,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP Feature Impact - Attack Class")
    plt.tight_layout()
    plt.savefig("results/shap_summary.png")
    plt.close()

    # SHAP Bar Plot (mean absolute impact)
    plt.figure()
    shap.summary_plot(
        attack_shap_values,
        sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Mean Feature Importance - Attack Class")
    plt.tight_layout()
    plt.savefig("results/shap_bar.png")
    plt.close()

    print("SHAP summary plot saved to results/shap_summary.png")
    print("SHAP bar plot saved to results/shap_bar.png")