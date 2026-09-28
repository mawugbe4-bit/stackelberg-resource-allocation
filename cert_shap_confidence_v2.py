"""
SHAP Feature Importance for the Classification Confidence Classifier -- v2
-------------------------------------------------------------------------
v2 change from v1 (Q14, signed SHAP): reports mean SIGNED SHAP value
alongside the existing mean ABSOLUTE SHAP value. The original version
only reported magnitude, so it was impossible to tell whether a feature
pushed the classifier's prediction TOWARD "malicious" or AWAY from it --
a feature could rank highly in importance while systematically pushing
predictions in either direction, and the original output could not
distinguish these. Both are now reported side by side. Also corrected
DATA_PATH to match the GroupKFold-corrected classifier's actual output
filename (plural "features").

Explains WHICH behavioural features (Impact, Propagation, Evasion -- the
three raw log-derived criteria, not Confidence itself, since Confidence
IS the model's output) drive the classifier's prediction of "malicious"
for a given user-week.

This is a genuinely different kind of analysis from the CATE work: CATE
asks "does the calibration effect vary across subgroups?"; SHAP asks
"what does the underlying Confidence classifier actually key off of, and
in which direction?". Both were committed to in Methods Section 2.14 as
supporting robustness checks -- neither replaces the primary AHP-
calibration finding (RQ1-RQ3).

Note: cert_confidence_classifier.py only produced OUT-OF-FOLD predictions
via cross-validation (deliberately, to avoid the classifier memorising
its own training labels when generating the Confidence feature). SHAP
needs one single fitted model to explain, so this script fits a final
model on the FULL labelled dataset -- appropriate here because the goal
is now interpretability (what does the classifier attend to, in
general?), not producing an unbiased per-row prediction. This final fit
does not use GroupKFold (there is no train/test split here at all), so
it is unaffected by the classifier's cross-validation scheme.

BEFORE RUNNING: edit DATA_PATH and INSIDERS_CSV below (same files as the
other CERT-side scripts). Requires: pip install shap
"""

import os
import numpy as np
import pandas as pd

DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
OUTPUT_DIR = r"A:\PhD IT\results"


def load_labeled_data():
    df = pd.read_csv(DATA_PATH)
    insiders = pd.read_csv(INSIDERS_CSV)
    insiders["dataset"] = insiders["dataset"].astype(str).str.strip()
    r42 = insiders[insiders["dataset"] == "4.2"].copy()
    r42["start"] = pd.to_datetime(r42["start"])
    r42["end"] = pd.to_datetime(r42["end"])

    windows_by_user = {}
    for _, row in r42.iterrows():
        windows_by_user.setdefault(row["user"], []).append((row["start"], row["end"]))

    split = df["threat_id"].str.split("_", n=1, expand=True)
    df["user"] = split[0]
    week_range = split[1].str.split("/", expand=True)
    df["week_start"] = pd.to_datetime(week_range[0])
    df["week_end"] = pd.to_datetime(week_range[1])

    def is_malicious(row):
        windows = windows_by_user.get(row["user"])
        if not windows:
            return 0
        for w_start, w_end in windows:
            if row["week_start"] <= w_end and row["week_end"] >= w_start:
                return 1
        return 0

    df["malicious"] = df.apply(is_malicious, axis=1)
    print(f"Loaded {len(df):,} rows, {df['malicious'].sum():,} malicious.")
    return df


def main():
    try:
        import shap
    except ImportError:
        print("[ERROR] shap is not installed. Run: pip install shap")
        return
    from sklearn.ensemble import RandomForestClassifier

    df = load_labeled_data()
    feature_cols = ["impact", "propagation", "evasion"]
    X = df[feature_cols].fillna(0.0)
    y = df["malicious"].to_numpy()

    print("Fitting final classifier on the full labelled dataset "
          "(for interpretability, not for unbiased per-row prediction)...")
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", max_depth=6,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # shap_values for a binary RF classifier: list [class0, class1] or a
    # single array depending on the sklearn/shap version -- handle both.
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 = malicious
    else:
        sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    mean_signed_shap = sv.mean(axis=0)  # NEW (Q14): preserves direction, not just magnitude
    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE (mean |SHAP value| AND mean signed SHAP) -- paste into Results")
    print("=" * 70)
    total = importance["mean_abs_shap"].sum()
    for _, row in importance.iterrows():
        pct = 100 * row["mean_abs_shap"] / total if total > 0 else 0
        direction = "TOWARD malicious" if row["mean_signed_shap"] > 0 else "AWAY FROM malicious"
        print(f"  {row['feature']:<15} |SHAP|={row['mean_abs_shap']:.5f} ({pct:.1f}% of total)  "
              f"signed={row['mean_signed_shap']:+.5f} (pushes predictions {direction}, on average)")
    print("=" * 70)

    # Save a summary plot (bar) -- viewable directly, no need to re-run
    # SHAP interactively to see it.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(sv, X, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_bar_v2.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"\nSaved bar plot: {bar_path}")

    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_beeswarm_v2.png")
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"Saved beeswarm plot: {beeswarm_path}")

    importance_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_v2.csv")
    importance.to_csv(importance_path, index=False)
    print(f"Saved: {importance_path}")


if __name__ == "__main__":
    main()
