"""
SHAP Feature Importance for the Classification Confidence Classifier -- v3
-------------------------------------------------------------------------
v3 change from v2 (Q14, direction fix): v2's mean-signed-SHAP statistic,
averaged across all 67,238 rows, was found to be uninformative in this
specific dataset -- with 99.52% of rows non-malicious, the global mean
is swamped by the majority class and shows all three features pushing
"away from malicious" almost by mathematical necessity, regardless of
what each feature actually does for malicious cases specifically. v3
adds the standard, meaningful direction statistic instead: the
correlation between each feature's own raw value and its SHAP value. A
positive correlation means higher values of that feature push the
classifier toward "malicious" -- this is what a SHAP beeswarm plot shows
visually (colour = feature value, position = SHAP value) but was never
previously stated as an actual number. Both v2's global signed mean and
v3's correlation are reported together, since they answer genuinely
different questions and neither alone is complete.
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
    mean_signed_shap = sv.mean(axis=0)  # global mean signed value (swamped by class imbalance -- see docstring)

    # NEW (v3): correlation between each feature's own value and its SHAP
    # value -- the standard, meaningful direction statistic. Positive =
    # higher feature value pushes toward "malicious"; not swamped by
    # class imbalance the way the global signed mean is, since it
    # measures within-feature variation rather than an unconditional average.
    X_arr = X.to_numpy()
    value_shap_corr = np.array([
        np.corrcoef(X_arr[:, j], sv[:, j])[0, 1] for j in range(len(feature_cols))
    ])

    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap,
        "value_shap_correlation": value_shap_corr,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE (magnitude, global signed mean, AND value-SHAP correlation)")
    print("=" * 70)
    total = importance["mean_abs_shap"].sum()
    for _, row in importance.iterrows():
        pct = 100 * row["mean_abs_shap"] / total if total > 0 else 0
        corr_direction = "higher values -> MORE evidence of malicious" if row["value_shap_correlation"] > 0 \
            else "higher values -> LESS evidence of malicious"
        print(f"  {row['feature']:<15} |SHAP|={row['mean_abs_shap']:.5f} ({pct:.1f}% of total)")
        print(f"    global signed mean = {row['mean_signed_shap']:+.5f} (swamped by class imbalance -- see note below)")
        print(f"    value-SHAP correlation = {row['value_shap_correlation']:+.4f} ({corr_direction})")
    print("=" * 70)
    pct_nonmalicious = 100 * (1 - y.mean())
    print(f"NOTE: the global signed mean is dominated by the non-malicious majority class")
    print(f"({pct_nonmalicious:.2f}% of rows here) almost regardless of feature behaviour, and should not be")
    print("read as the direction finding. The value-SHAP correlation is the meaningful")
    print("direction statistic: it answers 'does higher Impact make THIS classifier more")
    print("confident of malicious activity', which the global mean cannot answer in an")
    print("imbalanced dataset.")
    print("=" * 70)

    # Save a summary plot (bar) -- viewable directly, no need to re-run
    # SHAP interactively to see it.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(sv, X, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_bar_v3.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"\nSaved bar plot: {bar_path}")

    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_beeswarm_v3.png")
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"Saved beeswarm plot: {beeswarm_path}")

    importance_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_v3.csv")
    importance.to_csv(importance_path, index=False)
    print(f"Saved: {importance_path}")


if __name__ == "__main__":
    main()
