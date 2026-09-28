"""
CERT r4.2 Classification Confidence Builder
---------------------------------------------
Stage 3-5 of the Confidence pipeline:
  3. Label each user-week row in your feature table as malicious (1) or
     not (0), using the ground-truth insiders.csv windows.
  4. Train a classifier (behavioural features -> malicious label) that
     handles the severe class imbalance.
  5. Use its out-of-fold predicted probability as the Confidence value,
     replacing the NaN placeholder.

BEFORE RUNNING: edit the three paths below.
"""

import pandas as pd
import numpy as np
import os

# ----------------------------------------------------------------------
# 0. CONFIG -- edit these
# ----------------------------------------------------------------------
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
FEATURES_CSV = r"A:\PhD IT\results\cert_threat_resource_features.csv"  # <-- your existing output
OUTPUT_CSV = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"

TARGET_DATASET = "4.2"


# ----------------------------------------------------------------------
# 1. LOAD GROUND-TRUTH INSIDER WINDOWS FOR r4.2
# ----------------------------------------------------------------------
def load_insider_windows():
    df = pd.read_csv(INSIDERS_CSV)
    df["dataset"] = df["dataset"].astype(str).str.strip()
    r42 = df[df["dataset"] == TARGET_DATASET].copy()
    r42["start"] = pd.to_datetime(r42["start"])
    r42["end"] = pd.to_datetime(r42["end"])
    print(f"Loaded {len(r42)} insider windows for dataset {TARGET_DATASET}")
    return r42[["user", "start", "end"]]


# ----------------------------------------------------------------------
# 2. LABEL EACH USER-WEEK ROW
#    threat_id was built as f"{user}_{week}" where week looks like
#    "2010-08-23/2010-08-29" (a pandas week-period string). We split it
#    back apart rather than re-running the whole aggregation.
# ----------------------------------------------------------------------
def label_features(features, insiders):
    features = features.copy()
    split = features["threat_id"].str.split("_", n=1, expand=True)
    features["user"] = split[0]
    week_range = split[1].str.split("/", expand=True)
    features["week_start"] = pd.to_datetime(week_range[0])
    features["week_end"] = pd.to_datetime(week_range[1])

    # index insider windows by user for fast lookup
    windows_by_user = {}
    for _, row in insiders.iterrows():
        windows_by_user.setdefault(row["user"], []).append((row["start"], row["end"]))

    def is_malicious(row):
        windows = windows_by_user.get(row["user"])
        if not windows:
            return 0
        for w_start, w_end in windows:
            if row["week_start"] <= w_end and row["week_end"] >= w_start:
                return 1
        return 0

    features["malicious"] = features.apply(is_malicious, axis=1)
    n_pos = features["malicious"].sum()
    print(f"Labeled {len(features):,} rows: {n_pos:,} malicious user-weeks "
          f"({n_pos / len(features):.3%}), from {features['user'].isin(windows_by_user).sum():,} "
          f"rows belonging to a known insider user.")
    return features


# ----------------------------------------------------------------------
# 3. TRAIN THE CLASSIFIER WITH OUT-OF-FOLD PREDICTIONS
#    Out-of-fold (via cross-validation) so that "confidence" isn't just
#    the classifier memorising its own training labels -- each row's
#    score comes from a model that never saw that row during training.
# ----------------------------------------------------------------------
def build_confidence(features):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    feature_cols = ["impact", "propagation", "evasion"]
    X = features[feature_cols].fillna(0.0).to_numpy()
    y = features["malicious"].to_numpy()
    groups = features["user"].to_numpy()  # each person's weeks stay together in one fold

    n_splits = min(5, int(y.sum())) if y.sum() >= 2 else 2
    n_splits = max(n_splits, 2)
    # StratifiedGroupKFold: preserves class balance across folds (like the
    # StratifiedKFold this replaces) AND ensures the same person's rows
    # never span both the training and test side of a fold -- fixes a
    # leakage risk in the earlier plain StratifiedKFold version, where a
    # person's other weeks in the training fold could let the model
    # partially learn person-specific patterns for a row held out in the
    # test fold, rather than purely general malicious-vs-benign signal.
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )

    print(f"Training with {n_splits}-fold GROUP-stratified CV, grouped by user "
          f"({int(y.sum())} positive rows total, {len(set(groups))} distinct users)...")
    proba = cross_val_predict(clf, X, y, cv=sgkf, groups=groups, method="predict_proba")[:, 1]

    features["confidence"] = proba
    return features


def main():
    insiders = load_insider_windows()
    features = pd.read_csv(FEATURES_CSV)
    labeled = label_features(features, insiders)
    result = build_confidence(labeled)

    out = result[["threat_id", "resource_id", "impact", "propagation", "confidence", "evasion", "cost"]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(out["confidence"].describe())


if __name__ == "__main__":
    main()
