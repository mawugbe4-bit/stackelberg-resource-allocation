"""
Confusion Matrix / TPR-FPR for the Classification Confidence Classifier
---------------------------------------------------------------------------
Uses the out-of-fold Confidence probabilities already saved in your
feature CSV (from cert_confidence_classifier.py's cross_val_predict step)
-- no re-training or re-running the classifier needed.

Threshold: 0.5 by default (standard). A confusion matrix built on
out-of-fold predictions is a fair, non-overfit estimate, since each row's
prediction came from a model that never saw that row during training.

BEFORE RUNNING: edit DATA_PATH and INSIDERS_CSV below (same files as your
other CERT-side scripts).
"""

import numpy as np
import pandas as pd

DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_feature_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
THRESHOLD = 0.5


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
    return df


def main():
    df = load_labeled_data()
    y_true = df["malicious"].to_numpy()
    y_prob = df["confidence"].to_numpy()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")   # recall / sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else float("nan")

    print("=" * 70)
    print(f"CONFUSION MATRIX (threshold = {THRESHOLD}) -- out-of-fold predictions")
    print("=" * 70)
    print(f"                 Predicted: Not malicious   Predicted: Malicious")
    print(f"Actual: Not malicious    {tn:>10,}              {fp:>10,}")
    print(f"Actual: Malicious        {fn:>10,}              {tp:>10,}")
    print()
    print(f"TPR (recall/sensitivity): {tpr:.4f}")
    print(f"FPR:                      {fpr:.4f}")
    print(f"Precision:                {precision:.4f}")
    print(f"F1:                       {f1:.4f}")
    print(f"n = {len(df):,}  (malicious = {y_true.sum():,}, not malicious = {(y_true==0).sum():,})")
    print("=" * 70)

    pd.DataFrame({
        "metric": ["TP", "TN", "FP", "FN", "TPR", "FPR", "Precision", "F1", "threshold", "n"],
        "value": [tp, tn, fp, fn, tpr, fpr, precision, f1, THRESHOLD, len(df)],
    }).to_csv("confidence_confusion_matrix.csv", index=False)
    print("\nSaved: confidence_confusion_matrix.csv")


if __name__ == "__main__":
    main()
