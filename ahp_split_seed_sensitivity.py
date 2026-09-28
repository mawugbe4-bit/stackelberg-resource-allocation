"""
AHP Split-Seed Sensitivity Check (Q8)
-----------------------------------------------------------------------
Tests whether the calibration-wave weight vector (alpha_1-alpha_4) is an
artefact of the specific seed=42 70:30 split, by re-running the exact
same aggregation pipeline (CR<0.10 filter -> 70:30 split -> geometric-
mean aggregation) across 20 different seeds, and reporting the spread of
the resulting calibration-wave weights.

If the seed=42 result sits comfortably within the range produced by
other seeds, the reported weights are not a lucky/unlucky artefact of
that one split. If seed=42 is an outlier relative to the other 19 seeds,
that is reported honestly rather than hidden.

Uses the EXACT split method verified in ahp_cr_sensitivity.py to
reproduce the confirmed baseline (numpy permutation with a round(0.7*n)
cutoff, not sklearn's train_test_split).

BEFORE RUNNING: edit SURVEY_XLSX to your local path.
"""

import numpy as np
import pandas as pd
import openpyxl

SURVEY_XLSX = r"A:\PhD IT\Cybersecurity_Threat_Prioritisation_-_AHP_Survey.xlsx"  # <-- edit

RI = {4: 0.90}
N_CRITERIA = 4
CRITERIA = ["impact", "propagation", "confidence", "evasion"]
CR_THRESHOLD = 0.10  # the threshold actually used and reported
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 20 seeds, seed 42 first


def to_saaty(k):
    if k == 0:
        return 1.0
    val = abs(k) + 1
    return val if k > 0 else 1.0 / val


def build_matrix(ip, ic, ie, pc, pe, ce):
    M = np.ones((4, 4))
    M[0, 1] = to_saaty(ip); M[1, 0] = 1 / M[0, 1]
    M[0, 2] = to_saaty(ic); M[2, 0] = 1 / M[0, 2]
    M[0, 3] = to_saaty(ie); M[3, 0] = 1 / M[0, 3]
    M[1, 2] = to_saaty(pc); M[2, 1] = 1 / M[1, 2]
    M[1, 3] = to_saaty(pe); M[3, 1] = 1 / M[1, 3]
    M[2, 3] = to_saaty(ce); M[3, 2] = 1 / M[2, 3]
    return M


def eigen_priority(M):
    vals, vecs = np.linalg.eig(M)
    idx = np.argmax(vals.real)
    lam_max = vals.real[idx].real
    w = np.abs(vecs[:, idx].real)
    w = w / w.sum()
    CI = (lam_max - N_CRITERIA) / (N_CRITERIA - 1)
    CR = CI / RI[N_CRITERIA]
    return w, CR


def load_consistent_respondents():
    wb = openpyxl.load_workbook(SURVEY_XLSX, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    consistent_w = []
    for r in rows:
        ip, ic, ie, pc, pe, ce = r[10], r[11], r[12], r[13], r[14], r[15]
        M = build_matrix(ip, ic, ie, pc, pe, ce)
        w, cr = eigen_priority(M)
        if cr < CR_THRESHOLD:
            consistent_w.append(w)
    return np.array(consistent_w)


def agg(mat):
    gm = np.exp(np.mean(np.log(mat), axis=0))
    return gm / gm.sum()


def calibration_weights_for_seed(consistent_w, seed):
    rng = np.random.default_rng(seed)
    n = len(consistent_w)
    perm = rng.permutation(n)
    split = int(round(0.7 * n))
    cal_pos = perm[:split]
    return agg(consistent_w[cal_pos])


def main():
    consistent_w = load_consistent_respondents()
    print(f"Consistent respondents (CR < {CR_THRESHOLD}): n={len(consistent_w)}")

    results = []
    for seed in SEEDS:
        w = calibration_weights_for_seed(consistent_w, seed)
        results.append({"seed": seed, **dict(zip(CRITERIA, w))})
        marker = " <-- reported seed" if seed == 42 else ""
        print(f"  seed={seed:>3}: " + ", ".join(f"{c}={w[i]:.4f}" for i, c in enumerate(CRITERIA)) + marker)

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("SUMMARY ACROSS 20 SEEDS")
    print("=" * 70)
    for c in CRITERIA:
        vals = df[c].to_numpy()
        seed42_val = df[df.seed == 42][c].values[0]
        print(f"{c}: mean={vals.mean():.4f}, SD={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}], seed=42 value={seed42_val:.4f}")

    print("\n" + "=" * 70)
    print("Is seed=42 an outlier? (seed=42 value vs. mean +/- 2 SD of other 19 seeds)")
    print("=" * 70)
    for c in CRITERIA:
        other_vals = df[df.seed != 42][c].to_numpy()
        seed42_val = df[df.seed == 42][c].values[0]
        other_mean, other_sd = other_vals.mean(), other_vals.std()
        z = (seed42_val - other_mean) / other_sd if other_sd > 0 else 0
        flag = "OUTLIER" if abs(z) > 2 else "within normal range"
        print(f"{c}: seed=42 z-score relative to other 19 seeds = {z:+.2f} ({flag})")

    df.to_csv("ahp_split_seed_sensitivity.csv", index=False)
    print("\nSaved: ahp_split_seed_sensitivity.csv")


if __name__ == "__main__":
    main()
