"""
AHP Consistency-Ratio Sensitivity Check (Q2)
-----------------------------------------------------------------------
Tests whether the calibration-wave weight vector (alpha_1-alpha_4) is
sensitive to the CR<0.10 consistency-filtering threshold, by re-running
the identical aggregation pipeline (consistency filter -> 70:30
calibration/held-out split, seed=42 -> geometric-mean aggregation) under
three conditions:
  (a) CR < 0.10  -- the threshold actually used and reported (n=46)
  (b) CR < 0.20  -- a relaxed threshold, retaining more respondents
  (c) No filter  -- all 105 respondents, including inconsistent ones

If the calibration-wave weights shift only slightly across these three
conditions, the CR<0.10 threshold choice is not doing much work and the
reported weights are robust to it. If they shift substantially, that is
reported honestly as a real sensitivity, not smoothed over.

Uses the same Saaty eigenvector / CR computation as
ahp_weights_by_institution.py. The 70:30 calibration/held-out split uses
the EXACT original method (verified to reproduce the confirmed 0.224/
0.252/0.247/0.278 baseline exactly): numpy permutation with a fixed seed
(42) and a round(0.7*n) cutoff -- not sklearn's train_test_split, which
does not reproduce the original split.

BEFORE RUNNING: edit SURVEY_XLSX to your local path.
"""

import numpy as np
import pandas as pd
import openpyxl

SURVEY_XLSX = r"A:\PhD IT\Cybersecurity_Threat_Prioritisation_-_AHP_Survey.xlsx"  # <-- edit

RI = {4: 0.90}
N_CRITERIA = 4
CRITERIA = ["impact", "propagation", "confidence", "evasion"]
RANDOM_SEED = 42


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


def geomean_weights(sub):
    gm = np.exp(np.log(sub[CRITERIA]).mean(axis=0))
    return gm / gm.sum()


def load_all_respondents():
    wb = openpyxl.load_workbook(SURVEY_XLSX, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    records = []
    for r in rows:
        ip, ic, ie, pc, pe, ce = r[10], r[11], r[12], r[13], r[14], r[15]
        M = build_matrix(ip, ic, ie, pc, pe, ce)
        w, cr = eigen_priority(M)
        records.append({
            "CR": cr,
            "impact": w[0], "propagation": w[1], "confidence": w[2], "evasion": w[3],
        })
    return pd.DataFrame(records)


def calibration_weights_for_threshold(df, cr_threshold, label):
    if cr_threshold is None:
        subset = df.copy()
    else:
        subset = df[df["CR"] < cr_threshold].copy()

    if len(subset) < 4:
        print(f"{label}: only {len(subset)} respondents retained -- too few to split, skipping.")
        return None

    # Exact original split method (confirmed to reproduce 0.224/0.252/0.247/0.278
    # exactly): numpy permutation with a round(0.7*n) cutoff, NOT sklearn's
    # train_test_split (which does not reproduce the confirmed baseline).
    W = subset[CRITERIA].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(subset))
    split = int(round(0.7 * len(perm)))
    cal_pos, held_pos = perm[:split], perm[split:]

    def agg(mat):
        gm = np.exp(np.mean(np.log(mat), axis=0))
        return gm / gm.sum()

    calib_w = dict(zip(CRITERIA, agg(W[cal_pos])))
    heldout_w = dict(zip(CRITERIA, agg(W[held_pos])))

    print(f"\n{label}: n={len(subset)} retained (of 105) -> calibration n={len(cal_pos)}, held-out n={len(held_pos)}")
    print(f"  Calibration-wave weights: " + ", ".join(f"{c}={calib_w[c]:.4f}" for c in CRITERIA))
    print(f"  Held-out-wave weights:    " + ", ".join(f"{c}={heldout_w[c]:.4f}" for c in CRITERIA))
    return {"label": label, "n": len(subset), "n_calib": len(cal_pos), **{f"calib_{c}": calib_w[c] for c in CRITERIA}}


def main():
    df = load_all_respondents()
    print(f"Loaded {len(df)} total respondents.")
    print(f"CR distribution: min={df['CR'].min():.4f}, median={df['CR'].median():.4f}, max={df['CR'].max():.4f}")

    results = []
    r = calibration_weights_for_threshold(df, 0.10, "CR < 0.10 (as reported)")
    if r: results.append(r)
    r = calibration_weights_for_threshold(df, 0.20, "CR < 0.20 (relaxed)")
    if r: results.append(r)
    r = calibration_weights_for_threshold(df, None, "No filter (all 105)")
    if r: results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY -- calibration-wave weights across thresholds")
    print("=" * 70)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("Max absolute shift in any single weight, CR<0.10 vs. each alternative:")
    base = summary.iloc[0]
    for i in range(1, len(summary)):
        row = summary.iloc[i]
        max_shift = max(abs(row[f"calib_{c}"] - base[f"calib_{c}"]) for c in CRITERIA)
        print(f"  {row['label']}: max shift = {max_shift:.4f}")
    print("=" * 70)

    summary.to_csv("ahp_cr_sensitivity.csv", index=False)
    print("\nSaved: ahp_cr_sensitivity.csv")


if __name__ == "__main__":
    main()
