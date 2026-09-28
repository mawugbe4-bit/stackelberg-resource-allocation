"""
AHP-Elicited Weight Heterogeneity by Institution Type
---------------------------------------------------------
This is the methodologically valid place to test "institution type"
heterogeneity: in the AHP-elicited weights themselves (each survey
respondent has an institution type), NOT in the CERT threat-mitigation
outcome (CERT threats have no link to Ghanaian institutions at all --
see cert_cate_by_scenario.py for the valid CERT-side heterogeneity
dimension, threat scenario type).

Small-n warning, confirmed on the real data: among the 46 CR<0.10
consistent respondents, institution types split extremely unevenly
(technical_university n=33; polytechnic n=5; other/public/private
universities n=2-3 each). A 5-group test is close to uninformative for
the smallest groups. This script reports the full descriptive breakdown
(for transparency) alongside a better-powered binary comparison
(technical_university vs. all other types combined), and is explicit
about which result should actually be trusted.

BEFORE RUNNING: edit SURVEY_XLSX to your local path.
"""

import numpy as np
import pandas as pd
import openpyxl
from scipy.stats import kruskal, mannwhitneyu

SURVEY_XLSX = r"A:\PhD IT\Cybersecurity_Threat_Prioritisation_-_AHP_Survey.xlsx"  # <-- edit

RI = {4: 0.90}
N_CRITERIA = 4
CRITERIA = ["impact", "propagation", "confidence", "evasion"]


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


def main():
    wb = openpyxl.load_workbook(SURVEY_XLSX, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    records = []
    for r in rows:
        ip, ic, ie, pc, pe, ce = r[10], r[11], r[12], r[13], r[14], r[15]
        inst_type = r[4]
        M = build_matrix(ip, ic, ie, pc, pe, ce)
        w, cr = eigen_priority(M)
        records.append({
            "institution_type": inst_type, "CR": cr,
            "impact": w[0], "propagation": w[1], "confidence": w[2], "evasion": w[3],
        })

    df = pd.DataFrame(records)
    consistent = df[df["CR"] < 0.10].copy()
    print(f"Total respondents: {len(df)} | Consistent (CR<0.10): {len(consistent)}")
    print("\nConsistent respondents by institution type:")
    print(consistent["institution_type"].value_counts())

    print("\n" + "=" * 70)
    print("DESCRIPTIVE: geometric-mean weight vector per institution type")
    print("=" * 70)
    for inst, sub in consistent.groupby("institution_type"):
        gm = geomean_weights(sub)
        print(f"{inst} (n={len(sub)}): " + ", ".join(f"{c}={gm[c]:.3f}" for c in CRITERIA))

    print("\n" + "=" * 70)
    print("TEST 1 (descriptive only -- most groups too small to trust): "
          "Kruskal-Wallis across all institution types")
    print("=" * 70)
    for c in CRITERIA:
        groups = [sub[c].values for _, sub in consistent.groupby("institution_type")]
        stat, p = kruskal(*groups)
        print(f"{c}: H={stat:.3f}, p={p:.4f}")

    print("\n" + "=" * 70)
    print("TEST 2 (adequately powered -- this is the one to actually report): "
          "technical_university vs. all other types combined")
    print("=" * 70)
    consistent["group2"] = np.where(
        consistent["institution_type"] == "technical_university",
        "technical_university", "other_types_combined",
    )
    for grp, sub in consistent.groupby("group2"):
        gm = geomean_weights(sub)
        print(f"{grp} (n={len(sub)}): " + ", ".join(f"{c}={gm[c]:.3f}" for c in CRITERIA))
    print()
    for c in CRITERIA:
        a = consistent[consistent.group2 == "technical_university"][c].values
        b = consistent[consistent.group2 == "other_types_combined"][c].values
        stat, p = mannwhitneyu(a, b)
        print(f"{c}: Mann-Whitney U={stat:.1f}, p={p:.4f}")

    consistent.to_csv("ahp_weights_by_institution.csv", index=False)
    print("\nSaved: ahp_weights_by_institution.csv")


if __name__ == "__main__":
    main()
