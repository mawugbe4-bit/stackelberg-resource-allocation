"""
CATE by Threat Scenario Type -- v3
-----------------------------------------------------------------------
v2 changes from the original cert_cate_by_scenario.py, matching the same
fixes already applied to cert_bootstrap_ate_v2.py and
cert_sa_greedy_pipeline_v4.py:
  - Person-level aggregation (Q17): the allocation/outcome unit is now
    one row per person, not per user-week, matching the per-person-year
    catalog pricing.
  - Cluster bootstrap (Q6): resampling PERSONS with replacement is now
    inherently the cluster-bootstrap fix, since each resampled unit is a
    complete person rather than a row nested within one.
  - Two catalog scenarios (Section 2.7): Training-capacity unconstrained
    (A) vs. capacity-matched to Access Review (B), run and reported side
    by side rather than asserting a single answer.
  - Corrected DATA_PATH matching the GroupKFold-corrected classifier
    output.

Extends the paired-bootstrap ATE to check whether the AHP-calibration
effect on threat-mitigation rate is uniform across the three r4.2
insider scenarios, or concentrated in one:
  Scenario 1: rapid data theft shortly before leaving the organisation
  Scenario 2: gradual, long-running data theft
  Scenario 3: short-burst sabotage-style activity

Small-n warning: scenario 3 has only 10 insiders (vs. 30 each for
scenarios 1 and 2), a structural limitation of the CERT dataset itself
that no re-analysis can fix (see Methods, Chapter 6 limitations).

v3 changes from v2 (two real errors caught by supervisor review, both
fixed, matching cert_sa_greedy_pipeline_v5.py):
  1. Person-level aggregation for Impact/Propagation/Evasion changed from
     SUM to MEAN. Summing a weekly proportion (Evasion) is meaningless;
     more generally, summing any of these confounds tenure length with
     behavioural intensity, systematically under-scoring short-tenure
     people including the 30 rapid pre-departure-theft insiders
     (Scenario 1).
  2. Incident Response effectiveness corrected from 0.5991 to 0.3378.
     The earlier value used IBM's 2022 figure ($2.66M) mistakenly cited
     as 2025. The real 2025 IBM figure (confirmed directly from ibm.com)
     is $1.5M; $1.5M/$4.44M (2025 global average) = 0.3378.

BEFORE RUNNING: edit the paths below (same files as
cert_bootstrap_ate_v2.py / cert_sa_greedy_pipeline_v4.py).
"""

import numpy as np
import pandas as pd

DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
BUDGET = 7000.0
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

CALIBRATED_WEIGHTS = {"impact": 0.224, "propagation": 0.252, "confidence": 0.247, "evasion": 0.278}
BASELINE_WEIGHTS = {"impact": 0.25, "propagation": 0.25, "confidence": 0.25, "evasion": 0.25}

CATALOG_SCENARIOS = {
    "A_training_unconstrained": pd.DataFrame([
        {"resource_id": "training",      "cost": 24.0,  "effectiveness": 0.0428, "capacity": 100_000},
        {"resource_id": "monitoring",    "cost": 36.0,  "effectiveness": 0.0473, "capacity": 10},
        {"resource_id": "access_review", "cost": 114.0, "effectiveness": 0.0428, "capacity": 5},
        {"resource_id": "dlp",           "cost": 63.2,  "effectiveness": 0.0507, "capacity": 100_000},
        {"resource_id": "ir",            "cost": 15.0,  "effectiveness": 0.3378, "capacity": 10},
    ]),
    "B_training_constrained": pd.DataFrame([
        {"resource_id": "training",      "cost": 24.0,  "effectiveness": 0.0428, "capacity": 5},
        {"resource_id": "monitoring",    "cost": 36.0,  "effectiveness": 0.0473, "capacity": 10},
        {"resource_id": "access_review", "cost": 114.0, "effectiveness": 0.0428, "capacity": 5},
        {"resource_id": "dlp",           "cost": 63.2,  "effectiveness": 0.0507, "capacity": 100_000},
        {"resource_id": "ir",            "cost": 15.0,  "effectiveness": 0.3378, "capacity": 10},
    ]),
}


def load_and_aggregate_to_person_level_with_scenario():
    df = pd.read_csv(DATA_PATH)
    insiders = pd.read_csv(INSIDERS_CSV)
    insiders["dataset"] = insiders["dataset"].astype(str).str.strip()
    r42 = insiders[insiders["dataset"] == "4.2"].copy()

    r42_insider_users = set(r42["user"].unique())
    scenario_by_user = dict(zip(r42["user"], r42["scenario"].astype(int)))

    split = df["threat_id"].str.split("_", n=1, expand=True)
    df["user"] = split[0]

    # CORRECTED (was "sum" in an earlier version -- flagged as wrong and
    # fixed): Evasion is a weekly proportion, and summing a proportion
    # across weeks is meaningless; more generally, summing ANY of these
    # scales the score with weeks observed, confounding short-tenure
    # people (e.g. the 30 rapid pre-departure-theft insiders, Scenario 1)
    # with genuinely lower behavioural intensity. Per-week means remove
    # this confound for all three features consistently.
    agg = df.groupby("user").agg(
        impact=("impact", "mean"),
        propagation=("propagation", "mean"),
        evasion=("evasion", "mean"),
        confidence=("confidence", "mean"),
    ).reset_index()

    agg["malicious"] = agg["user"].isin(r42_insider_users).astype(int)
    agg["scenario"] = agg["user"].map(scenario_by_user).fillna(-1).astype(int)

    for col in ["impact", "propagation", "evasion"]:
        lo, hi = agg[col].min(), agg[col].max()
        agg[col] = (agg[col] - lo) / (hi - lo) if hi > lo else 0.0

    counts = agg[agg["malicious"] == 1]["scenario"].value_counts().sort_index()
    print(f"Aggregated {len(df):,} user-week records to {len(agg):,} person-level records.")
    print(f"Malicious persons by scenario: {dict(counts)}")
    return agg


def threat_scores(df, weights):
    return (
        weights["impact"] * df["impact"] + weights["propagation"] * df["propagation"]
        + weights["confidence"] * df["confidence"] + weights["evasion"] * df["evasion"]
    ).to_numpy()


def greedy_allocate(threat_score, budget, catalog):
    n = len(threat_score)
    costs = catalog["cost"].to_numpy()
    effs = catalog["effectiveness"].to_numpy()
    caps = catalog["capacity"].to_numpy()
    n_res = len(catalog)

    value_matrix = threat_score[:, None] * effs[None, :]
    cost_matrix = np.tile(costs, (n, 1))
    ratio_matrix = value_matrix / cost_matrix
    order = np.argsort(-ratio_matrix.ravel())
    t_idx_all, r_idx_all = np.unravel_index(order, (n, n_res))

    assigned = np.full(n, -1, dtype=np.int8)
    used_capacity = np.zeros(n_res, dtype=np.int64)
    spent = 0.0
    for t_idx, r_idx in zip(t_idx_all, r_idx_all):
        if assigned[t_idx] != -1 or used_capacity[r_idx] >= caps[r_idx]:
            continue
        cost = costs[r_idx]
        if spent + cost <= budget:
            assigned[t_idx] = r_idx
            used_capacity[r_idx] += 1
            spent += cost
    return assigned


def mitigation_rate_by_scenario(df, assigned, scenario):
    mask = (df["malicious"].to_numpy() == 1) & (df["scenario"].to_numpy() == scenario)
    if mask.sum() == 0:
        return np.nan
    return (assigned[mask] != -1).mean()


def run_for_scenario_catalog(df, catalog, catalog_name):
    n = len(df)
    rng = np.random.default_rng(RANDOM_SEED)
    scenarios = [1, 2, 3]
    diffs = {s: [] for s in scenarios}

    print(f"\nRunning {N_BOOTSTRAP} paired cluster-bootstrap replicates (resampling persons) "
          f"per insider-scenario -- catalog {catalog_name}...")
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)  # resample PERSONS with replacement
        sample = df.iloc[idx].reset_index(drop=True)

        base_assigned = greedy_allocate(threat_scores(sample, BASELINE_WEIGHTS), BUDGET, catalog)
        cal_assigned = greedy_allocate(threat_scores(sample, CALIBRATED_WEIGHTS), BUDGET, catalog)

        for s in scenarios:
            base_rate = mitigation_rate_by_scenario(sample, base_assigned, s)
            cal_rate = mitigation_rate_by_scenario(sample, cal_assigned, s)
            diffs[s].append(cal_rate - base_rate)

        if (b + 1) % 200 == 0:
            print(f"  ...{b + 1}/{N_BOOTSTRAP}")

    print("\n" + "=" * 70)
    print(f"CATE BY SCENARIO -- catalog {catalog_name} -- paste into Results chapter")
    print("=" * 70)
    print(f"BUDGET used for this run: {BUDGET}")
    print("=" * 70)
    scenario_labels = {1: "Scenario 1 (rapid theft before leaving)",
                        2: "Scenario 2 (gradual long-running theft)",
                        3: "Scenario 3 (short-burst sabotage)"}
    for s in scenarios:
        d = np.array(diffs[s])
        d = d[~np.isnan(d)]
        if len(d) == 0:
            print(f"{scenario_labels[s]}: no valid replicates (too few malicious cases in this scenario)")
            continue
        ate = d.mean()
        ci_low, ci_high = np.percentile(d, [2.5, 97.5])
        print(f"{scenario_labels[s]} [{len(d)}/{N_BOOTSTRAP} valid replicates]:")
        if abs(ci_high - ci_low) < 0.001:
            print(f"    No detectable effect at this sample size (point estimate {ate:+.4f}).")
        else:
            print(f"    CATE = {ate:+.4f}, 95% CI [{ci_low:+.4f}, {ci_high:+.4f}]")
    print("=" * 70)

    out_rows = []
    for s in scenarios:
        for v in diffs[s]:
            out_rows.append({"scenario": s, "difference": v})
    pd.DataFrame(out_rows).to_csv(f"cate_by_scenario_results_{catalog_name}.csv", index=False)
    print(f"Saved: cate_by_scenario_results_{catalog_name}.csv")


def main():
    df = load_and_aggregate_to_person_level_with_scenario()
    for catalog_name, catalog in CATALOG_SCENARIOS.items():
        print("\n" + "#" * 70)
        print(f"CATALOG SCENARIO: {catalog_name}")
        print("#" * 70)
        run_for_scenario_catalog(df, catalog, catalog_name)


if __name__ == "__main__":
    main()
