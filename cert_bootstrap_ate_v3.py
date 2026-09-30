"""
Weight-Set Effect on Threat-Mitigation Rate -- Paired Bootstrap ATE -- v3
-----------------------------------------------------------------------
v2 change from the original cert_bootstrap_ate.py: resamples PERSONS
with replacement instead of individual user-week rows (Q6, cluster
bootstrap fix). The original version treated 67,238 user-week rows as
independent observations, when 322 malicious rows actually came from
only 70 distinct people -- resampling rows ignored this clustering and
likely understated the true sampling variance (a narrower CI than the
data actually supports). Since this pipeline now aggregates to one row
per person (Q17, granularity fix -- same change already made in
cert_sa_greedy_pipeline_v4.py), resampling persons with replacement IS
the cluster bootstrap: each resampled unit is a complete person, so
there is no within-person clustering left for the resampling to ignore.

Also carries the same catalog fix as v4 (two Training-capacity
scenarios, Section 2.7) and the same GroupKFold-corrected Confidence
input file.

Answers RQ2/Objective 3: does switching from baseline (equal) weights to
AHP-calibrated weights change the rate at which genuinely malicious
PERSONS receive a resource allocation, under a fixed budget?

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

BEFORE RUNNING: edit DATA_PATH and INSIDERS_CSV below (same files as
cert_sa_greedy_pipeline_v4.py).
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
BUDGET = 7000.0
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

CALIBRATED_WEIGHTS = {"impact": 0.224, "propagation": 0.252, "confidence": 0.247, "evasion": 0.278}
BASELINE_WEIGHTS = {"impact": 0.25, "propagation": 0.25, "confidence": 0.25, "evasion": 0.25}

# Same two-scenario catalog as cert_sa_greedy_pipeline_v4.py (Section 2.7)
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


# ----------------------------------------------------------------------
# 1. LOAD + RE-AGGREGATE TO PERSON LEVEL (same as cert_sa_greedy_pipeline_v4.py)
# ----------------------------------------------------------------------
def load_and_aggregate_to_person_level():
    df = pd.read_csv(DATA_PATH)
    insiders = pd.read_csv(INSIDERS_CSV)
    insiders["dataset"] = insiders["dataset"].astype(str).str.strip()
    r42_insider_users = set(insiders[insiders["dataset"] == "4.2"]["user"].unique())

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

    for col in ["impact", "propagation", "evasion"]:
        lo, hi = agg[col].min(), agg[col].max()
        agg[col] = (agg[col] - lo) / (hi - lo) if hi > lo else 0.0

    print(f"Aggregated {len(df):,} user-week records to {len(agg):,} person-level records.")
    print(f"Malicious persons: {agg['malicious'].sum():,} of {len(agg):,}")
    return agg


# ----------------------------------------------------------------------
# 2. GREEDY ALLOCATION (same logic as cert_sa_greedy_pipeline_v4.py --
#    SA is not re-run here, since it was already shown to add nothing
#    over greedy at full scale in both catalog scenarios)
# ----------------------------------------------------------------------
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


def mitigation_rate(df, assigned):
    malicious_mask = df["malicious"].to_numpy() == 1
    if malicious_mask.sum() == 0:
        return np.nan
    covered = assigned[malicious_mask] != -1
    return covered.mean()


# ----------------------------------------------------------------------
# 3. PAIRED CLUSTER BOOTSTRAP ATE (resamples PERSONS, not rows)
# ----------------------------------------------------------------------
def run_bootstrap_for_scenario(df, catalog, scenario_name):
    n = len(df)
    rng = np.random.default_rng(RANDOM_SEED)

    diffs = []
    baseline_rates, calibrated_rates = [], []

    print(f"\nRunning {N_BOOTSTRAP} paired cluster-bootstrap replicates (resampling persons) "
          f"-- scenario {scenario_name}...")
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)  # resample PERSONS with replacement
        sample = df.iloc[idx].reset_index(drop=True)

        base_scores = threat_scores(sample, BASELINE_WEIGHTS)
        cal_scores = threat_scores(sample, CALIBRATED_WEIGHTS)

        base_assigned = greedy_allocate(base_scores, BUDGET, catalog)
        cal_assigned = greedy_allocate(cal_scores, BUDGET, catalog)

        base_rate = mitigation_rate(sample, base_assigned)
        cal_rate = mitigation_rate(sample, cal_assigned)

        baseline_rates.append(base_rate)
        calibrated_rates.append(cal_rate)
        diffs.append(cal_rate - base_rate)

        if (b + 1) % 200 == 0:
            print(f"  ...{b + 1}/{N_BOOTSTRAP} replicates done")

    diffs = np.array(diffs)
    diffs_clean = diffs[~np.isnan(diffs)]

    ate = diffs_clean.mean()
    ci_low, ci_high = np.percentile(diffs_clean, [2.5, 97.5])

    print("\n" + "=" * 70)
    print(f"RESULTS -- scenario {scenario_name} -- paste into Results chapter / RQ2 write-up")
    print("=" * 70)
    print(f"BUDGET used for this run: {BUDGET}")
    print(f"Mean baseline mitigation rate:   {np.nanmean(baseline_rates):.4f}")
    print(f"Mean calibrated mitigation rate: {np.nanmean(calibrated_rates):.4f}")
    print(f"ATE (calibrated - baseline):     {ate:+.4f}")
    print(f"95% cluster-bootstrap CI:        [{ci_low:+.4f}, {ci_high:+.4f}]")
    if ci_low > 0:
        print("-> CI excludes zero: the calibration effect is distinguishable from no effect.")
    elif ci_low <= 0 <= ci_high:
        print("-> CI touches or straddles zero: a borderline/marginal effect -- report it as such.")
    print("NOTE: this bootstrap resamples PERSONS (Q6 cluster-bootstrap fix), not individual")
    print("user-week rows, correctly accounting for the fact that malicious user-weeks are")
    print("clustered within a small number of people rather than independent observations.")
    print("=" * 70)

    pd.DataFrame({
        "baseline_rate": baseline_rates,
        "calibrated_rate": calibrated_rates,
        "difference": np.concatenate([diffs_clean, [np.nan] * (N_BOOTSTRAP - len(diffs_clean))]),
    }).to_csv(f"bootstrap_ate_results_{scenario_name}.csv", index=False)
    print(f"Saved per-replicate results to bootstrap_ate_results_{scenario_name}.csv")


def main():
    df = load_and_aggregate_to_person_level()
    for scenario_name, catalog in CATALOG_SCENARIOS.items():
        print("\n" + "#" * 70)
        print(f"CATALOG SCENARIO: {scenario_name}")
        print("#" * 70)
        run_bootstrap_for_scenario(df, catalog, scenario_name)


if __name__ == "__main__":
    main()
