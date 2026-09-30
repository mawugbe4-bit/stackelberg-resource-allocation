"""
SA/Greedy Allocation Pipeline -- v5
-----------------------------------------------------------------------
v4 change from v3: DATA_PATH default corrected to match the actual output
filename from the GroupKFold-corrected cert_confidence_classifier.py
(cert_threat_resource_features_with_confidence.csv, plural "features") --
v3's default pointed to a singular-"feature" filename that the classifier
never actually produced, which would have caused a file-not-found error.
No other logic changed from v3.

Rebuilt to resolve two linked audit findings:

  (Q17, granularity mismatch): the allocation unit moves from user-WEEK
  to user (person), since the new resource catalog's costs are priced
  per-person-per-year (real vendor/retainer pricing) -- allocating a
  weekly "Security Awareness Training" assignment made no real-world
  sense, since nobody buys training week-by-week.

  (GATE-3a(2)/GATE-4/Q1/Q20/Q23/Q24, degenerate catalog): the five
  resources' cost/effectiveness/capacity values are replaced with real,
  cited figures (see the accompanying write-up for full sourcing), so
  the value/cost ratio is no longer monotonically decreasing in cost --
  Incident Response is the highest-ratio resource, not the cheapest one.

Also resolves Q8 (seed-count): SA is now run at 10 seeds, not 5, per the
audit's stated minimum for a null result to be reportable.

v5 changes from v4 (two real errors caught by supervisor review, both
fixed):
  1. Person-level aggregation for Impact/Propagation/Evasion changed from
     SUM to MEAN. Summing a weekly proportion (Evasion) is meaningless;
     more generally, summing any of these confounds tenure length with
     behavioural intensity, systematically under-scoring short-tenure
     people including the 30 rapid pre-departure-theft insiders
     (Scenario 1) -- a plausible explanation for that scenario's earlier
     null result, which needs re-checking under this fix.
  2. Incident Response effectiveness corrected from 0.5991 to 0.3378.
     The earlier value used $2.66M, which is IBM's 2022 Cost of a Data
     Breach Report figure, mistakenly cited as 2025 after being carried
     forward uncritically across multiple secondary sources that had
     not re-verified it against the current report. The real 2025 IBM
     figure (confirmed directly from ibm.com) is $1.5M; $1.5M/$4.44M
     (the 2025 global average) = 0.3378.

BEFORE RUNNING: edit DATA_PATH and INSIDERS_CSV below (same files as your
other CERT-side scripts).
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
BUDGET = 7000.0  # unchanged nominal value for comparability; real institutional
                 # budget figure is a SEPARATE open item, not resolved here

CALIBRATED_WEIGHTS = {"impact": 0.224, "propagation": 0.252, "confidence": 0.247, "evasion": 0.278}
BASELINE_WEIGHTS = {"impact": 0.25, "propagation": 0.25, "confidence": 0.25, "evasion": 0.25}

# TWO catalog scenarios for Training's capacity, since no specific real
# source exists for it (Section 2.7 write-up explains this fully):
#   Scenario A: unconstrained (software delivery has no per-assignment
#     scarcity) -- the original reasoning, kept as one arm of the check.
#   Scenario B: constrained by the SAME administrative-bandwidth logic
#     already used for Access Review (NinjaOne: 1 admin per 200 users),
#     since enrolling someone still requires some administrative action
#     even if delivery itself is automated. Capacity = 5, same as Access
#     Review, since both draw on the same real staffing ratio.
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
RESOURCE_CATALOG = CATALOG_SCENARIOS["A_training_unconstrained"]  # default; main() loops both

SA_PARAMS = dict(initial_temp=1000.0, cooling_rate=0.95, min_temp=0.01, max_iter=10000)
SA_SEEDS = [42, 123, 456, 789, 1010, 2024, 3141, 5926, 5358, 9793]  # 10 seeds, per Q8


# ----------------------------------------------------------------------
# 1. LOAD + RE-AGGREGATE TO PERSON LEVEL (resolves Q17)
# ----------------------------------------------------------------------
def load_and_aggregate_to_person_level():
    df = pd.read_csv(DATA_PATH)
    insiders = pd.read_csv(INSIDERS_CSV)
    insiders["dataset"] = insiders["dataset"].astype(str).str.strip()
    r42_insider_users = set(insiders[insiders["dataset"] == "4.2"]["user"].unique())

    split = df["threat_id"].str.split("_", n=1, expand=True)
    df["user"] = split[0]

    # Person-level aggregation choices, corrected (was SUMMED in an earlier
    # version -- flagged as wrong and fixed):
    #  - Impact/Propagation/Evasion: MEANED across all weeks for that user
    #    (average weekly behavioural intensity), NOT summed. Evasion is a
    #    weekly proportion, and summing a proportion across weeks has no
    #    defined meaning as an "annual total". More importantly, summing
    #    ANY of these three features mechanically scales the resulting
    #    score with how many weeks a person was observed -- a real
    #    confound, since the 30 rapid pre-departure-theft insiders
    #    (Scenario 1) have systematically fewer active weeks than everyone
    #    else, and would score lower purely from shorter tenure, not from
    #    genuinely lower behavioural intensity. Using per-week means
    #    removes this confound for all three features consistently.
    #  - Confidence: MEANED across weeks (it is a probability, not a count;
    #    summing probabilities has no defined meaning) -- unchanged, this
    #    one was already correct.
    #  - malicious: 1 if the person is one of the 70 planted r4.2 insiders
    #    (independent of which specific weeks were flagged), 0 otherwise --
    #    this is the natural person-level ground truth, not re-derived from
    #    week-level labels.
    agg = df.groupby("user").agg(
        impact=("impact", "mean"),
        propagation=("propagation", "mean"),
        evasion=("evasion", "mean"),
        confidence=("confidence", "mean"),
    ).reset_index()

    agg["malicious"] = agg["user"].isin(r42_insider_users).astype(int)

    # Re-normalise to [0,1] at the NEW person level, since the original
    # min-max normalisation was fit at the user-week level and does not
    # carry over to person-level means.
    for col in ["impact", "propagation", "evasion"]:
        lo, hi = agg[col].min(), agg[col].max()
        agg[col] = (agg[col] - lo) / (hi - lo) if hi > lo else 0.0

    print(f"Aggregated {len(df):,} user-week records to {len(agg):,} person-level records.")
    print(f"Malicious persons: {agg['malicious'].sum():,} of {len(agg):,}")
    return agg


# ----------------------------------------------------------------------
# 2. GREEDY + SA (same allocation logic as before; only the unit and
#    catalog changed, per Algorithm 1's own note that the calibrated-
#    weight substitution and catalog values are the only intended
#    differences from prior versions)
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
    return assigned, used_capacity, spent


def sa_allocate(threat_score, budget, initial_assigned, initial_temp, cooling_rate, min_temp, max_iter, seed, catalog):
    rng = np.random.default_rng(seed)
    n = len(threat_score)
    costs = catalog["cost"].to_numpy()
    effs = catalog["effectiveness"].to_numpy()
    caps = catalog["capacity"].to_numpy()
    n_res = len(catalog)

    def pair_value(t_idx, r_idx):
        return threat_score[t_idx] * effs[r_idx] if r_idx >= 0 else 0.0

    def pair_cost(r_idx):
        return costs[r_idx] if r_idx >= 0 else 0.0

    assigned = initial_assigned.copy()
    spent = sum(pair_cost(r) for r in assigned)
    value = sum(pair_value(i, r) for i, r in enumerate(assigned))
    used_capacity = np.array([(assigned == r).sum() for r in range(n_res)], dtype=np.int64)
    best_assigned, best_value = assigned.copy(), value

    per_threat_value_if_assigned = np.array(
        [pair_value(i, r) if r >= 0 else -np.inf for i, r in enumerate(assigned)])
    pool_size = min(500, n)
    marginal_admitted = np.argsort(per_threat_value_if_assigned)[:pool_size]
    marginal_admitted = marginal_admitted[assigned[marginal_admitted] != -1]
    best_alt_value = np.full(n, -np.inf)
    for r in range(n_res):
        v = threat_score * effs[r]
        best_alt_value = np.maximum(best_alt_value, np.where(assigned != r, v, -np.inf))
    unassigned_mask = assigned == -1
    marginal_rejected = np.argsort(-np.where(unassigned_mask, best_alt_value, -np.inf))[:pool_size]
    boundary_pool = np.unique(np.concatenate([marginal_admitted, marginal_rejected]))
    if len(boundary_pool) < 2:
        boundary_pool = np.arange(n)

    temp = initial_temp
    it = 0
    while temp > min_temp and it < max_iter:
        use_boundary = rng.random() < 0.8
        pool = boundary_pool if use_boundary else np.arange(n)
        if rng.random() < 0.5:
            t_idx = rng.choice(pool)
            new_r = rng.integers(-1, n_res)
            old_r = assigned[t_idx]
            if new_r != old_r:
                cap_ok = new_r == -1 or used_capacity[new_r] < caps[new_r]
                delta_cost = pair_cost(new_r) - pair_cost(old_r)
                if cap_ok and spent + delta_cost <= budget:
                    delta_value = pair_value(t_idx, new_r) - pair_value(t_idx, old_r)
                    if delta_value > 0 or rng.random() < np.exp(delta_value / max(temp, 1e-9)):
                        if old_r != -1:
                            used_capacity[old_r] -= 1
                        if new_r != -1:
                            used_capacity[new_r] += 1
                        assigned[t_idx] = new_r
                        spent += delta_cost
                        value += delta_value
                        if value > best_value:
                            best_assigned, best_value = assigned.copy(), value
        else:
            i, j = rng.choice(pool, size=2, replace=False) if len(pool) >= 2 else (0, 0)
            if i != j:
                ri, rj = assigned[i], assigned[j]
                if ri != rj:
                    delta_value = (pair_value(i, rj) + pair_value(j, ri)) - (pair_value(i, ri) + pair_value(j, rj))
                    if delta_value > 0 or rng.random() < np.exp(delta_value / max(temp, 1e-9)):
                        assigned[i], assigned[j] = rj, ri
                        value += delta_value
                        if value > best_value:
                            best_assigned, best_value = assigned.copy(), value
        temp *= cooling_rate
        it += 1
    return best_assigned, best_value


def resource_usage_report(assigned, label, catalog):
    n_res = len(catalog)
    print(f"  Resource usage under {label}:")
    for r in range(n_res):
        n_used = (assigned == r).sum()
        rid = catalog.iloc[r]["resource_id"]
        cap = catalog.iloc[r]["capacity"]
        print(f"    {rid:15} used {n_used:>7,} of capacity {cap:>7,}")
    n_unassigned = (assigned == -1).sum()
    print(f"    {'(unassigned)':15} {n_unassigned:>7,}")


def main():
    df = load_and_aggregate_to_person_level()
    n = len(df)

    for scenario_name, catalog in CATALOG_SCENARIOS.items():
        print("\n" + "#" * 70)
        print(f"CATALOG SCENARIO: {scenario_name}")
        print("#" * 70)

        for label, weights in [("baseline", BASELINE_WEIGHTS), ("calibrated", CALIBRATED_WEIGHTS)]:
            print("\n" + "=" * 70)
            print(f"WEIGHT SET: {label}")
            print("=" * 70)
            scores = threat_scores(df, weights)

            g_assigned, g_used_cap, g_spent = greedy_allocate(scores, BUDGET, catalog)
            g_value = sum(scores[i] * catalog.iloc[r]["effectiveness"]
                          for i, r in enumerate(g_assigned) if r != -1)
            print(f"Greedy: value={g_value:.4f}, spent={g_spent:.1f}/{BUDGET}")
            resource_usage_report(g_assigned, "greedy", catalog)

            sa_values = []
            for seed in SA_SEEDS:
                sa_assigned, sa_value = sa_allocate(scores, BUDGET, initial_assigned=g_assigned,
                                                      seed=seed, catalog=catalog, **SA_PARAMS)
                sa_values.append(sa_value)
            sa_values = np.array(sa_values)
            print(f"SA (n={len(SA_SEEDS)} seeds): mean={sa_values.mean():.4f}, "
                  f"SD={sa_values.std():.4f}, min={sa_values.min():.4f}, max={sa_values.max():.4f}")
            print(f"SA improvement over greedy: {sa_values.mean() - g_value:+.6f}")

    print("\n" + "=" * 70)
    print("Paste all of the above into your Results chapter / supervisor update.")
    print("=" * 70)


if __name__ == "__main__":
    main()
