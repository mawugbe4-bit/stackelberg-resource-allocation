"""
Computational Timing -- v2
-----------------------------------------------------------------------
v2 changes from v1, matching every other pipeline correction this session:
  - Person-level aggregation (Q17): timing now reflects the REAL pipeline,
    which operates on ~1,000 person-level records, not 67,238 user-week
    records -- the old v1 timings measured a pipeline that no longer
    matches what is actually run.
  - Two-scenario catalog (Section 2.7): both Training-capacity scenarios
    timed separately, since they are both part of the real reported
    results now.
  - 10 seeds, not 5 (Q8).
  - Classifier timing uses StratifiedGroupKFold, matching the actual
    GroupKFold-corrected classifier (Q9iii/GATE-2), not plain
    StratifiedKFold.
  - Corrected DATA_PATH (plural "features").

BEFORE RUNNING: edit DATA_PATH and INSIDERS_CSV below (same files as your
other CERT-side scripts).
"""

import time
import numpy as np
import pandas as pd

DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
BUDGET = 7000.0

CALIBRATED_WEIGHTS = {"impact": 0.224, "propagation": 0.252, "confidence": 0.247, "evasion": 0.278}
BASELINE_WEIGHTS = {"impact": 0.25, "propagation": 0.25, "confidence": 0.25, "evasion": 0.25}

CATALOG_SCENARIOS = {
    "A_training_unconstrained": pd.DataFrame([
        {"resource_id": "training",      "cost": 24.0,  "effectiveness": 0.0428, "capacity": 100_000},
        {"resource_id": "monitoring",    "cost": 36.0,  "effectiveness": 0.0473, "capacity": 10},
        {"resource_id": "access_review", "cost": 114.0, "effectiveness": 0.0428, "capacity": 5},
        {"resource_id": "dlp",           "cost": 63.2,  "effectiveness": 0.0507, "capacity": 100_000},
        {"resource_id": "ir",            "cost": 15.0,  "effectiveness": 0.5991, "capacity": 10},
    ]),
    "B_training_constrained": pd.DataFrame([
        {"resource_id": "training",      "cost": 24.0,  "effectiveness": 0.0428, "capacity": 5},
        {"resource_id": "monitoring",    "cost": 36.0,  "effectiveness": 0.0473, "capacity": 10},
        {"resource_id": "access_review", "cost": 114.0, "effectiveness": 0.0428, "capacity": 5},
        {"resource_id": "dlp",           "cost": 63.2,  "effectiveness": 0.0507, "capacity": 100_000},
        {"resource_id": "ir",            "cost": 15.0,  "effectiveness": 0.5991, "capacity": 10},
    ]),
}

SA_PARAMS = dict(initial_temp=1000.0, cooling_rate=0.95, min_temp=0.01, max_iter=10000)
SA_SEEDS = [42, 123, 456, 789, 1010, 2024, 3141, 5926, 5358, 9793]  # 10 seeds, per Q8


def load_and_aggregate_to_person_level():
    df = pd.read_csv(DATA_PATH)
    insiders = pd.read_csv(INSIDERS_CSV)
    insiders["dataset"] = insiders["dataset"].astype(str).str.strip()
    r42_insider_users = set(insiders[insiders["dataset"] == "4.2"]["user"].unique())

    split = df["threat_id"].str.split("_", n=1, expand=True)
    df["user"] = split[0]

    agg = df.groupby("user").agg(
        impact=("impact", "sum"), propagation=("propagation", "sum"),
        evasion=("evasion", "sum"), confidence=("confidence", "mean"),
    ).reset_index()
    agg["malicious"] = agg["user"].isin(r42_insider_users).astype(int)

    for col in ["impact", "propagation", "evasion"]:
        lo, hi = agg[col].min(), agg[col].max()
        agg[col] = (agg[col] - lo) / (hi - lo) if hi > lo else 0.0
    return agg, df  # also return the raw user-week df for classifier timing


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
                        if old_r != -1: used_capacity[old_r] -= 1
                        if new_r != -1: used_capacity[new_r] += 1
                        assigned[t_idx] = new_r
                        spent += delta_cost
                        value += delta_value
                        if value > best_value: best_assigned, best_value = assigned.copy(), value
        else:
            i, j = rng.choice(pool, size=2, replace=False) if len(pool) >= 2 else (0, 0)
            if i != j:
                ri, rj = assigned[i], assigned[j]
                if ri != rj:
                    delta_value = (pair_value(i, rj) + pair_value(j, ri)) - (pair_value(i, ri) + pair_value(j, rj))
                    if delta_value > 0 or rng.random() < np.exp(delta_value / max(temp, 1e-9)):
                        assigned[i], assigned[j] = rj, ri
                        value += delta_value
                        if value > best_value: best_assigned, best_value = assigned.copy(), value
        temp *= cooling_rate
        it += 1
    return best_assigned, best_value


def main():
    df_person, df_week = load_and_aggregate_to_person_level()
    n = len(df_person)
    print(f"Loaded {len(df_week):,} user-week records, aggregated to {n:,} person-level records.\n")

    results = {}

    for catalog_name, catalog in CATALOG_SCENARIOS.items():
        print("=" * 70)
        print(f"CATALOG SCENARIO: {catalog_name}")
        print("=" * 70)
        for label, weights in [("baseline", BASELINE_WEIGHTS), ("calibrated", CALIBRATED_WEIGHTS)]:
            scores = threat_scores(df_person, weights)
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                greedy_allocate(scores, BUDGET, catalog)
                times.append(time.perf_counter() - t0)
            times = np.array(times)
            print(f"  Greedy [{label}]: {times.mean():.5f}s +/- {times.std():.5f}s (5 repeats, n={n:,} persons)")
            results[f"greedy_{catalog_name}_{label}_mean_s"] = times.mean()
            results[f"greedy_{catalog_name}_{label}_sd_s"] = times.std()

            g_assigned = greedy_allocate(scores, BUDGET, catalog)
            t0 = time.perf_counter()
            for seed in SA_SEEDS:
                sa_allocate(scores, BUDGET, initial_assigned=g_assigned, seed=seed, catalog=catalog, **SA_PARAMS)
            elapsed = time.perf_counter() - t0
            print(f"  SA [{label}]: {elapsed:.5f}s total for {len(SA_SEEDS)} seeds ({elapsed/len(SA_SEEDS):.5f}s/seed)")
            results[f"sa_{catalog_name}_{label}_total_s"] = elapsed
            results[f"sa_{catalog_name}_{label}_per_seed_s"] = elapsed / len(SA_SEEDS)
        print()

    print("=" * 70)
    print("CLASSIFICATION CONFIDENCE CLASSIFIER TIMING (GroupKFold, matching actual classifier)")
    print("=" * 70)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    feature_cols = ["impact", "propagation", "evasion"]
    X = df_week[feature_cols].fillna(0.0).to_numpy()
    groups = df_week["user"].to_numpy()
    insiders = pd.read_csv(INSIDERS_CSV)
    insiders["dataset"] = insiders["dataset"].astype(str).str.strip()
    r42_users = set(insiders[insiders["dataset"] == "4.2"]["user"].unique())
    y = df_week["user"].isin(r42_users).astype(int).to_numpy()

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=6, random_state=42, n_jobs=-1)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    t0 = time.perf_counter()
    cross_val_predict(clf, X, y, cv=sgkf, groups=groups, method="predict_proba")
    cv_elapsed = time.perf_counter() - t0
    print(f"  5-fold GROUP-stratified CV training + out-of-fold prediction (total): {cv_elapsed:.4f}s")
    print(f"  ~{cv_elapsed/5:.4f}s per fold")
    results["classifier_cv_total_s"] = cv_elapsed
    results["classifier_cv_per_fold_s"] = cv_elapsed / 5

    clf_final = RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=6, random_state=42, n_jobs=-1)
    clf_final.fit(X, y)
    t0 = time.perf_counter()
    clf_final.predict_proba(X)
    infer_elapsed = time.perf_counter() - t0
    per_instance_ms = (infer_elapsed / len(X)) * 1000
    print(f"  Inference on {len(X):,} user-week instances: {infer_elapsed:.4f}s total, {per_instance_ms:.5f} ms/instance")
    results["inference_total_s"] = infer_elapsed
    results["inference_per_instance_ms"] = per_instance_ms
    results["n_user_weeks"] = len(X)
    results["n_persons"] = n

    print("\n" + "=" * 70)
    print("Paste all of the above into your Results section.")
    print("=" * 70)

    pd.DataFrame([results]).to_csv("cert_timing_results_v2.csv", index=False)
    print("\nSaved: cert_timing_results_v2.csv")


if __name__ == "__main__":
    main()
