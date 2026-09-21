"""
Computational Timing -- Greedy Allocation, SA Refinement, Classifier
-----------------------------------------------------------------------
Measures real wall-clock time for the three computational components
that matter for a "Computational Efficiency" subsection:
  1. Greedy allocation (baseline vs. calibrated weights)
  2. SA refinement (5 seeds x 10,000 iterations, baseline vs. calibrated)
  3. Classification Confidence classifier: 5-fold CV training time, and
     per-instance inference time (final full-fit model)

Reports mean +/- SD across repeated timed runs where repetition is cheap
(greedy, classifier inference), and single measured wall-clock time where
repetition would be expensive (SA, CV training) -- noted per section.

BEFORE RUNNING: edit DATA_PATH and INSIDERS_CSV below (same files as your
other CERT-side scripts).
"""

import time
import numpy as np
import pandas as pd

DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_feature_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
BUDGET = 7000.0

CALIBRATED_WEIGHTS = {"impact": 0.224, "propagation": 0.252, "confidence": 0.247, "evasion": 0.278}
BASELINE_WEIGHTS = {"impact": 0.25, "propagation": 0.25, "confidence": 0.25, "evasion": 0.25}

RESOURCE_CATALOG = pd.DataFrame([
    {"resource_id": "training",      "cost": 1,  "effectiveness": 0.15, "capacity": 100_000},
    {"resource_id": "monitoring",    "cost": 3,  "effectiveness": 0.30, "capacity": 5_000},
    {"resource_id": "access_review", "cost": 5,  "effectiveness": 0.45, "capacity": 1_000},
    {"resource_id": "dlp",           "cost": 8,  "effectiveness": 0.55, "capacity": 200},
    {"resource_id": "ir",            "cost": 12, "effectiveness": 0.70, "capacity": 50},
])

SA_PARAMS = dict(initial_temp=1000.0, cooling_rate=0.95, min_temp=0.01, max_iter=10000)
SA_SEEDS = [42, 123, 456, 789, 1010]


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


def threat_scores(df, weights):
    return (
        weights["impact"] * df["impact"] + weights["propagation"] * df["propagation"]
        + weights["confidence"] * df["confidence"] + weights["evasion"] * df["evasion"]
    ).to_numpy()


def greedy_allocate(threat_score, budget):
    n = len(threat_score)
    costs = RESOURCE_CATALOG["cost"].to_numpy()
    effs = RESOURCE_CATALOG["effectiveness"].to_numpy()
    caps = RESOURCE_CATALOG["capacity"].to_numpy()
    n_res = len(RESOURCE_CATALOG)
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


def sa_allocate(threat_score, budget, initial_assigned, initial_temp, cooling_rate, min_temp, max_iter, seed):
    rng = np.random.default_rng(seed)
    n = len(threat_score)
    costs = RESOURCE_CATALOG["cost"].to_numpy()
    effs = RESOURCE_CATALOG["effectiveness"].to_numpy()
    caps = RESOURCE_CATALOG["capacity"].to_numpy()
    n_res = len(RESOURCE_CATALOG)

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


def main():
    df = load_labeled_data()
    n = len(df)
    print(f"Loaded {n:,} threats.\n")

    results = {}

    # ---- 1. Greedy allocation timing (5 repeats each, mean +/- SD) ----
    print("=" * 70)
    print("1. GREEDY ALLOCATION TIMING")
    print("=" * 70)
    for label, weights in [("baseline", BASELINE_WEIGHTS), ("calibrated", CALIBRATED_WEIGHTS)]:
        scores = threat_scores(df, weights)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            greedy_allocate(scores, BUDGET)
            times.append(time.perf_counter() - t0)
        times = np.array(times)
        print(f"  {label}: {times.mean():.4f}s +/- {times.std():.4f}s  (5 repeats)")
        results[f"greedy_{label}_mean_s"] = times.mean()
        results[f"greedy_{label}_sd_s"] = times.std()

    # ---- 2. SA refinement timing (single measured run per weight set,
    #    since each run is already 5 seeds x 10,000 iterations -- repeating
    #    this 5x would take 25x longer for a very similar number) ----
    print("\n" + "=" * 70)
    print("2. SA REFINEMENT TIMING (5 seeds x 10,000 iterations each)")
    print("=" * 70)
    for label, weights in [("baseline", BASELINE_WEIGHTS), ("calibrated", CALIBRATED_WEIGHTS)]:
        scores = threat_scores(df, weights)
        g_assigned = greedy_allocate(scores, BUDGET)
        t0 = time.perf_counter()
        for seed in SA_SEEDS:
            sa_allocate(scores, BUDGET, initial_assigned=g_assigned, seed=seed, **SA_PARAMS)
        elapsed = time.perf_counter() - t0
        print(f"  {label}: {elapsed:.4f}s total for 5 seeds ({elapsed/5:.4f}s/seed)")
        results[f"sa_{label}_total_s"] = elapsed
        results[f"sa_{label}_per_seed_s"] = elapsed / 5

    # ---- 3. Classifier training + inference timing ----
    print("\n" + "=" * 70)
    print("3. CLASSIFICATION CONFIDENCE CLASSIFIER TIMING")
    print("=" * 70)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    feature_cols = ["impact", "propagation", "evasion"]
    X = df[feature_cols].fillna(0.0).to_numpy()
    y = df["malicious"].to_numpy()

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=6,
                                  random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    t0 = time.perf_counter()
    cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    cv_elapsed = time.perf_counter() - t0
    print(f"  5-fold CV training + out-of-fold prediction (total): {cv_elapsed:.4f}s")
    print(f"  ~{cv_elapsed/5:.4f}s per fold")
    results["classifier_cv_total_s"] = cv_elapsed
    results["classifier_cv_per_fold_s"] = cv_elapsed / 5

    # Final full-fit model, for inference timing
    clf_final = RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=6,
                                        random_state=42, n_jobs=-1)
    clf_final.fit(X, y)
    t0 = time.perf_counter()
    clf_final.predict_proba(X)
    infer_elapsed = time.perf_counter() - t0
    per_instance_ms = (infer_elapsed / len(X)) * 1000
    print(f"  Inference on {len(X):,} instances: {infer_elapsed:.4f}s total, {per_instance_ms:.5f} ms/instance")
    results["inference_total_s"] = infer_elapsed
    results["inference_per_instance_ms"] = per_instance_ms
    results["n_threats"] = len(X)

    print("\n" + "=" * 70)
    print("Paste all of the above into your Results \u00a73.8.")
    print("=" * 70)

    out_df = pd.DataFrame([results])
    out_df.to_csv("cert_timing_results.csv", index=False)
    print("\nSaved: cert_timing_results.csv")


if __name__ == "__main__":
    main()
