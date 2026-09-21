"""
CATE by Threat Scenario Type
-------------------------------
Extends the paired-bootstrap ATE (cert_bootstrap_ateV2.py) to check
whether the AHP-calibration effect on threat-mitigation rate is uniform
across the three r4.2 insider scenarios, or concentrated in one:
  Scenario 1: rapid data theft shortly before leaving the organisation
  Scenario 2: gradual, long-running data theft
  Scenario 3: short-burst sabotage-style activity

This is real, available heterogeneity (from insiders.csv's own 'scenario'
column) -- unlike institution type, which has no CERT-threat-level
counterpart (see ahp_weights_by_institution.py for the valid place to
test institution-type heterogeneity: in the AHP weights themselves).

Small-n warning: scenario 3 has only 10 insiders (vs. 30 each for
scenarios 1 and 2). Its confidence interval will be wide -- report that
honestly rather than reading too much into a single scenario's estimate.

BEFORE RUNNING: edit the paths below (same files as cert_bootstrap_ateV2.py).
"""

import numpy as np
import pandas as pd

DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_feature_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
BUDGET = 5000.0          # <-- must match the budget you used for the primary ATE result
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

CALIBRATED_WEIGHTS = {"impact": 0.224, "propagation": 0.252, "confidence": 0.247, "evasion": 0.278}
BASELINE_WEIGHTS = {"impact": 0.25, "propagation": 0.25, "confidence": 0.25, "evasion": 0.25}

RESOURCE_CATALOG = pd.DataFrame([
    {"resource_id": "training",      "cost": 1,  "effectiveness": 0.15, "capacity": 100_000},
    {"resource_id": "monitoring",    "cost": 3,  "effectiveness": 0.30, "capacity": 5_000},
    {"resource_id": "access_review", "cost": 5,  "effectiveness": 0.45, "capacity": 1_000},
    {"resource_id": "dlp",           "cost": 8,  "effectiveness": 0.55, "capacity": 200},
    {"resource_id": "ir",            "cost": 12, "effectiveness": 0.70, "capacity": 50},
])


def load_data_with_scenario():
    df = pd.read_csv(DATA_PATH)
    insiders = pd.read_csv(INSIDERS_CSV)
    insiders["dataset"] = insiders["dataset"].astype(str).str.strip()
    r42 = insiders[insiders["dataset"] == "4.2"].copy()
    r42["start"] = pd.to_datetime(r42["start"])
    r42["end"] = pd.to_datetime(r42["end"])

    windows_by_user = {}
    scenario_by_user = {}
    for _, row in r42.iterrows():
        windows_by_user.setdefault(row["user"], []).append((row["start"], row["end"]))
        scenario_by_user[row["user"]] = int(row["scenario"])

    split = df["threat_id"].str.split("_", n=1, expand=True)
    df["user"] = split[0]
    week_range = split[1].str.split("/", expand=True)
    df["week_start"] = pd.to_datetime(week_range[0])
    df["week_end"] = pd.to_datetime(week_range[1])

    def label(row):
        windows = windows_by_user.get(row["user"])
        if not windows:
            return 0, -1
        for w_start, w_end in windows:
            if row["week_start"] <= w_end and row["week_end"] >= w_start:
                return 1, scenario_by_user[row["user"]]
        return 0, -1

    labels = df.apply(label, axis=1, result_type="expand")
    df["malicious"] = labels[0]
    df["scenario"] = labels[1]

    counts = df[df["malicious"] == 1]["scenario"].value_counts().sort_index()
    print(f"Malicious user-weeks by scenario: {dict(counts)}")
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


def mitigation_rate_by_scenario(df, assigned, scenario):
    mask = (df["malicious"].to_numpy() == 1) & (df["scenario"].to_numpy() == scenario)
    if mask.sum() == 0:
        return np.nan
    return (assigned[mask] != -1).mean()


def main():
    df = load_data_with_scenario()
    n = len(df)
    rng = np.random.default_rng(RANDOM_SEED)
    scenarios = [1, 2, 3]

    diffs = {s: [] for s in scenarios}

    print(f"\nRunning {N_BOOTSTRAP} paired bootstrap replicates per scenario...")
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx].reset_index(drop=True)

        base_assigned = greedy_allocate(threat_scores(sample, BASELINE_WEIGHTS), BUDGET)
        cal_assigned = greedy_allocate(threat_scores(sample, CALIBRATED_WEIGHTS), BUDGET)

        for s in scenarios:
            base_rate = mitigation_rate_by_scenario(sample, base_assigned, s)
            cal_rate = mitigation_rate_by_scenario(sample, cal_assigned, s)
            diffs[s].append(cal_rate - base_rate)

        if (b + 1) % 200 == 0:
            print(f"  ...{b + 1}/{N_BOOTSTRAP}")

    print("\n" + "=" * 70)
    print("CATE BY SCENARIO -- paste into your Results chapter")
    print("=" * 70)
    scenario_labels = {1: "Scenario 1 (rapid theft before leaving)",
                        2: "Scenario 2 (gradual long-running theft)",
                        3: "Scenario 3 (short-burst sabotage)"}
    n_insiders = {1: 30, 2: 30, 3: 10}
    for s in scenarios:
        d = np.array(diffs[s])
        d = d[~np.isnan(d)]
        if len(d) == 0:
            print(f"{scenario_labels[s]}: no valid replicates (too few malicious cases in this scenario)")
            continue
        ate = d.mean()
        ci_low, ci_high = np.percentile(d, [2.5, 97.5])
        print(f"{scenario_labels[s]} [n={n_insiders[s]} insiders, {len(d)}/{N_BOOTSTRAP} valid replicates]:")
        print(f"    CATE = {ate:+.4f}, 95% CI [{ci_low:+.4f}, {ci_high:+.4f}]")
    print("=" * 70)
    print("Small-n caveat: Scenario 3's estimate is based on only 10 insiders and will")
    print("have a much wider interval than Scenarios 1/2 -- this reflects real data")
    print("scarcity, not an error. Do not treat a wide interval as 'no effect'; treat it")
    print("as 'not enough data to distinguish an effect from noise in this subgroup'.")

    out_rows = []
    for s in scenarios:
        for v in diffs[s]:
            out_rows.append({"scenario": s, "difference": v})
    pd.DataFrame(out_rows).to_csv("cate_by_scenario_results.csv", index=False)
    print("\nSaved: cate_by_scenario_results.csv")


if __name__ == "__main__":
    main()
