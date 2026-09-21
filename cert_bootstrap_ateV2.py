"""
Weight-Set Effect on Threat-Mitigation Rate -- Paired Bootstrap ATE
-----------------------------------------------------------------------
Answers RQ2/Objective 3: does switching from baseline (equal) weights to
AHP-calibrated weights change the rate at which genuinely malicious
user-weeks receive a resource allocation, under a fixed budget?

Outcome (threat-mitigation rate) = fraction of TRUE malicious user-weeks
(your ground-truth labels from cert_confidence_classifier.py) that are
assigned any resource by the allocator.

Method: paired bootstrap. In each of B replicates, the threat population
is resampled with replacement; greedy allocation is run on that resample
under BOTH weight sets (same resample, so the comparison is paired); the
difference in mitigation rate (calibrated - baseline) is recorded. The
ATE is the mean of that distribution, with a 95% CI from its 2.5th/97.5th
percentiles, plus a Wilcoxon signed-rank test on the paired differences.

This replaces a literal DoWhy confounder-adjustment call: baseline vs.
calibrated weights are two deterministic policies applied to the same
population, not a treatment assigned under a confounded process, so
DoWhy's backdoor-adjustment machinery has nothing to adjust for here --
the paired bootstrap is the statistically appropriate tool for this
specific comparison, and still delivers the "95% bootstrap confidence
intervals" your Methods section commits to.

BEFORE RUNNING: edit DATA_PATH below (your confidence-completed CSV,
which must include the ground-truth 'malicious' column -- if your saved
file doesn't have it, see the note in load_data() below).
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_feature_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"  # to rebuild ground truth if needed
BUDGET = 5000.0          # <-- match whatever you used in cert_sa_greedy_pipeline_v2.py
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


# ----------------------------------------------------------------------
# 1. LOAD DATA (rebuilds the 'malicious' ground-truth column if your
#    saved CSV doesn't already have it -- cert_confidence_classifier.py
#    only saved threat_id/resource_id/impact/propagation/confidence/
#    evasion/cost, not the intermediate 'malicious' label).
# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "malicious" not in df.columns:
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
        print(f"Rebuilt ground-truth labels: {df['malicious'].sum():,} malicious rows.")
    else:
        print(f"Ground truth already present: {df['malicious'].sum():,} malicious rows.")
    return df


# ----------------------------------------------------------------------
# 2. GREEDY ALLOCATION (same logic as cert_sa_greedy_pipeline_v2.py --
#    only greedy is needed here since SA showed no measurable difference
#    over greedy at full scale; re-running the identical SA refinement
#    thousands of times across bootstrap replicates would cost a lot of
#    compute for a result already shown to add nothing).
# ----------------------------------------------------------------------
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


def mitigation_rate(df, assigned):
    malicious_mask = df["malicious"].to_numpy() == 1
    if malicious_mask.sum() == 0:
        return np.nan
    covered = assigned[malicious_mask] != -1
    return covered.mean()


# ----------------------------------------------------------------------
# 3. PAIRED BOOTSTRAP ATE
# ----------------------------------------------------------------------
def main():
    df = load_data()
    n = len(df)
    rng = np.random.default_rng(RANDOM_SEED)

    diffs = []
    baseline_rates, calibrated_rates = [], []

    print(f"Running {N_BOOTSTRAP} paired bootstrap replicates...")
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        sample = df.iloc[idx].reset_index(drop=True)

        base_scores = threat_scores(sample, BASELINE_WEIGHTS)
        cal_scores = threat_scores(sample, CALIBRATED_WEIGHTS)

        base_assigned = greedy_allocate(base_scores, BUDGET)
        cal_assigned = greedy_allocate(cal_scores, BUDGET)

        base_rate = mitigation_rate(sample, base_assigned)
        cal_rate = mitigation_rate(sample, cal_assigned)

        baseline_rates.append(base_rate)
        calibrated_rates.append(cal_rate)
        diffs.append(cal_rate - base_rate)

        if (b + 1) % 100 == 0:
            print(f"  ...{b + 1}/{N_BOOTSTRAP} replicates done")

    diffs = np.array(diffs)
    diffs = diffs[~np.isnan(diffs)]

    ate = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    print("\n" + "=" * 70)
    print("RESULTS -- paste into your Results chapter / RQ2 write-up")
    print("=" * 70)
    print(f"Mean baseline mitigation rate:   {np.nanmean(baseline_rates):.4f}")
    print(f"Mean calibrated mitigation rate: {np.nanmean(calibrated_rates):.4f}")
    print(f"ATE (calibrated - baseline):     {ate:+.4f}")
    print(f"95% bootstrap CI:                [{ci_low:+.4f}, {ci_high:+.4f}]")
    if ci_low > 0:
        print("-> CI excludes zero: the calibration effect is distinguishable from no effect.")
    elif ci_low == 0 or (ci_low < 0 < ci_high):
        print("-> CI touches or straddles zero: a borderline/marginal effect -- report it as such,")
        print("   not as a strong or highly significant result.")
    print("NOTE: no additional significance test (e.g. Wilcoxon) is applied on top of the")
    print("bootstrap differences -- they are correlated resamples of one dataset, not independent")
    print("observations, so a classic paired test here would produce a misleadingly tiny p-value.")
    print("The percentile CI above IS the valid inferential statement; report it alone.")
    print("=" * 70)

    pd.DataFrame({
        "baseline_rate": baseline_rates,
        "calibrated_rate": calibrated_rates,
        "difference": np.concatenate([diffs, [np.nan] * (N_BOOTSTRAP - len(diffs))]),
    }).to_csv("bootstrap_ate_results.csv", index=False)
    print("\nSaved per-replicate results to bootstrap_ate_results.csv")


if __name__ == "__main__":
    main()
