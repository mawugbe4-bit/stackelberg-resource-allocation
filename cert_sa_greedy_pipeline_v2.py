"""
AHP-Calibrated Stackelberg-SA Allocation Pipeline -- v2 (resource catalog)
----------------------------------------------------------------------------
Assigns AT MOST ONE resource from a 5-item catalog to each threat
(user-week), under a total budget, comparing:
  - baseline weights (equal, 0.25 each) vs. calibration-wave AHP weights
  - greedy heuristic vs. Simulated Annealing

Prints an environment block for Methods Section 2.9, and saves full
results to allocation_results.json.

WHAT TO EDIT BEFORE RUNNING
----------------------------
1. DATA_PATH  -> your threat-level feature file (one row per user-week,
                 with impact/propagation/confidence/evasion columns) --
                 the output of cert_confidence_classifier.py.
2. BUDGET     -> total resource budget for one allocation round. This is
                 a real modelling decision (what a semester's security
                 budget looks like in these cost units) -- the value
                 below is illustrative, not a recommendation. Decide
                 this with your supervisor.
"""

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 0. CONFIG -- edit this block for your real run
# ----------------------------------------------------------------------
DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_feature_with_confidence.csv"  # <-- your file
BUDGET = 5000.0   # <-- illustrative only; decide the real number with your supervisor

# AHP calibration-wave weights, computed from the real survey (Table 1)
CALIBRATED_WEIGHTS = {"impact": 0.224, "propagation": 0.252, "confidence": 0.247, "evasion": 0.278}

# "No calibration" baseline: equal weighting (Methods Section 2.7)
BASELINE_WEIGHTS = {"impact": 0.25, "propagation": 0.25, "confidence": 0.25, "evasion": 0.25}

# The 5-resource catalog agreed on for this study. `capacity` reflects a
# real operational constraint beyond money: e.g. Incident Response is
# capped by how many skilled investigators the institution has, not just
# budget. This is what makes the allocation problem genuinely combinatorial
# (a multi-dimensional knapsack) rather than one that ratio-sorted greedy
# can solve near-exactly by construction.
RESOURCE_CATALOG = pd.DataFrame([
    {"resource_id": "training",      "name": "Security Awareness Training",           "cost": 1,  "effectiveness": 0.15, "capacity": 100_000},  # scales freely
    {"resource_id": "monitoring",    "name": "Enhanced Logging & Monitoring",          "cost": 3,  "effectiveness": 0.30, "capacity": 5_000},    # SOC review throughput
    {"resource_id": "access_review", "name": "Access Review & Privilege Restriction",  "cost": 5,  "effectiveness": 0.45, "capacity": 1_000},    # IT admin time
    {"resource_id": "dlp",           "name": "Data Loss Prevention tooling",           "cost": 8,  "effectiveness": 0.55, "capacity": 200},      # license seats
    {"resource_id": "ir",            "name": "Dedicated Incident Response",            "cost": 12, "effectiveness": 0.70, "capacity": 50},       # skilled investigators
])

SA_PARAMS = dict(
    initial_temp=1000.0,
    cooling_rate=0.95,
    min_temp=0.01,
    max_iter=10000,
)

RANDOM_SEEDS = [42, 123, 456, 789, 1010]


# ----------------------------------------------------------------------
# 1. ENVIRONMENT CAPTURE -- prints the block to paste into Methods §2.9
# ----------------------------------------------------------------------
def get_package_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "not installed"


def get_ram_gb() -> str:
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(["wmic", "computersystem", "get", "TotalPhysicalMemory"]).decode()
            bytes_ = int([l for l in out.split("\n") if l.strip().isdigit()][0].strip())
            return f"{bytes_ / 1024**3:.1f} GB"
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return f"{int(line.split()[1]) / 1024 / 1024:.1f} GB"
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "hw.memsize"]).decode()
            return f"{int(out.strip().split(':')[1]) / 1024**3:.1f} GB"
    except Exception:
        pass
    return "unknown (record manually)"


def get_cpu_model() -> str:
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode()
            lines = [l.strip() for l in out.split("\n") if l.strip() and "Name" not in l]
            return lines[0] if lines else platform.processor()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def capture_environment() -> dict:
    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_model": get_cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram": get_ram_gb(),
        "numpy_version": get_package_version("numpy"),
        "pandas_version": get_package_version("pandas"),
        "sklearn_version": get_package_version("scikit-learn"),
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def print_methods_2_9_block(env: dict, sa_params: dict) -> None:
    print("\n" + "=" * 70)
    print("PASTE THIS BLOCK INTO METHODS SECTION 2.9")
    print("=" * 70)
    print(
        f"All experiments were conducted on a {env['cpu_model']} "
        f"({env['cpu_count_logical']} logical cores, {env['ram']} RAM) "
        f"running {env['os']}. The software environment used Python "
        f"{env['python_version']}, NumPy {env['numpy_version']}, pandas "
        f"{env['pandas_version']}, and scikit-learn {env['sklearn_version']}. "
        f"Simulated Annealing was configured with an initial temperature of "
        f"{sa_params['initial_temp']:.0f}, a geometric cooling rate of "
        f"{sa_params['cooling_rate']}, a minimum temperature of "
        f"{sa_params['min_temp']}, and a maximum of {sa_params['max_iter']:,} "
        f"iterations. Run captured on {env['captured_at']}."
    )
    print("=" * 70 + "\n")


# ----------------------------------------------------------------------
# 2. THREAT SCORE  U(T_i) = sum(alpha_k * criterion_k)
# ----------------------------------------------------------------------
def threat_scores(df: pd.DataFrame, weights: dict) -> np.ndarray:
    return (
        weights["impact"] * df["impact"]
        + weights["propagation"] * df["propagation"]
        + weights["confidence"] * df["confidence"]
        + weights["evasion"] * df["evasion"]
    ).to_numpy()


# ----------------------------------------------------------------------
# 3. GREEDY: rank every (threat, resource) pair by value/cost, assign
#    greedily, at most one resource per threat, within budget.
# ----------------------------------------------------------------------
def greedy_allocate(threat_score: np.ndarray, budget: float):
    n = len(threat_score)
    costs = RESOURCE_CATALOG["cost"].to_numpy()
    effs = RESOURCE_CATALOG["effectiveness"].to_numpy()
    caps = RESOURCE_CATALOG["capacity"].to_numpy()
    n_res = len(RESOURCE_CATALOG)

    # (n * n_res) candidate pairs -- vectorised value/cost/ratio
    value_matrix = threat_score[:, None] * effs[None, :]           # n x n_res
    cost_matrix = np.tile(costs, (n, 1))                            # n x n_res
    ratio_matrix = value_matrix / cost_matrix

    flat_ratio = ratio_matrix.ravel()
    order = np.argsort(-flat_ratio)
    t_idx_all, r_idx_all = np.unravel_index(order, (n, n_res))

    assigned = np.full(n, -1, dtype=np.int8)
    used_capacity = np.zeros(n_res, dtype=np.int64)
    spent = 0.0
    total_value = 0.0
    for t_idx, r_idx in zip(t_idx_all, r_idx_all):
        if assigned[t_idx] != -1:
            continue
        if used_capacity[r_idx] >= caps[r_idx]:
            continue
        cost = costs[r_idx]
        if spent + cost <= budget:
            assigned[t_idx] = r_idx
            used_capacity[r_idx] += 1
            spent += cost
            total_value += value_matrix[t_idx, r_idx]

    return assigned, total_value, spent


# ----------------------------------------------------------------------
# 4. SIMULATED ANNEALING (greedy-seeded refinement)
#    Starts from the greedy solution rather than empty, then searches for
#    local improvements via single-threat re-rolls. Plain SA started from
#    an empty allocation is known to underperform sorted-greedy on
#    knapsack-style problems like this one unless given a very large
#    search budget; seeding from greedy makes the comparison meaningful
#    (does SA refine on top of greedy?) rather than an artifact of a weak
#    starting point. This reframes the comparison as "greedy alone" vs.
#    "greedy + SA refinement" -- report it that way in your Methods/
#    Results text rather than as two fully independent allocators.
# ----------------------------------------------------------------------
def sa_allocate(
    threat_score: np.ndarray,
    budget: float,
    initial_assigned: np.ndarray,
    initial_temp: float,
    cooling_rate: float,
    min_temp: float,
    max_iter: int,
    seed: int,
):
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

    # --- Identify the "boundary" of the greedy solution: threats whose
    # assignment is only marginal (lowest value among admitted, for each
    # resource) and threats that were excluded despite a decent ratio
    # (would rank highly if any capacity/budget freed up). Any real gap
    # between greedy and optimal, for this kind of constrained knapsack,
    # is mathematically concentrated at exactly this boundary -- so SA's
    # search time is spent where it can actually matter, rather than
    # uniformly across thousands of already-settled items. ---
    per_threat_value_if_assigned = np.array(
        [pair_value(i, r) if r >= 0 else -np.inf for i, r in enumerate(assigned)]
    )
    boundary_pool_size = min(500, n)
    marginal_admitted = np.argsort(per_threat_value_if_assigned)[:boundary_pool_size]
    marginal_admitted = marginal_admitted[assigned[marginal_admitted] != -1]

    best_alt_value = np.full(n, -np.inf)
    for r in range(n_res):
        v = threat_score * effs[r]
        best_alt_value = np.maximum(best_alt_value, np.where(assigned != r, v, -np.inf))
    unassigned_mask = assigned == -1
    marginal_rejected = np.argsort(-np.where(unassigned_mask, best_alt_value, -np.inf))[:boundary_pool_size]

    boundary_pool = np.unique(np.concatenate([marginal_admitted, marginal_rejected]))
    if len(boundary_pool) < 2:
        boundary_pool = np.arange(n)

    temp = initial_temp
    it = 0
    while temp > min_temp and it < max_iter:
        use_boundary = rng.random() < 0.8
        pool = boundary_pool if use_boundary else np.arange(n)

        if rng.random() < 0.5:
            # Move type A: re-roll one threat's assignment.
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
            # Move type B: swap two threats' assigned resources.
            i, j = rng.choice(pool, size=2, replace=False) if len(pool) >= 2 else (0, 0)
            if i != j:
                ri, rj = assigned[i], assigned[j]
                if ri != rj:
                    delta_value = (
                        (pair_value(i, rj) + pair_value(j, ri))
                        - (pair_value(i, ri) + pair_value(j, rj))
                    )
                    if delta_value > 0 or rng.random() < np.exp(delta_value / max(temp, 1e-9)):
                        assigned[i], assigned[j] = rj, ri
                        value += delta_value
                        if value > best_value:
                            best_assigned, best_value = assigned.copy(), value

        temp *= cooling_rate
        it += 1

    return best_assigned, best_value


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------
def main():
    env = capture_environment()
    print_methods_2_9_block(env, SA_PARAMS)

    if not os.path.exists(DATA_PATH):
        print(f"[NOTE] '{DATA_PATH}' not found -- edit DATA_PATH at the top of this script.")
        return

    df = pd.read_csv(DATA_PATH)
    required = {"threat_id", "impact", "propagation", "confidence", "evasion"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing expected columns: {missing}")
        return

    results = {}
    for label, weights in [("baseline", BASELINE_WEIGHTS), ("calibrated", CALIBRATED_WEIGHTS)]:
        scores = threat_scores(df, weights)

        g_assigned, g_value, g_spent = greedy_allocate(scores, BUDGET)
        n_assigned_greedy = int((g_assigned != -1).sum())

        sa_values = []
        for seed in RANDOM_SEEDS:
            _, v = sa_allocate(scores, BUDGET, initial_assigned=g_assigned, seed=seed, **SA_PARAMS)
            sa_values.append(v)
        sa_improvement = float(np.mean(sa_values)) - g_value

        results[label] = {
            "greedy_total_value": float(g_value),
            "greedy_budget_spent": float(g_spent),
            "greedy_n_threats_covered": n_assigned_greedy,
            "sa_refined_value_mean": float(np.mean(sa_values)),
            "sa_refined_value_sd": float(np.std(sa_values)),
            "sa_improvement_over_greedy": sa_improvement,
            "sa_runs": sa_values,
            "n_threats_total": int(len(df)),
            "budget": BUDGET,
        }
        print(f"\n[{label}] greedy value={g_value:.2f} (covers {n_assigned_greedy:,}/{len(df):,} "
              f"threats, spent {g_spent:.0f}/{BUDGET:.0f}) | "
              f"SA-refined mean={np.mean(sa_values):.2f} sd={np.std(sa_values):.2f} "
              f"(+{sa_improvement:.2f} over greedy)")

    with open("allocation_results.json", "w") as f:
        json.dump({"environment": env, "resource_catalog": RESOURCE_CATALOG.to_dict(orient="records"),
                    "results": results}, f, indent=2)
    print("\nSaved: allocation_results.json")


if __name__ == "__main__":
    main()
