# AHP-Calibrated Stackelberg-SA Cybersecurity Resource Allocation

Code accompanying the PhD thesis *"A Stackelberg Game-Theoretic Model for
Optimizing Cybersecurity Resource Allocation in Resource-Constrained
Organizations: A Causal ML and Simulated Annealing Approach"* (Dennis
Redeemer Korda, PG5145120, Department of Computer Science, Kwame Nkrumah
University of Science and Technology, Kumasi).

## Repository history note (read this first)

An earlier commit to this repository (21 September) contained only a
partial, now-superseded version of this pipeline — the person-level
aggregation, the corrected resource catalog, GroupKFold classifier
correction, cluster bootstrap, 10-seed SA, and both AHP sensitivity
checks were developed after that commit but were not pushed at the time.
**This commit replaces that gap with the complete, current pipeline.**
Every script below is the exact version that produced the numbers in the
current Methods and Results drafts.

## Pipeline order

| # | Script | What it does | Input | Output |
|---|---|---|---|---|
| 1 | `cert_local_aggregation.py` | Streams the CERT r4.2 archive and aggregates raw logon/device/email/file/http records into per-user-week behavioural features | Raw CERT `.tar.bz2` archive | `cert_threat_resource_features.csv` |
| 2 | `cert_confidence_classifier.py` | Labels each user-week against the CERT ground-truth insider answer key, then trains a class-weighted Random Forest via **StratifiedGroupKFold** (grouped by user — closes a person-level leakage path present in an earlier ungrouped version) to produce the Classification Confidence criterion | Output of #1 + `insiders.csv` | `cert_threat_resource_features_with_confidence.csv` |
| 3 | `cert_sa_greedy_pipeline_v4.py` | Re-aggregates to **person-level** records (one row per person, matching the catalog's per-person-per-year pricing), runs greedy + 10-seed Simulated Annealing under **two Training-capacity scenarios** and both weight sets | Output of #2 | Console output; resource-usage report |
| 4 | `cert_bootstrap_ate_v2.py` | **Cluster bootstrap by person** (not by row) estimate of the causal effect of weight-set on threat-mitigation rate, both catalog scenarios | Output of #2 | `bootstrap_ate_results_{scenario}.csv` |
| 5 | `cert_cate_by_scenario_v2.py` | Same cluster-bootstrap design, stratified by the three r4.2 insider scenarios, both catalog scenarios | Output of #2 | `cate_by_scenario_results_{scenario}.csv` |
| 6 | `cert_confusion_matrix_v2.py` | Confusion matrix / TPR-FPR for the GroupKFold-corrected classifier's out-of-fold predictions | Output of #2 | `confidence_confusion_matrix.csv` |
| 7 | `cert_shap_confidence_v3.py` | SHAP magnitude **and direction** (value-SHAP correlation) — the global mean signed value was found to be uninformative given 99.5% class imbalance; correlation is the meaningful direction statistic | Output of #2 | SHAP plots + `shap_feature_importance_v3.csv` |
| 8 | `cert_timing_v2.py` | Wall-clock timing of the person-level allocator, both scenarios, and the GroupKFold classifier | Output of #2 | `cert_timing_results_v2.csv` |
| — | `ahp_weights_by_institution.py` | AHP priority vectors from the raw survey export; tests weight heterogeneity by institution type | Raw AHP survey `.xlsx` | `ahp_weights_by_institution.csv` |
| — | `ahp_cr_sensitivity.py` | Re-aggregates calibration weights at CR<0.10 (reported), CR<0.20, and no filter, to test consistency-threshold sensitivity | Raw AHP survey `.xlsx` | `ahp_cr_sensitivity.csv` |
| — | `ahp_split_seed_sensitivity.py` | Re-runs the 70:30 calibration/held-out split across 20 seeds to test whether seed=42 is an outlier | Raw AHP survey `.xlsx` | `ahp_split_seed_sensitivity.csv` |

## Data (not included in this repository)

- **CERT Insider Threat Test Dataset (r4.2 release)**: too large for a code
  repository; third-party dataset with its own citation requirement.
  Lindauer, B. (2020). *Insider Threat Test Dataset* [Data set].
  https://doi.org/10.1184/R1/12841247.v1
- **AHP practitioner survey (n = 105)**: the raw KoboToolbox export is
  committed in `data/` for independent verification of every AHP-side
  number (consistency filtering, calibration/held-out split, all three
  sensitivity checks).

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Each script has file paths and key parameters (e.g. `BUDGET`, `DATA_PATH`)
set as plain constants near the top — edit these before running.

## Key results (full detail in the thesis, Chapters 3–4)

- AHP calibration-wave weights: Impact = 0.224, Propagation = 0.252,
  Confidence = 0.247, Evasion = 0.278 (n = 46 of 105; reproduced exactly
  from the committed raw survey export)
- Person-level population: 1,000 (aggregated from 67,238 user-weeks)
- Causal effect on threat-mitigation rate (cluster bootstrap by person):
  Scenario A (Training unconstrained): +0.0012, 95% CI [0.0000, 0.0228]
  Scenario B (Training capacity = 5): −0.0026, 95% CI [−0.0400, +0.0308]
  — the primary effect is sensitive to this one capacity assumption and
  is reported as such, not smoothed over.
- Simulated Annealing vs. greedy: no measurable improvement under either
  weight set or catalog scenario (SD = 0.0000 across all 10 seeds)

## Superseded files (not included, per prior convention)

`cert_local_aggregation_v2.py` (abandoned alternative design),
`cert_sa_greedy_pipeline.py`/`v2`/`v3` (superseded by v4),
`cert_bootstrap_ate.py`, `cert_cate_by_scenario.py`, `cert_shap_confidence.py`/`v2`,
`cert_confusion_matrix.py`, `cert_timing.py` (all superseded by their v2/v3
successors above). Available on request for development-history context.

## Citation

If you use this pipeline, please cite the thesis (full citation to be
added once finalised) and the CERT dataset DOI above.
