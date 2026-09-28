# Data directory

**The raw KoboToolbox AHP survey export is already committed here** (your
supervisor independently recomputed every AHP-side number directly from
this file and confirmed they reproduce exactly — keep it committed, do
not remove it).

**Not included, and should not be committed:**
- `r4.2.tar.bz2` — the CERT Insider Threat Test Dataset archive
  (DOI: https://doi.org/10.1184/R1/12841247.v1). Too large for a code
  repository (~2-8GB compressed); scripts stream it, so full extraction
  is neither required nor recommended.
- `answers/` — the CERT ground-truth insider answer key, extracted from
  the same release (small, but still third-party data — request/download
  separately from the same DOI).

If reproducing the CERT-side pipeline, place the CERT archive and its
answer key here (matching each script's `DATA_PATH`/`INSIDERS_CSV`
constants) before running scripts 1–8.
