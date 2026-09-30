# Data directory

**The raw KoboToolbox AHP survey export has been removed from this repository.**

It was previously committed here, and an earlier version of this README
incorrectly stated that the supervisor's verification of the AHP numbers
meant the file should be kept public ("do not remove it"). That was my
own inference, not something the supervisor said, and it was wrong.

The supervisor's actual position, stated directly: the file needed to
come down. It contained data collected before ethics approval was
granted, included timestamps, and had 20 respondents who were each the
only person with their specific combination of institution type,
region, and role — making them individually re-identifiable despite no
names being present.

The survey's consistency-filtered numbers (CR<0.10, n=46; the 32/14
calibration/held-out split; all downstream AHP weight calculations) were
independently verified by the supervisor against the raw file before its
removal, and reproduce exactly. That verification stands; the file
itself does not need to remain public for the verification to have been
real.

**Not included, and should not be committed:**
- `r4.2.tar.bz2` — the CERT Insider Threat Test Dataset archive
  (DOI: https://doi.org/10.1184/R1/12841247.v1). Too large for a code
  repository; scripts stream it, so full extraction is neither required
  nor recommended.
- `answers/` — the CERT ground-truth insider answer key, extracted from
  the same release.

If reproducing the CERT-side pipeline, place the CERT archive and its
answer key here (matching each script's `DATA_PATH`/`INSIDERS_CSV`
constants) before running the CERT-side scripts.
