"""
CERT r4.2 Local Streaming Aggregation (Windows-friendly, no Colab needed)
--------------------------------------------------------------------------
Runs entirely on your own machine. Extracts the small log files normally,
but streams http.csv straight out of the .tar.bz2 archive in chunks --
it is never fully written to disk, so you don't need 87GB of free space.

BEFORE RUNNING: edit the two paths below.
"""

import tarfile
import pandas as pd
import numpy as np
import os

# ----------------------------------------------------------------------
# 0. CONFIG -- edit these two lines
# ----------------------------------------------------------------------
ARCHIVE_PATH = r"C:\Users\YOURNAME\Downloads\r4.2.tar.bz2"   # <-- paste your real path here
OUTPUT_DIR = r"C:\Users\YOURNAME\Documents\cert_features"     # <-- where results get saved

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. LIST WHAT'S INSIDE THE ARCHIVE (run this first, on its own, to check
#    the exact member names before trusting the rest of the script)
# ----------------------------------------------------------------------
def list_contents():
    with tarfile.open(ARCHIVE_PATH, "r:bz2") as tf:
        for member in tf.getmembers():
            print(f"{member.size / 1024 / 1024:8.1f} MB   {member.name}")


# ----------------------------------------------------------------------
# 2. EXTRACT THE SMALL FILES NORMALLY
#    Edit SMALL_MEMBERS to match the exact names printed by list_contents()
#    -- the leading folder name can differ (e.g. "r4.2/" vs "r4.2.2/").
# ----------------------------------------------------------------------
SMALL_MEMBERS = [
    "r4.2/logon.csv",
    "r4.2/device.csv",
    "r4.2/email.csv",
    "r4.2/file.csv",
    "r4.2/psychometric.csv",
]

def extract_small_files():
    with tarfile.open(ARCHIVE_PATH, "r:bz2") as tf:
        for member_name in SMALL_MEMBERS:
            try:
                member = tf.getmember(member_name)
            except KeyError:
                print(f"[WARNING] '{member_name}' not found in archive -- "
                      f"check the exact name against list_contents() output.")
                continue
            tf.extract(member, path=OUTPUT_DIR)
            print(f"Extracted: {member_name}")


# ----------------------------------------------------------------------
# 3. STREAM-AGGREGATE http.csv WITHOUT EVER WRITING IT TO DISK
#    Reads it in chunks straight from inside the archive, does a running
#    groupby-sum per user/week. Unique user-week combinations are small
#    (~1,000 users x ~90 weeks at most), so the running total stays tiny
#    even though http.csv itself is tens of GB.
# ----------------------------------------------------------------------
def stream_aggregate_http(http_member_name="r4.2/http.csv", chunksize=200_000):
    running = None
    with tarfile.open(ARCHIVE_PATH, "r:bz2") as tf:
        fileobj = tf.extractfile(http_member_name)
        if fileobj is None:
            raise FileNotFoundError(
                f"'{http_member_name}' not found -- check the exact name "
                f"against list_contents() output."
            )
        reader = pd.read_csv(fileobj, chunksize=chunksize)
        n_chunks = 0
        for chunk in reader:
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk["week"] = chunk["date"].dt.to_period("W").astype(str)
            partial = (
                chunk.groupby(["user", "week"])
                .agg(
                    http_events=("date", "count"),
                    http_after_hours=("date", lambda s: _after_hours_mask(s).sum()),
                )
                .reset_index()
            )
            if running is None:
                running = partial
            else:
                running = (
                    pd.concat([running, partial])
                    .groupby(["user", "week"], as_index=False)
                    .sum()
                )
            n_chunks += 1
            if n_chunks % 20 == 0:
                print(f"  ...processed {n_chunks * chunksize:,} http.csv rows so far")

    return running


# ----------------------------------------------------------------------
# 4. AGGREGATE THE SMALL FILES THE SAME WAY (they're small enough to load
#    fully once extracted)
# ----------------------------------------------------------------------
def _after_hours_mask(dt_series, lo=6, hi=20):
    return (dt_series.dt.hour < lo) | (dt_series.dt.hour > hi)


def _parse_recipients(row):
    """Union of unique addresses across to/cc/bcc for one email row."""
    parts = set()
    for col in ("to", "cc", "bcc"):
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            parts.update(p.strip() for p in val.split(";") if p.strip())
    return parts


def aggregate_small_files():
    # --- logon.csv: id, date, user, pc, activity ---
    # Used for: Evasion (after-hours share) and Propagation (distinct PCs).
    logon = pd.read_csv(os.path.join(OUTPUT_DIR, "r4.2", "logon.csv"))
    logon["date"] = pd.to_datetime(logon["date"])
    logon["week"] = logon["date"].dt.to_period("W").astype(str)
    logon_weekly = (
        logon.groupby(["user", "week"])
        .agg(
            logon_events=("date", "count"),
            logon_after_hours=("date", lambda s: _after_hours_mask(s).sum()),
            distinct_pcs=("pc", "nunique"),
        )
        .reset_index()
    )

    # --- device.csv: id, date, user, pc, activity ---
    # Used for: Evasion only (this release has no removable-media file-copy
    # flag, so device activity contributes to the after-hours ratio but not
    # directly to Impact).
    device = pd.read_csv(os.path.join(OUTPUT_DIR, "r4.2", "device.csv"))
    device["date"] = pd.to_datetime(device["date"])
    device["week"] = device["date"].dt.to_period("W").astype(str)
    device_weekly = (
        device.groupby(["user", "week"])
        .agg(
            device_events=("date", "count"),
            device_after_hours=("date", lambda s: _after_hours_mask(s).sum()),
        )
        .reset_index()
    )

    # --- email.csv: id, date, user, pc, to, cc, bcc, from, size, attachments, content ---
    # Used for: Impact (large attachments), Propagation (distinct recipients
    # reached that week), Evasion (after-hours share).
    email = pd.read_csv(os.path.join(OUTPUT_DIR, "r4.2", "email.csv"))
    email["date"] = pd.to_datetime(email["date"])
    email["week"] = email["date"].dt.to_period("W").astype(str)
    email["recipients"] = email.apply(_parse_recipients, axis=1)
    email_weekly = (
        email.groupby(["user", "week"])
        .agg(
            email_events=("date", "count"),
            email_after_hours=("date", lambda s: _after_hours_mask(s).sum()),
            large_attachments=("size", lambda s: (s > 1_000_000).sum()),
        )
        .reset_index()
    )
    recipients_weekly = (
        email.groupby(["user", "week"])["recipients"]
        .apply(lambda sets: len(set.union(*sets)) if len(sets) else 0)
        .reset_index(name="distinct_recipients")
    )
    email_weekly = email_weekly.merge(recipients_weekly, on=["user", "week"], how="left")

    # --- file.csv: id, date, user, pc, filename, content ---
    # This release has no activity/to_removable_media column, so file_events
    # is a plain count -- used as the main Impact signal, and its timestamps
    # also feed the Evasion after-hours ratio.
    filecsv = pd.read_csv(os.path.join(OUTPUT_DIR, "r4.2", "file.csv"))
    filecsv["date"] = pd.to_datetime(filecsv["date"])
    filecsv["week"] = filecsv["date"].dt.to_period("W").astype(str)
    file_weekly = (
        filecsv.groupby(["user", "week"])
        .agg(
            file_events=("date", "count"),
            file_after_hours=("date", lambda s: _after_hours_mask(s).sum()),
        )
        .reset_index()
    )

    return logon_weekly, device_weekly, email_weekly, file_weekly


# ----------------------------------------------------------------------
# 5. JOIN EVERYTHING + COMPUTE YOUR FOUR CRITERIA
#
#    Impact       = file_events + large_attachments
#                   (volume of data touched: files accessed + big email
#                   attachments sent, per user-week)
#    Propagation  = distinct_pcs + distinct_recipients
#                   (how far the user's activity reaches: machines logged
#                   into + distinct people emailed, per user-week)
#    Evasion      = share of all logged events that fall outside working
#                   hours (6am-8pm), across logon/device/email/file/http
#    Confidence   = NOT computed here. Per the dataset justification
#                   section, this is meant to be the output of a trained
#                   classifier (validated against CIC-IDS2017 features),
#                   not a raw count -- that's a separate stage. Left as
#                   NaN and flagged, not silently filled with a guess.
# ----------------------------------------------------------------------
def build_feature_table():
    logon_weekly, device_weekly, email_weekly, file_weekly = aggregate_small_files()
    http_weekly = stream_aggregate_http()

    merged = (
        logon_weekly.merge(device_weekly, on=["user", "week"], how="outer")
        .merge(email_weekly, on=["user", "week"], how="outer")
        .merge(file_weekly, on=["user", "week"], how="outer")
        .merge(http_weekly, on=["user", "week"], how="outer")
        .fillna(0)
    )

    merged["impact"] = merged["file_events"] + merged["large_attachments"]
    merged["propagation"] = merged["distinct_pcs"] + merged["distinct_recipients"]

    total_events = (
        merged["logon_events"] + merged["device_events"] + merged["email_events"]
        + merged["file_events"] + merged["http_events"]
    )
    total_after_hours = (
        merged["logon_after_hours"] + merged["device_after_hours"] + merged["email_after_hours"]
        + merged["file_after_hours"] + merged["http_after_hours"]
    )
    merged["evasion"] = np.where(total_events > 0, total_after_hours / total_events, 0.0)

    merged["confidence"] = np.nan  # placeholder -- see note above; Stage 2 task

    for col in ["impact", "propagation", "evasion"]:  # confidence intentionally excluded
        rng = merged[col].max() - merged[col].min()
        merged[col] = (merged[col] - merged[col].min()) / rng if rng > 0 else 0.0

    merged["threat_id"] = merged["user"] + "_" + merged["week"]
    merged["resource_id"] = "TBD"
    merged["cost"] = 1.0

    out_path = os.path.join(OUTPUT_DIR, "cert_threat_resource_features.csv")
    merged[["threat_id", "resource_id", "impact", "propagation", "confidence", "evasion", "cost"]].to_csv(
        out_path, index=False
    )
    print(f"\nSaved: {out_path}")
    print(merged.shape)
    return merged


if __name__ == "__main__":
    print("Step 1: listing archive contents...\n")
    list_contents()
    input("\nCheck the names above against SMALL_MEMBERS in this script, then press Enter to continue...")

    print("\nStep 2: extracting small files...")
    extract_small_files()

    print("\nStep 3-5: streaming http.csv and building the feature table (this is the slow part)...")
    build_feature_table()
