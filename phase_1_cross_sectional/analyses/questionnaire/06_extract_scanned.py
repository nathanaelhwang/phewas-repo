# %% [markdown]
# Step 6 - Extract the Scanned paper-survey dataset into the target-code frame.
#
# `scanned_dataset.csv` is wide (one row per paper survey) under friendly names
# that map to target codes via `load_scanned_seed()`. Most items are already
# digit-coded; a few need Scanned-specific handling:
#   - BEDTIME / WAKETIME : split Hour(HHMM) + AM/PM columns -> combine.
#   - LATENCY            : decimal HOURS (0.25 = 15 min), not KPHC's bins.
#   - FOSQ1-10           : Scanned uses a 0-4 scale (KPHC is 0-3) -> SKIP here,
#                          rely on KPHC (authoritative). Documented, not silent.
# Lowest merge precedence (KPHC > Somnoware > Scanned).
#
# Emits: results/questionnaire_results/scanned_wide.parquet
# DE-IDENTIFIED stdout only (never DOB/Name/MRN or row values).

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import questionnaire_utils as U

IDENT = {"ParticipantID", "SurveyPageID", "Schedued Service", "DOB", "MRN",
         "FirstName", "LastName"}
FOSQ_CODES = {f"FOSQ{i}" for i in range(1, 11)}   # Scanned digit 0-4 = standard value
# Scanned friendly cols the seed can't reach (their quest_id is the uncoded RLS stem)
SCANNED_MANUAL = {"RLSUnpleasant": "RLS1", "RLSHowOften": "RLS4"}


def scanned_time(hour, ampm):
    """Combine Scanned split time hour 'HHMM' + 'am'/'pm' -> minutes since midnight."""
    h = U.norm(hour)
    if not h or not h.replace(".", "").isdigit():
        return pd.NA
    digits = h.split(".")[0]
    if len(digits) <= 2:                 # bare hour
        hh, mm = digits, "00"
    else:                                # HHMM
        hh, mm = digits[:-2], digits[-2:]
    ap = U.norm(ampm)
    ap = "pm" if ap.startswith("p") else ("am" if ap.startswith("a") else "")
    return U.decode_time_minutes(f"{hh}:{mm} {ap}".strip())


def scanned_fosq(value):
    """Scanned FOSQ digit is already the standard 0-4 value (4=No..1=extreme,0=other)."""
    s = U.norm(value)
    return int(s) if s.isdigit() and 0 <= int(s) <= 4 else pd.NA


def scanned_latency(value):
    """Scanned latency = decimal hours -> minutes (fallback to bin map)."""
    s = U.norm(value)
    if not s:
        return pd.NA
    if U.decode("LATENCY", value) is not pd.NA and any(c.isalpha() for c in s):
        return U.decode("LATENCY", value)          # categorical bin form
    try:
        mins = round(float(s) * 60)
    except ValueError:
        return pd.NA
    return mins if 0 <= mins <= 1440 else pd.NA


def main() -> None:
    U.ensure_dirs()
    df = pd.read_csv(U.RESULTS / "scanned_dataset.csv", dtype=str)
    seed = U.load_scanned_seed()
    code2cols: dict[str, list[str]] = {}
    for _, r in seed.iterrows():
        if pd.isna(r["code"]) or r["code"] not in U.TARGET_CODES:
            continue
        if r["friendly"] in IDENT or r["friendly"] not in df.columns:
            continue
        code2cols.setdefault(r["code"], []).append(r["friendly"])
    for friendly, code in SCANNED_MANUAL.items():   # RLSUnpleasant->RLS1, RLSHowOften->RLS4
        if friendly in df.columns:
            code2cols.setdefault(code, []).append(friendly)

    out = pd.DataFrame(index=df.index)
    out["MRN"] = U.normalize_mrn(df["MRN"]) if "MRN" in df.columns else pd.NA
    out["STUDY_DATE"] = pd.NaT
    out["CONTACT_DATE"] = pd.NaT      # Scanned has no encounter date ('Schedued Service'=service type)

    populated = []
    for code in U.TARGET_CODES:
        cols = code2cols.get(code, [])
        if not cols:
            out[code] = pd.array([pd.NA] * len(df), dtype=U.code_dtype(code))
            continue
        if code in ("BEDTIME_CONT", "WAKETIME_CONT"):
            hh = next((c for c in cols if "hour" in c.lower()), cols[0])
            ap = next((c for c in cols if "ampm" in c.lower()), None)
            col = df.apply(lambda r: scanned_time(r[hh], r[ap] if ap else ""), axis=1)
        elif code == "LATENCY":
            col = df[cols[0]].map(scanned_latency)
        elif code in FOSQ_CODES:
            col = df[cols[0]].map(scanned_fosq)
        else:
            # coalesce across friendly cols (first non-null decoded)
            dec = pd.DataFrame({c: df[c].map(lambda v, k=code: U.decode(k, v)) for c in cols})
            col = dec.bfill(axis=1).iloc[:, 0]
        out[code] = col.astype(U.code_dtype(code))
        if out[code].notna().any():
            populated.append(code)

    out["source"] = "scanned"
    out = out[U.TARGET_COLUMNS + ["source"]]
    out_path = U.RESULTS / "scanned_wide.parquet"
    out.to_parquet(out_path, index=False)

    print(f"[scanned] {len(out):,} surveys x {len(U.TARGET_COLUMNS)} cols -> {out_path.name}")
    print(f"[scanned] populated {len(populated)} codes; "
          f"skipped (FOSQ scale / no col): "
          f"{sorted(set(U.TARGET_CODES) - set(populated))}")
    U.log_milestone(
        f"06_extract_scanned: {len(out)} surveys; {len(populated)} codes populated; "
        f"FOSQ deferred to KPHC (scale differs).")


if __name__ == "__main__":
    main()
