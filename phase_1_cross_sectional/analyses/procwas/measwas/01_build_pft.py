# %% [markdown]
# MeasWAS Step 01 - per-patient spirometry feature table for the 5-year pre-index window.
#
# Reads pft.csv (96,307 tests / 58,654 patients / 1998-10..2019-03), applies the user's column
# contract (encoded in measwas_utils as PFT_RENAME / PFT_DERIVED / PFT_FEATURES / PFT_INPUTS_ONLY /
# PFT_GATES), restricts to v2-cohort tests in [index - 5yr, index) -- lower inclusive, upper
# exclusive, the same convention as analyses/procwas/01_build_windowed.py -- and collapses to ONE
# ROW PER PATIENT by the mean of each feature across that patient's in-window tests (mirrors the
# LabWAS per-patient mean; also absorbs the 1,778 multi-session patient-days).
#
# EXPLORATORY ARM: pft.csv ends 2019-03 while index dates run to 2025-03, so only ~3.6% of the
# 222,109-patient cohort has an in-window PFT, and coverage is strongly confounded by index year.
# Every downstream output must carry that caveat and adjust for study_year.
#
# -> results/procwas_results/measwas/store/pft_5yr.parquet   (one row per patient)
# DE-IDENTIFIED summary to stdout only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import measwas_utils as U

# Raw pft.csv columns we need = the tested phenome minus the engineered ratios, plus the
# derivation-only inputs. That set happens to be exactly the gated set.
RAW_COLS = sorted((set(U.PFT_FEATURES) - set(U.PFT_DERIVED)) | set(U.PFT_INPUTS_ONLY))
INV_RENAME = {v: k for k, v in U.PFT_RENAME.items()}          # PEF_REF -> PEF_PREDICTED, etc.
SRC_COLS = [INV_RENAME.get(c, c) for c in RAW_COLS]
OUT_COLS = U.PFT_FEATURES + U.PFT_INPUTS_ONLY


# %%
def main() -> None:
    U.ensure_dirs()
    print("=" * 78)
    print("EXPLORATORY ARM - SPIROMETRY (pft.csv)")
    print("pft.csv ends 2019-03; cohort index dates run to 2025-03. Only ~3.6% of the 222,109")
    print("cohort patients have a PFT in the 5yr pre-index window, and coverage falls with index")
    print("year. Treat every PFT result as hypothesis-generating, and adjust for study_year.")
    print("=" * 78)

    # ---- read (MRN MUST be str: pandas coerces it to int64 and strips the leading zeros) ----
    df = pd.read_csv(
        U.PFT_SRC,
        usecols=["MRN", "VISITDATE", "TEST_DATE", "PreFEV1pred"] + SRC_COLS,
        dtype={"MRN": str, "VISITDATE": str, "TEST_DATE": str},
        encoding="latin-1",
        low_memory=False,
    ).rename(columns=U.PFT_RENAME)
    n_read = len(df)
    for c in RAW_COLS + ["PreFEV1pred"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    key_len = df["MRN"].str.len().value_counts().sort_index().to_dict()
    print(f"\nrows read                       : {n_read:,}")
    print(f"MRN key length distribution     : {key_len}  (expect all 12-char zero-padded)")

    # ---- drop the predicted-value-only shells (empty TEST_DATE, zero spirometry) ----
    has_test = df["TEST_DATE"].notna() & (df["TEST_DATE"].str.strip() != "")
    df = df.loc[has_test].copy()
    print(f"rows after empty-TEST_DATE drop : {len(df):,}  (dropped {n_read - len(df):,}; expect 92,299 kept)")

    # ---- event date = VISITDATE (100% complete, == TEST_DATE at day resolution) ----
    # Fractional seconds are MANDATORY here; the project's stock '%d%b%Y:%H:%M:%S' yields NaT.
    df["event_date"] = pd.to_datetime(df["VISITDATE"], format=U.SAS_MS, errors="coerce").dt.normalize()
    n_bad_date = int(df["event_date"].isna().sum())
    if n_bad_date:
        print(f"unparseable VISITDATE dropped   : {n_bad_date:,}")
        df = df.loc[df["event_date"].notna()].copy()

    # ---- arithmetic self-check: are these columns what we think they are? ----
    chk = df[["FEV1_PRE", "FVC_PRE", "FEV1_FVC_pre"]].dropna()
    chk = chk[chk["FVC_PRE"] > 0]
    resid = (100.0 * chk["FEV1_PRE"] / chk["FVC_PRE"] - chk["FEV1_FVC_pre"]).abs()
    print(f"\nself-check 100*FEV1_PRE/FVC_PRE vs FEV1_FVC_pre on {len(chk):,} paired rows: "
          f"median |diff|={resid.median():.4g}, {100 * (resid < 0.5).mean():.2f}% within 0.5, "
          f"max |diff|={resid.max():.4g}")

    # ---- reference guard: null non-positive references BEFORE dividing ----
    for ref in ("PEF_REF", "FEV1_REF"):
        bad = df[ref].notna() & (df[ref] <= 0)
        print(f"non-positive {ref:<9} nulled  : {int(bad.sum()):,}")
        df.loc[bad, ref] = np.nan

    # ---- physiologic gates: out-of-range VALUES -> NaN, never a row drop ----
    print("\nvalues gated to NaN by U.PFT_GATES (row kept - a bad PEF must not cost us the FEV1):")
    for col, (lo, hi) in U.PFT_GATES.items():
        before = df[col].notna()
        df.loc[before & ~df[col].between(lo, hi), col] = np.nan
        n_gated = int(before.sum() - df[col].notna().sum())
        print(f"  {col:<14} [{lo:g}, {hi:g}]  gated={n_gated:,}  "
              f"remaining non-null={int(df[col].notna().sum()):,}")

    # ---- engineered percent-of-reference columns (from the GATED inputs) ----
    # These are RATIOS, per spec - fractions, not percents. The file's own PreFEV1pred /
    # PostFEV1pred are 100x these, so switching to the percent scale is a one-line change
    # (multiply by 100 below).
    for new, (num, ref) in U.PFT_DERIVED.items():
        df[new] = df[num] / df[ref]
    scale = df[["FEV1_PRE_PREDICTED", "PreFEV1pred"]].dropna()
    scale = scale[scale["FEV1_PRE_PREDICTED"] > 0]
    print(f"\nscale check: FEV1_PRE_PREDICTED vs file's PreFEV1pred/100 on {len(scale):,} rows -> "
          f"median ratio={(scale['PreFEV1pred'] / 100.0 / scale['FEV1_PRE_PREDICTED']).median():.4f} "
          f"(1.0 = our ratios are fractions of the file's percent)")

    # ---- cohort join + [index - 5yr, index) window ----
    coh = pd.read_parquet(U.COHORT, columns=["id", "STUDY_DATE", "study_year", "severity"])
    coh["idx"] = coh["STUDY_DATE"].dt.normalize()
    coh["lo"] = coh["idx"] - pd.DateOffset(years=U.WINDOW_YEARS)

    m = df.merge(coh[["id", "idx", "lo"]], left_on="MRN", right_on="id", how="inner")
    m = m.loc[(m["event_date"] >= m["lo"]) & (m["event_date"] < m["idx"])].copy()
    print(f"\nrows after cohort join + [index-{U.WINDOW_YEARS}yr, index) filter: {len(m):,}"
          f"   patients={m['id'].nunique():,}")

    # ---- one row per patient: mean across in-window tests ----
    out = m.groupby("id", as_index=False)[OUT_COLS].mean()
    out["n_tests"] = m.groupby("id").size().reindex(out["id"]).to_numpy()
    out = out[["id", "n_tests"] + OUT_COLS]
    out[OUT_COLS] = out[OUT_COLS].astype("float64")
    out["n_tests"] = out["n_tests"].astype("int64")
    out.to_parquet(U.PFT_FEAT, index=False)

    n_pat, n_coh = len(out), len(coh)
    print(f"\nwrote {U.PFT_FEAT}")
    print(f"patients in output              : {n_pat:,} of {n_coh:,} cohort patients "
          f"({100 * n_pat / n_coh:.2f}% - expect ~8,091 / 3.6%)")
    print(f"tests per patient               : mean={out['n_tests'].mean():.2f} "
          f"median={out['n_tests'].median():.0f} max={out['n_tests'].max():,}")

    # ---- per-feature coverage vs the MIN_PATIENTS floor ----
    print(f"\nper-feature patient coverage (floor = U.MIN_PATIENTS = {U.MIN_PATIENTS}):")
    print(f"  {'feature':<22} {'role':<12} {'n_patients':>10}  {'%cohort':>8}  passes")
    for c in OUT_COLS:
        n = int(out[c].notna().sum())
        role = "phenome" if c in U.PFT_FEATURES else "input-only"
        print(f"  {c:<22} {role:<12} {n:>10,}  {100 * n / n_coh:>7.2f}%  "
              f"{'YES' if n >= U.MIN_PATIENTS else 'no'}")
    n_pass = sum(int(out[c].notna().sum()) >= U.MIN_PATIENTS for c in U.PFT_FEATURES)
    print(f"  -> {n_pass}/{len(U.PFT_FEATURES)} phenome features clear the {U.MIN_PATIENTS}-patient floor")

    # ---- severity composition of the retained patients ----
    ret = coh.merge(out[["id"]], on="id", how="inner")
    print("\nseverity composition of retained patients:")
    for sev in U.SEVERITIES:
        k = int((ret["severity"] == sev).sum())
        base = int((coh["severity"] == sev).sum())
        print(f"  {sev:<9} {k:>7,}  ({100 * k / n_pat:5.1f}% of retained; "
              f"{100 * k / base:4.1f}% of the {base:,} such cohort patients)")

    # ---- index-year distribution: makes the coverage-vs-index-year confound visible ----
    print("\nindex-year distribution, retained vs whole cohort (the coverage confound):")
    print(f"  {'year':<6} {'retained':>9} {'cohort':>9} {'coverage':>9}")
    yr = pd.DataFrame({"cohort": coh["study_year"].value_counts(),
                       "retained": ret["study_year"].value_counts()}).fillna(0).astype(int).sort_index()
    for y, r in yr.iterrows():
        print(f"  {y:<6} {r['retained']:>9,} {r['cohort']:>9,} "
              f"{100 * r['retained'] / max(r['cohort'], 1):>8.1f}%")

    U.log_milestone(
        f"Step 01 spirometry features ({U.WINDOW_LABEL} pre-index): {len(m):,} in-window tests -> "
        f"{n_pat:,}/{n_coh:,} patients ({100 * n_pat / n_coh:.1f}%); "
        f"{n_pass}/{len(U.PFT_FEATURES)} phenome features clear the {U.MIN_PATIENTS} floor. "
        f"EXPLORATORY - coverage confounded by index year.")


if __name__ == "__main__":
    main()
