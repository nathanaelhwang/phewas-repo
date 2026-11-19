# %% [markdown]
# MeasWAS Step 02 - per-patient echocardiogram feature table for the 5-year pre-index window.
#
# Reads the repaired echo store (U.ECHO_CLEAN, 284,028 studies / 141,775 patients / 2010-2020,
# built by step 00) plus ef.csv (534,610 LVEF readings / 211,503 patients / 1995-2025), and
# collapses both to ONE ROW PER COHORT PATIENT over [index - 5yr, index) -- lower inclusive,
# upper exclusive, the same convention as step 01 and analyses/procwas/01_build_windowed.py.
#
# Per study, in this order (the order matters):
#   1. plausibility filter on age_performed (U.ECHO_AGE_GATE; raw min is -7),
#   2. deterministic x100 unit repair of the four contaminated columns the phenome uses
#      (U.ECHO_UNIT_REPAIR / U.echo_repair_units) - tr_velocity, av_peak_velocity,
#      mv_peak_e_wave, mv_annulus_septal_e_prime,
#   3. gate the three derivation INPUTS (U.ECHO_INPUT_GATES) - the ASE cube amplifies input error,
#   4. derive (U.ECHO_DERIVED): LVEF coalesce, ASE LV mass, RWT, TR gradient, septal e', E/e',
#      AV Vmax,
#   5. gate every feature to its own range (U.ECHO_GATES). Out-of-range -> NaN, NEVER a row drop:
#      a study with a bad LVIDd may still have a good LAVi.
# Then per patient: the MEAN across in-window studies (ICC is low for almost every feature, so
# averaging k studies cuts the error variance ~k-fold and strictly beats nearest-to-index).
#
# ef.csv is kept as a SEPARATE feature (ef_value), NOT pooled into the echo LVEF: the two are
# independent sources with different coverage and only Spearman 0.664 on same-day pairs. Their
# agreement is quantified in analysis/ef_source_agreement.csv - that is the evidence for whether
# they could ever be pooled.
#
# CAVEAT for the write-up: every echo feature is conditional on having had an echo, which is a
# strong function of cardiac symptoms. Echo rate by severity is non-monotonic, and the project's
# IPW sub-analysis found severe-OSA patients have ~18% FEWER encounters - so ascertainment bias
# runs OPPOSITE to the expected effect: a null is uninformative, a positive finding conservative.
# Availability is also mechanically coupled to index year (see 'era' in U.ECHO_FEATURES), so
# study_year adjustment is mandatory downstream.
#
# -> results/procwas_results/measwas/store/echo_5yr.parquet   (one row per patient)
# -> results/procwas_results/measwas/analysis/echo_feature_completeness.csv
# -> results/procwas_results/measwas/analysis/ef_source_agreement.csv
# DE-IDENTIFIED summary to stdout only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import measwas_utils as U

RAW_ECHO_COLS = sorted({c for f in U.ECHO_FEATURES if f["kind"] != "ef" for c in f["sources"]}
                       | set(U.ECHO_INPUT_GATES))
FEATURES = [f["name"] for f in U.ECHO_FEATURES]


# %%
def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Steps 2-4: unit repair, input gates, derived features. Returns df with the new columns."""
    for col in U.ECHO_UNIT_REPAIR:
        if col in df.columns:
            df[col] = U.echo_repair_units(df[col], col)

    for col, (lo, hi) in U.ECHO_INPUT_GATES.items():
        before = int(df[col].notna().sum())
        df.loc[df[col].notna() & ~df[col].between(lo, hi), col] = np.nan
        print(f"  input gate {col:<24} [{lo:g}, {hi:g}]  "
              f"nulled={before - int(df[col].notna().sum()):,}")

    # LVEF: prefer the calculated (volumetric) value, fall back to the visual estimate. Each is
    # gated first so a wild -629,900 cannot be preferred over a good estimate.
    lo, hi = U.ECHO_GATES["LVEF_pct"]
    calc = df["lv_ef_calculated"].where(df["lv_ef_calculated"].between(lo, hi))
    est = df["lv_ef_estimated"].where(df["lv_ef_estimated"].between(lo, hi))
    df["LVEF_pct"] = calc.fillna(est)

    d, pwt, ivs = (df["lv_diastolic_dimension"], df["lv_post_wall_thick"], df["ivs_thick"])
    df["LV_mass_ASE_g"] = 0.8 * 1.04 * ((d + pwt + ivs) ** 3 - d ** 3) + 0.6
    df["RWT"] = 2.0 * pwt / d.where(d > 0)

    df["TR_grad_mmHg"] = 4.0 * (df["tr_velocity"] / 100.0) ** 2
    df["e_prime_septal_cms"] = df["mv_annulus_septal_e_prime"]
    ep = df["mv_annulus_septal_e_prime"]
    df["E_over_e_prime_septal"] = df["mv_peak_e_wave"] / ep.where(ep > 0)
    df["AV_Vmax_cms"] = df["av_peak_velocity"]
    return df


# %%
def load_ef(coh: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """ef.csv -> one row per in-window cohort patient (mean gated EF, n_ef)."""
    ef = pd.read_csv(U.EF_SRC, usecols=["MRN", "ECHO_DATE", "EF_VALUE"],
                     dtype={"MRN": str, "ECHO_DATE": str}, encoding="latin-1")
    n_read = len(ef)
    ef = ef.drop_duplicates()
    print(f"\nef.csv rows read                : {n_read:,}")
    # De-dup key is (MRN, ECHO_DATE, EF_VALUE): ORIGINAL_VALUE is free text and is never read.
    # 886 rows are duplicated across all four columns; the rest differ only in the free text.
    print(f"  duplicate readings dropped    : {n_read - len(ef):,}  "
          f"(on MRN+date+value; 886 are duplicated across all 4 raw columns)")
    key_len = ef["MRN"].str.len().value_counts().sort_index().to_dict()
    print(f"  MRN key length distribution   : {key_len}  (expect all 12-char zero-padded)")

    ef["event_date"] = pd.to_datetime(ef["ECHO_DATE"], format=U.US_DATE, errors="coerce")
    n_bad = int(ef["event_date"].isna().sum())
    ef = ef.loc[ef["event_date"].notna()].copy()
    print(f"  unparseable/blank ECHO_DATE   : {n_bad:,} dropped ({100 * n_bad / n_read:.1f}%)")
    ef["EF_VALUE"] = pd.to_numeric(ef["EF_VALUE"], errors="coerce")

    lo, hi = U.ECHO_GATES["ef_value"]
    before = int(ef["EF_VALUE"].notna().sum())
    ef.loc[ef["EF_VALUE"].notna() & ~ef["EF_VALUE"].between(lo, hi), "EF_VALUE"] = np.nan
    n_gated = before - int(ef["EF_VALUE"].notna().sum())
    print(f"  EF_VALUE gated to NaN         : {n_gated:,} of {before:,} "
          f"({100 * n_gated / before:.3f}%) outside [{lo:g}, {hi:g}]")
    print(f"  reading years                 : {ef['event_date'].dt.year.min()}.."
          f"{ef['event_date'].dt.year.max()}")

    m = ef.merge(coh[["id", "idx", "lo"]], left_on="MRN", right_on="id", how="inner")
    m = m.loc[(m["event_date"] >= m["lo"]) & (m["event_date"] < m["idx"])]
    print(f"  in-window readings            : {len(m):,}   patients={m['id'].nunique():,}")

    out = m.groupby("id", as_index=False).agg(ef_value=("EF_VALUE", "mean"),
                                              n_ef=("EF_VALUE", "size"))
    return out, n_gated


# %%
def agreement(win: pd.DataFrame, ef_pat: pd.DataFrame) -> None:
    """Per-patient echo LVEF vs ef.csv EF on patients with both in-window -> Bland-Altman."""
    echo_ef = win.groupby("id", as_index=False)["LVEF_pct"].mean().dropna()
    both = echo_ef.merge(ef_pat[["id", "ef_value"]].dropna(), on="id", how="inner")
    diff = both["ef_value"] - both["LVEF_pct"]                 # ef.csv minus echo.csv
    r = both["ef_value"].corr(both["LVEF_pct"])
    rho = both["ef_value"].corr(both["LVEF_pct"], method="spearman")
    md, sd = diff.mean(), diff.std(ddof=1)
    row = pd.DataFrame([{
        "n_patients_both": len(both), "pearson_r": round(r, 4), "spearman_rho": round(rho, 4),
        "mean_diff_ef_minus_echo": round(md, 4), "sd_diff": round(sd, 4),
        "median_abs_diff": round(diff.abs().median(), 4),
        "loa_lower": round(md - 1.96 * sd, 4), "loa_upper": round(md + 1.96 * sd, 4),
        "pct_within_5_points": round(100 * (diff.abs() <= 5).mean(), 2),
        "mean_echo_LVEF_pct": round(both["LVEF_pct"].mean(), 3),
        "mean_ef_value": round(both["ef_value"].mean(), 3),
    }])
    out_csv = U.ANALYSIS / "ef_source_agreement.csv"
    row.to_csv(out_csv, index=False)
    print("\nEF SOURCE AGREEMENT (per-patient in-window means; ef.csv minus echo.csv):")
    print(f"  patients with both            : {len(both):,}")
    print(f"  Pearson r                     : {r:.4f}   (Spearman {rho:.4f})")
    print(f"  mean difference               : {md:+.3f} EF points   SD {sd:.3f}")
    print(f"  Bland-Altman 95% LoA          : [{md - 1.96 * sd:+.2f}, {md + 1.96 * sd:+.2f}] "
          f"EF points")
    print(f"  median |difference|           : {diff.abs().median():.3f}   within 5 points: "
          f"{100 * (diff.abs() <= 5).mean():.1f}%")
    print("  -> LoA this wide means the two sources are NOT interchangeable patient-by-patient;")
    print("     they stay separate features. -> " + str(out_csv))


# %%
def main() -> None:
    U.ensure_dirs()
    print("=" * 78)
    print("ECHO ARM (echo.csv + ef.csv) - 5yr pre-index window")
    print("Conditional on having had an echo, which tracks cardiac symptoms; echo rate by")
    print("severity is non-monotonic and severe-OSA patients have ~18% FEWER encounters, so")
    print("ascertainment bias runs OPPOSITE the expected effect. Availability is also coupled")
    print("to index year -> study_year adjustment is mandatory downstream.")
    print("=" * 78)

    coh = pd.read_parquet(U.COHORT, columns=["id", "STUDY_DATE", "study_year", "severity"])
    coh["idx"] = coh["STUDY_DATE"].dt.normalize()
    coh["lo"] = coh["idx"] - pd.DateOffset(years=U.WINDOW_YEARS)
    n_coh = len(coh)

    # ---- echo studies: read, age filter, window ----
    echo = pd.read_parquet(U.ECHO_CLEAN, columns=["id", "echo_date", "age_performed"]
                           + RAW_ECHO_COLS)
    echo["echo_date"] = pd.to_datetime(echo["echo_date"])
    print(f"\necho studies read               : {len(echo):,}")

    win = echo.merge(coh[["id", "idx", "lo"]], on="id", how="inner")
    win = win.loc[(win["echo_date"] >= win["lo"]) & (win["echo_date"] < win["idx"])].copy()
    print(f"  in-window studies             : {len(win):,}   "
          f"patients={win['id'].nunique():,}  (step 00 reported 44,412 / 30,627)")

    # Study-level age plausibility filter (the one row filter in this script). FLAGGED: the
    # cohort is adults-only AT INDEX, but the window reaches back 5 years, so a 14-17 y/o echo on
    # a patient whose index age is 18-22 is legitimate, not an error - and [18,105] discards it.
    # Only the negative ages are unambiguously bad. Kept as specified; widen U.ECHO_AGE_GATE to
    # (13, 105) to retain them.
    alo, ahi = U.ECHO_AGE_GATE
    bad_age = win["age_performed"].notna() & ~win["age_performed"].between(alo, ahi)
    n_impl = int((win.loc[bad_age, "age_performed"] < 0).sum())
    n_pat_lost = win["id"].nunique() - win.loc[~bad_age, "id"].nunique()
    win = win.loc[~bad_age].copy()
    n_win_std, n_win_pat = len(win), win["id"].nunique()
    print(f"  age_performed outside [{alo:g}, {ahi:g}]: {int(bad_age.sum()):,} studies dropped "
          f"({n_impl} of them negative, i.e. truly implausible; the rest are real 14-17 y/o")
    print(f"    echoes on patients aged 18+ at index) -> {n_pat_lost} patients lost entirely")
    print(f"  in-window studies analysed    : {n_win_std:,}   patients={n_win_pat:,} "
          f"(expect ~30,621)")

    # ---- unit repair, input gates, derivations ----
    print("\nunit repair + derivation input gates:")
    win = derive(win)

    # ---- per-feature gates: out-of-range -> NaN, never a row drop ----
    print(f"\nGATING AUDIT - study-level values nulled per feature "
          f"(n = {n_win_std:,} in-window studies):")
    print(f"  {'feature':<24} {'gate':<16} {'nonnull_pre':>12} {'nulled':>9} {'%':>7} "
          f"{'nonnull_post':>13}")
    gated = {}
    for f in U.ECHO_FEATURES:
        name = f["name"]
        if f["kind"] == "ef":
            continue
        lo, hi = f["gate"]
        pre = int(win[name].notna().sum())
        win.loc[win[name].notna() & ~win[name].between(lo, hi), name] = np.nan
        post = int(win[name].notna().sum())
        gated[name] = pre - post
        print(f"  {name:<24} [{lo:g}, {hi:g}]".ljust(44)
              + f"{pre:>12,} {pre - post:>9,} {100 * (pre - post) / max(pre, 1):>6.2f}% "
                f"{post:>13,}")

    # ---- one row per patient: mean across in-window studies ----
    echo_feats = [f["name"] for f in U.ECHO_FEATURES if f["kind"] != "ef"]
    out = win.groupby("id", as_index=False)[echo_feats].mean()
    out["n_echo"] = win.groupby("id").size().reindex(out["id"]).to_numpy()
    print(f"\nechoes per patient              : mean={out['n_echo'].mean():.2f} "
          f"median={out['n_echo'].median():.0f} max={out['n_echo'].max():,}   "
          f">1 echo: {100 * (out['n_echo'] > 1).mean():.1f}%")

    # ---- ef.csv, windowed + reduced the same way, kept as a SEPARATE feature ----
    ef_pat, n_ef_gated = load_ef(coh)
    gated["ef_value"] = n_ef_gated
    agreement(win, ef_pat)

    # ---- assemble + write ----
    res = out.merge(ef_pat, on="id", how="outer")
    res["n_echo"] = res["n_echo"].fillna(0).astype("int64")
    res["n_ef"] = res["n_ef"].fillna(0).astype("int64")
    res[FEATURES] = res[FEATURES].astype("float64")
    res = res[["id", "n_echo", "n_ef"] + FEATURES].sort_values("id").reset_index(drop=True)
    res.to_parquet(U.ECHO_FEAT, index=False)

    n_pat = len(res)
    n_e, n_f = int((res["n_echo"] > 0).sum()), int((res["n_ef"] > 0).sum())
    n_both = int(((res["n_echo"] > 0) & (res["n_ef"] > 0)).sum())
    print(f"\nwrote {U.ECHO_FEAT}")
    print(f"patients in output              : {n_pat:,} of {n_coh:,} cohort "
          f"({100 * n_pat / n_coh:.2f}%)")
    print(f"  with >=1 in-window echo       : {n_e:,} ({100 * n_e / n_coh:.2f}%) "
          f"[expect ~30,621]")
    print(f"  with >=1 in-window ef reading : {n_f:,} ({100 * n_f / n_coh:.2f}%)")
    print(f"  with both                     : {n_both:,} ({100 * n_both / n_coh:.2f}%)")
    print(f"  echo only / ef only           : {n_e - n_both:,} / {n_f - n_both:,}")

    # ---- per-feature coverage vs the MIN_PATIENTS floor ----
    print(f"\nper-feature patient coverage (floor = U.MIN_PATIENTS = {U.MIN_PATIENTS}):")
    print(f"  {'feature':<24} {'concept':<38} {'n_pts':>7} {'%coh':>6}  pass")
    rows = []
    for f in U.ECHO_FEATURES:
        name = f["name"]
        v = res[name]
        n = int(v.notna().sum())
        rows.append({
            "feature": name, "concept": f["concept"], "kind": f["kind"],
            "sources": "|".join(f["sources"]), "gate_low": f["gate"][0], "gate_high": f["gate"][1],
            "n_patients": n, "pct_cohort": round(100 * n / n_coh, 2),
            "pct_of_retained": round(100 * n / n_pat, 2),
            "n_values_gated_to_nan": gated[name],
            "passes_min_patients": bool(n >= U.MIN_PATIENTS),
            "p1": v.quantile(0.01), "median": v.median(), "p99": v.quantile(0.99),
            "era_availability": f["era"], "note": f["note"],
        })
        print(f"  {name:<24} {f['concept'][:38]:<38} {n:>7,} {100 * n / n_coh:>5.2f}%  "
              f"{'YES' if n >= U.MIN_PATIENTS else 'NO'}")
    comp = pd.DataFrame(rows).sort_values("n_patients", ascending=False)
    out_csv = U.ANALYSIS / "echo_feature_completeness.csv"
    comp.to_csv(out_csv, index=False)
    n_pass = int(comp["passes_min_patients"].sum())
    print(f"  -> {n_pass}/{len(FEATURES)} features clear the {U.MIN_PATIENTS}-patient floor")
    print(f"  -> {out_csv}")

    # ---- severity composition + the ascertainment pattern ----
    ret = coh.merge(res[["id", "n_echo", "n_ef"]], on="id", how="inner")
    print("\nseverity composition of retained patients, and the echo rate WITHIN each severity")
    print("(the ascertainment pattern - known to be non-monotonic):")
    print(f"  {'severity':<9} {'cohort':>8} {'retained':>9} {'%ret':>6} {'echo>=1':>8} "
          f"{'echo rate':>10} {'ef>=1':>8} {'ef rate':>8}")
    for sev in U.SEVERITIES:
        base = int((coh["severity"] == sev).sum())
        sub = ret.loc[ret["severity"] == sev]
        k, ke, kf = len(sub), int((sub["n_echo"] > 0).sum()), int((sub["n_ef"] > 0).sum())
        print(f"  {sev:<9} {base:>8,} {k:>9,} {100 * k / n_pat:>5.1f}% {ke:>8,} "
              f"{100 * ke / base:>9.1f}% {kf:>8,} {100 * kf / base:>7.1f}%")

    # ---- index-year coverage: the era/index-year confound, on the record ----
    print("\nindex-year coverage (echo availability is mechanically coupled to index year):")
    print(f"  {'year':<6} {'cohort':>8} {'echo>=1':>9} {'rate':>7} {'ef>=1':>9} {'rate':>7}")
    yr = pd.DataFrame({
        "cohort": coh["study_year"].value_counts(),
        "echo": ret.loc[ret["n_echo"] > 0, "study_year"].value_counts(),
        "ef": ret.loc[ret["n_ef"] > 0, "study_year"].value_counts(),
    }).fillna(0).astype(int).sort_index()
    for y, r in yr.iterrows():
        print(f"  {y:<6} {r['cohort']:>8,} {r['echo']:>9,} "
              f"{100 * r['echo'] / max(r['cohort'], 1):>6.1f}% {r['ef']:>9,} "
              f"{100 * r['ef'] / max(r['cohort'], 1):>6.1f}%")

    U.log_milestone(
        f"Step 02 echo+EF features ({U.WINDOW_LABEL} pre-index): {n_win_std:,} in-window echo "
        f"studies on {n_e:,} patients + ef.csv on {n_f:,} patients -> {n_pat:,}/{n_coh:,} "
        f"({100 * n_pat / n_coh:.1f}%) in echo_5yr.parquet; {n_pass}/{len(FEATURES)} features "
        f"clear the {U.MIN_PATIENTS} floor. ef.csv kept SEPARATE from the echo LVEF "
        f"(agreement in analysis/ef_source_agreement.csv).")


if __name__ == "__main__":
    main()
