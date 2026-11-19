# %% [markdown]
# Step 6 - Write SUMMARY.md (de-identified) and run the verification suite.
# Structural checks are hard asserts; LOINC spot-checks are warnings (the
# name-only mapping is approximate by design).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import labwas_utils as U

EXPECT = {  # descriptor (upper) -> expected LOINC for spot-checks
    "718-7": ["HGB", "HB", "HEMOGLOBIN"],
    "2345-7": ["GLUCOSE", "GLU"],
    "2160-0": ["CREATININE", "CR", "CREAT"],
    "2951-2": ["SODIUM", "NA"],
    "2823-3": ["POTASSIUM", "K"],
}


def main() -> None:
    try:  # Windows console defaults to cp1252; make prints unicode-safe
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    lines: list[str] = []

    def p(msg: str = "") -> None:
        print(msg)
        lines.append(str(msg))

    R = U.RESULTS
    m1 = pd.read_parquet(R / "labs_for_phewas_1yr.parquet")
    m5 = pd.read_parquet(R / "labs_for_phewas_5yrs.parquet")
    a1 = pd.read_parquet(U.LABS_DIR / "labs_agg_1yr_long.parquet")
    a5 = pd.read_parquet(U.LABS_DIR / "labs_agg_5yrs_long.parquet")
    anchor = pd.read_parquet(U.COHORT_DIR / "lab_anchor.parquet")
    numeric = pd.read_parquet(U.LABS_DIR / "labs_long_numeric.parquet")
    quar = pd.read_parquet(U.LABS_DIR / "labs_nonnumeric_quarantine.parquet")
    cw = pd.read_csv(U.CROSSWALK_DIR / "loinc_crosswalk.csv", dtype=str).fillna("")
    ddict = pd.read_csv(R / "column_data_dictionary.csv")

    # ---------- SUMMARY ----------
    p("# labs_for_phewas — build summary")
    p("")
    p(f"- inclusion threshold (per window): >= {U.MIN_PATIENTS} patients AND >= {U.MIN_OBS} observations per analyte")
    p(f"- anchored MRNs (index study >= {U.MIN_STUDY_YEAR}): {len(anchor):,}")
    p(f"- distinct de-exploded numeric lab events: {len(numeric):,}")
    p(f"- quarantined non-numeric lab events: {len(quar):,}")
    n_evt = len(numeric) + len(quar)
    p(f"  - numeric fraction: {100 * len(numeric) / n_evt:.1f}%")
    p("")
    p("## Crosswalk")
    n_desc = len(cw)
    n_map = int((cw["loinc_code"] != "").sum())
    n_high = int(((cw["loinc_code"] != "") & (cw["confidence"] == "high")).sum())
    p(f"- descriptors: {n_desc:,}; mapped: {n_map:,} ({100 * n_map / n_desc:.1f}%); "
      f"high-confidence: {n_high:,}")
    p(f"- distinct LOINC analytes in matrices: {len(ddict):,}")
    p("")
    for label, m, a in [("1yr", m1, a1), ("5yrs", m5, a5)]:
        p(f"## labs_for_phewas_{label}")
        p(f"- MRNs (rows): {len(m):,}")
        p(f"- LOINC analytes (kept): {sum(c.endswith('_mean') for c in m.columns):,}")
        p(f"- feature columns: {sum(c.endswith(('_mean', '_count')) for c in m.columns):,}")
        p("")

    p("## Top 15 analytes by 5yr coverage")
    top = ddict.sort_values("n_mrn_5yrs", ascending=False).head(15)
    for r in top.itertuples(index=False):
        p(f"- {r.loinc_code}  {str(r.loinc_long_common_name)[:60]}  (n_mrn_5yrs={r.n_mrn_5yrs:,})")
    p("")

    p("## Top 10 quarantined descriptors (non-numeric)")
    qtop = quar.groupby("lab").size().sort_values(ascending=False).head(10)
    for lab, c in qtop.items():
        p(f"- {lab}: {c:,} events")

    (R / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {R / 'SUMMARY.md'}")

    # ---------- VERIFICATION ----------
    print("\n==== verification ====")
    fails = []

    def check(name: str, ok: bool) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)

    check("1yr id unique", m1["id"].is_unique)
    check("5yrs id unique", m5["id"].is_unique)
    check("ids are 12-char", (m1["id"].str.len() == 12).all() and (m5["id"].str.len() == 12).all())
    check("index_date >= 2010",
          (pd.to_datetime(m1["index_date"]).dt.year >= U.MIN_STUDY_YEAR).all())
    check("de-explosion held (< 200M numeric rows)", len(numeric) < 200_000_000)

    cnt_iff_mean = True
    for m in (m1, m5):
        for c in [c for c in m.columns if c.endswith("_mean")]:
            cc = c[:-5] + "_count"
            if not (m[c].notna() == m[cc].notna()).all():
                cnt_iff_mean = False
                break
    check("count present iff mean present", cnt_iff_mean)

    mg = a1.merge(a5, on=["id", "loinc_code"], suffixes=("_1", "_5"))
    check("5yr count >= 1yr count (windowed monotonicity)", bool((mg["n_obs_5"] >= mg["n_obs_1"]).all()))

    # LOINC spot-checks (warnings only)
    print("  -- LOINC spot-checks (warnings) --")
    cw_up = cw.assign(u=cw["lab"].str.upper().str.strip())
    for code, names in EXPECT.items():
        hit = cw_up[cw_up["u"].isin(names)]
        mapped = sorted(set(hit["loinc_code"]) - {""})
        status = "OK" if code in mapped else ("MISS" if len(hit) else "n/a-descriptor-absent")
        print(f"    {names[0]:>12} -> expect {code:>8}: {status}  (got {mapped or 'none'})")

    U.log_milestone(f"Step 6 verify: {len(fails)} hard-check failures; "
                    f"1yr={len(m1):,} MRNs, 5yrs={len(m5):,} MRNs.")
    if fails:
        raise SystemExit(f"VERIFICATION FAILED: {fails}")
    print("\nAll hard checks passed.")


if __name__ == "__main__":
    main()
