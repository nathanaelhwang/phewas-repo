# %% [markdown]
# Step 04 - UNADJUSTED significance of utilization vs OSA severity (None<Mild<Moderate<Severe),
# per analysis x window, off the windowed store.
#
# Three complementary tests (per analysis x window):
#   omnibus  : Kruskal-Wallis across the 4 severity groups on per-patient encounter COUNT
#              ("do the groups differ at all?"). Effect size = epsilon^2.
#   trend    : Cuzick / linear-rank trend (tie-corrected) on per-patient count with scores
#              None=0..Severe=3 ("is there a monotonic dose-response?"). Effect r = Z/sqrt(N-1).
#   any-use  : Cochran-Armitage trend on the 0/1 "has >=1 encounter" indicator (extensive margin).
# Each of omnibus+trend is run on BOTH margins:
#   population  = all cohort patients (zeros included) -> matches per_pt_per_yr
#   among_users = patients with >=1 encounter only     -> matches per_user_per_yr (intensity)
#
# CAVEATS (printed): N ~ 222k so p-values are tiny for any real gradient -- read effect sizes and
# per-group medians, not just p. UNADJUSTED for age/sex/BMI (confounders of both OSA & utilization);
# the adjusted regression is the real dose-response test. 5yr counts are partially confounded by
# varying pre-index lookback (obs_window); 1yr is clean (all patients have >=1yr).
#
# Emits results/utilwas_results/analysis/tables/significance.csv (tests) and
# severity_descriptives.csv (per-group n / prevalence / median / mean). DE-IDENTIFIED.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata, norm

import utilwas_utils as U

ICD = U.sql_in(U.ICD_CODE_SUBTYPES)
NONICD = f"COALESCE(CODE_SUBTYPE,'~NULL~') NOT IN ({ICD})"
OUTPATIENT_WHERE = f"care_type IN ({U.sql_in(U.OUTPATIENT_CARE_TYPES)}) AND care_subtype IN ({U.sql_in(U.OUTPATIENT_SUBTYPES)})"
ANALYSES = [
    ("physician_level", f"{OUTPATIENT_WHERE} AND has_icd = 1"),
    ("allied_health",   f"{OUTPATIENT_WHERE} AND has_nonicd = 1"),
    ("other",           f"care_type = '{U.OTHER_CARE_TYPE}'"),
    ("messages_phone",  f"care_subtype IN ({U.sql_in(U.MESSAGES_SUBTYPES)})"),
    ("ED",              f"care_subtype IN ({U.sql_in(U.ED_SUBTYPES)})"),
    ("inpatient",       f"care_subtype IN ({U.sql_in(U.INPATIENT_SUBTYPES)})"),
    ("urgent_care",     "is_urgent = 1"),
]
GROUPS = U.SEVERITIES                       # ["None","Mild","Moderate","Severe"], ordered
SCORE = {g: i for i, g in enumerate(GROUPS)}
MIN_USERS = 20                              # skip among-users test if a group has < this many


def cuzick_trend(values: np.ndarray, scores: np.ndarray):
    """Linear-rank (Cuzick) trend test, tie-corrected. Returns (Z, two-sided p, r)."""
    n = len(values)
    if n < 3 or np.ptp(scores) == 0:
        return np.nan, np.nan, np.nan
    R = rankdata(values)
    z = scores.astype(float)
    ss_z = ((z - z.mean()) ** 2).sum()
    ss_R = ((R - R.mean()) ** 2).sum()
    if ss_z == 0 or ss_R == 0:
        return np.nan, np.nan, np.nan
    T = float((z * R).sum())
    ET = n * z.mean() * R.mean()
    varT = ss_z * ss_R / (n - 1)
    Z = (T - ET) / np.sqrt(varT)
    return Z, 2 * norm.sf(abs(Z)), Z / np.sqrt(n - 1)


def cochran_armitage(binary: np.ndarray, scores: np.ndarray):
    """Cochran-Armitage trend test on a 0/1 outcome across ordered scores. (Z, two-sided p)."""
    N = len(binary)
    M = float(binary.sum())
    if M == 0 or M == N:
        return np.nan, np.nan
    z = scores.astype(float)
    T = float((binary * z).sum())
    sz, sz2 = float(z.sum()), float((z * z).sum())
    ET = M / N * sz
    varT = M * (N - M) / (N * N * (N - 1)) * (N * sz2 - sz * sz)
    if varT <= 0:
        return np.nan, np.nan
    Z = (T - ET) / np.sqrt(varT)
    return Z, 2 * norm.sf(abs(Z))


def kruskal_eps2(values: np.ndarray, ord_arr: np.ndarray):
    """Kruskal-Wallis across severity groups. Returns (H, p, epsilon^2)."""
    samples = [values[ord_arr == i] for i in range(len(GROUPS)) if (ord_arr == i).any()]
    if len(samples) < 2 or all(np.ptp(s) == 0 for s in samples) or len(values) < 2:
        return np.nan, np.nan, np.nan
    H, p = kruskal(*samples)
    k, n = len(samples), len(values)
    return H, p, (H - k + 1) / (n - k)


def direction(z: float) -> str:
    return "increasing" if z > 0 else ("decreasing" if z < 0 else "flat")


def main() -> None:
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=96, threads=12)

    # encounter-level table (as in step 02)
    con.execute(f"""
        CREATE TEMP TABLE enc AS
        SELECT id, UTIL_ID, any_value(severity) AS severity, bool_or(in_1yr) AS in_1yr,
               any_value(CARE_TYPE) AS care_type, any_value(CARE_SUBTYPE) AS care_subtype,
               max(CASE WHEN CODE_SUBTYPE IN ({ICD}) THEN 1 ELSE 0 END) AS has_icd,
               max(CASE WHEN {NONICD} THEN 1 ELSE 0 END) AS has_nonicd,
               max(CASE WHEN DEPT_SPECIALTY = '{U.URGENT_CARE_SPECIALTY}' THEN 1 ELSE 0 END) AS is_urgent
        FROM read_parquet('{U.STORE_5YR}') GROUP BY id, UTIL_ID
    """)
    cohort = con.sql(f"SELECT id, severity FROM read_parquet('{U.COHORT}')").df()
    cohort["ord"] = cohort["severity"].map(SCORE).astype("int8")

    tests, descr = [], []
    for name, where in ANALYSES:
        # per-patient 1yr & 5yr counts for this analysis (only patients with >=1 in 5yr)
        pc = con.sql(f"""
            SELECT id, sum(CASE WHEN in_1yr THEN 1 ELSE 0 END) AS c1, count(*) AS c5
            FROM enc WHERE {where} GROUP BY id
        """).df()
        m = cohort.merge(pc, on="id", how="left").fillna({"c1": 0, "c5": 0})
        m["c1"] = m["c1"].astype("int32"); m["c5"] = m["c5"].astype("int32")
        ordv = m["ord"].values

        for wlabel, col in [("1yr", "c1"), ("5yr", "c5")]:
            vals = m[col].values

            # per-group descriptives (population + among-users)
            for g in GROUPS:
                gv = vals[m["severity"].values == g]
                users = gv[gv > 0]
                descr.append(dict(analysis=name, window=wlabel, severity=g, n_pat=len(gv),
                                  n_users=int((gv > 0).sum()), pct_any=round((gv > 0).mean(), 4),
                                  median_all=float(np.median(gv)), mean_all=round(float(gv.mean()), 3),
                                  median_users=(float(np.median(users)) if len(users) else np.nan),
                                  mean_users=(round(float(users.mean()), 3) if len(users) else np.nan)))

            # any-use (extensive margin): Cochran-Armitage trend
            anyv = (vals > 0).astype(np.int8)
            zca, pca = cochran_armitage(anyv, ordv)
            tests.append(dict(analysis=name, window=wlabel, margin="any_use", test="cochran_armitage_trend",
                              statistic=round(zca, 3) if zca == zca else np.nan, p_value=pca,
                              effect=np.nan, direction=direction(zca) if zca == zca else "", n=len(vals)))

            # population (with zeros): KW omnibus + Cuzick trend
            H, pk, eps2 = kruskal_eps2(vals, ordv)
            tests.append(dict(analysis=name, window=wlabel, margin="population", test="kruskal_omnibus",
                              statistic=round(H, 2) if H == H else np.nan, p_value=pk,
                              effect=round(eps2, 5) if eps2 == eps2 else np.nan, direction="", n=len(vals)))
            zc, pc, r = cuzick_trend(vals, ordv)
            tests.append(dict(analysis=name, window=wlabel, margin="population", test="cuzick_trend",
                              statistic=round(zc, 3) if zc == zc else np.nan, p_value=pc,
                              effect=round(r, 5) if r == r else np.nan,
                              direction=direction(zc) if zc == zc else "", n=len(vals)))

            # among-users (intensive margin): KW + Cuzick on count>0
            mask = vals > 0
            g_ok = all((mask & (ordv == i)).sum() >= MIN_USERS for i in range(len(GROUPS)))
            if mask.sum() >= 3 and g_ok:
                uv, uo = vals[mask], ordv[mask]
                H2, pk2, eps22 = kruskal_eps2(uv, uo)
                tests.append(dict(analysis=name, window=wlabel, margin="among_users", test="kruskal_omnibus",
                                  statistic=round(H2, 2) if H2 == H2 else np.nan, p_value=pk2,
                                  effect=round(eps22, 5) if eps22 == eps22 else np.nan, direction="", n=int(mask.sum())))
                zc2, pc2, r2 = cuzick_trend(uv, uo)
                tests.append(dict(analysis=name, window=wlabel, margin="among_users", test="cuzick_trend",
                                  statistic=round(zc2, 3) if zc2 == zc2 else np.nan, p_value=pc2,
                                  effect=round(r2, 5) if r2 == r2 else np.nan,
                                  direction=direction(zc2) if zc2 == zc2 else "", n=int(mask.sum())))

    tests = pd.DataFrame(tests)
    descr = pd.DataFrame(descr)
    U.TABLES.mkdir(parents=True, exist_ok=True)
    tests.to_csv(U.TABLES / "significance.csv", index=False)
    descr.to_csv(U.TABLES / "severity_descriptives.csv", index=False)

    # ---- console summary (de-identified) -----------------------------------
    pd.set_option("display.width", 200)
    print("N ~ 222k -> p-values are tiny for any real gradient; read EFFECT + per-group medians.\n"
          "UNADJUSTED for age/sex/BMI. 5yr partially confounded by lookback (obs_window).\n")
    print("=== trend tests (Cuzick population, Cochran-Armitage any-use, Cuzick among-users) ===")
    show = tests[tests.test.isin(["cuzick_trend", "cochran_armitage_trend"])].copy()
    piv = show.assign(key=show.margin).pivot_table(
        index=["analysis", "window"], columns="key", values="p_value", aggfunc="first")
    eff = (tests[(tests.test == "cuzick_trend") & (tests.margin == "population")]
           .set_index(["analysis", "window"])[["effect", "direction"]])
    out = piv.join(eff).reset_index()
    print(out.to_string(index=False))

    U.log_milestone("Step 04 significance: KW omnibus + Cuzick trend (population & among-users) + "
                    "Cochran-Armitage any-use trend, per analysis x window (unadjusted).")


if __name__ == "__main__":
    main()
