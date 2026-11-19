# %% [markdown]
# Step 05 - Per-category diagnosis counts, phecode-mapped, off the windowed store.
#
# For five categories, count how often each disease (phecode) appears as the requested diagnosis
# type, using the SAME pyPheWAS v1.2 ICD->phecode maps as the ICD PheWAS (icd9 + icd10_beta):
#   physician_level : PRIMARY diagnosis (PRIMARY_DX_IND='Y')  -- outpatient encounters
#   allied_health   : PRIMARY diagnosis (PRIMARY_DX_IND='Y')  -- outpatient enc with >=1 non-ICD row
#   ED              : SECONDARY diagnosis (PRIN_DX_IND='N')
#   inpatient       : PRIMARY/principal diagnosis (PRIN_DX_IND='Y')
#   urgent_care     : SECONDARY diagnosis (PRIMARY_DX_IND='N')  -- Urgent Care specialty encounters
# (PRIMARY_DX_IND = professional/outpatient primary flag; PRIN_DX_IND = facility principal flag.)
#
# Count grain = distinct encounters (UTIL_ID) + distinct patients (id) carrying that phecode, per
# window (1yr, 5yr), Overall. Only ICD9/ICD10 rows map; non-disease Z/V "encounter" codes have no
# phecode (dropped by design) -> coverage reported per category in dx_coverage.csv.
#
# Emits results/utilwas_results/analysis/tables/dx_<category>.csv + dx_coverage.csv. DE-IDENTIFIED
# (disease names + counts only; no patient rows).

# %%
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import utilwas_utils as U

ICD = U.sql_in(U.ICD_CODE_SUBTYPES)
NONICD = f"COALESCE(CODE_SUBTYPE,'~NULL~') NOT IN ({ICD})"
_res = os.path.join(os.path.dirname(importlib.util.find_spec("pyPheWAS").origin), "resources")
ICD9_MAP = os.path.join(_res, "phecode_map_v1_2_icd9.csv").replace("\\", "/")
ICD10_MAP = os.path.join(_res, "phecode_map_v1_2_icd10_beta.csv").replace("\\", "/")

OUTP = f"CARE_TYPE IN ({U.sql_in(U.OUTPATIENT_CARE_TYPES)}) AND CARE_SUBTYPE IN ({U.sql_in(U.OUTPATIENT_SUBTYPES)})"
# (name, dx-type label, WHERE over the dxall table -- excludes the `mapped` condition)
CATEGORIES = [
    ("physician_level", "primary (PRIMARY_DX_IND=Y)",   f"{OUTP} AND pdx='Y'"),
    ("allied_health",   "primary (PRIMARY_DX_IND=Y)",   f"{OUTP} AND has_nonicd=1 AND pdx='Y'"),
    ("ED",              "secondary (PRIN_DX_IND=N)",     f"CARE_SUBTYPE IN ({U.sql_in(U.ED_SUBTYPES)}) AND prin='N'"),
    ("inpatient",       "primary (PRIN_DX_IND=Y)",       f"CARE_SUBTYPE IN ({U.sql_in(U.INPATIENT_SUBTYPES)}) AND prin='Y'"),
    ("urgent_care",     "secondary (PRIMARY_DX_IND=N)",  "is_urgent=1 AND pdx='N'"),
]


def main() -> None:
    U.ensure_dirs()
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=96, threads=12)

    # ICD -> phecode lookup (v1.2 icd9 + icd10_beta), deduped on (type, code)
    con.execute(f"""
        CREATE TEMP TABLE map AS
        SELECT cst, code, any_value(phecode) phecode, any_value(phenotype) phenotype,
               any_value(category_string) category
        FROM (
          SELECT 'ICD9' cst, upper(trim("ICD9")) code, CAST("PheCode" AS VARCHAR) phecode,
                 "Phenotype" phenotype, category_string
          FROM read_csv_auto('{ICD9_MAP}', ALL_VARCHAR=TRUE) WHERE "ICD9" IS NOT NULL
          UNION ALL
          SELECT 'ICD10', upper(trim("ICD10")), CAST("PheCode" AS VARCHAR), "Phenotype", category_string
          FROM read_csv_auto('{ICD10_MAP}', ALL_VARCHAR=TRUE) WHERE "ICD10" IS NOT NULL
        ) GROUP BY cst, code
    """)

    # per-encounter flags (need non-ICD rows for allied, DEPT_SPECIALTY for urgent)
    con.execute(f"""
        CREATE TEMP TABLE encflag AS
        SELECT UTIL_ID,
               max(CASE WHEN {NONICD} THEN 1 ELSE 0 END) AS has_nonicd,
               max(CASE WHEN DEPT_SPECIALTY = '{U.URGENT_CARE_SPECIALTY}' THEN 1 ELSE 0 END) AS is_urgent
        FROM read_parquet('{U.STORE_5YR}') GROUP BY UTIL_ID
    """)

    # all ICD diagnosis rows + normalized flags + phecode (NULL if unmapped)
    con.execute(f"""
        CREATE TEMP TABLE dxall AS
        SELECT s.id, s.UTIL_ID, s.in_1yr, s.severity, s.CARE_TYPE, s.CARE_SUBTYPE,
               upper(trim(s.PRIMARY_DX_IND)) AS pdx, upper(trim(s.PRIN_DX_IND)) AS prin,
               e.has_nonicd, e.is_urgent,
               upper(trim(s.CODE)) AS code, upper(trim(s.CODE_SUBTYPE)) AS cst,
               m.phecode, m.phenotype, m.category, (m.phecode IS NOT NULL) AS mapped
        FROM read_parquet('{U.STORE_5YR}') s
        JOIN encflag e ON s.UTIL_ID = e.UTIL_ID
        LEFT JOIN map m ON upper(trim(s.CODE_SUBTYPE)) = m.cst AND upper(trim(s.CODE)) = m.code
        WHERE upper(trim(s.CODE_SUBTYPE)) IN ({ICD})
    """)

    # cohort denominators (for the per-patient-per-year rate = encounters / N_cohort[sev] / years)
    den = con.sql(f"SELECT severity, count(*) n FROM read_parquet('{U.COHORT}') GROUP BY 1").df()
    N = {r.severity: int(r.n) for r in den.itertuples()}
    N["Overall"] = int(sum(N.values()))
    YEARS = {"5yr": 5, "1yr": 1}

    SEV_ORDER = ["Overall"] + U.SEVERITIES
    METRIC_PREFIX = {"n_enc_5yr": "enc_5yr", "n_enc_1yr": "enc_1yr",
                     "n_pat_5yr": "pat_5yr", "n_pat_1yr": "pat_1yr"}

    def dx_table(where: str) -> pd.DataFrame:
        # distinct-count per (phecode, severity); Overall = sum across severities (patients/encounters
        # partition by severity, since each patient has exactly one severity).
        long = con.sql(f"""
            SELECT phecode, any_value(phenotype) AS phenotype, any_value(category) AS category, severity,
                   count(DISTINCT UTIL_ID) FILTER (WHERE in_1yr) AS n_enc_1yr,
                   count(DISTINCT id)      FILTER (WHERE in_1yr) AS n_pat_1yr,
                   count(DISTINCT UTIL_ID) AS n_enc_5yr,
                   count(DISTINCT id)      AS n_pat_5yr
            FROM dxall WHERE mapped AND ({where})
            GROUP BY phecode, severity
        """).df()
        if long.empty:
            return long
        metrics = list(METRIC_PREFIX)
        meta = long.groupby("phecode")[["phenotype", "category"]].first()
        wide = long.pivot(index="phecode", columns="severity", values=metrics)
        wide = wide.reindex(columns=pd.MultiIndex.from_product([metrics, U.SEVERITIES])).fillna(0)
        out = {}
        for m in metrics:
            p, block = METRIC_PREFIX[m], wide[m]
            out[f"{p}_Overall"] = block.sum(axis=1)
            for s in U.SEVERITIES:
                out[f"{p}_{s}"] = block[s]
        res = pd.DataFrame(out).astype("int64").join(meta).reset_index()
        # per-patient-per-year rate: encounters / N_cohort[severity] / window-years (population denom)
        rate_cols = []
        for win in ("5yr", "1yr"):
            for s in SEV_ORDER:
                col = f"rate_{win}_{s}"
                res[col] = (res[f"enc_{win}_{s}"] / N[s] / YEARS[win]).round(6)
                rate_cols.append(col)
        enc_cols = [f"enc_{win}_{s}" for win in ("5yr", "1yr") for s in SEV_ORDER]
        pat_cols = [f"pat_{win}_{s}" for win in ("5yr", "1yr") for s in SEV_ORDER]
        res = res[["phecode", "phenotype", "category"] + enc_cols + rate_cols + pat_cols]
        return res.sort_values("enc_5yr_Overall", ascending=False).reset_index(drop=True)

    def coverage(name: str, dxtype: str, where: str) -> dict:
        tot, mp, nphe = con.sql(f"""
            SELECT count(*), sum(CASE WHEN mapped THEN 1 ELSE 0 END),
                   count(DISTINCT CASE WHEN mapped THEN phecode END)
            FROM dxall WHERE {where}""").fetchone()
        zv = con.sql(f"""SELECT sum(CASE WHEN NOT mapped AND (code LIKE 'Z%' OR code LIKE 'V%')
                         THEN 1 ELSE 0 END) FROM dxall WHERE {where}""").fetchone()[0] or 0
        return dict(analysis=name, dx_type=dxtype, dx_rows=int(tot), mapped_rows=int(mp or 0),
                    pct_mapped=round(100 * (mp or 0) / tot, 2) if tot else None,
                    n_phecodes=int(nphe or 0), unmapped_rows=int(tot - (mp or 0)),
                    unmapped_ZV_rows=int(zv), pct_unmapped_that_is_ZV=(round(100 * zv / (tot - (mp or 0)), 1)
                                                                       if tot - (mp or 0) else None))

    cov = []
    for name, dxtype, where in CATEGORIES:
        t = dx_table(where)
        t.insert(0, "dx_type", dxtype)
        t.to_csv(U.TABLES / f"dx_{name}.csv", index=False)
        cov.append(coverage(name, dxtype, where))
        top = t.head(5)[["phenotype", "enc_5yr_Overall"]].to_dict("records")
        print(f"{name:16s} [{dxtype}]: {len(t):4d} phecodes; top: "
              + "; ".join(f"{r['phenotype']}({r['enc_5yr_Overall']:,})" for r in top))

    covdf = pd.DataFrame(cov)
    covdf.to_csv(U.TABLES / "dx_coverage.csv", index=False)
    print("\n=== coverage (mapped diagnosis rows; unmapped is mostly Z/V non-disease codes) ===")
    print(covdf.to_string(index=False))

    U.log_milestone("Step 05 diagnosis phecodes: per-category primary/secondary phecode counts "
                    "(pyPheWAS v1.2 maps) for physician_level/allied_health/ED/inpatient/urgent_care.")


if __name__ == "__main__":
    main()
