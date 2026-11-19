# %% [markdown]
# Step 06 - Build per-patient UtilWAS association targets off the windowed store (+ phecode map).
#
# Two target families, each for windows 1yr & 5yr, keyed on cohort id:
#   SPECIALTY encounter counts -- per (id, family, DEPT_SPECIALTY) distinct-UTIL_ID counts, for
#     family in {physician_level (outpatient, has ICD), allied_health (outpatient, has non-ICD)}.
#     -> rate outcome (encounters/yr per specialty); kept specialties = >=500 patients in-window.
#   PHECODE diagnosis -- per (id, combo, phecode) distinct-UTIL_ID counts, combos:
#     physician_primary (PRIMARY_DX_IND=Y),
#     ED_primary/ED_secondary (PRIN_DX_IND=Y/N), inpatient_primary/inpatient_secondary (PRIN_DX_IND=Y/N),
#     urgent_primary/urgent_secondary (PRIMARY_DX_IND=Y/N). pyPheWAS v1.2 maps.
#     -> two-part outcome (binary presence + count among-present); kept phecodes = >=500 patients in-window.
#
# We store only the presence rows (count>=1) over the 222,109 cohort patients; the model step
# reconstructs the whole-cohort 0/1 and 0-count vectors. DE-IDENTIFIED summary only.

# %%
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import utilwas_utils as U

MIN_PATIENTS = 500
ICD = U.sql_in(U.ICD_CODE_SUBTYPES)
NONICD = f"COALESCE(CODE_SUBTYPE,'~NULL~') NOT IN ({ICD})"
OUTP = f"care_type IN ({U.sql_in(U.OUTPATIENT_CARE_TYPES)}) AND care_subtype IN ({U.sql_in(U.OUTPATIENT_SUBTYPES)})"
_res = os.path.join(os.path.dirname(importlib.util.find_spec("pyPheWAS").origin), "resources")
ICD9_MAP = os.path.join(_res, "phecode_map_v1_2_icd9.csv").replace("\\", "/")
ICD10_MAP = os.path.join(_res, "phecode_map_v1_2_icd10_beta.csv").replace("\\", "/")

# combo -> encounter/dx predicate over the dxall table (pdx=PRIMARY_DX_IND, prin=PRIN_DX_IND)
COMBOS = {
    "physician_primary":   f"{OUTP} AND pdx='Y'",
    "ED_primary":          f"care_subtype IN ({U.sql_in(U.ED_SUBTYPES)}) AND prin='Y'",
    "ED_secondary":        f"care_subtype IN ({U.sql_in(U.ED_SUBTYPES)}) AND prin='N'",
    "urgent_primary":      "is_urgent=1 AND pdx='Y'",
    "urgent_secondary":    "is_urgent=1 AND pdx='N'",
    "inpatient_primary":   f"care_subtype IN ({U.sql_in(U.INPATIENT_SUBTYPES)}) AND prin='Y'",
    "inpatient_secondary": f"care_subtype IN ({U.sql_in(U.INPATIENT_SUBTYPES)}) AND prin='N'",
}


def main() -> None:
    U.ensure_dirs()
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=96, threads=12)

    # ICD -> phecode map (v1.2)
    con.execute(f"""
        CREATE TEMP TABLE map AS
        SELECT cst, code, any_value(phecode) phecode, any_value(phenotype) phenotype, any_value(category) category
        FROM (
          SELECT 'ICD9' cst, upper(trim("ICD9")) code, CAST("PheCode" AS VARCHAR) phecode,
                 "Phenotype" phenotype, category_string category
          FROM read_csv_auto('{ICD9_MAP}', ALL_VARCHAR=TRUE) WHERE "ICD9" IS NOT NULL
          UNION ALL
          SELECT 'ICD10', upper(trim("ICD10")), CAST("PheCode" AS VARCHAR), "Phenotype", category_string
          FROM read_csv_auto('{ICD10_MAP}', ALL_VARCHAR=TRUE) WHERE "ICD10" IS NOT NULL
        ) GROUP BY cst, code
    """)

    # encounter-level table: family flags + one DEPT_SPECIALTY per encounter
    con.execute(f"""
        CREATE TEMP TABLE enc AS
        SELECT id, UTIL_ID, bool_or(in_1yr) AS in_1yr,
               any_value(CARE_TYPE) AS care_type, any_value(CARE_SUBTYPE) AS care_subtype,
               max(CASE WHEN CODE_SUBTYPE IN ({ICD}) THEN 1 ELSE 0 END) AS has_icd,
               max(CASE WHEN {NONICD} THEN 1 ELSE 0 END) AS has_nonicd,
               max(DEPT_SPECIALTY) AS dept_specialty
        FROM read_parquet('{U.STORE_5YR}') GROUP BY id, UTIL_ID
    """)

    # ICD diagnosis rows + phecode + normalized flags (+ encounter flags for allied/urgent)
    con.execute(f"""
        CREATE TEMP TABLE encx AS
        SELECT UTIL_ID, max(CASE WHEN {NONICD} THEN 1 ELSE 0 END) has_nonicd,
               max(CASE WHEN DEPT_SPECIALTY = '{U.URGENT_CARE_SPECIALTY}' THEN 1 ELSE 0 END) is_urgent
        FROM read_parquet('{U.STORE_5YR}') GROUP BY UTIL_ID
    """)
    con.execute(f"""
        CREATE TEMP TABLE dxall AS
        SELECT s.id, s.UTIL_ID, s.in_1yr, s.CARE_TYPE AS care_type, s.CARE_SUBTYPE AS care_subtype,
               upper(trim(s.PRIMARY_DX_IND)) AS pdx, upper(trim(s.PRIN_DX_IND)) AS prin,
               e.has_nonicd, e.is_urgent, m.phecode, m.phenotype, m.category,
               (m.phecode IS NOT NULL) AS mapped
        FROM read_parquet('{U.STORE_5YR}') s
        JOIN encx e ON s.UTIL_ID = e.UTIL_ID
        LEFT JOIN map m ON upper(trim(s.CODE_SUBTYPE)) = m.cst AND upper(trim(s.CODE)) = m.code
        WHERE upper(trim(s.CODE_SUBTYPE)) IN ({ICD})
    """)

    n_spec_rows = {}
    n_phe_rows = {}
    for label, _ in U.WINDOWS:
        wf = "in_1yr" if label == "1yr" else "TRUE"

        # ---- specialty targets (physician_level + allied_health) ----
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE spec AS
            SELECT id, family, dept_specialty, count(DISTINCT UTIL_ID) AS n_enc
            FROM (
              SELECT id, UTIL_ID, dept_specialty, 'physician_level' AS family
              FROM enc WHERE {wf} AND {OUTP} AND has_icd=1 AND dept_specialty IS NOT NULL
              UNION ALL
              SELECT id, UTIL_ID, dept_specialty, 'allied_health' AS family
              FROM enc WHERE {wf} AND {OUTP} AND has_nonicd=1 AND dept_specialty IS NOT NULL
            ) GROUP BY id, family, dept_specialty
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE spec_kept AS
            SELECT family, dept_specialty FROM spec
            GROUP BY family, dept_specialty HAVING count(DISTINCT id) >= {MIN_PATIENTS}
        """)
        spec_out = (U.ANALYSIS / f"specialty_targets_{label}.parquet").as_posix()
        con.execute(f"""COPY (SELECT s.* FROM spec s JOIN spec_kept k USING (family, dept_specialty))
                        TO '{spec_out}' (FORMAT PARQUET)""")
        con.execute(f"""COPY (SELECT k.family, k.dept_specialty, count(DISTINCT s.id) n_pat
                             FROM spec s JOIN spec_kept k USING (family, dept_specialty)
                             GROUP BY 1,2 ORDER BY 3 DESC)
                        TO '{(U.ANALYSIS / f"kept_specialties_{label}.csv").as_posix()}' (HEADER, DELIMITER ',')""")
        n_spec_rows[label] = con.sql("SELECT count(*), count(DISTINCT (family||dept_specialty)) FROM spec s JOIN spec_kept k USING (family, dept_specialty)").fetchone()

        # ---- phecode targets (7 combos) ----
        union = " UNION ALL ".join(
            f"SELECT '{c}' AS combo, id, phecode, phenotype, category, UTIL_ID FROM dxall WHERE mapped AND ({pred}) AND {wf}"
            for c, pred in COMBOS.items())
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE phe AS
            SELECT combo, id, phecode, any_value(phenotype) phenotype, any_value(category) category,
                   count(DISTINCT UTIL_ID) AS n_enc
            FROM ({union}) GROUP BY combo, id, phecode
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE phe_kept AS
            SELECT combo, phecode FROM phe GROUP BY combo, phecode HAVING count(DISTINCT id) >= {MIN_PATIENTS}
        """)
        phe_out = (U.ANALYSIS / f"phecode_targets_{label}.parquet").as_posix()
        con.execute(f"""COPY (SELECT p.combo, p.id, p.phecode, p.n_enc
                             FROM phe p JOIN phe_kept k USING (combo, phecode))
                        TO '{phe_out}' (FORMAT PARQUET)""")
        con.execute(f"""COPY (SELECT p.combo, p.phecode, any_value(p.phenotype) phenotype,
                                    any_value(p.category) category, count(DISTINCT p.id) n_pat
                             FROM phe p JOIN phe_kept k USING (combo, phecode)
                             GROUP BY p.combo, p.phecode ORDER BY p.combo, n_pat DESC)
                        TO '{(U.ANALYSIS / f"kept_phecodes_{label}.csv").as_posix()}' (HEADER, DELIMITER ',')""")
        n_phe_rows[label] = con.sql("SELECT count(*), count(DISTINCT (combo||phecode)) FROM phe p JOIN phe_kept k USING (combo, phecode)").fetchone()

        # de-identified per-combo kept counts
        kp = con.sql("SELECT combo, count(DISTINCT phecode) n_kept FROM phe p JOIN phe_kept k USING (combo, phecode) GROUP BY 1 ORDER BY 1").df()
        sk = con.sql("SELECT family, count(DISTINCT dept_specialty) n_kept FROM spec s JOIN spec_kept k USING (family, dept_specialty) GROUP BY 1").df()
        print(f"\n[{label}] specialty targets: {n_spec_rows[label][0]:,} present rows, "
              f"{n_spec_rows[label][1]} kept specialties (>={MIN_PATIENTS} pts)")
        print(sk.to_string(index=False))
        print(f"[{label}] phecode targets: {n_phe_rows[label][0]:,} present rows, "
              f"{n_phe_rows[label][1]} kept (combo,phecode)")
        print(kp.to_string(index=False))

    U.log_milestone(f"Step 06 targets: specialty encounter counts (2 families) + phecode presence/counts "
                    f"(7 combos), 1yr & 5yr, >={MIN_PATIENTS}-patient floor.")


if __name__ == "__main__":
    main()
