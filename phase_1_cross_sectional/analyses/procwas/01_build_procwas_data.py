# %% [markdown]
# ProcWAS Step 1 - Build procwas_data.csv from proc.csv (11.7 GB, diagnosis-line grain).
#
# Pipeline (one streaming DuckDB pass over proc.csv):
#   1. Strip Python bytes-repr wrappers  b'...'  /  b"..."  from every string field.
#   2. Keep only ORDER_STATUS == 'COMPLETED'.
#   3. Drop the requested order/administrative columns; MRN -> int64.
#   4. LEFT-join the procedure crosswalk (procedures_classified_v4.xlsx, sheet
#      'Procedures') on PROC_CODE == 'CPT primary' to add Clear descriptor /
#      Procedure type / Imaging modality / Body system.
#   5. INNER-join the v2-cohort index date (icd_analysis_v2_results cohort.parquet,
#      222,109 patients) and keep only CONTACT_DATE <= STUDY_DATE (pre-index).
#   6. Write results/procwas_results/procwas_data.csv (DE-IDENTIFIED prints only).
#
# STUDY_DATE = the v2 analytic-cohort spine reused by every WAS stage (id = 12-char
# MRN). Data is real EHR/PHI: only aggregate counts/distributions are printed; the
# patient-level CSV lands on disk.

# %%
import sys
import datetime as dt
from pathlib import Path

import pandas as pd
import duckdb

# reuse canonical project locations without touching that package
_ICD = Path(__file__).resolve().parent.parent / "icd_analysis"
sys.path.insert(0, str(_ICD))
import utils as U  # noqa: E402

DATA = Path(r"D:\phewas_data")
PROC = (DATA / "proc.csv").as_posix()
XW = DATA / "procedures_classified_v4.xlsx"
# v2 analytic cohort (222,109; id = 12-char MRN, STUDY_DATE) -- the shared WAS spine.
COHORT = (U.PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet").as_posix()

OUT_DIR = U.PROJECT / "results" / "procwas_results"
OUT_CSV = (OUT_DIR / "procwas_data.csv").as_posix()
SUMMARY = OUT_DIR / "procwas_build_summary.md"
TMP = Path(__file__).resolve().parent / "_duckdb_tmp"

# Required output columns (step 8 spec) + STUDY_DATE (index date, kept for windowing).
CLASS_COLS = ["Clear descriptor", "Procedure type", "Imaging modality", "Body system"]


# %%
def build_cohort_index() -> tuple[pd.DataFrame, int]:
    """v2-cohort index date -> (mrn_int int64, study_date); one row per cohort patient.

    Reuses the analytic spine every WAS stage shares:
    results/icd_analysis_v2_results/cohort/cohort.parquet (id = 12-char MRN, STUDY_DATE).
    """
    c = pd.read_parquet(COHORT, columns=["id", "STUDY_DATE"])
    idx = pd.DataFrame({
        "mrn_int": pd.to_numeric(c["id"], errors="coerce").astype("int64"),
        "study_date": pd.to_datetime(c["STUDY_DATE"]).dt.normalize(),
    })
    return idx, len(idx)


def build_crosswalk() -> pd.DataFrame:
    """De-duplicated procedure crosswalk keyed on UPPER('CPT primary')."""
    xw = pd.read_excel(XW, sheet_name="Procedures")
    xw["key_u"] = xw["CPT primary"].astype("string").str.strip().str.upper()
    xw = xw[xw["key_u"].notna() & (xw["key_u"] != "")]
    # one row per key: keep the dominant procedure (max 'N performed').
    xw = xw.sort_values("N performed", ascending=False).drop_duplicates("key_u", keep="first")
    out = pd.DataFrame({
        "key_u": xw["key_u"].astype(str).to_numpy(),
        "clear_descriptor": xw["Clear descriptor"].astype("string").to_numpy(),
        "procedure_type": xw["Procedure type"].astype("string").to_numpy(),
        "imaging_modality": xw["Imaging modality"].astype("string").to_numpy(),
        "body_system": xw["Body system"].astype("string").to_numpy(),
    })
    return out


# %%
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print("Loading v2 cohort index (STUDY_DATE) ...")
    sd, n_cohort = build_cohort_index()
    print(f"  v2 cohort patients: {n_cohort:,}")

    print("Building de-duplicated procedure crosswalk ...")
    xw = build_crosswalk()
    print(f"  crosswalk keys (1 per CPT primary): {len(xw):,}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='32GB'")
    con.execute(f"PRAGMA temp_directory='{TMP.as_posix()}'")
    con.register("sd", sd)
    con.register("xw", xw)

    # b'...' or b"..." -> inner text; leaves non-wrapped values (dates, blanks) untouched.
    con.execute(r"""CREATE MACRO strip_b(x) AS regexp_replace(x, '^b[''"](.*)[''"]$', '\1')""")

    # ---- single 11.7 GB scan: strip + COMPLETED filter + typed projection ----
    print("Scanning proc.csv (COMPLETED rows) ...")
    con.execute(f"""
        CREATE TEMP TABLE comp AS
        SELECT
            TRY_CAST(strip_b(MRN) AS BIGINT) AS MRN,
            TRY_CAST(CONTACT_DATE AS DATE)   AS CONTACT_DATE,
            strip_b(PROC_CODE)               AS PROC_CODE,
            strip_b(CODE_TYPE)               AS CODE_TYPE,
            strip_b(CODE)                    AS CODE,
            strip_b(DX_NAME)                 AS DX_NAME
        FROM read_csv('{PROC}', ALL_VARCHAR=TRUE, quote='"', escape='"', header=true)
        WHERE strip_b(ORDER_STATUS) = 'COMPLETED'
    """)
    n_completed = con.sql("SELECT count(*) FROM comp").fetchone()[0]
    n_bad_mrn = con.sql("SELECT count(*) FROM comp WHERE MRN IS NULL").fetchone()[0]
    n_bad_date = con.sql("SELECT count(*) FROM comp WHERE CONTACT_DATE IS NULL").fetchone()[0]

    # ---- joins: crosswalk (left) + study index (inner) + pre-index filter ----
    con.execute("""
        CREATE TEMP TABLE procwas AS
        SELECT
            c.MRN, c.CONTACT_DATE, c.PROC_CODE, c.CODE_TYPE, c.CODE, c.DX_NAME,
            x.clear_descriptor AS "Clear descriptor",
            x.procedure_type   AS "Procedure type",
            x.imaging_modality AS "Imaging modality",
            x.body_system      AS "Body system",
            CAST(s.study_date AS DATE) AS STUDY_DATE
        FROM comp c
        JOIN sd  s ON c.MRN = s.mrn_int
        LEFT JOIN xw x ON upper(c.PROC_CODE) = x.key_u
        WHERE c.CONTACT_DATE <= CAST(s.study_date AS DATE)
    """)

    print(f"Writing {OUT_CSV} ...")
    con.execute(f"COPY procwas TO '{OUT_CSV}' (HEADER, DELIMITER ',')")

    # ---- DE-IDENTIFIED funnel + QA -----------------------------------------
    n_studyjoin = con.sql("SELECT count(*) FROM comp c JOIN sd s ON c.MRN = s.mrn_int").fetchone()[0]
    n_final, n_pat = con.sql('SELECT count(*), count(DISTINCT "MRN") FROM procwas').fetchone()
    n_xw_hit = con.sql('SELECT count(*) FROM procwas WHERE "Procedure type" IS NOT NULL').fetchone()[0]
    n_blank_code = con.sql("SELECT count(*) FROM procwas WHERE CODE IS NULL").fetchone()[0]
    codetype = con.sql("""SELECT COALESCE(CODE_TYPE,'<blank>') ct, count(*) n
                          FROM procwas GROUP BY 1 ORDER BY 2 DESC""").df()
    cd_lo, cd_hi = con.sql("SELECT min(CONTACT_DATE), max(CONTACT_DATE) FROM procwas").fetchone()
    sd_lo, sd_hi = con.sql("SELECT min(STUDY_DATE), max(STUDY_DATE) FROM procwas").fetchone()
    ptype = con.sql('''SELECT COALESCE("Procedure type",'<unmatched>') t, count(*) n
                       FROM procwas GROUP BY 1 ORDER BY 2 DESC''').df()

    L: list[str] = []
    def p(m: str = "") -> None:
        print(m); L.append(m)

    p("# ProcWAS Step 1 build summary\n")
    p("## Funnel")
    p(f"- COMPLETED rows after strip: {n_completed:,}")
    p(f"  - dropped (non-numeric MRN): {n_bad_mrn:,}")
    p(f"  - dropped (unparseable CONTACT_DATE): {n_bad_date:,}")
    p(f"- after inner-join to v2 cohort ({n_cohort:,} patients): {n_studyjoin:,}")
    p(f"- **after pre-index filter (CONTACT_DATE <= STUDY_DATE): {n_final:,}** rows / {n_pat:,} patients")
    p(f"\n## Crosswalk coverage")
    p(f"- rows with a crosswalk match: {n_xw_hit:,} ({100*n_xw_hit/max(n_final,1):.1f}% of output rows)")
    p(f"\n## QA")
    p(f"- rows with blank CODE (procedure, no ICD dx line): {n_blank_code:,} ({100*n_blank_code/max(n_final,1):.1f}%)")
    p(f"- CONTACT_DATE range: {cd_lo} .. {cd_hi}")
    p(f"- STUDY_DATE  range: {sd_lo} .. {sd_hi}")
    p(f"- CODE_TYPE distribution: {dict(zip(codetype.ct, codetype.n))}")
    p(f"\n## Procedure type distribution (output rows)")
    for t, n in zip(ptype.t, ptype.n):
        p(f"  - {t:42s}: {int(n):,} ({100*n/n_final:.1f}%)")
    p(f"\nWrote {OUT_CSV}")

    SUMMARY.write_text("\n".join(L), encoding="utf-8")

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(U.PROJECT / "logs" / "activity_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [procwas] Step 1 built procwas_data.csv: "
                f"{n_final:,} pre-index completed-procedure rows / {n_pat:,} patients.\n")
    con.close()


if __name__ == "__main__":
    main()
