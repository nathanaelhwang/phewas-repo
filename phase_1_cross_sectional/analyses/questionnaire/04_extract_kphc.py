# %% [markdown]
# Step 4 - Extract the KPHC (authoritative) source into the target-code wide frame.
#
# Stream `sleep_history_quest.csv` with DuckDB (all_varchar), keep rows whose
# QUEST_ID is one of the 59 KPHC-mapped target codes, aggregate multi-LINE answers,
# pivot to one row per (MRN, encounter), decode each cell, attach STUDY_DATE /
# CONTACT_DATE. Empty codes (STOP5/6/8) and Somnoware-only NAPDURATION stay <NA>.
#
# Emits: results/questionnaire_results/kphc_wide.parquet  (+ a small CSV sample)
# DE-IDENTIFIED stdout only (shapes / fill counts; never MRN or answer values).

# %%
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import questionnaire_utils as U


def kphc_id2code() -> dict[int, str]:
    qm = U.load_kphc_quest_map()
    qm = qm[qm["code"].isin(U.TARGET_CODES)]
    return {int(q): c for q, c in zip(qm["quest_id"], qm["code"])}


def cast_column(col: pd.Series, code: str) -> pd.Series:
    """Cast a decoded column to its nullable dtype (string for text/timebin, else Int64)."""
    return col.astype(U.code_dtype(code))


def main() -> None:
    U.ensure_dirs()
    id2code = kphc_id2code()
    id_list = ", ".join(str(q) for q in id2code)
    t0 = time.time()

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{(U.RESULTS / '_duckdb_tmp').as_posix()}'")
    src = (f"read_csv('{U.KPHC_CSV.as_posix()}', header=true, all_varchar=true, "
           f"ignore_errors=true, quote='\"', escape='\"')")

    print("KPHC: streaming + aggregating answers ...", flush=True)
    con.execute(f"""
        CREATE TEMP TABLE src AS
        SELECT {U.MRN_SQL} AS MRN,
               PAT_ENC_CSN_ID AS enc,
               CONTACT_DATE AS contact, STUDY_DATE AS study,
               try_cast(try_cast(QUEST_ID AS DOUBLE) AS BIGINT) AS qid,
               QUEST_ANSWER AS ans, try_cast(LINE AS INT) AS line
        FROM {src}
        WHERE try_cast(try_cast(QUEST_ID AS DOUBLE) AS BIGINT) IN ({id_list})
          AND PAT_ENC_CSN_ID IS NOT NULL
    """)
    # one answer per (MRN, enc, qid): multi-LINE joined in LINE order
    ans = con.execute("""
        SELECT MRN, enc, qid,
               string_agg(ans, '; ' ORDER BY line) AS ans
        FROM src WHERE ans IS NOT NULL GROUP BY MRN, enc, qid
    """).df()
    meta = con.execute("""
        SELECT MRN, enc, min(contact) AS CONTACT_DATE, min(study) AS STUDY_DATE
        FROM src GROUP BY MRN, enc
    """).df()
    con.close()
    print(f"  fetched {len(ans):,} (enc,code) answers, {len(meta):,} encounters "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # pivot to wide: one row per (MRN, enc)
    ans["code"] = ans["qid"].map(id2code)
    wide = ans.pivot_table(index=["MRN", "enc"], columns="code",
                           values="ans", aggfunc="first").reset_index()

    # decode each present code column
    for code in [c for c in U.TARGET_CODES if c in wide.columns]:
        wide[code] = cast_column(wide[code].map(lambda v: U.decode(code, v)), code)

    # attach dates, parse (SAS or ISO), normalize MRN already done in SQL
    wide = wide.merge(meta, on=["MRN", "enc"], how="left")
    wide["STUDY_DATE"] = U.parse_sas_datetime(wide["STUDY_DATE"])
    wide["CONTACT_DATE"] = U.parse_sas_datetime(wide["CONTACT_DATE"])
    wide["source"] = "kphc"

    # conform to target column order (+ SLEEPPARALYSIS), fill absent codes as NA
    for code in U.TARGET_CODES:
        if code not in wide.columns:
            wide[code] = pd.array([pd.NA] * len(wide), dtype=U.code_dtype(code))
    out = wide[U.TARGET_COLUMNS + ["source"]].copy()

    out_path = U.RESULTS / "kphc_wide.parquet"
    out.to_parquet(out_path, index=False)
    U.safe_to_csv(out.head(200), U.RESULTS / "kphc_wide_sample.csv")

    # de-identified fill report
    fills = {c: int(out[c].notna().sum()) for c in U.TARGET_CODES}
    zero = [c for c, n in fills.items() if n == 0]
    print(f"[kphc] {len(out):,} encounters x {len(U.TARGET_COLUMNS)} cols -> {out_path.name}")
    print(f"[kphc] codes with 0 fills (expected STOP5/6/8, NAPDURATION): {zero}")
    print(f"[kphc] median fill across item codes: "
          f"{int(pd.Series(fills).median()):,}")
    U.log_milestone(
        f"04_extract_kphc: {len(out)} encounters; {len([c for c in fills if fills[c]>0])}"
        f"/{len(U.TARGET_CODES)} codes populated; zero-fill={zero}.")


if __name__ == "__main__":
    main()
