# %% [markdown]
# MeasWAS Step 00 - One-time repair of echo.csv -> a clean, PHI-free, typed parquet.
#
# echo.csv (484 MB) is NOT corrupted, but it is not RFC 4180 either: the record terminator is a
# bare LF while newlines inside free-text report bodies are CRLF. pandas.read_csv silently
# returns 926,702 NaN-padded junk rows and duckdb ignore_errors=true keeps a facility-confounded
# 61.6%. U.echo_records()/U.echo_fields() recover 284,031/284,031 records at 228 fields each.
#
# This step streams the file once and:
#   - drops U.ECHO_PHI_COLS (direct identifiers + free-text bodies) AT READ TIME,
#   - drops the four no-information constants (is_deleted, is_finalized_flg, modality_name,
#     original_performed_start_time) and the then-constant procedure_type_name,
#   - excludes the 3 procedure_type_name='Peripheral Vascular US' studies,
#   - renames MRN -> id, kept as a 12-char zero-padded VARCHAR (casting it to a number destroys
#     the leading zeros and collapses the cohort join to ~0),
#   - builds echo_date = performed_date (US_DATE), falling back to the date part of
#     creation_datetime (SAS_MS) for the 266 studies with no performed_date -> 100% non-null,
#   - parses the five surviving SAS timestamps to real timestamps (no raw date strings left),
#   - casts the 150 quantitative measurement columns to DOUBLE (blank -> NULL).
#
# Values are left RAW: real clinical outliers exist (lv_ef_calculated min -629,900,
# heart_rate_rest max 7,212,881, age_performed min -7) and tr_velocity carries mixed units.
# Gating is step 02's job; this step reports the p1/p99 tails so 02 can set the gates.
#
# -> U.ECHO_CLEAN (results/procwas_results/measwas/store/echo_clean.parquet)
# -> U.ANALYSIS/echo_column_completeness.csv
# DE-IDENTIFIED summary to stdout only.

# %%
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import measwas_utils as U

BATCH = 25_000

# no-information constants (verified: single value across all 284,031 records)
CONST_DROP = ["is_deleted", "is_finalized_flg", "modality_name", "original_performed_start_time"]
# 'Echo' on 284,028 rows; the 3 'Peripheral Vascular US' studies are excluded, so it then carries
# no information and is dropped from the output.
PV_US = "Peripheral Vascular US"
# consumed into echo_date / parsed in place
DATE_SRC = "performed_date"
TS_COLS = ["performed_start_time", "performed_end_time", "creation_datetime",
           "update_datetime", "finalization_dt"]
# the quantitative block, by name so the contract is self-documenting
MEAS_FIRST, MEAS_LAST = "anterior_per_effusion", "lv_esv_index"
EXTRA_NUM = ["age_performed", "tr_velocity", "tr_grad"]

# velocity/gradient pairs: simplified Bernoulli grad = 4*v^2 with v in m/s, so the pair pins the
# velocity column's units. tr_velocity is known to carry cm/s; this checks the other pairs.
# Only the PEAK pairs are diagnostic - a MEAN gradient is the time-average of the instantaneous
# gradients, not 4*(mean velocity)^2, so the mean pairs are expected to disagree. They are printed
# for completeness, tagged 'mean'.
VG_PAIRS = [
    ("av_peak_velocity", "av_peak_gradient", "peak"),
    ("pv_peak_velocity", "pv_peak_gradient", "peak"),
    ("pv_peak_pi_vel", "pv_peak_pi_grad", "peak"),
    ("lvot_peak_velocity", "lvot_peak_gradient", "peak"),
    ("rvot_peak_velocity", "rvot_peak_gradient", "peak"),
    ("pulmonary_artery_peak_velocity", "pulmonary_artery_peak_gradient", "peak"),
    ("tr_velocity", "tr_grad", "peak"),
    ("av_mean_velocity", "av_mean_gradient", "mean"),
    ("pv_mean_velocity", "pv_mean_gradient", "mean"),
    ("lvot_mean_velocity", "lvot_mean_gradient", "mean"),
    ("rvot_mean_velocity", "rvot_mean_gradient", "mean"),
    ("mv_mean_velocity", "mv_mean_grad", "mean"),
    ("tv_mean_velocity", "tv_mean_grad", "mean"),
]


# %%
def plan(header: list[str]) -> list[tuple[str, str, int]]:
    """(output name, kind, raw index) in raw-file order. kind: str|num|ts|date|year."""
    meas = set(header[header.index(MEAS_FIRST): header.index(MEAS_LAST) + 1]) | set(EXTRA_NUM)
    drop = set(U.ECHO_PHI_COLS) | set(CONST_DROP) | {"procedure_type_name"}
    out = []
    for i, h in enumerate(header):
        if h in drop:
            continue
        if h == "MRN":
            out.append(("id", "str", i))
        elif h == DATE_SRC:
            out.append(("echo_date", "date", i))
        elif h == "performed_year":
            out.append(("performed_year", "year", i))
        elif h in TS_COLS:
            out.append((h, "ts", i))
        elif h in meas:
            out.append((h, "num", i))
        else:
            out.append((h, "str", i))
    return out


_ARROW = {"str": pa.string(), "num": pa.float64(), "ts": pa.timestamp("us"),
          "date": pa.date32(), "year": pa.int32()}


def _dates(vals, fallback, memo, stats):
    """performed_date, falling back to the date part of creation_datetime."""
    out = []
    for v, cd in zip(vals, fallback):
        d = memo.get(v, 0)
        if d == 0:
            try:
                d = dt.datetime.strptime(v, U.US_DATE).date()
            except ValueError:
                d = None
            memo[v] = d
        if d is None:
            stats["date_fallback"] += 1
            try:
                d = dt.datetime.strptime(cd, U.SAS_MS).date()
            except ValueError:
                d = None
                stats["date_null"] += 1
        out.append(d)
    return pa.array(out, type=pa.date32())


def _ts(vals, name, stats):
    out = []
    for v in vals:
        try:
            out.append(dt.datetime.strptime(v, U.SAS_MS))
        except ValueError:
            out.append(None)
            if v.strip():
                stats["ts_unparsed"][name] = stats["ts_unparsed"].get(name, 0) + 1
    return pa.array(out, type=pa.timestamp("us"))


def _num(vals, name, stats):
    """Fast vectorised string->double; falls back to per-cell only if that cast raises."""
    try:
        return pa.array([v if v else None for v in vals], type=pa.string()).cast(pa.float64())
    except (pa.ArrowInvalid, ValueError):
        stats["slow_cast"][name] = stats["slow_cast"].get(name, 0) + 1
        out = []
        for v in vals:
            s = v.strip()
            if not s:
                out.append(None)
                continue
            try:
                out.append(float(s))
            except ValueError:
                out.append(None)
                stats["cast_fail"][name] = stats["cast_fail"].get(name, 0) + 1
        return pa.array(out, type=pa.float64())


def _years(vals, dates, stats):
    out = []
    for v, d in zip(vals, dates.to_pylist()):
        try:
            out.append(int(float(v)))
        except ValueError:
            stats["year_filled"] += 1
            out.append(d.year if d is not None else None)
    return pa.array(out, type=pa.int32())


# %%
def build() -> dict:
    gen = U.echo_records()
    header = U.echo_fields(next(gen))
    if len(header) != U.ECHO_N_FIELDS:
        raise SystemExit(f"HARD FAIL: header has {len(header)} fields, expected {U.ECHO_N_FIELDS}")

    cols = plan(header)
    schema = pa.schema([(n, _ARROW[k]) for n, k, _ in cols])
    i_ptn, i_date, i_crea = (header.index(x) for x in ("procedure_type_name", DATE_SRC,
                                                       "creation_datetime"))
    stats = {"read": 0, "written": 0, "pv_us": 0, "date_fallback": 0, "date_null": 0,
             "year_filled": 0, "cast_fail": {}, "slow_cast": {}, "ts_unparsed": {}}
    memo: dict[str, dt.date | None] = {}

    def flush(rows, writer):
        raw = list(zip(*rows))
        d = _dates(raw[i_date], raw[i_crea], memo, stats)
        arrays = []
        for name, kind, i in cols:
            if kind == "num":
                arrays.append(_num(raw[i], name, stats))
            elif kind == "ts":
                arrays.append(_ts(raw[i], name, stats))
            elif kind == "date":
                arrays.append(d)
            elif kind == "year":
                arrays.append(_years(raw[i], d, stats))
            else:
                arrays.append(pa.array([v if v else None for v in raw[i]], type=pa.string()))
        writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
        stats["written"] += len(rows)

    buf = []
    with pq.ParquetWriter(U.ECHO_CLEAN, schema, compression="zstd") as writer:
        for rec in gen:
            f = U.echo_fields(rec)
            if len(f) != U.ECHO_N_FIELDS:
                raise SystemExit(
                    f"HARD FAIL: record {stats['read'] + 1} parsed to {len(f)} fields, expected "
                    f"{U.ECHO_N_FIELDS}. Field alignment is not what was established - stop.")
            stats["read"] += 1
            if f[i_ptn] == PV_US:
                stats["pv_us"] += 1
                continue
            buf.append(f)
            if len(buf) >= BATCH:
                flush(buf, writer)
                buf = []
        if buf:
            flush(buf, writer)
    stats["n_cols"] = len(cols)
    stats["n_num"] = sum(1 for _, k, _ in cols if k == "num")
    return stats


# %%
def verify(stats: dict | None) -> None:
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=32, threads=8)
    e = f"read_parquet('{U.ECHO_CLEAN}')"

    n_rows, n_pat, y0, y1, n_date = con.sql(
        f"SELECT count(*), count(DISTINCT id), min(year(echo_date)), max(year(echo_date)), "
        f"count(echo_date) FROM {e}").fetchone()
    n_study = con.sql(f"SELECT count(DISTINCT procedure_id) FROM {e}").fetchone()[0]
    head = con.sql(f"SELECT * FROM {e} LIMIT 0")
    n_cols = len(head.columns)
    leaked = sorted(set(head.columns) & set(U.ECHO_PHI_COLS))
    if leaked:
        raise SystemExit(f"HARD FAIL: PHI columns reached disk: {leaked}")

    print(f"records read    : {stats['read']:,}" if stats else
          "records read    : n/a (cached parquet; pass --force to rebuild)")
    if stats:
        print(f"  excluded '{PV_US}': {stats['pv_us']}")
    print(f"records written : {n_rows:,}   distinct procedure_id: {n_study:,}")
    print(f"distinct patients: {n_pat:,}   columns: {n_cols}")
    print(f"echo_date       : {y0}..{y1}   non-null {100 * n_date / n_rows:.2f}% "
          f"({n_rows - n_date} null)")
    if stats:
        print(f"  performed_date blank -> creation_datetime fallback: {stats['date_fallback']:,}"
              f"  still null: {stats['date_null']:,}")
        print(f"  performed_year blank -> filled from echo_date: {stats['year_filled']:,}")
        fails = sum(stats["cast_fail"].values())
        print(f"numeric columns cast to DOUBLE: {stats['n_num']}   cast failures: {fails}")
        if fails:
            raise SystemExit(f"HARD FAIL: {fails} numeric cells failed to cast "
                             f"{sorted(stats['cast_fail'].items())} - field alignment is wrong.")
        if stats["slow_cast"]:
            print(f"  (vectorised cast fell back per-cell for: {sorted(stats['slow_cast'])})")
        if stats["ts_unparsed"]:
            print(f"  unparsed non-blank timestamps: {sorted(stats['ts_unparsed'].items())}")
    else:
        print("numeric cast failures: 0 (parquet is typed; enforced at build time)")

    # ---- cohort join: proves the 12-char MRN survived as a string ----
    con.execute(f"CREATE TEMP TABLE c AS SELECT id, CAST(STUDY_DATE AS DATE) AS idx "
                f"FROM read_parquet('{U.COHORT}')")
    n_cohort = con.sql("SELECT count(*) FROM c").fetchone()[0]
    n_any = con.sql(f"SELECT count(DISTINCT c.id) FROM c JOIN {e} x ON c.id = x.id").fetchone()[0]
    n_win_pat, n_win_std = con.sql(f"""
        SELECT count(DISTINCT c.id), count(*) FROM c JOIN {e} x ON c.id = x.id
        WHERE x.echo_date >= (c.idx - to_years({U.WINDOW_YEARS}))::DATE AND x.echo_date < c.idx
    """).fetchone()
    print(f"cohort spine    : {n_cohort:,} patients")
    print(f"  with >=1 echo (any date): {n_any:,} ({100 * n_any / n_cohort:.1f}%)")
    print(f"  with >=1 echo in [index-{U.WINDOW_YEARS}yr, index): {n_win_pat:,} "
          f"({100 * n_win_pat / n_cohort:.1f}%)  studies: {n_win_std:,}")
    if n_any < 1000:
        raise SystemExit("HARD FAIL: cohort join is ~empty - the MRN was cast to a number.")

    # ---- per-column completeness + tails for the measurement block ----
    num_cols = [c for c, t in zip(con.sql(f"SELECT * FROM {e} LIMIT 0").columns,
                                  con.sql(f"SELECT * FROM {e} LIMIT 0").types)
                if str(t) == "DOUBLE"]
    df = pd.read_parquet(U.ECHO_CLEAN, columns=num_cols)
    d = df.describe(percentiles=[0.01, 0.5, 0.99]).T
    comp = pd.DataFrame({
        "column": d.index,
        "n_nonnull": d["count"].astype("int64").to_numpy(),
        "pct_nonnull": (100 * d["count"] / n_rows).round(2).to_numpy(),
        "min": d["min"].to_numpy(), "p1": d["1%"].to_numpy(), "median": d["50%"].to_numpy(),
        "p99": d["99%"].to_numpy(), "max": d["max"].to_numpy(),
    }).sort_values("n_nonnull", ascending=False)
    out_csv = U.ANALYSIS / "echo_column_completeness.csv"
    comp.to_csv(out_csv, index=False)
    filled = 100 * df.notna().to_numpy().sum() / (n_rows * len(num_cols))
    print(f"measurement cells populated: {filled:.1f}% over {len(num_cols)} numeric columns")
    print(f"  columns with >=10% completeness: {(comp['pct_nonnull'] >= 10).sum()}"
          f"  >=1%: {(comp['pct_nonnull'] >= 1).sum()}")
    print(f"  -> {out_csv}")
    print(comp.head(12).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ---- velocity/gradient unit check (tr_velocity is known-mixed) ----
    print("velocity/gradient pairs (grad == 4*v^2 => v in m/s; == 4*(v/100)^2 => v in cm/s;")
    print("  only 'peak' rows are diagnostic - a mean gradient is not 4*(mean velocity)^2):")
    for v, g, kind in VG_PAIRS:
        if v not in df.columns or g not in df.columns:
            continue
        s = df[[v, g]].dropna()
        s = s[(s[v] > 0) & (s[g] > 0)]
        if len(s) < 50:
            continue
        ms = (abs(4 * s[v] ** 2 - s[g]) / s[g] < 0.05).mean()
        cms = (abs(4 * (s[v] / 100) ** 2 - s[g]) / s[g] < 0.05).mean()
        flag = ("  <-- units not consistent"
                if kind == "peak" and max(ms, cms) < 0.98 else "")
        print(f"  [{kind}] {v:32s} n={len(s):>7,}  m/s {100 * ms:5.1f}%  "
              f"cm/s {100 * cms:5.1f}%{flag}")
    con.close()


# %%
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if the parquet exists")
    args = ap.parse_args()
    U.ensure_dirs()

    stats = None
    if args.force or not Path(U.ECHO_CLEAN).exists():
        stats = build()
    else:
        print(f"{U.ECHO_CLEAN} exists - skipping rebuild (--force to redo).")
    verify(stats)

    if stats:
        U.log_milestone(
            f"Step 00 echo repair: {stats['read']:,} records recovered from echo.csv "
            f"({stats['pv_us']} Peripheral Vascular US excluded), {stats['written']:,} written to "
            f"echo_clean.parquet at {stats['n_cols']} PHI-free columns "
            f"({stats['n_num']} DOUBLE, 0 cast failures).")


if __name__ == "__main__":
    main()
