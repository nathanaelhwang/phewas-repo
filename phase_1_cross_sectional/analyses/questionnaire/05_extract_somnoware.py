# %% [markdown]
# Step 5 - Extract the Somnoware source into the target-code wide frame.
#
# ONE streaming pass over kaiser_questionnaire.csv (^||^), classifying each row
# (cached per distinct question) as:
#   - LABELED: anchors (match_question) or the reviewer FEEDBACK map -> a code.
#   - MATRIX : a collapsed-matrix stem (MATRIX_SPECS) -> staged for positional
#              reconstruction.
# Rows are staged to parquet in chunks (memory-safe); reconstruction then runs in
# DuckDB (dedup by responseid + row_number windows). Matrix value = the selected
# row's patient_response; item = pos // scale (validated pos % scale == rating for
# ESS). Only clean blocks (size == n_items x scale) are kept.
#
# Emits: results/questionnaire_results/somnoware_wide.parquet
# DE-IDENTIFIED stdout only.

# %%
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import questionnaire_utils as U

STAGE = U.RESULTS / "_somno_stage"
CHUNK = 1_000_000


class Staged:
    """Chunked parquet writer for staged rows."""
    def __init__(self, path: Path, schema: pa.Schema):
        self.path, self.schema, self.buf, self.w = path, schema, [], None

    def add(self, rec: dict):
        self.buf.append(rec)
        if len(self.buf) >= CHUNK:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        tbl = pa.Table.from_pylist(self.buf, schema=self.schema)
        if self.w is None:
            self.w = pq.ParquetWriter(self.path, self.schema)
        self.w.write_table(tbl)
        self.buf = []

    def close(self):
        self.flush()
        if self.w:
            self.w.close()


def classify(cform: str, qnorm: str, fb_q2code: dict, cache: dict):
    """(kind, payload): ('labeled', code) | ('matrix', spec_idx) | None."""
    key = (cform, qnorm)
    if key in cache:
        return cache[key]
    res = None
    if qnorm in fb_q2code:
        res = ("labeled", fb_q2code[qnorm])
    else:
        hits = U.match_question(qnorm)
        if len(hits) == 1:
            res = ("labeled", hits[0])
        else:
            for si, spec in enumerate(U.MATRIX_SPECS):
                if cform != spec["form"]:
                    continue
                stem = spec["stem"]
                ok = (qnorm == stem) if spec.get("mode") == "exact" else (stem in qnorm)
                if ok:
                    res = ("matrix", si)
                    break
    cache[key] = res
    return res


def stream_and_stage() -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    fb = U.load_feedback()["q2code"]
    cache: dict = {}
    lab = Staged(STAGE / "labeled.parquet", pa.schema(
        [("fid", pa.string()), ("code", pa.string()), ("val", pa.string())]))
    mat = Staged(STAGE / "matrix.parquet", pa.schema(
        [("spec", pa.int16()), ("fid", pa.string()),
         ("rid", pa.int64()), ("val", pa.string())]))
    meta: dict = {}
    t0 = time.time()
    n = 0
    with U.SOMNO_CSV.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        H = U.somno_split(fh.readline())
        ncol = len(H)
        ix = {h: i for i, h in enumerate(H)}
        i_form, i_q, i_opt, i_resp = ix["form_title"], ix["question"], ix["option_text"], ix["patient_response"]
        i_rid, i_fid = ix["pk_patientQuestionnaireResponseId"], ix["pk_patientQuestFormId"]
        i_mrn, i_date = ix["medical_record_number"], ix["formcompletetiondate"]
        for line in fh:
            n += 1
            if n % 5_000_000 == 0:
                print(f"  scanned {n:,} [{time.time()-t0:.0f}s]", flush=True)
            row = U.somno_split(line)
            if len(row) != ncol:
                continue
            cform = U.canon_form(row[i_form])
            if cform in U.SOMNO_FORM_DENYLIST or not cform:
                continue
            fid = row[i_fid]
            if fid not in meta:
                meta[fid] = (row[i_mrn], row[i_date])
            q = row[i_q] or ""
            qn = " ".join(q.strip().lower().split())
            if not qn:
                continue
            kind = classify(cform, qn, fb, cache)
            if kind is None:
                continue
            if kind[0] == "labeled":
                # the SELECTED option is the row with a non-blank patient_response;
                # non-selected option rows are blank -> never use option_text here.
                resp = (row[i_resp] or "").strip()
                if resp:
                    lab.add({"fid": fid, "code": kind[1], "val": resp})
            else:  # matrix
                try:
                    rid = int(row[i_rid])
                except ValueError:
                    continue
                mat.add({"spec": kind[1], "fid": fid,
                         "rid": rid, "val": (row[i_resp] or "").strip()})
    lab.close()
    mat.close()
    print(f"  staged {n:,} rows scanned [{time.time()-t0:.0f}s]; "
          f"{len(meta):,} form instances", flush=True)
    return meta


def process_labeled() -> pd.DataFrame:
    p = STAGE / "labeled.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["fid"])
    df = pd.read_parquet(p)
    df["dec"] = [U.decode(c, v) for c, v in zip(df["code"], df["val"])]
    df = df.dropna(subset=["dec"])
    text_codes = {c for c in U.TARGET_CODES if U.FAMILY[c] == "text"}
    num = df[~df["code"].isin(text_codes)]
    wide = num.pivot_table(index="fid", columns="code", values="dec", aggfunc="first")
    txt = df[df["code"].isin(text_codes)]
    if len(txt):                          # multi-select text (CC1) -> join distinct
        wtxt = (txt.groupby(["fid", "code"])["dec"]
                .agg(lambda s: "; ".join(dict.fromkeys(map(str, s)))).unstack("code"))
        wide = wide.join(wtxt, how="outer")
    return wide.reset_index()


def process_matrix() -> pd.DataFrame:
    p = STAGE / "matrix.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["fid"])
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{(STAGE / '_ddb').as_posix()}'")
    frames = []
    report = []
    for si, spec in enumerate(U.MATRIX_SPECS):
        codes = spec["codes"]
        if not codes:                     # item order unknown -> defer (reported)
            report.append((spec["form"], spec["stem"], "deferred: item order TBD"))
            continue
        n_items = len(codes)
        # dedup by (fid, rid); positional pos; clean block size = n_items*scale
        base = con.execute(f"""
            WITH d AS (SELECT DISTINCT fid, rid, val FROM read_parquet('{p.as_posix()}')
                       WHERE spec = {si}),
                 p AS (SELECT fid, rid, val,
                              row_number() OVER (PARTITION BY fid ORDER BY rid)-1 AS pos,
                              count(*) OVER (PARTITION BY fid) AS bsize
                       FROM d)
            SELECT fid, pos, bsize, val FROM p
        """).df()
        if base.empty:
            report.append((spec["form"], spec["stem"], "no rows"))
            continue
        scale = spec["scale"]
        if scale is None:                 # infer from dominant clean block size
            sizes = base.groupby("fid")["bsize"].first().value_counts()
            cand = [(bs, cnt) for bs, cnt in sizes.items() if bs % n_items == 0]
            scale = (max(cand, key=lambda x: x[1])[0] // n_items) if cand else None
        if not scale:
            report.append((spec["form"], spec["stem"], "scale inference failed"))
            continue
        clean = base[base["bsize"] == n_items * scale].copy()
        sel = clean[clean["val"].astype(str).str.strip() != ""].copy()
        sel["item_idx"] = sel["pos"] // scale
        sel = sel[sel["item_idx"] < n_items]
        sel["code"] = sel["item_idx"].map(lambda i: codes[i])
        sel = sel[sel["code"].notna()]            # skip None positions (non-target items)
        # matrix value = digitmap(option index). The selected row's patient_response is
        # the option index as a bare digit ('2') or digit-prefixed ('0 - None'); take
        # the LEADING digit (the misleading freq label text is blank in the matrix).
        dm = spec.get("digitmap") or (lambda d: d)

        def _mval(v, _dm=dm, _sc=scale):
            m = re.match(r"(\d+)", str(v).strip())
            if not m:
                return pd.NA
            d = int(m.group(1))
            return _dm(d) if 0 <= d < _sc else pd.NA
        sel["dec"] = sel["val"].map(_mval)
        sel = sel.dropna(subset=["dec"])
        if sel.empty:
            report.append((spec["form"], spec["stem"], f"scale={scale}: 0 decoded"))
            continue
        wide = sel.pivot_table(index="fid", columns="code", values="dec", aggfunc="first")
        frames.append(wide.reset_index())
        report.append((spec["form"], spec["stem"],
                       f"scale={scale}, items={n_items}, "
                       f"{clean['fid'].nunique():,} clean blocks"))
    con.close()
    print("[matrix] reconstruction report:")
    for f, s, msg in report:
        print(f"   {f[:26]:26s}/{s[:24]:24s} -> {msg}")
    if not frames:
        return pd.DataFrame(columns=["fid"])
    out = frames[0].set_index("fid")
    for f in frames[1:]:                  # combine specs; coalesce same code
        out = out.combine_first(f.set_index("fid"))
    return out.reset_index()


def main() -> None:
    U.ensure_dirs()
    meta = stream_and_stage()
    lab = process_labeled()
    mat = process_matrix()

    # labeled has priority; matrix fills the collapsed-grid codes (ESS/FOSQ/screen)
    if mat.empty:
        wide = lab
    else:
        wide = (lab.set_index("fid").combine_first(mat.set_index("fid"))
                .reset_index())

    # attach meta (mrn, completion date) and conform to target schema
    md = pd.DataFrame([(f, m, d) for f, (m, d) in meta.items()],
                      columns=["fid", "MRN", "CONTACT_DATE"])
    md["MRN"] = U.normalize_mrn(md["MRN"])
    md["CONTACT_DATE"] = pd.to_datetime(md["CONTACT_DATE"], errors="coerce")
    wide = wide.merge(md, on="fid", how="left")
    wide["STUDY_DATE"] = pd.NaT
    wide["source"] = "somnoware"

    for code in U.TARGET_CODES:
        if code not in wide.columns:
            wide[code] = pd.NA
        wide[code] = wide[code].astype(U.code_dtype(code))
    out = wide[U.TARGET_COLUMNS + ["source"]].copy()

    out_path = U.RESULTS / "somnoware_wide.parquet"
    out.to_parquet(out_path, index=False)
    fills = {c: int(out[c].notna().sum()) for c in U.TARGET_CODES}
    pop = [c for c, nfill in fills.items() if nfill > 0]
    print(f"[somnoware] {len(out):,} form instances x {len(U.TARGET_COLUMNS)} cols -> {out_path.name}")
    print(f"[somnoware] populated {len(pop)}/{len(U.TARGET_CODES)} codes; "
          f"empty: {sorted(set(U.TARGET_CODES)-set(pop))}")
    U.log_milestone(
        f"05_extract_somnoware: {len(out)} form instances; {len(pop)} codes populated "
        f"(labeled + matrix reconstruction).")


if __name__ == "__main__":
    main()
