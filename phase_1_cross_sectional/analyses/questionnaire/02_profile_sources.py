# %% [markdown]
# Step 2 - PHI-safe profiling of the two in-scope source CSVs.
#
# Purpose: confirm the target QUEST_IDs really appear in the full KPHC file, and
# capture the **distinct answer-option set per target code** that finalizes each
# decoder. Also inventory the Somnoware forms/questions to seed text anchors.
#
# KPHC (sleep_history_quest.csv, comma)  -> DuckDB streaming (full file).
# Somnoware (kaiser_questionnaire.csv, ^||^) -> python chunked SAMPLE (DuckDB
#   can't parse a 4-char delimiter); step 05 does the authoritative full pass.
#
# Emits (all de-identified -> counts & option labels only, never identifiers):
#   profile/kphc_profile.json
#   profile/somnoware_profile.json
#   profile/PROFILE.md
# The 51 target codes are all structured (likert/yes-no/time/count) items, so
# their distinct answers are metadata; free-text items are never target codes.

# %%
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import questionnaire_utils as U

CAP_OPTS = 30          # max distinct answer options recorded per code
TRUNC = 80             # truncate long answer strings in the report
SOMNO_SAMPLE_ROWS = int(os.environ.get("SOMNO_SAMPLE_ROWS", "12000000"))


# --------------------------------------------------------------------------- #
def profile_kphc() -> dict:
    qmap = U.load_kphc_quest_map()
    target = qmap[qmap["code"].isin(U.TARGET_CODES)].copy()
    id2code = dict(zip(target["quest_id"], target["code"]))
    id_list = ", ".join(str(int(q)) for q in target["quest_id"])

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{(U.PROFILE_DIR / '_duckdb_tmp').as_posix()}'")
    src = (f"read_csv('{U.KPHC_CSV.as_posix()}', header=true, all_varchar=true, "
           f"ignore_errors=true, quote='\"', escape='\"')")

    print("KPHC: scanning for target quest_ids ...", flush=True)
    t0 = time.time()
    # one slim scan -> normalize quest_id, keep only target ids
    con.execute(f"""
        CREATE TEMP TABLE k AS
        SELECT try_cast(try_cast(QUEST_ID AS DOUBLE) AS BIGINT) AS qid,
               FORM_NAME AS form, QUESTION AS question, QUEST_ANSWER AS ans,
               PAT_ENC_CSN_ID AS enc
        FROM {src}
        WHERE try_cast(try_cast(QUEST_ID AS DOUBLE) AS BIGINT) IN ({id_list})
    """)
    n_rows = con.execute("SELECT count(*) FROM k").fetchone()[0]
    print(f"  matched {n_rows:,} answer rows [{time.time()-t0:.0f}s]", flush=True)

    forms = con.execute(
        "SELECT form, count(*) n FROM k GROUP BY 1 ORDER BY n DESC").fetchall()

    # per-code: distinct encounters, n rows, distinct answer options (capped)
    per_code = {}
    rows = con.execute("""
        SELECT qid, ans, count(*) n FROM k GROUP BY 1,2
    """).fetchall()
    opt = defaultdict(list)
    for qid, ans, n in rows:
        opt[qid].append((ans, n))
    enc_counts = dict(con.execute(
        "SELECT qid, count(DISTINCT enc) FROM k GROUP BY 1").fetchall())
    row_counts = dict(con.execute(
        "SELECT qid, count(*) FROM k GROUP BY 1").fetchall())

    for qid, code in id2code.items():
        qi = int(qid)
        options = sorted(opt.get(qi, []), key=lambda x: -x[1])
        per_code[code] = {
            "quest_id": qi,
            "family": U.FAMILY[code],
            "question": target.loc[target["quest_id"] == qid, "question"].iloc[0],
            "n_rows": row_counts.get(qi, 0),
            "n_encounters": enc_counts.get(qi, 0),
            "n_distinct_answers": len(options),
            "answer_options": [[(a or "")[:TRUNC], n] for a, n in options[:CAP_OPTS]],
        }
    con.close()
    return {"matched_rows": n_rows, "forms": forms, "per_code": per_code}


# --------------------------------------------------------------------------- #
def profile_somnoware() -> dict:
    print(f"Somnoware: sampling first {SOMNO_SAMPLE_ROWS:,} rows ...", flush=True)
    t0 = time.time()
    form_counts: Counter = Counter()
    # form -> question -> count
    q_by_form: dict = defaultdict(Counter)
    # form -> question -> set(option_text) (answer-option labels = metadata)
    opt_by_q: dict = defaultdict(lambda: defaultdict(set))
    # matrix block sizes: spec_idx -> {formid -> row count} (for scale/item inference)
    mblocks: dict = {i: defaultdict(int) for i in range(len(U.MATRIX_SPECS))}
    n = 0
    with U.SOMNO_CSV.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        header = U.somno_split(fh.readline())
        ncol = len(header)
        ix = {h: i for i, h in enumerate(header)}
        i_form, i_q, i_opt = ix["form_title"], ix["question"], ix["option_text"]
        i_fid = ix["pk_patientQuestFormId"]
        for line in fh:
            n += 1
            if n > SOMNO_SAMPLE_ROWS:
                break
            if n % 4_000_000 == 0:
                print(f"  scanned {n:,} [{time.time()-t0:.0f}s]", flush=True)
            row = U.somno_split(line)
            if len(row) != ncol:
                continue
            form = (row[i_form] or "").strip()
            form_counts[form] += 1
            cf = U.canon_form(form)                       # KP(1)->KP fold
            if cf in U.SOMNO_FORM_DENYLIST or not form:
                continue
            q = (row[i_q] or "").strip()
            if not q:
                continue
            q_by_form[form][q] += 1
            ot = (row[i_opt] or "").strip()
            if ot and len(opt_by_q[form][q]) < 12:
                opt_by_q[form][q].add(ot[:40])
            ql = q.lower()
            for si, spec in enumerate(U.MATRIX_SPECS):     # matrix block sizes
                if cf == spec["form"] and spec["stem"] in ql:
                    mblocks[si][row[i_fid]] += 1
    print(f"  done: {n:,} rows scanned [{time.time()-t0:.0f}s]", flush=True)

    matrix_out = []
    for si, spec in enumerate(U.MATRIX_SPECS):
        sizes = Counter(mblocks[si].values())
        matrix_out.append({
            "form": spec["form"], "stem": spec["stem"], "note": spec["note"],
            "codes": spec["codes"], "declared_scale": spec["scale"],
            "n_form_instances": len(mblocks[si]),
            "top_block_sizes": sizes.most_common(8),
        })

    forms_out = {}
    for form, qcounter in q_by_form.items():
        forms_out[form] = {
            "n_rows_sampled": form_counts[form],
            "n_distinct_questions": len(qcounter),
            "questions": [
                {"question": q[:120], "n": c,
                 "option_text_samples": sorted(opt_by_q[form][q])}
                for q, c in qcounter.most_common(60)
            ],
        }
    return {"sampled_rows": min(n, SOMNO_SAMPLE_ROWS),
            "form_counts": form_counts.most_common(),
            "in_scope_forms": forms_out,
            "matrix_blocks": matrix_out}


# --------------------------------------------------------------------------- #
def write_markdown(kphc: dict, somno: dict) -> None:
    L = ["# Questionnaire source profile", ""]
    L.append("## KPHC — `sleep_history_quest.csv` (full scan)")
    L.append(f"- target answer-rows matched: **{kphc['matched_rows']:,}**")
    L.append(f"- FORM_NAME of matched rows: " +
             ", ".join(f"`{f}` ({n:,})" for f, n in kphc["forms"][:5]))
    L.append("")
    L.append("### Per-target-code answer-option sets (decoder driver)")
    L.append("| code | family | #enc | #ans | distinct answer options (top) |")
    L.append("|---|---|---|---|---|")
    for code in U.TARGET_CODES:
        c = kphc["per_code"].get(code)
        if not c:
            L.append(f"| {code} | ? | 0 | 0 | **MISSING** |")
            continue
        opts = "; ".join(f"{a!r}×{n}" for a, n in c["answer_options"][:8])
        L.append(f"| {code} | {c['family']} | {c['n_encounters']:,} | "
                 f"{c['n_distinct_answers']} | {opts} |")
    L.append("")
    L.append(f"## Somnoware — `kaiser_questionnaire.csv` (SAMPLE of "
             f"{somno['sampled_rows']:,} rows)")
    L.append("Form distribution (sampled):")
    for f, n in somno["form_counts"][:15]:
        L.append(f"- `{f}` — {n:,}")
    L.append("")
    L.append("In-scope forms — distinct question counts:")
    for form, d in somno["in_scope_forms"].items():
        L.append(f"- **{form}** — {d['n_distinct_questions']} questions "
                 f"({d['n_rows_sampled']:,} rows)")
    L.append("")
    L.append("### Collapsed-matrix block sizes (scale/item inference for step 05)")
    L.append("block_size = n_items × scale; e.g. ESS 8×4=32.")
    for mb in somno.get("matrix_blocks", []):
        codes = mb["codes"]
        cs = f"{len(codes)} items" if codes else "items TBD"
        L.append(f"- **{mb['form']}** / `{mb['stem']}` ({cs}; declared scale "
                 f"{mb['declared_scale']}) — {mb['n_form_instances']:,} instances; "
                 f"top block sizes: {mb['top_block_sizes']}")
    (U.PROFILE_DIR / "PROFILE.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    U.ensure_dirs()
    kphc = profile_kphc()
    somno = profile_somnoware()
    (U.PROFILE_DIR / "kphc_profile.json").write_text(
        json.dumps(kphc, indent=1, default=str), encoding="utf-8")
    (U.PROFILE_DIR / "somnoware_profile.json").write_text(
        json.dumps(somno, indent=1, default=str), encoding="utf-8")
    write_markdown(kphc, somno)

    missing = [c for c in U.TARGET_CODES if kphc["per_code"].get(c, {}).get("n_rows", 0) == 0]
    print(f"\n[profile] KPHC matched {kphc['matched_rows']:,} rows; "
          f"codes with 0 KPHC rows: {missing or 'none'}")
    print(f"[profile] Somnoware in-scope forms: {list(somno['in_scope_forms'])}")
    U.log_milestone(
        f"02_profile_sources: KPHC matched {kphc['matched_rows']} target rows; "
        f"{len(missing)} codes absent in KPHC; profiled "
        f"{len(somno['in_scope_forms'])} Somnoware forms (sampled).")


if __name__ == "__main__":
    main()
