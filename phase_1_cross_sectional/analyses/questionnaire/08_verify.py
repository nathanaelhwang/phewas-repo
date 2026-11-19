# %% [markdown]
# Step 8 - Verify the unified frame + write SUMMARY.md and the data dictionary.
#
# Checks: exact target schema; per-family value ranges; STOP5/6/8 all-NA;
# per-code fill rates overall and by source. Emits a de-identified SUMMARY.md and
# column_data_dictionary.csv. No identifiers or row values are printed/written.

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import questionnaire_utils as U

FAM_DESC = {
    "ess": "0-3 Epworth doze likelihood", "freq": "0-4 how-often frequency",
    "yesno": "0/1 (No/Yes)", "isi_sev": "0-4 ISI severity",
    "isi_sat": "0-4 ISI satisfaction (reverse)", "isi_imp": "0-4 ISI impact",
    "phq": "0-3 PHQ-2 frequency", "fosq": "0-4 FOSQ standard (4=No difficulty..0=other)",
    "time": "minutes since midnight", "latency": "minutes (bin midpoint)",
    "duration": "minutes (nap bin midpoint)", "count": "integer count (0-9)",
    "neck": "0/1 collar >=16in", "hours": "hours of sleep 0-24",
    "text": "raw chief-complaint text", "empty": "EHR-derived (left blank)",
    "timebin": "2-hour clock bin label", "rlsfreq": "RLS frequency 0-5",
    "cc_binary": "chief-complaint present(1)/absent(0)",
}


def main() -> None:
    df = pd.read_parquet(U.RESULTS / "questionnaire_wide.parquet")
    problems = []

    item_cols = U.TARGET_CODES + U.CC_COLUMNS

    # 1. schema
    if list(df.columns) != U.FULL_COLUMNS + ["source"]:
        problems.append("column set/order != full schema + source")

    # 2. ranges + empty codes + CC binaries
    for code in U.TARGET_CODES:
        fam = U.FAMILY[code]
        if fam == "empty":
            if df[code].notna().any():
                problems.append(f"{code} should be all-NA (EHR-derived) but has values")
            continue
        if fam in U.NONNUMERIC_FAMILIES:
            continue
        s = pd.to_numeric(df[code], errors="coerce")
        if s.notna().any():
            lo, hi = U.FAMILY_RANGE[fam]
            if s.min() < lo or s.max() > hi:
                problems.append(f"{code} out of range [{lo},{hi}]: {s.min()}..{s.max()}")
    for c in U.CC_COLUMNS:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any() and (s.min() < 0 or s.max() > 1):
            problems.append(f"{c} not binary")

    # 3. fill rates overall + by source
    srcs = list(df["source"].unique())
    rows = []
    for col in item_cols:
        rec = {"code": col, "family": U.FAMILY.get(col, "cc_binary"),
               "n_filled": int(df[col].notna().sum())}
        rec["pct"] = round(100 * rec["n_filled"] / len(df), 1)
        for s in srcs:
            rec[f"n_{s}"] = int(df.loc[df["source"] == s, col].notna().sum())
        rows.append(rec)
    fill = pd.DataFrame(rows)

    # 4. data dictionary (with per-column value labels)
    dd_rows = []
    for col in U.FULL_COLUMNS:
        if col in U.KEY_COLS:
            dd_rows.append({"column": col, "role": "key", "family": "key",
                            "description": {"MRN": "12-digit MRN (join key)",
                                            "STUDY_DATE": "sleep-study date (KPHC only)",
                                            "CONTACT_DATE": "encounter/completion date"}[col],
                            "value_labels": ""})
        else:
            fam = U.FAMILY.get(col, "cc_binary")
            labels = U.value_labels(col) if col in U.FAMILY else \
                f"0={col[3:].lower().replace('_',' ')} not a chief complaint; 1=is"
            f = fill[fill["code"] == col].iloc[0]
            dd_rows.append({"column": col, "role": "item", "family": fam,
                            "description": FAM_DESC[fam], "value_labels": labels,
                            "n_filled": int(f["n_filled"]), "pct_filled": f["pct"]})
    dd = pd.DataFrame(dd_rows)
    U.safe_to_csv(dd, U.RESULTS / "column_data_dictionary.csv")

    # 5. SUMMARY.md
    by_src = df["source"].value_counts().to_dict()
    L = ["# Questionnaire cleaning — SUMMARY", "",
         f"- **{len(df):,} encounters** x {len(U.FULL_COLUMNS)} columns "
         f"(+ `source`); **{df['MRN'].nunique():,} distinct MRNs**.",
         f"- By source: {by_src}.",
         f"- Verification: {'**PASS** (no problems)' if not problems else '**ISSUES**'}."]
    if problems:
        L += [f"  - {p}" for p in problems]
    L += ["", "## Coverage / encoding notes",
          "- STOP5/STOP6/STOP8 (BMI/Age/Gender) intentionally blank — derive from EHR.",
          "- FOSQ = standard 0-4 (4=No difficulty..0=other), harmonized across all 3 sources.",
          "- BEDTIME/WAKETIME = 2-hour bin LABELS; BEDTIME_CONT/WAKETIME_CONT = continuous minutes.",
          "- CC1 kept as raw text AND one-hot into CC_<category> binaries (present=1/absent=0).",
          "- RLS4 = Scanned RLSHowOften (0-5). RLS1 now includes Scanned RLSUnpleasant.",
          "- Deferred: only KP abbreviated ESS/FOSQ matrices — KPHC covers those codes.",
          "- Screen freq items: KPHC/Intake/Sleep-History give 0-4; KP-narrative Yes/No not blended.",
          "", "## Fill rate by code (top + bottom)"]
    fs = fill.sort_values("pct", ascending=False)
    L.append("| code | family | filled | % | kphc | somnoware | scanned |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in pd.concat([fs.head(20), fs.tail(15)]).iterrows():
        L.append(f"| {r['code']} | {r['family']} | {r['n_filled']:,} | {r['pct']} | "
                 f"{r.get('n_kphc', 0):,} | {r.get('n_somnoware', 0):,} | {r.get('n_scanned', 0):,} |")
    (U.RESULTS / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")

    print(f"[verify] {len(df):,} encounters; problems: {problems or 'NONE'}")
    print(f"[verify] wrote SUMMARY.md + column_data_dictionary.csv")
    U.log_milestone(
        f"08_verify: {len(df)} encounters; verification "
        f"{'PASS' if not problems else 'ISSUES: '+str(problems)}.")


if __name__ == "__main__":
    main()
