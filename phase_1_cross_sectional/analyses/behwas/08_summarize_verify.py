# %% [markdown]
# Step 08 - Verification gate + SUMMARY.md (run before trusting results).
#
# Asserts: 12-char ids on every feature; targets keyed to the cohort (no leakage); windowed
# presence monotonic (1yr subset of 5yr) for flu / FMLA / adherence-users; a flu windowing
# recompute matches the 5yr targets. Any failure -> SystemExit. Then writes a de-identified
# SUMMARY.md of the headline M4 effects. DE-IDENTIFIED only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import behwas_utils as U

FAILS = []


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


def _norm(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def main() -> None:
    cohort = pd.read_parquet(U.COHORT, columns=["id", "STUDY_DATE"])
    cohort["idx"] = _norm(cohort["STUDY_DATE"])
    cohort_ids = set(cohort["id"])

    print("== feature id hygiene ==")
    for fn in ["adherence.parquet", "fmla.parquet", "flu_events.parquet", "popcohort_person.parquet"]:
        p = U.FEATURES_DIR / fn
        if not p.exists():
            check(f"{fn} exists", False)
            continue
        ids = pd.read_parquet(p, columns=["id"])["id"].astype(str)
        check(f"{fn} ids all 12-char", bool((ids.str.len() == 12).all()))

    print("== targets keyed to cohort ==")
    tg = {}
    for label in ["1yr", "5yr", "lifetime"]:
        tg[label] = pd.read_parquet(U.ANALYSIS / f"targets_{label}.parquet")
        leak = (~tg[label]["id"].isin(cohort_ids)).sum()
        check(f"targets_{label} ids all in cohort", leak == 0)

    print("== windowed presence monotonic (1yr subset of 5yr) ==")
    t1, t5 = tg["1yr"], tg["5yr"]
    for col in ["flu_received", "fmla_filed"]:
        s1 = set(t1.loc[t1[col] == 1, "id"]); s5 = set(t5.loc[t5[col] == 1, "id"])
        check(f"{col}: 1yr subset of 5yr", s1 <= s5)
    u1 = set(t1.loc[t1["mean_pdc"].notna(), "id"]); u5 = set(t5.loc[t5["mean_pdc"].notna(), "id"])
    check("adherence users: 1yr subset of 5yr", u1 <= u5)

    print("== flu windowing recompute (5yr) ==")
    flu = pd.read_parquet(U.FEATURES_DIR / "flu_events.parquet").merge(
        cohort[["id", "idx"]], on="id", how="inner")
    flu["immune_date"] = _norm(flu["immune_date"])
    lower = flu["idx"] - pd.DateOffset(years=5)
    got = set(flu.loc[(flu["immune_date"] >= lower) & (flu["immune_date"] < flu["idx"]), "id"])
    want = set(t5.loc[t5["flu_received"] == 1, "id"])
    check("recomputed 5yr flu-received matches targets", got == want)

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))

    # ---- SUMMARY.md ------------------------------------------------------------
    res = pd.read_csv(U.ANALYSIS / "results_all.csv")
    lines = ["# BehWAS summary\n",
             "Forest-plot prevalence scan of 5 behavioral outcomes vs OSA severity "
             "(reuses the v2 cohort, 7 contrasts, M1/M3/M4). Both binary (OR) and continuous "
             "(beta) forms are computed; the headline effect scale is still an open decision.\n",
             "## Coverage (whole cohort unless noted)"]
    for label in ["1yr", "5yr"]:
        t = tg[label]
        lines.append(f"- {label}: flu {int(t['flu_received'].sum()):,} "
                     f"({100*t['flu_received'].mean():.1f}%), fmla {int(t['fmla_filed'].sum()):,} "
                     f"({100*t['fmla_filed'].mean():.1f}%), adherence users {int(t['mean_pdc'].notna().sum()):,}")
    lt = tg["lifetime"]
    lines.append(f"- lifetime: any no-show {int(lt['any_noshow'].sum()):,} "
                 f"({100*lt['any_noshow'].mean():.1f}%), any off-work {int(lt['any_off_work'].sum()):,} "
                 f"({100*lt['any_off_work'].mean():.1f}%)\n")

    lines.append("## Headline M4 effects (severe_vs_none & trend)")
    head = (res.query("model=='m4' and contrast in ['severe_vs_none','trend'] and effect_type in ['OR','beta']")
               .sort_values(["effect_type", "outcome", "window", "contrast"]))
    for _, r in head.iterrows():
        star = "*" if r["sig_fdr"] else ""
        lines.append(f"- {r['outcome']} [{r['window']}] {r['contrast']}: "
                     f"{r['effect_type']}={r['effect']:.3f} "
                     f"({r['ci_low']:.3f}, {r['ci_high']:.3f}) p={r['p']:.2g}{star}")
    (U.ANALYSIS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", U.ANALYSIS / "SUMMARY.md")

    U.log_milestone(f"Step 08 verify: {'PASS' if not FAILS else 'FAIL ' + str(FAILS)}; SUMMARY.md written.")
    if FAILS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
