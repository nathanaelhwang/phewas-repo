# %% [markdown]
# Step 05 - BehWAS association driver.
#
# Runs every behavioral outcome against the seven OSA-severity contrasts under the unadjusted
# screen + nested M1/M3/M4 models, using the shared engine (_behwas_engine.py): logistic OR for
# binary forms, OLS-on-INT beta for continuous forms. Both forms of each feature are run so the
# forest-plot effect scale (all-binary->OR vs native mixed) is a downstream choice.
#
# Windowed outcomes (adherence/flu/FMLA) run on the 1yr & 5yr targets; lifetime outcomes
# (no-show/off-work) run once on the lifetime targets. Adherence is TESTED-ONLY (med-users);
# all other outcomes are whole-cohort (non-events = 0). BH-FDR + Bonferroni are applied within
# each (window, contrast, model, effect_type) family (note: with so few outcomes the correction
# is minor and is reported for completeness).
#
# Emits analysis/results_{1yr,5yr,lifetime}.csv + analysis/results_all.csv (16/17-col schema).
# DE-IDENTIFIED aggregates only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import behwas_utils as U
import _behwas_engine as E

MIN_N = 4 * E.MIN_GRP  # need at least this many patients to test an outcome

# (outcome, kind, column, population): population "users" drops NaN, "cohort" fills 0.
WINDOWED_OUTCOMES = [
    ("adherence_pdc", "continuous", "mean_pdc", "users"),
    ("adherent", "binary", "adherent", "users"),
    ("flu_received", "binary", "flu_received", "cohort"),
    ("flu_count", "continuous", "flu_count", "cohort"),
    ("fmla_filed", "binary", "fmla_filed", "cohort"),
    ("fmla_dur_days", "continuous", "fmla_dur_days", "cohort"),
    ("fmla_admitted", "binary", "fmla_admitted", "cohort"),
    ("fmla_pregnancy", "binary", "fmla_pregnancy", "cohort"),
]
LIFETIME_OUTCOMES = [
    ("any_noshow", "binary", "any_noshow", "cohort"),
    ("no_show_pct", "continuous", "no_show_pct", "cohort"),
    ("any_off_work", "binary", "any_off_work", "cohort"),
    ("off_work_days", "continuous", "off_work_days", "cohort"),
]

RESULT_COLS = ["window", "outcome", "kind", "contrast", "model", "n", "effect_type",
               "effect", "se", "ci_low", "ci_high", "p", "neglog10p",
               "sig_bon", "sig_fdr", "unstable"]


def run_one(d: pd.DataFrame, outcome: str, kind: str, col: str) -> list:
    if len(d) < MIN_N or d[col].nunique() < 2:
        return [E.row(outcome, "omnibus", "unadjusted", len(d), "skip",
                      np.nan, np.nan, np.nan, np.nan, np.nan, "low_n_or_zero_variance")]
    if kind == "binary":
        rows = E.binary_unadjusted(d, col, outcome) + E.binary_adjusted(d, col, outcome)
    else:
        rows = E.cont_unadjusted(d, col, outcome) + E.cont_adjusted(d, col, outcome)
    for r in rows:
        r["kind"] = kind
    return rows


def run_scope(cohort: pd.DataFrame, targets: pd.DataFrame, outcomes, label: str) -> pd.DataFrame:
    df = cohort.merge(targets, on="id", how="inner")
    rows = []
    for outcome, kind, col, pop in outcomes:
        if pop == "users":
            d = df[df[col].notna() & np.isfinite(df[col])].copy()
        else:
            d = df.assign(**{col: df[col].fillna(0)})
        out = run_one(d, outcome, kind, col)
        for r in out:
            r.setdefault("kind", kind)
            r["window"] = label
        rows.extend(out)
    res = pd.DataFrame(rows)
    res = E.add_fdr_bonferroni(res, ["window", "contrast", "model", "effect_type"])
    return res[RESULT_COLS]


def main() -> None:
    U.ensure_dirs()
    cohort = E.prep_cohort()
    print(f"cohort used: {len(cohort):,}")

    all_res = []
    for label, _years in U.WINDOWS:
        tg = pd.read_parquet(U.ANALYSIS / f"targets_{label}.parquet")
        res = run_scope(cohort, tg, WINDOWED_OUTCOMES, label)
        res.to_csv(U.ANALYSIS / f"results_{label}.csv", index=False)
        all_res.append(res)
        n_sig = res.query("model=='m4' and sig_fdr").shape[0]
        print(f"[{label}] {len(res):,} tests -> results_{label}.csv (m4 FDR-sig: {n_sig})")

    life = pd.read_parquet(U.ANALYSIS / "targets_lifetime.parquet")
    res_life = run_scope(cohort, life, LIFETIME_OUTCOMES, "lifetime")
    res_life.to_csv(U.ANALYSIS / "results_lifetime.csv", index=False)
    all_res.append(res_life)
    print(f"[lifetime] {len(res_life):,} tests -> results_lifetime.csv")

    combined = pd.concat(all_res, ignore_index=True)
    combined.to_csv(U.ANALYSIS / "results_all.csv", index=False)
    print(f"\nwrote results_all.csv ({len(combined):,} rows)")

    U.log_milestone(
        "Step 05 BehWAS: association results for 5 behavioral outcomes (binary OR + continuous "
        "beta forms) across 1yr/5yr/lifetime written to analysis/results_*.csv.")


if __name__ == "__main__":
    main()
