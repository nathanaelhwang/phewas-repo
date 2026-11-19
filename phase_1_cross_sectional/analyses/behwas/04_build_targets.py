# %% [markdown]
# Step 04 - Assemble per-patient BehWAS targets, joined to the v2 cohort spine.
#
# Three of the five features are WINDOWED on their own event date ([idx - N yr, idx), where
# idx = STUDY_DATE); two are LIFETIME per-person values (no usable date). For every feature we
# store BOTH a continuous and a binary form, so the forest-plot effect scale can be chosen later
# without re-extracting.
#
#   WINDOWED (1yr & 5yr):
#     adherence  -> among med-users only: mean_pdc (continuous), adherent = PDC>=0.80 (binary).
#                   Non-users are left NaN (PDC is undefined off medication -> tested-only pop).
#     flu shot   -> whole cohort: flu_count (continuous), flu_received (binary).
#     FMLA       -> whole cohort: fmla_dur_days (continuous), fmla_filed / fmla_admitted /
#                   fmla_pregnancy (binary).
#   LIFETIME (no window):
#     no-show    -> whole cohort: no_show_pct (continuous), any_noshow (binary).
#     off-work   -> whole cohort: off_work_days (continuous), any_off_work (binary).
#
# Emits: analysis/targets_1yr.parquet, analysis/targets_5yr.parquet, analysis/targets_lifetime.parquet
# DE-IDENTIFIED coverage summary only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import behwas_utils as U


def _norm(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _in_window(dates: pd.Series, idx: pd.Series, years: int) -> pd.Series:
    lower = idx - pd.DateOffset(years=years)
    return (dates >= lower) & (dates < idx)


def main() -> None:
    U.ensure_dirs()

    cohort = pd.read_parquet(U.COHORT, columns=["id", "STUDY_DATE"])
    cohort["idx"] = _norm(cohort["STUDY_DATE"])
    base = cohort[["id"]].copy()
    n_cohort = len(base)
    print(f"cohort patients: {n_cohort:,}")

    adh = pd.read_parquet(U.FEATURES_DIR / "adherence.parquet").merge(
        cohort[["id", "idx"]], on="id", how="inner")
    adh["med_start_date"] = _norm(adh["med_start_date"])
    flu = pd.read_parquet(U.FEATURES_DIR / "flu_events.parquet").merge(
        cohort[["id", "idx"]], on="id", how="inner")
    flu["immune_date"] = _norm(flu["immune_date"])
    fmla = pd.read_parquet(U.FEATURES_DIR / "fmla.parquet").merge(
        cohort[["id", "idx"]], on="id", how="inner")
    fmla["contact_date"] = _norm(fmla["contact_date"])

    # ---- windowed targets ------------------------------------------------------
    for label, years in U.WINDOWS:
        # adherence: among users (mean PDC + n classes; adherent flag). NaN for non-users.
        aw = adh[_in_window(adh["med_start_date"], adh["idx"], years)]
        a = (aw.groupby("id")
               .agg(mean_pdc=("adherence", "mean"), n_adh_classes=("gpi", "nunique"))
               .reset_index())
        a["adherent"] = (a["mean_pdc"] >= U.ADHERENT_PDC).astype(int)

        # flu: whole cohort (count + received)
        fw = flu[_in_window(flu["immune_date"], flu["idx"], years)]
        f = (fw.groupby("id").agg(flu_count=("immune_date", "count")).reset_index())
        f["flu_received"] = 1

        # fmla: whole cohort (filed + duration + admitted/pregnancy attributes)
        mw = fmla[_in_window(fmla["contact_date"], fmla["idx"], years)]
        m = (mw.groupby("id")
               .agg(fmla_dur_days=("prob_cond_dur_days", "max"),
                    fmla_admitted=("admitted", "max"),
                    fmla_pregnancy=("pregnancy", "max"))
               .reset_index())
        m["fmla_filed"] = 1

        tg = (base.merge(a, on="id", how="left")
                  .merge(f, on="id", how="left")
                  .merge(m, on="id", how="left"))
        # whole-cohort outcomes: non-events = 0 (adherence stays NaN for non-users)
        for col, fill in [("flu_count", 0), ("flu_received", 0), ("n_adh_classes", 0),
                          ("fmla_filed", 0), ("fmla_dur_days", 0),
                          ("fmla_admitted", 0), ("fmla_pregnancy", 0)]:
            tg[col] = tg[col].fillna(fill)
        tg[["fmla_filed", "fmla_admitted", "fmla_pregnancy", "flu_received"]] = \
            tg[["fmla_filed", "fmla_admitted", "fmla_pregnancy", "flu_received"]].astype(int)

        out_path = U.ANALYSIS / f"targets_{label}.parquet"
        tg.to_parquet(out_path, index=False)

        n_users = tg["mean_pdc"].notna().sum()
        print(f"\n[{label}] window [idx-{years}yr, idx)")
        print(f"  adherence med-users:     {n_users:,} ({100*n_users/n_cohort:.1f}%); "
              f"adherent (PDC>=0.80): {int(tg['adherent'].sum()):,}")
        print(f"  flu received:            {int(tg['flu_received'].sum()):,} "
              f"({100*tg['flu_received'].mean():.1f}%)")
        print(f"  fmla filed:              {int(tg['fmla_filed'].sum()):,} "
              f"({100*tg['fmla_filed'].mean():.1f}%); admitted {int(tg['fmla_admitted'].sum()):,}, "
              f"pregnancy {int(tg['fmla_pregnancy'].sum()):,}")

    # ---- lifetime targets (no-show + off-work) ---------------------------------
    person = pd.read_parquet(U.FEATURES_DIR / "popcohort_person.parquet")
    life = base.merge(person, on="id", how="left")
    for col in ["no_show", "total_noshow", "no_show_pct", "total_noshow_pct",
                "office_visit", "off_work_days"]:
        life[col] = life[col].fillna(0)
    life["any_noshow"] = ((life["total_noshow"] > 0) | (life["no_show"] > 0)).astype(int)
    life["any_off_work"] = life["off_work"].isin([1, 2]).astype(int)
    life_path = U.ANALYSIS / "targets_lifetime.parquet"
    life.to_parquet(life_path, index=False)

    print("\n[lifetime] no-show + off-work (per-person, non-windowed)")
    print(f"  any no-show:  {int(life['any_noshow'].sum()):,} ({100*life['any_noshow'].mean():.1f}%); "
          f"no_show_pct median {life['no_show_pct'].median():.3f}")
    print(f"  any off-work: {int(life['any_off_work'].sum()):,} ({100*life['any_off_work'].mean():.1f}%); "
          f"off_work_days median-among-positive "
          f"{life.loc[life['off_work_days'] > 0, 'off_work_days'].median():.0f}")

    U.log_milestone(
        "Step 04 targets: windowed (adherence/flu/fmla, 1yr & 5yr) + lifetime (no-show/off-work) "
        "per-patient targets built with both continuous and binary forms.")


if __name__ == "__main__":
    main()
