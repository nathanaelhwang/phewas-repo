"""Unadjusted log-rank tests for the exported incidence curves.

The tests answer two descriptive questions for each exported PheCode:

1. Do event-free survival curves differ by OSA status (No OSA vs any OSA) or
   severity (No OSA, Mild, Moderate, Severe) through six years?
2. Among landmark-eligible OSA patients, do curves differ across the five CPAP
   adherence groups through five years after the 90- or 180-day landmark?

The plotted curves are Aalen-Johansen cumulative-incidence functions with death
as a competing event. As requested, these are ordinary cause-specific log-rank
tests: incident disease (etype == 1) is the event, while censoring and competing
death (etype in {0, 2}) are censored. Gray's test would be the directly matched
test of equality of the plotted cumulative-incidence functions.

Inputs are the no-ID long exports created by steps 11 and 13. Outputs are
aggregate internal result tables; the source outcomes were preselected by the
severity Cox screen, so these p-values are descriptive/post-selection.

Usage:  python 16_incidence_logrank.py
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "incwas_results"
ANALYSIS = RESULTS / "analysis"
SEVERITY_INPUT = RESULTS / "survival_curves" / "survival_long.parquet"
LANDMARK_INPUT = RESULTS / "survival_curves_landmark" / "survival_long_landmark.parquet"
PHECODE_META = RESULTS / "survival_curves" / "fdr_phecodes.csv"

SEVERITY_OUTPUT = ANALYSIS / "logrank_osa_status_severity.csv"
LANDMARK_OUTPUT = ANALYSIS / "logrank_cpap_landmark.csv"

SEVERITY_ORDER = ["No OSA", "Mild", "Moderate", "Severe"]
ADHERENCE_ORDER = [
    "4+ hr/night",
    "2-4 hr/night",
    "0-2 hr/night",
    "never-started",
    "unknown",
]
SEVERITY_HORIZON = 6.0
LANDMARK_HORIZON = 5.0


def _group_summary(
    data: pd.DataFrame, group_col: str, order: list[str], horizon: float
) -> tuple[str, int, int, bool]:
    """Return aggregate group counts and test-sparsity diagnostics."""
    parts: list[str] = []
    min_n = len(data)
    min_events = int((data["etype"] == 1).sum())
    sparse = False
    for group in order:
        sub = data[data[group_col] == group]
        within = sub["time_years"] <= horizon
        n = len(sub)
        events = int(((sub["etype"] == 1) & within).sum())
        deaths = int(((sub["etype"] == 2) & within).sum())
        min_n = min(min_n, n)
        min_events = min(min_events, events)
        sparse |= n < 20 or events < 5
        parts.append(f"{group}:n={n},events={events},deaths={deaths}")
    return " | ".join(parts), min_n, min_events, sparse


def _logrank_row(
    data: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    horizon: float,
) -> dict:
    """Run one horizon-truncated k-sample log-rank test."""
    observed_groups = set(data[group_col].dropna().unique())
    expected_groups = set(group_order)
    if observed_groups != expected_groups:
        missing = sorted(expected_groups - observed_groups)
        extra = sorted(observed_groups - expected_groups)
        raise ValueError(f"Unexpected {group_col} levels; missing={missing}, extra={extra}")

    original_time = data["time_years"].to_numpy(float)
    durations = np.minimum(original_time, horizon)
    events = ((data["etype"].to_numpy() == 1) & (original_time <= horizon)).astype(int)
    groups = data[group_col]
    result = multivariate_logrank_test(durations, groups, events)

    summary, min_n, min_events, sparse = _group_summary(
        data, group_col, group_order, horizon
    )
    within = data["time_years"] <= horizon
    n_groups = len(group_order)
    return {
        "n_groups": n_groups,
        "df": n_groups - 1,
        "n_atrisk": len(data),
        "n_events": int(((data["etype"] == 1) & within).sum()),
        "n_deaths_competing": int(((data["etype"] == 2) & within).sum()),
        "min_group_n": min_n,
        "min_group_events": min_events,
        "sparse_test_group": sparse,
        "chi2": float(result.test_statistic),
        "p": float(result.p_value),
        "group_summary": summary,
    }


def _adjust_pvalues(data: pd.DataFrame, family_cols: list[str]) -> pd.DataFrame:
    """Adjust substantive outcomes and distinguish sparse descriptive tests."""
    data = data.copy()
    data["multiplicity_included"] = ~data["osa_control"].astype(bool)
    data["p_fdr_bh"] = np.nan
    data["p_bonferroni"] = np.nan
    data["sig_nominal"] = data["p"] < 0.05
    data["sig_fdr_bh"] = pd.Series(pd.NA, index=data.index, dtype="boolean")
    data["sig_bonferroni"] = pd.Series(pd.NA, index=data.index, dtype="boolean")

    substantive = data[data["multiplicity_included"]]
    for _, idx in substantive.groupby(family_cols, sort=False, dropna=False).groups.items():
        p = data.loc[idx, "p"].to_numpy(float)
        reject_fdr, p_fdr, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
        reject_bon, p_bon, _, _ = multipletests(p, alpha=0.05, method="bonferroni")
        data.loc[idx, "p_fdr_bh"] = p_fdr
        data.loc[idx, "p_bonferroni"] = p_bon
        data.loc[idx, "sig_fdr_bh"] = reject_fdr
        data.loc[idx, "sig_bonferroni"] = reject_bon

    # Sparse tests remain in the prespecified multiplicity family to avoid selecting
    # that family from the observed event counts. Their raw/adjusted values are
    # retained, but they are not counted as inferential calls.
    data["inference_eligible"] = (
        data["multiplicity_included"] & ~data["sparse_test_group"]
    )
    data["sig_nominal_inference"] = data["sig_nominal"] & data["inference_eligible"]
    data["sig_fdr_bh_inference"] = (
        data["sig_fdr_bh"].fillna(False) & data["inference_eligible"]
    )
    data["sig_bonferroni_inference"] = (
        data["sig_bonferroni"].fillna(False) & data["inference_eligible"]
    )
    return data


def _validate_levels(data: pd.DataFrame, column: str, expected: list[str]) -> None:
    """Fail before categorical conversion can hide an unexpected label as NA."""
    observed = set(data[column].dropna().unique())
    missing = sorted(set(expected) - observed)
    extra = sorted(observed - set(expected))
    if data[column].isna().any() or missing or extra:
        raise ValueError(
            f"Unexpected {column} levels; missing={missing}, extra={extra}, "
            f"missing_values={int(data[column].isna().sum())}"
        )


def severity_tests(osa_control: dict[str, bool]) -> pd.DataFrame:
    columns = ["phecode", "phenotype", "time_years", "etype", "severity"]
    data = pd.read_parquet(SEVERITY_INPUT, columns=columns)
    _validate_levels(data, "severity", SEVERITY_ORDER)
    data["severity"] = pd.Categorical(data["severity"], SEVERITY_ORDER, ordered=True)

    rows: list[dict] = []
    for phecode, sub in data.groupby("phecode", sort=True, observed=True):
        phenotype = sub["phenotype"].iloc[0]

        status = sub.assign(
            osa_status=np.where(sub["severity"] == "No OSA", "No OSA", "Any OSA")
        )
        row = _logrank_row(
            status, "osa_status", ["No OSA", "Any OSA"], SEVERITY_HORIZON
        )
        rows.append(
            {
                "analysis": "osa_status",
                "phecode": phecode,
                "phenotype": phenotype,
                "osa_control": osa_control.get(str(phecode), str(phecode).startswith("327.3")),
                "horizon_years": SEVERITY_HORIZON,
                **row,
            }
        )

        row = _logrank_row(sub, "severity", SEVERITY_ORDER, SEVERITY_HORIZON)
        rows.append(
            {
                "analysis": "osa_severity",
                "phecode": phecode,
                "phenotype": phenotype,
                "osa_control": osa_control.get(str(phecode), str(phecode).startswith("327.3")),
                "horizon_years": SEVERITY_HORIZON,
                **row,
            }
        )

    del data
    gc.collect()
    result = _adjust_pvalues(pd.DataFrame(rows), ["analysis"])
    return result.sort_values(["analysis", "p", "phecode"], kind="stable")


def landmark_tests(osa_control: dict[str, bool]) -> pd.DataFrame:
    columns = [
        "phecode",
        "phenotype",
        "window_days",
        "time_years",
        "etype",
        "adherence",
    ]
    data = pd.read_parquet(LANDMARK_INPUT, columns=columns)
    _validate_levels(data, "adherence", ADHERENCE_ORDER)
    data["adherence"] = pd.Categorical(data["adherence"], ADHERENCE_ORDER, ordered=True)

    rows: list[dict] = []
    keys = ["window_days", "phecode"]
    for (window, phecode), sub in data.groupby(keys, sort=True, observed=True):
        row = _logrank_row(sub, "adherence", ADHERENCE_ORDER, LANDMARK_HORIZON)
        rows.append(
            {
                "analysis": "landmark_cpap_adherence",
                "analysis_role": "primary" if int(window) == 180 else "sensitivity",
                "window_days": int(window),
                "phecode": phecode,
                "phenotype": sub["phenotype"].iloc[0],
                "osa_control": osa_control.get(str(phecode), str(phecode).startswith("327.3")),
                "horizon_years": LANDMARK_HORIZON,
                **row,
            }
        )

    del data
    gc.collect()
    result = _adjust_pvalues(pd.DataFrame(rows), ["window_days"])
    return result.sort_values(["window_days", "p", "phecode"], kind="stable")


def _print_summary(data: pd.DataFrame, family_cols: list[str]) -> None:
    substantive = data[~data["osa_control"]]
    for keys, sub in substantive.groupby(family_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = ", ".join(f"{col}={value}" for col, value in zip(family_cols, keys))
        print(
            f"{label}: {len(sub)} substantive tests; "
            f"nominal={int(sub['sig_nominal'].sum())} "
            f"(inference-eligible={int(sub['sig_nominal_inference'].sum())}), "
            f"BH-FDR={int(sub['sig_fdr_bh'].sum())} "
            f"(inference-eligible={int(sub['sig_fdr_bh_inference'].sum())}), "
            f"Bonferroni={int(sub['sig_bonferroni'].sum())} "
            f"(inference-eligible={int(sub['sig_bonferroni_inference'].sum())}), "
            f"sparse-group flags={int(sub['sparse_test_group'].sum())}"
        )


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(PHECODE_META, dtype={"phecode": str})
    osa_control = meta.set_index("phecode")["osa_control"].astype(bool).to_dict()

    severity = severity_tests(osa_control)
    severity.to_csv(SEVERITY_OUTPUT, index=False, float_format="%.12g")
    landmark = landmark_tests(osa_control)
    landmark.to_csv(LANDMARK_OUTPUT, index=False, float_format="%.12g")

    _print_summary(severity, ["analysis"])
    _print_summary(landmark, ["window_days", "analysis_role"])
    print(f"wrote {len(severity)} tests -> {SEVERITY_OUTPUT}")
    print(f"wrote {len(landmark)} tests -> {LANDMARK_OUTPUT}")


if __name__ == "__main__":
    main()
