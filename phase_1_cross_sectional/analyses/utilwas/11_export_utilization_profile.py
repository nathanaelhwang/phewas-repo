"""Build the disclosure-reviewed UtilWAS utilization-profile website export.

The script reads the existing v2 cohort and five-year windowed utilization store,
but writes aggregate outputs only. It never writes or prints patient-level rows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utilwas_utils as U


SCHEMA_VERSION = "1.1.0"
DISCLOSURE_THRESHOLD = 11
SEVERITY_ORDER = ["Overall", "None", "Mild", "Moderate", "Severe"]
WINDOW_ORDER = ["1yr", "5yr"]

EXPORT_DIR = U.RESULTS / "website_exports"
EXPORT_CSV = EXPORT_DIR / "utilwas_utilization_profile.csv"
MANIFEST_JSON = EXPORT_DIR / "utilwas_utilization_profile_manifest.json"
AUDIT_MD = EXPORT_DIR / "UTILWAS_UTILIZATION_PROFILE_AUDIT.md"
CANDIDATE_CROSSWALK = EXPORT_DIR / "utilwas_primary_care_crosswalk_candidate.csv"

REQUIRED_COLUMNS = [
    "metric_id",
    "metric_label",
    "window",
    "severity",
    "n_denominator",
    "n_nonmissing",
    "n_with_use",
    "mean",
    "sd",
    "median",
    "unit",
    "denominator_definition",
    "metric_definition",
    "suppressed",
    "suppression_reason",
]

METRICS = {
    "outpatient_rate": {
        "label": "Total outpatient encounter rate",
        "count_column": "outpatient_count",
        "unit": "encounters/person-year",
        "definition": (
            "Distinct-union outpatient encounters with CARE_TYPE in {CLINIC, VIRTUAL} "
            "and CARE_SUBTYPE in {OFFICE VISIT, TAV, VIDEO}; physician/allied-health "
            "subcounts are not added because they overlap."
        ),
    },
    "inpatient_rate": {
        "label": "Inpatient encounter rate",
        "count_column": "inpatient_count",
        "unit": "encounters/person-year",
        "definition": "Distinct encounters with a canonical UtilWAS inpatient CARE_SUBTYPE.",
    },
    "hospital_los_days": {
        "label": "Hospital length of stay",
        "unit": "days/admission",
        "definition": (
            "LOS in days across distinct canonical inpatient encounters with non-null LOS > 0; "
            "mean, sample SD, and median are admission-level statistics."
        ),
    },
    "ed_rate": {
        "label": "Emergency department encounter rate",
        "count_column": "ed_count",
        "unit": "encounters/person-year",
        "definition": "Distinct encounters with a canonical UtilWAS emergency-department CARE_SUBTYPE.",
    },
    "urgent_care_rate": {
        "label": "Urgent-care encounter rate",
        "count_column": "urgent_count",
        "unit": "encounters/person-year",
        "definition": (
            "Distinct encounters with DEPT_SPECIALTY exactly 'Urgent Care'; "
            "'Pediatric Urgent Care' is excluded."
        ),
    },
    "primary_care_rate": {
        "label": "Primary-care encounter rate",
        "count_column": "primary_care_count",
        "unit": "encounters/person-year",
        "definition": (
            "Distinct canonical outpatient encounters with DEPT_SPECIALTY in {Family Practice, "
            "Internal Medicine, Pediatrics}, per the owner-approved crosswalk."
        ),
    },
    "specialty_rate": {
        "label": "Specialty-care encounter rate",
        "count_column": "specialty_count",
        "unit": "encounters/person-year",
        "definition": (
            "Distinct canonical outpatient encounters with a nonblank DEPT_SPECIALTY outside the "
            "owner-approved primary-care list, excluding exact 'Urgent Care'."
        ),
    },
}

RATE_DENOMINATOR = (
    "All v2 cohort patients in the severity stratum, including patients with zero qualifying "
    "encounters. Each patient's distinct-encounter count is divided by min(obs_window, "
    "window_years), floored at 1/365 year, before patient-level rates are summarized."
)
LOS_DENOMINATOR = (
    "n_denominator is all distinct canonical inpatient encounters in the cell; n_nonmissing is "
    "the subset with non-null LOS > 0 used for the admission-level statistics; n_with_use is the "
    "number of distinct patients contributing at least one such valid-LOS admission."
)

APPROVED_PRIMARY_CARE_LABELS = {
    "Family Practice",
    "Internal Medicine",
    "Pediatrics",
}
URGENT_LABELS = {"Urgent Care"}

PENDING_CROSSWALK_DEFINITIONS = {
    "primary_care_rate": "Pending an owner-approved DEPT_SPECIALTY crosswalk; candidate labels are not used.",
    "specialty_rate": (
        "Pending an owner-approved DEPT_SPECIALTY crosswalk; intended as the complementary "
        "approved outpatient-specialty group with urgent care excluded."
    ),
}

VALIDATION_PATHS = {
    "counts_tidy": U.TABLES / "counts_tidy.csv",
    "outpatient_decomposition": U.TABLES / "outpatient_decomposition.csv",
    "los_tidy": U.TABLES / "los_tidy.csv",
    "severity_descriptives": U.TABLES / "severity_descriptives.csv",
    "specialty_descriptives_1yr": U.ANALYSIS / "specialty_descriptives_1yr.csv",
    "specialty_descriptives_5yr": U.ANALYSIS / "specialty_descriptives_5yr.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_aggregate_csv(path: Path) -> pd.DataFrame:
    """Preserve the literal severity label 'None' while parsing blank numeric cells."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def build_encounter_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE TEMP TABLE enc AS
        SELECT id,
               UTIL_ID,
               bool_or(in_1yr) AS in_1yr,
               any_value(CARE_TYPE) AS care_type,
               any_value(CARE_SUBTYPE) AS care_subtype,
               max(los_days) AS los_days,
               max(DEPT_SPECIALTY) AS dept_specialty,
               count(DISTINCT DEPT_SPECIALTY) FILTER (
                   WHERE DEPT_SPECIALTY IS NOT NULL AND trim(DEPT_SPECIALTY) <> ''
               ) AS n_specialty_labels,
               max(CASE WHEN DEPT_SPECIALTY = '{U.URGENT_CARE_SPECIALTY}' THEN 1 ELSE 0 END) AS is_urgent
        FROM read_parquet('{U.STORE_5YR}')
        GROUP BY id, UTIL_ID
        """
    )


def write_candidate_crosswalk(
    con: duckdb.DuckDBPyConnection, approve_proposed_crosswalk: bool
) -> tuple[pd.DataFrame, int]:
    labels = con.sql(
        f"""
        SELECT DISTINCT DEPT_SPECIALTY AS dept_specialty
        FROM read_parquet('{U.STORE_5YR}')
        WHERE CARE_TYPE IN ({U.sql_in(U.OUTPATIENT_CARE_TYPES)})
          AND CARE_SUBTYPE IN ({U.sql_in(U.OUTPATIENT_SUBTYPES)})
          AND DEPT_SPECIALTY IS NOT NULL
          AND trim(DEPT_SPECIALTY) <> ''
        ORDER BY DEPT_SPECIALTY
        """
    ).df()

    def classify(label: str) -> tuple[str, str]:
        if label in APPROVED_PRIMARY_CARE_LABELS:
            return (
                "primary_care" if approve_proposed_crosswalk else "primary_care_candidate",
                (
                    "Owner-approved primary-care department label."
                    if approve_proposed_crosswalk
                    else "Generalist department-name proposal only; owner approval is required before use."
                ),
            )
        if label in URGENT_LABELS:
            return (
                "exclude_urgent_care" if approve_proposed_crosswalk else "exclude_urgent_care_candidate",
                (
                    "Owner-approved exclusion from specialty care to avoid urgent-care double counting."
                    if approve_proposed_crosswalk
                    else "Proposed exclusion from specialty care to avoid urgent-care double counting; owner review required."
                ),
            )
        return (
            "specialty_care" if approve_proposed_crosswalk else "specialty_care_candidate",
            (
                "Owner-approved complement of the primary-care and exact urgent-care labels."
                if approve_proposed_crosswalk
                else "Proposed complement of candidate primary/urgent labels; owner approval is required before use."
            ),
        )

    proposed = labels["dept_specialty"].astype(str).map(classify)
    labels["proposed_classification"] = [x[0] for x in proposed]
    labels["classification_rationale"] = [x[1] for x in proposed]
    labels["approval_status"] = (
        "owner_approved_cli_override" if approve_proposed_crosswalk else "requires_owner_approval"
    )
    labels["owner_approval_required"] = not approve_proposed_crosswalk
    labels.to_csv(CANDIDATE_CROSSWALK, index=False)

    multi_specialty_encounters = int(
        con.sql(
            f"""
            SELECT count(*)
            FROM enc
            WHERE n_specialty_labels > 1
              AND care_type IN ({U.sql_in(U.OUTPATIENT_CARE_TYPES)})
              AND care_subtype IN ({U.sql_in(U.OUTPATIENT_SUBTYPES)})
            """
        ).fetchone()[0]
    )
    return labels, multi_specialty_encounters


def build_patient_counts(con: duckdb.DuckDBPyConnection) -> None:
    outpatient = (
        f"care_type IN ({U.sql_in(U.OUTPATIENT_CARE_TYPES)}) "
        f"AND care_subtype IN ({U.sql_in(U.OUTPATIENT_SUBTYPES)})"
    )
    inpatient = f"care_subtype IN ({U.sql_in(U.INPATIENT_SUBTYPES)})"
    emergency = f"care_subtype IN ({U.sql_in(U.ED_SUBTYPES)})"
    primary = f"dept_specialty IN ({U.sql_in(sorted(APPROVED_PRIMARY_CARE_LABELS))})"
    specialty = (
        "dept_specialty IS NOT NULL AND trim(dept_specialty) <> '' "
        f"AND dept_specialty NOT IN ({U.sql_in(sorted(APPROVED_PRIMARY_CARE_LABELS))}) "
        f"AND dept_specialty NOT IN ({U.sql_in(sorted(URGENT_LABELS))})"
    )
    con.execute(
        f"""
        CREATE TEMP TABLE cohort_windows AS
        SELECT id, severity, '1yr' AS window_label,
               CASE WHEN obs_window IS NULL THEN NULL
                    ELSE greatest(least(CAST(obs_window AS DOUBLE), 1.0), 1.0/365.0) END AS obs_years
        FROM read_parquet('{U.COHORT}')
        UNION ALL
        SELECT id, severity, '5yr' AS window_label,
               CASE WHEN obs_window IS NULL THEN NULL
                    ELSE greatest(least(CAST(obs_window AS DOUBLE), 5.0), 1.0/365.0) END AS obs_years
        FROM read_parquet('{U.COHORT}')
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE counts_by_id AS
        SELECT id,
               sum(CASE WHEN {outpatient} THEN 1 ELSE 0 END) AS outpatient_5yr,
               sum(CASE WHEN in_1yr AND {outpatient} THEN 1 ELSE 0 END) AS outpatient_1yr,
               sum(CASE WHEN {inpatient} THEN 1 ELSE 0 END) AS inpatient_5yr,
               sum(CASE WHEN in_1yr AND {inpatient} THEN 1 ELSE 0 END) AS inpatient_1yr,
               sum(CASE WHEN {emergency} THEN 1 ELSE 0 END) AS ed_5yr,
               sum(CASE WHEN in_1yr AND {emergency} THEN 1 ELSE 0 END) AS ed_1yr,
               sum(CASE WHEN is_urgent = 1 THEN 1 ELSE 0 END) AS urgent_5yr,
               sum(CASE WHEN in_1yr AND is_urgent = 1 THEN 1 ELSE 0 END) AS urgent_1yr,
               sum(CASE WHEN {outpatient} AND {primary} THEN 1 ELSE 0 END) AS primary_care_5yr,
               sum(CASE WHEN in_1yr AND {outpatient} AND {primary} THEN 1 ELSE 0 END) AS primary_care_1yr,
               sum(CASE WHEN {outpatient} AND {specialty} THEN 1 ELSE 0 END) AS specialty_5yr,
               sum(CASE WHEN in_1yr AND {outpatient} AND {specialty} THEN 1 ELSE 0 END) AS specialty_1yr
        FROM enc
        GROUP BY id
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE patient_counts AS
        SELECT cw.id,
               cw.severity,
               cw.window_label,
               cw.obs_years,
               CASE WHEN cw.window_label = '1yr' THEN coalesce(x.outpatient_1yr, 0)
                    ELSE coalesce(x.outpatient_5yr, 0) END AS outpatient_count,
               CASE WHEN cw.window_label = '1yr' THEN coalesce(x.inpatient_1yr, 0)
                    ELSE coalesce(x.inpatient_5yr, 0) END AS inpatient_count,
               CASE WHEN cw.window_label = '1yr' THEN coalesce(x.ed_1yr, 0)
                    ELSE coalesce(x.ed_5yr, 0) END AS ed_count,
               CASE WHEN cw.window_label = '1yr' THEN coalesce(x.urgent_1yr, 0)
                    ELSE coalesce(x.urgent_5yr, 0) END AS urgent_count,
               CASE WHEN cw.window_label = '1yr' THEN coalesce(x.primary_care_1yr, 0)
                    ELSE coalesce(x.primary_care_5yr, 0) END AS primary_care_count,
               CASE WHEN cw.window_label = '1yr' THEN coalesce(x.specialty_1yr, 0)
                    ELSE coalesce(x.specialty_5yr, 0) END AS specialty_count
        FROM cohort_windows cw
        LEFT JOIN counts_by_id x USING (id)
        """
    )


def rate_rows_and_totals(
    con: duckdb.DuckDBPyConnection, include_approved_crosswalk: bool,
) -> tuple[list[dict], dict[tuple[str, str, str], int]]:
    rows: list[dict] = []
    event_totals: dict[tuple[str, str, str], int] = {}
    metric_ids = ["outpatient_rate", "inpatient_rate", "ed_rate", "urgent_care_rate"]
    if include_approved_crosswalk:
        metric_ids.extend(["primary_care_rate", "specialty_rate"])
    for metric_id in metric_ids:
        meta = METRICS[metric_id]
        column = meta["count_column"]
        stats = con.sql(
            f"""
            SELECT window_label,
                   CASE WHEN GROUPING(severity) = 1 THEN 'Overall' ELSE severity END AS severity,
                   count(*) AS n_denominator,
                   count(obs_years) AS n_nonmissing,
                   count(*) FILTER (WHERE {column} > 0) AS n_with_use,
                   avg({column} / obs_years) AS mean,
                   stddev_samp({column} / obs_years) AS sd,
                   median({column} / obs_years) AS median,
                   sum({column}) AS n_events
            FROM patient_counts
            GROUP BY GROUPING SETS ((window_label, severity), (window_label))
            """
        ).df()
        for record in stats.to_dict("records"):
            key = (metric_id, str(record["window_label"]), str(record["severity"]))
            event_totals[key] = int(record["n_events"])
            rows.append(
                {
                    "metric_id": metric_id,
                    "metric_label": meta["label"],
                    "window": record["window_label"],
                    "severity": record["severity"],
                    "n_denominator": int(record["n_denominator"]),
                    "n_nonmissing": int(record["n_nonmissing"]),
                    "n_with_use": int(record["n_with_use"]),
                    "mean": float(record["mean"]),
                    "sd": float(record["sd"]),
                    "median": float(record["median"]),
                    "unit": meta["unit"],
                    "denominator_definition": RATE_DENOMINATOR,
                    "metric_definition": meta["definition"],
                }
            )
    return rows, event_totals


def los_rows(con: duckdb.DuckDBPyConnection) -> list[dict]:
    stats = con.sql(
        f"""
        WITH admissions AS (
          SELECT c.severity, '1yr' AS window_label, e.id, e.UTIL_ID, e.los_days
          FROM enc e
          JOIN read_parquet('{U.COHORT}') c USING (id)
          WHERE e.in_1yr
            AND e.care_subtype IN ({U.sql_in(U.INPATIENT_SUBTYPES)})
          UNION ALL
          SELECT c.severity, '5yr' AS window_label, e.id, e.UTIL_ID, e.los_days
          FROM enc e
          JOIN read_parquet('{U.COHORT}') c USING (id)
          WHERE e.care_subtype IN ({U.sql_in(U.INPATIENT_SUBTYPES)})
        )
        SELECT window_label,
               CASE WHEN GROUPING(severity) = 1 THEN 'Overall' ELSE severity END AS severity,
               count(*) AS n_denominator,
               count(*) FILTER (WHERE los_days IS NOT NULL AND los_days > 0) AS n_nonmissing,
               count(DISTINCT id) FILTER (WHERE los_days IS NOT NULL AND los_days > 0) AS n_with_use,
               avg(los_days) FILTER (WHERE los_days IS NOT NULL AND los_days > 0) AS mean,
               stddev_samp(los_days) FILTER (WHERE los_days IS NOT NULL AND los_days > 0) AS sd,
               median(los_days) FILTER (WHERE los_days IS NOT NULL AND los_days > 0) AS median
        FROM admissions
        GROUP BY GROUPING SETS ((window_label, severity), (window_label))
        """
    ).df()
    meta = METRICS["hospital_los_days"]
    rows = []
    for record in stats.to_dict("records"):
        rows.append(
            {
                "metric_id": "hospital_los_days",
                "metric_label": meta["label"],
                "window": record["window_label"],
                "severity": record["severity"],
                "n_denominator": int(record["n_denominator"]),
                "n_nonmissing": int(record["n_nonmissing"]),
                "n_with_use": int(record["n_with_use"]),
                "mean": float(record["mean"]),
                "sd": float(record["sd"]),
                "median": float(record["median"]),
                "unit": meta["unit"],
                "denominator_definition": LOS_DENOMINATOR,
                "metric_definition": meta["definition"],
            }
        )
    return rows


def unresolved_crosswalk_rows(rate_rows: list[dict]) -> list[dict]:
    denominators = {
        (row["window"], row["severity"]): (row["n_denominator"], row["n_nonmissing"])
        for row in rate_rows
        if row["metric_id"] == "outpatient_rate"
    }
    reason = "definition_not_approved: no owner-approved DEPT_SPECIALTY crosswalk"
    rows = []
    for metric_id in ("primary_care_rate", "specialty_rate"):
        meta = METRICS[metric_id]
        for window in WINDOW_ORDER:
            for severity in SEVERITY_ORDER:
                n_denominator, n_nonmissing = denominators[(window, severity)]
                rows.append(
                    {
                        "metric_id": metric_id,
                        "metric_label": meta["label"],
                        "window": window,
                        "severity": severity,
                        "n_denominator": n_denominator,
                        "n_nonmissing": n_nonmissing,
                        "n_with_use": pd.NA,
                        "mean": np.nan,
                        "sd": np.nan,
                        "median": np.nan,
                        "unit": meta["unit"],
                        "denominator_definition": RATE_DENOMINATOR,
                        "metric_definition": PENDING_CROSSWALK_DEFINITIONS[metric_id],
                        "suppressed": True,
                        "suppression_reason": reason,
                    }
                )
    return rows


def apply_disclosure(rows: list[dict]) -> list[dict]:
    for row in rows:
        if "suppressed" in row:
            continue
        supporting_counts = [row["n_denominator"], row["n_nonmissing"], row["n_with_use"]]
        failed = any(count is None or int(count) < DISCLOSURE_THRESHOLD for count in supporting_counts)
        row["suppressed"] = bool(failed)
        row["suppression_reason"] = (
            f"disclosure_threshold: one or more supporting counts are below {DISCLOSURE_THRESHOLD}"
            if failed
            else ""
        )
        if failed:
            row["mean"] = np.nan
            row["sd"] = np.nan
            row["median"] = np.nan
    return rows


def assemble_export(
    rate_rows: list[dict], los: list[dict], approve_proposed_crosswalk: bool
) -> pd.DataFrame:
    rows = apply_disclosure(rate_rows + los)
    if not approve_proposed_crosswalk:
        rows.extend(unresolved_crosswalk_rows(rate_rows))
    frame = pd.DataFrame(rows)
    frame = frame[REQUIRED_COLUMNS]
    metric_order = {metric: index for index, metric in enumerate(METRICS)}
    window_order = {window: index for index, window in enumerate(WINDOW_ORDER)}
    severity_order = {severity: index for index, severity in enumerate(SEVERITY_ORDER)}
    frame = (
        frame.assign(
            _metric=frame["metric_id"].map(metric_order),
            _window=frame["window"].map(window_order),
            _severity=frame["severity"].map(severity_order),
        )
        .sort_values(["_metric", "_window", "_severity"])
        .drop(columns=["_metric", "_window", "_severity"])
        .reset_index(drop=True)
    )
    for column in ("n_denominator", "n_nonmissing", "n_with_use"):
        frame[column] = frame[column].astype("Int64")
    for column in ("mean", "sd", "median"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round(6)
    return frame


def structural_checks(
    frame: pd.DataFrame,
    candidate: pd.DataFrame,
    approve_proposed_crosswalk: bool,
    multi_specialty_encounters: int,
) -> list[dict]:
    checks = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail})

    add("required_schema", list(frame.columns) == REQUIRED_COLUMNS, "Required columns are present in exact order.")
    add("row_count", len(frame) == 70, f"Observed {len(frame)} rows; expected 70.")
    add(
        "complete_unique_grid",
        not frame.duplicated(["metric_id", "window", "severity"]).any()
        and set(frame["metric_id"]) == set(METRICS)
        and set(frame["window"]) == set(WINDOW_ORDER)
        and set(frame["severity"]) == set(SEVERITY_ORDER),
        "Metric/window/severity keys form the expected unique 7 x 2 x 5 grid.",
    )
    identifier_columns = {
        "id",
        "patient_id",
        "person_id",
        "mrn",
        "util_id",
        "enc_sid",
        "encounter_id",
        "ssn",
        "dob",
        "name",
        "address",
    }
    bad_columns = sorted(set(column.lower() for column in frame.columns) & identifier_columns)
    add("no_identifier_like_columns", not bad_columns, f"Identifier-like columns found: {bad_columns or 'none'}.")
    suppressed = frame["suppressed"]
    add(
        "suppressed_statistics_are_null",
        frame.loc[suppressed, ["mean", "sd", "median"]].isna().all().all(),
        "All suppressed rows have null mean, SD, and median.",
    )
    add(
        "released_statistics_are_finite",
        np.isfinite(frame.loc[~suppressed, ["mean", "sd", "median"]].to_numpy(dtype=float)).all(),
        "All unsuppressed statistics are finite.",
    )
    supporting = frame.loc[~suppressed, ["n_denominator", "n_nonmissing", "n_with_use"]]
    add(
        "disclosure_threshold",
        (supporting >= DISCLOSURE_THRESHOLD).all().all(),
        f"Every unsuppressed supporting count is at least {DISCLOSURE_THRESHOLD}.",
    )
    add(
        "candidate_crosswalk_unique",
        candidate["dept_specialty"].notna().all() and not candidate["dept_specialty"].duplicated().any(),
        "Candidate crosswalk has one row per nonblank relevant DEPT_SPECIALTY label.",
    )
    if approve_proposed_crosswalk:
        approved_primary = set(
            candidate.loc[candidate["proposed_classification"] == "primary_care", "dept_specialty"]
        )
        add(
            "approved_crosswalk_exact",
            approved_primary == APPROVED_PRIMARY_CARE_LABELS
            and candidate["approval_status"].eq("owner_approved_cli_override").all()
            and candidate["owner_approval_required"].eq(False).all(),
            "Approved crosswalk uses the exact three-label primary-care list and records the CLI override.",
        )
        add(
            "approved_crosswalk_single_label_rule",
            multi_specialty_encounters == 0,
            f"Canonical outpatient encounters with multiple nonblank specialty labels: {multi_specialty_encounters}.",
        )
    else:
        add(
            "candidate_crosswalk_not_approved",
            candidate["approval_status"].eq("requires_owner_approval").all()
            and candidate["owner_approval_required"].eq(True).all(),
            "Every candidate classification is explicitly marked as requiring owner approval.",
        )
    return checks


def validate_against_aggregates(
    frame: pd.DataFrame,
    event_totals: dict[tuple[str, str, str], int],
    approve_proposed_crosswalk: bool,
) -> list[dict]:
    validations: list[dict] = []
    lookup = frame.set_index(["metric_id", "window", "severity"])

    counts = read_aggregate_csv(VALIDATION_PATHS["counts_tidy"])
    mapping = {"inpatient_rate": "inpatient", "ed_rate": "ED", "urgent_care_rate": "urgent_care"}
    count_diffs = []
    user_diffs = []
    denominator_diffs = []
    one_year_mean_diffs = []
    five_year_deltas = []
    for metric_id, analysis in mapping.items():
        source = counts[(counts["analysis"] == analysis) & (counts["subcategory"] == "Total")]
        for record in source.to_dict("records"):
            key = (metric_id, str(record["window"]), str(record["severity"]))
            exported = lookup.loc[key]
            count_diffs.append(abs(event_totals[key] - int(record["n_encounters"])))
            user_diffs.append(abs(int(exported["n_with_use"]) - int(record["n_pat_ge1"])))
            denominator_diffs.append(abs(int(exported["n_denominator"]) - int(record["n_cohort"])))
            if record["window"] == "1yr":
                one_year_mean_diffs.append(abs(float(exported["mean"]) - float(record["per_pt_per_yr"])))
            else:
                five_year_deltas.append(float(exported["mean"]) - float(record["per_pt_per_yr"]))
    counts_pass = max(count_diffs + user_diffs + denominator_diffs, default=0) == 0
    validations.append(
        {
            "check_id": "counts_tidy_counts_users_denominators",
            "source": str(VALIDATION_PATHS["counts_tidy"].relative_to(U.PROJECT)).replace("\\", "/"),
            "status": "pass" if counts_pass else "fail",
            "detail": (
                f"Exact reconciliation for inpatient, ED, and urgent-care event totals, users, and cohort "
                f"denominators; maximum absolute integer difference {max(count_diffs + user_diffs + denominator_diffs, default=0)}."
            ),
        }
    )
    validations.append(
        {
            "check_id": "counts_tidy_rate_means",
            "source": str(VALIDATION_PATHS["counts_tidy"].relative_to(U.PROJECT)).replace("\\", "/"),
            "status": "expected_difference"
            if max(one_year_mean_diffs, default=0.0) <= 1.1e-6 and min(five_year_deltas, default=0.0) >= -1.1e-6
            else "fail",
            "detail": (
                "The 1yr means reconcile within CSV rounding (max absolute difference "
                f"{max(one_year_mean_diffs, default=0.0):.6g}). The requested 5yr person-specific means "
                "are expected to exceed or equal nominal count/(N*5) rates because some obs_window values "
                f"are below five years (delta range {min(five_year_deltas, default=0.0):.6g} to "
                f"{max(five_year_deltas, default=0.0):.6g})."
            ),
        }
    )

    decomp = read_aggregate_csv(VALIDATION_PATHS["outpatient_decomposition"])
    outpatient_diffs = []
    for record in decomp.to_dict("records"):
        key = ("outpatient_rate", str(record["window"]), str(record["severity"]))
        outpatient_diffs.append(abs(event_totals[key] - int(record["total"])))
    validations.append(
        {
            "check_id": "outpatient_distinct_union",
            "source": str(VALIDATION_PATHS["outpatient_decomposition"].relative_to(U.PROJECT)).replace("\\", "/"),
            "status": "pass" if max(outpatient_diffs, default=0) == 0 else "fail",
            "detail": (
                "Canonical outpatient distinct-union encounter totals reconcile without adding overlapping "
                f"physician/allied counts; maximum absolute difference {max(outpatient_diffs, default=0)}."
            ),
        }
    )

    los = read_aggregate_csv(VALIDATION_PATHS["los_tidy"])
    los = los[(los["analysis"] == "inpatient") & (los["subcategory"] == "Total") & (los["rule"] == "los>0")]
    los_count_diffs = []
    los_mean_diffs = []
    los_median_diffs = []
    for record in los.to_dict("records"):
        key = ("hospital_los_days", str(record["window"]), str(record["severity"]))
        exported = lookup.loc[key]
        los_count_diffs.append(abs(int(exported["n_nonmissing"]) - int(record["n_enc_los"])))
        los_mean_diffs.append(abs(float(exported["mean"]) - float(record["mean_los"])))
        los_median_diffs.append(abs(float(exported["median"]) - float(record["median_los"])))
    los_pass = (
        max(los_count_diffs, default=0) == 0
        and max(los_mean_diffs, default=0.0) <= 0.005001
        and max(los_median_diffs, default=0.0) <= 0.005001
    )
    validations.append(
        {
            "check_id": "los_tidy_inpatient_total",
            "source": str(VALIDATION_PATHS["los_tidy"].relative_to(U.PROJECT)).replace("\\", "/"),
            "status": "pass" if los_pass else "fail",
            "detail": (
                "Valid-LOS admission counts reconcile exactly; mean and median reconcile to the source's "
                f"two-decimal precision (max differences {max(los_mean_diffs, default=0.0):.6g} and "
                f"{max(los_median_diffs, default=0.0):.6g}). The source has no SD for independent comparison."
            ),
        }
    )

    severity = read_aggregate_csv(VALIDATION_PATHS["severity_descriptives"])
    sev_count_diffs = []
    sev_user_diffs = []
    sev_mean_count_diffs = []
    for metric_id, analysis in mapping.items():
        source = severity[severity["analysis"] == analysis]
        for record in source.to_dict("records"):
            key = (metric_id, str(record["window"]), str(record["severity"]))
            exported = lookup.loc[key]
            sev_count_diffs.append(abs(int(exported["n_denominator"]) - int(record["n_pat"])))
            sev_user_diffs.append(abs(int(exported["n_with_use"]) - int(record["n_users"])))
            source_mean_count = float(record["mean_all"])
            derived_mean_count = event_totals[key] / int(exported["n_denominator"])
            sev_mean_count_diffs.append(abs(source_mean_count - derived_mean_count))
    severity_pass = (
        max(sev_count_diffs + sev_user_diffs, default=0) == 0
        and max(sev_mean_count_diffs, default=0.0) <= 0.000501
    )
    validations.append(
        {
            "check_id": "severity_descriptives_counts",
            "source": str(VALIDATION_PATHS["severity_descriptives"].relative_to(U.PROJECT)).replace("\\", "/"),
            "status": "pass" if severity_pass else "fail",
            "detail": (
                "Clinical-severity denominators, users, and unannualized mean counts reconcile for inpatient, "
                f"ED, and urgent care (max mean-count rounding difference {max(sev_mean_count_diffs, default=0.0):.6g}). "
                "Five-year rate means intentionally differ after person-specific annualization."
            ),
        }
    )

    for key in ("specialty_descriptives_1yr", "specialty_descriptives_5yr"):
        specialty = read_aggregate_csv(VALIDATION_PATHS[key])
        validations.append(
            {
                "check_id": key,
                "source": str(VALIDATION_PATHS[key].relative_to(U.PROJECT)).replace("\\", "/"),
                "status": "not_comparable",
                "detail": (
                    f"Source loaded successfully ({len(specialty)} aggregate rows), but it contains only "
                    ">=500-user retained specialties split into overlapping physician/allied families. It cannot "
                    "directly validate the grouped primary/specialty distinct-union rates"
                    + (" from the owner-approved crosswalk." if approve_proposed_crosswalk else ", and no approved crosswalk exists.")
                ),
            }
        )
    return validations


def source_hashes() -> dict[str, dict]:
    sources = {
        "generator": Path(__file__).resolve(),
        "utilwas_utils": Path(U.__file__).resolve(),
        "cohort": Path(U.COHORT),
        "windowed_utilization_store": Path(U.STORE_5YR),
        **VALIDATION_PATHS,
    }
    result = {}
    for name, path in sources.items():
        resolved = Path(path).resolve()
        result[name] = {
            "path": str(resolved.relative_to(U.PROJECT)).replace("\\", "/"),
            "sha256": sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    return result


def write_audit(
    manifest: dict,
    candidate: pd.DataFrame,
    multi_specialty_encounters: int,
    approve_proposed_crosswalk: bool,
) -> None:
    validation_lines = ["| Check | Status | Result |", "|---|---|---|"]
    for result in manifest["validation_results"]:
        detail = str(result["detail"]).replace("|", "\\|")
        validation_lines.append(f"| `{result['check_id']}` | **{result['status']}** | {detail} |")

    proposed_primary = sorted(
        candidate.loc[candidate["dept_specialty"].isin(APPROVED_PRIMARY_CARE_LABELS), "dept_specialty"].astype(str)
    )
    proposed_text = ", ".join(f"`{label}`" for label in proposed_primary) or "None"
    unresolved = "\n".join(f"- {item}" for item in manifest["unresolved_items"]) or "- None."
    if multi_specialty_encounters:
        multi_specialty_note = (
            f"Canonical outpatient encounters carrying more than one nonblank specialty label: "
            f"**{multi_specialty_encounters}**. The owner must approve a multi-label classification rule."
        )
    else:
        multi_specialty_note = (
            "No canonical outpatient encounters carried more than one nonblank specialty label, so the "
            "current analytic store does not require a multi-label classification rule."
        )
    if approve_proposed_crosswalk:
        outcome = (
            "It is structurally and disclosure validated and approved for public release under the "
            "explicit project-owner crosswalk override."
        )
        crosswalk_heading = "Approved primary-care crosswalk"
        crosswalk_text = (
            f"The explicit approval override accepts **{len(candidate)}** classified nonblank "
            "DEPT_SPECIALTY labels. Primary-care labels: {proposed_text}. Exact `Urgent Care` is "
            f"excluded from specialty care; all other nonblank canonical outpatient labels form the "
            f"specialty-care complement. {multi_specialty_note}"
        )
    else:
        outcome = (
            "It is structurally and disclosure validated, but it is **not approved for release** "
            "because no owner-approved primary-care crosswalk exists. The 20 primary-/specialty-care "
            "rows are retained and definition-gated with null mean, SD, and median."
        )
        crosswalk_heading = "Primary-care crosswalk gate"
        crosswalk_text = (
            f"No approved crosswalk or approved label list was supplied. The candidate contains "
            f"**{len(candidate)}** nonblank DEPT_SPECIALTY labels observed on canonical outpatient "
            f"encounters in the five-year analytic store. Proposed primary-care labels: "
            f"{proposed_text}. The candidate treats remaining labels as the specialty-care complement "
            f"after excluding exact `Urgent Care`. {multi_specialty_note}"
        )
    audit = f"""# UtilWAS utilization profile audit

Generated: {manifest['generated_at_utc']}

## Outcome

The export contains **{manifest['row_count']} rows** (7 metrics x 2 windows x 5 strata). {outcome}

- `release_approved`: **{str(manifest['release_approved']).lower()}**
- `public_website_release_approved`: **{str(manifest['public_website_release_approved']).lower()}**
- Suppressed rows: **{manifest['suppressed_cell_count']}** ({manifest['disclosure_suppressed_cell_count']} threshold-driven; {manifest['definition_suppressed_cell_count']} definition-gated)

## Aggregate-only safety controls

- The generator writes only the required aggregate CSV, manifest, audit, and label-only candidate crosswalk.
- No patient IDs, encounter IDs, or patient-level rates are written or printed.
- The final CSV has the exact required schema and no identifier-like columns.
- Existing patient-level windowed or overlap files are not copied into `website_exports`.
- Literal severity `None` is preserved when validation CSVs are read.

## Definitions

- Cohort: the existing v2 ICD analytic spine, including all cohort patients in each stratum.
- Windows: encounter date in `[index - N years, index)`, lower-inclusive and index-exclusive, for N = 1 and 5.
- Rates: calculate each person's distinct-encounter count divided by `min(obs_window, N)` (floor 1/365 year), include zero-use patients, then report mean, sample SD (`ddof=1`), and median.
- Outpatient: the distinct union of `CARE_TYPE` in {{`CLINIC`, `VIRTUAL`}} and `CARE_SUBTYPE` in {{`OFFICE VISIT`, `TAV`, `VIDEO`}}. Physician/allied counts are not added.
- ED and inpatient use the exact subtype lists in `analyses/utilwas/utilwas_utils.py`. The categories intentionally overlap on the shared `ED-*` subtypes.
- Urgent care requires `DEPT_SPECIALTY == 'Urgent Care'` exactly; `Pediatric Urgent Care` is excluded.
- LOS: admission-level LOS in days for canonical inpatient encounters with non-null LOS > 0. `n_denominator` counts all qualifying inpatient encounters, `n_nonmissing` counts valid positive LOS admissions, and `n_with_use` counts distinct contributing patients.

## {crosswalk_heading}

{crosswalk_text}

## Disclosure review

The project threshold is **{DISCLOSURE_THRESHOLD}**. An unsuppressed row must have `n_denominator`, `n_nonmissing`, and `n_with_use` all at or above the threshold. Suppressed rows retain their expected key but have null mean, SD, and median and a nonblank reason.

## Validation

{chr(10).join(validation_lines)}

The existing aggregate tables do not contain SD, so sample SD was recomputed from the in-memory patient-level rate distribution (or admission-level LOS distribution) and cannot be independently matched to those aggregates. Five-year legacy rates use a nominal five-year denominator; differences from the requested person-specific annualization are expected and documented above.

## Unresolved items

{unresolved}
"""
    AUDIT_MD.write_text(audit, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve-proposed-crosswalk",
        action="store_true",
        help=(
            "Record explicit owner approval of the three-label primary-care definition and compute "
            "primary-/specialty-care aggregates. Omit to retain the definition gate."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    missing_sources = [str(path) for path in [Path(U.COHORT), Path(U.STORE_5YR), *VALIDATION_PATHS.values()] if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"Required sources are missing: {missing_sources}")

    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=48, threads=12)
    build_encounter_table(con)
    candidate, multi_specialty_encounters = write_candidate_crosswalk(
        con, args.approve_proposed_crosswalk
    )
    build_patient_counts(con)
    rate_rows, event_totals = rate_rows_and_totals(con, args.approve_proposed_crosswalk)
    los = los_rows(con)
    frame = assemble_export(rate_rows, los, args.approve_proposed_crosswalk)

    structural = structural_checks(
        frame, candidate, args.approve_proposed_crosswalk, multi_specialty_encounters
    )
    aggregate_validation = validate_against_aggregates(
        frame, event_totals, args.approve_proposed_crosswalk
    )
    validation_results = structural + aggregate_validation

    failed = [result for result in validation_results if result["status"] == "fail"]
    suppressed_reason = frame["suppression_reason"].fillna("")
    definition_suppressed = int(suppressed_reason.str.startswith("definition_not_approved").sum())
    disclosure_suppressed = int(suppressed_reason.str.startswith("disclosure_threshold").sum())
    suppressed_count = int(frame["suppressed"].sum())

    frame.to_csv(EXPORT_CSV, index=False, na_rep="", float_format="%.6f")

    proposed_primary = sorted(
        candidate.loc[candidate["dept_specialty"].isin(APPROVED_PRIMARY_CARE_LABELS), "dept_specialty"].astype(str)
    )
    unresolved_items = []
    if not args.approve_proposed_crosswalk:
        unresolved_items.extend(
            [
                "A UtilWAS owner must approve an exact DEPT_SPECIALTY crosswalk and primary-care label list.",
                "After approval, primary- and specialty-care rates must be recomputed and revalidated before release.",
            ]
        )
    if multi_specialty_encounters:
        unresolved_items.append(
            "The owner must approve how encounters carrying multiple nonblank DEPT_SPECIALTY labels are classified."
        )
    if failed:
        unresolved_items.append(
            "One or more structural or aggregate validation checks failed: "
            + ", ".join(result["check_id"] for result in failed)
            + "."
        )

    release_approved = (
        args.approve_proposed_crosswalk and not failed and definition_suppressed == 0
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "row_count": len(frame),
        "required_row_count": 70,
        "source_hashes": source_hashes(),
        "disclosure_threshold": DISCLOSURE_THRESHOLD,
        "cohort_definition": (
            "Existing v2 ICD analytic cohort; OSA-diagnosis-naive at index, at least one year of "
            "pre-index ICD observability, at least two pre-index ICD encounters; Overall plus AHI-based "
            "None/Mild/Moderate/Severe strata."
        ),
        "window_definitions": {
            "1yr": "Encounter date in [index - 1 year, index), lower-inclusive and index-exclusive.",
            "5yr": "Encounter date in [index - 5 years, index), lower-inclusive and index-exclusive.",
        },
        "rate_denominator_definition": RATE_DENOMINATOR,
        "los_denominator_definition": LOS_DENOMINATOR,
        "exact_primary_care_specialty_list": proposed_primary if args.approve_proposed_crosswalk else [],
        "proposed_primary_care_specialty_list": proposed_primary,
        "primary_care_crosswalk_status": (
            "owner_approved_cli_override"
            if args.approve_proposed_crosswalk
            else "requires_owner_approval"
        ),
        "crosswalk_approval_basis": (
            "Explicit project-owner public-release override, applied with --approve-proposed-crosswalk."
            if args.approve_proposed_crosswalk
            else None
        ),
        "primary_care_crosswalk_candidate": str(CANDIDATE_CROSSWALK.relative_to(U.PROJECT)).replace("\\", "/"),
        "specialty_care_definition": (
            "Canonical outpatient encounters with a nonblank DEPT_SPECIALTY outside the exact approved "
            "primary-care list, excluding exact 'Urgent Care'."
            if args.approve_proposed_crosswalk
            else "Pending owner approval. Intended definition: canonical outpatient encounters assigned to the "
            "approved outpatient-specialty label universe, excluding approved primary-care labels and urgent care."
        ),
        "ed_inpatient_overlap_warning": (
            "ED and inpatient intentionally overlap on ED-IP, ED-OB, ED-OB-IP, ED-OB-SU, ED-OP, "
            "ED-OP-IP, ED-SU, and ED-SU-IP; do not add the categories as if mutually exclusive."
        ),
        "suppressed_cell_count": suppressed_count,
        "disclosure_suppressed_cell_count": disclosure_suppressed,
        "definition_suppressed_cell_count": definition_suppressed,
        "validation_results": validation_results,
        "unresolved_items": unresolved_items,
        "release_approved": bool(release_approved),
        "public_website_release_approved": bool(release_approved),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_audit(manifest, candidate, multi_specialty_encounters, args.approve_proposed_crosswalk)

    if len(frame) != 70 or failed:
        raise RuntimeError("Export was written for audit but did not pass all structural/aggregate validation checks.")
    print(
        f"Aggregate export complete: rows={len(frame)}, suppressed={suppressed_count}, "
        f"release_approved={str(release_approved).lower()}."
    )


if __name__ == "__main__":
    main()
