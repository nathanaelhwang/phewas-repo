"""Disclosure-controlled aggregate CPAP/PAP summary for phenotype octants.

All patient-level joins remain in memory and are reduced immediately to the requested
9-group by 5-window aggregate table.  No patient row, identifier, or date is printed or
written.  Release is fail-closed: serialized disclosure checks, two integer
reconstruction audits, source-definition gates, manifest hash preservation, and
post-write verification all run before the artifact is considered complete.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


PROJECT = Path(__file__).resolve().parents[2]
PHENOTYPES = (PROJECT / "latent-class-analysis" / "results" / "phenotypes"
              / "cross_domain_phenotypes.csv")
C1_AUX = PROJECT / "latent-class-analysis" / "work" / "c1_aux.parquet"
FOLLOWUP = PROJECT / "results" / "incwas_results" / "followup" / "followup_frame.parquet"
CPAP = PROJECT / "results" / "incwas_results" / "followup" / "cpap_usage.parquet"
OUT = (PROJECT / "results" / "incwas_results" / "phenotype_exposure"
       / "website_exports")
CSV_NAME = "octant_cpap_summary.csv"
AUDIT_NAME = "octant_cpap_summary_audit.json"
MANIFEST_NAME = "phenotype_website_export_manifest.json"
CSV_PATH = OUT / CSV_NAME
AUDIT_PATH = OUT / AUDIT_NAME
MANIFEST_PATH = OUT / MANIFEST_NAME
TARGET_NAMES = {CSV_NAME, AUDIT_NAME}

DISCLOSURE_MIN = 11
EXPECTED_COHORT_N = 70_880
ROUND_DIGITS = 6
OVERALL = "overall"
ORDER = [
    "mild-all",
    "hypoxemia-predominant",
    "symptom-predominant",
    "comorbidity-predominant",
    "hypoxemic-symptomatic",
    "hypoxemic-comorbid",
    "symptomatic-comorbid",
    "high-all",
]
EXPECTED_OCTANT_N = {
    "mild-all": 9_249,
    "hypoxemia-predominant": 8_349,
    "symptom-predominant": 8_685,
    "comorbidity-predominant": 8_728,
    "hypoxemic-symptomatic": 9_157,
    "hypoxemic-comorbid": 9_114,
    "symptomatic-comorbid": 8_778,
    "high-all": 8_820,
}
GROUPS = ORDER + [OVERALL]

WINDOWS = [
    {
        "key": "before_or_on_index",
        "days": 0,
        "label": "documented setup before/on sleep-study index",
    },
    {
        "key": "within_90_days",
        "days": 90,
        "label": "documented new setup after index through 90 days",
    },
    {
        "key": "within_180_days",
        "days": 180,
        "label": "documented new setup after index through 180 days",
    },
    {
        "key": "within_365_days",
        "days": 365,
        "label": "documented new setup after index through 365 days",
    },
    {
        "key": "ever_after_index",
        "days": None,
        "label": "documented setup after index through observed follow-up",
    },
]
WINDOW_BY_KEY = {item["key"]: item for item in WINDOWS}
WINDOW_KEYS = [item["key"] for item in WINDOWS]

SCHEMA = [
    "octant",
    "n_octant",
    "initiation_window_days",
    "n_observable_for_window",
    "n_cpap_record_present",
    "cpap_record_coverage_pct",
    "n_documented_setup",
    "documented_setup_pct",
    "initiation_label",
    "n_started_with_adherence_flag",
    "n_adherent",
    "adherent_pct_among_evaluable_starters",
    "adherence_flag_missing",
    "adherence_coverage_pct",
    "n_started_with_usage",
    "usage_coverage_pct",
    "mean_usage_hours_night",
    "sd_usage_hours_night",
    "median_usage_hours_night",
    "q1_usage_hours_night",
    "q3_usage_hours_night",
    "usage_window_definition",
    "usage_unit_source",
    "usage_unit_confirmed",
    "adherence_definition",
    "adherence_definition_confirmed",
    "suppressed",
    "suppression_reason",
]

COUNT_COLUMNS = [
    "n_observable_for_window",
    "n_cpap_record_present",
    "n_documented_setup",
    "n_started_with_adherence_flag",
    "adherence_flag_missing",
    "n_started_with_usage",
]
# Exact complements that are not exported as columns but are recoverable by subtracting
# two released counts.  They receive the same primary and complementary disclosure
# protection as explicit counts.  The final specification activates automatically once
# n_adherent is added to COUNT_COLUMNS by a future, definition-approved release.
IMPLICIT_COMPLEMENT_SPECS = [
    (
        "observable_without_cpap_record",
        "n_observable_for_window",
        "n_cpap_record_present",
    ),
    (
        "cpap_record_without_documented_setup",
        "n_cpap_record_present",
        "n_documented_setup",
    ),
    (
        "setup_without_adherence_flag",
        "n_documented_setup",
        "n_started_with_adherence_flag",
    ),
    (
        "setup_without_usage",
        "n_documented_setup",
        "n_started_with_usage",
    ),
    (
        "evaluable_nonadherent",
        "n_started_with_adherence_flag",
        "n_adherent",
    ),
]
NESTED_OBSERVATION_WINDOW_PAIRS = [
    ("ever_after_index", "within_90_days"),
    ("within_90_days", "within_180_days"),
    ("within_180_days", "within_365_days"),
]
NONSTRUCTURAL_OBSERVATION_WINDOWS = [
    "within_90_days",
    "within_180_days",
    "within_365_days",
    "ever_after_index",
]
NON_NESTED_INITIATION_SERIES = [
    "n_documented_setup",
    "n_started_with_adherence_flag",
    "adherence_flag_missing",
    "n_started_with_usage",
]
STRUCTURAL_ZERO_COMPLEMENT_RULE = (
    "For the before/on-index row, n_observable_for_window equals n_octant by "
    "construction because every cohort member has the required index. Its zero "
    "complement is structural rather than an observed small cell and cannot be hidden "
    "while both that denominator definition and the public octant total are retained."
)
NON_NESTED_INITIATION_INCREMENT_RULE = (
    "Setup, adherence-evaluable, adherence-missing, and usage counts use different "
    "sufficient-observation cohorts at each fixed window. Adjacent differences are net "
    "changes across non-nested analytic populations, not exact patient-count increments, "
    "so they are not treated as disclosure complements."
)
ADHERENCE_IDENTITY = [
    "n_documented_setup",
    "n_started_with_adherence_flag",
    "adherence_flag_missing",
]
DERIVED_DEPENDENCIES = {
    "cpap_record_coverage_pct": [
        "n_cpap_record_present", "n_observable_for_window"
    ],
    "documented_setup_pct": ["n_documented_setup", "n_observable_for_window"],
    "adherence_coverage_pct": [
        "n_started_with_adherence_flag", "n_documented_setup"
    ],
    "usage_coverage_pct": ["n_started_with_usage", "n_documented_setup"],
}
USAGE_DISTRIBUTION_COLUMNS = [
    "mean_usage_hours_night",
    "sd_usage_hours_night",
    "median_usage_hours_night",
    "q1_usage_hours_night",
    "q3_usage_hours_night",
]
SCIENTIFIC_NULL_COLUMNS = [
    "n_adherent",
    "adherent_pct_among_evaluable_starters",
    *USAGE_DISTRIBUTION_COLUMNS,
    "usage_window_definition",
    "adherence_definition",
]
SOURCE_METADATA_FILE = "Description.xlsx"
SOURCE_METADATA_PATH = Path("D:/Active files to transfer") / SOURCE_METADATA_FILE
SOURCE_METADATA_SHA256 = (
    "ab0bceabe91da932a8197faa4ad0ead1599dd0966325f3ef3f196a3e59f617f0"
)
SOURCE_METADATA_SHEET = "Sheet3"
SOURCE_METADATA_RANGE = "A81:G84"
CMS_LCD_ID = "L33718"
CMS_LCD_URL = (
    "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=33718"
)
CMS_DEFINITION_VERBATIM_EXCERPT = (
    "use of PAP ≥4 hours per night on 70% of nights during a consecutive thirty "
    "(30) day period anytime during the first three (3) months"
)
CMS_DEFINITION_PARAPHRASE = (
    "At least four hours of PAP use per night on at least 70% of nights in any "
    "consecutive 30-day period during the first three months of initial use."
)
USAGE_UNIT_SOURCE = (
    "Source metadata Description.xlsx, Sheet3 G82: Average minutes for PAP usage; "
    "the averaging denominator and window are not specified."
)
GENERIC_SUPPRESSION_REASON = "disclosure_control"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_definition_sources() -> None:
    if (not SOURCE_METADATA_PATH.is_file()
            or _sha256_file(SOURCE_METADATA_PATH) != SOURCE_METADATA_SHA256):
        raise RuntimeError("the pinned PAP source-metadata workbook is missing or changed")


def _verify_manifest_files(manifest: dict) -> dict[str, str]:
    expected: dict[str, str] = {}
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError("manifest files registry is missing")
    for entry in files:
        name = entry.get("name")
        recorded = entry.get("sha256")
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded)):
            raise RuntimeError("manifest contains an invalid file entry")
        if name in expected:
            raise RuntimeError("manifest contains duplicate file names")
        path = OUT / name
        if not path.is_file() or _sha256_file(path) != recorded:
            raise RuntimeError("an existing manifest artifact failed hash verification")
        expected[name] = recorded
    return expected


def _load_verified_manifest() -> tuple[dict, bytes, dict[str, str]]:
    if not OUT.is_dir() or not MANIFEST_PATH.is_file():
        raise RuntimeError("website export directory or manifest is missing")
    payload = MANIFEST_PATH.read_bytes()
    manifest = json.loads(payload.decode("utf-8"))
    if manifest.get("validation", {}).get("status") != "PASS":
        raise RuntimeError("existing manifest validation status is not PASS")
    return manifest, payload, _verify_manifest_files(manifest)


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return np.nan
    return round(100.0 * numerator / denominator, ROUND_DIGITS)


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a narrow projection through DuckDB for repository Parquet compatibility."""
    selected = ", ".join(f'"{column}"' for column in columns)
    escaped = path.as_posix().replace("'", "''")
    connection = duckdb.connect()
    try:
        return connection.execute(
            f"SELECT {selected} FROM read_parquet('{escaped}')"
        ).df()
    finally:
        connection.close()


def _aggregate() -> tuple[pd.DataFrame, dict]:
    """Return only aggregate rows plus aggregate reconciliation metadata."""
    ph = pd.read_csv(PHENOTYPES, usecols=["MRN", "octant"], dtype={"MRN": str})
    if (len(ph) != EXPECTED_COHORT_N or ph["MRN"].isna().any()
            or ph["MRN"].duplicated().any() or ph["octant"].isna().any()):
        raise RuntimeError("phenotype cohort identity or size check failed")
    octant_n = ph.groupby("octant", observed=True).size().reindex(ORDER)
    if (octant_n.isna().any() or octant_n.astype(int).to_dict() != EXPECTED_OCTANT_N
            or set(ph["octant"]) != set(ORDER)):
        raise RuntimeError("phenotype octant reconciliation failed")

    aux = _read_parquet_columns(C1_AUX, ["MRN", "STUDY_DATE"])
    if aux["MRN"].isna().any() or aux["MRN"].duplicated().any():
        raise RuntimeError("sleep-study index source identity check failed")
    aux["MRN"] = aux["MRN"].astype(str)
    aux["idx"] = pd.to_datetime(aux["STUDY_DATE"], errors="coerce").dt.normalize()
    aux = aux[["MRN", "idx"]]

    follow = _read_parquet_columns(FOLLOWUP, ["id", "idx", "censor_date"])
    if follow["id"].isna().any() or follow["id"].duplicated().any():
        raise RuntimeError("follow-up frame identity check failed")
    follow["id"] = follow["id"].astype(str)
    follow["followup_idx"] = pd.to_datetime(follow["idx"], errors="coerce").dt.normalize()
    follow["censor_date"] = pd.to_datetime(
        follow["censor_date"], errors="coerce"
    ).dt.normalize()
    follow = follow[["id", "followup_idx", "censor_date"]]

    cpap = _read_parquet_columns(
        CPAP, ["id", "setup_date", "adherence_flag", "mean_pap_usage"]
    )
    if cpap["id"].isna().any() or cpap["id"].duplicated().any():
        raise RuntimeError("CPAP extract identity check failed")
    cpap["id"] = cpap["id"].astype(str)
    cpap["setup_date"] = pd.to_datetime(cpap["setup_date"], errors="coerce").dt.normalize()
    adherence = pd.to_numeric(cpap["adherence_flag"], errors="coerce")
    observed_codes = set(adherence.dropna().unique().tolist())
    if observed_codes != {0.0, 1.0}:
        raise RuntimeError("adherence_flag does not contain exactly the documented 0/1 codes")
    cpap["adherence_flag"] = adherence
    cpap["mean_pap_usage"] = pd.to_numeric(cpap["mean_pap_usage"], errors="coerce")

    joined = ph.merge(aux, on="MRN", how="left", validate="one_to_one")
    if joined["idx"].isna().any():
        raise RuntimeError("not every phenotype member has a sleep-study index")
    joined = joined.merge(
        follow, left_on="MRN", right_on="id", how="left", validate="one_to_one"
    ).drop(columns=["id"])
    overlap = joined["followup_idx"].notna()
    followup_overlap = int(overlap.sum())
    followup_index_matches = int(
        (joined.loc[overlap, "idx"] == joined.loc[overlap, "followup_idx"]).sum()
    )
    followup_index_mismatches = followup_overlap - followup_index_matches
    joined = joined.drop(columns=["followup_idx"])
    joined = joined.merge(
        cpap, left_on="MRN", right_on="id", how="left", validate="one_to_one",
        indicator="_cpap_merge",
    ).drop(columns=["id"])
    joined["cpap_record_present"] = joined["_cpap_merge"].eq("both")
    joined = joined.drop(columns=["_cpap_merge"])

    rows: list[dict] = []
    for group in GROUPS:
        group_frame = joined if group == OVERALL else joined[joined["octant"].eq(group)]
        n_group = EXPECTED_COHORT_N if group == OVERALL else EXPECTED_OCTANT_N[group]
        if len(group_frame) != n_group:
            raise RuntimeError("aggregate group size drifted during the CPAP join")
        for window in WINDOWS:
            if window["key"] == "before_or_on_index":
                eligible = group_frame
                starter = eligible["setup_date"].notna() & (
                    eligible["setup_date"] <= eligible["idx"]
                )
            elif window["key"] == "ever_after_index":
                observable = (group_frame["censor_date"].notna()
                              & (group_frame["censor_date"] > group_frame["idx"]))
                eligible = group_frame.loc[observable]
                starter = (eligible["setup_date"].notna()
                           & (eligible["setup_date"] > eligible["idx"])
                           & (eligible["setup_date"] <= eligible["censor_date"]))
            else:
                boundary = group_frame["idx"] + pd.to_timedelta(window["days"], unit="D")
                observable = (group_frame["censor_date"].notna()
                              & (group_frame["censor_date"] >= boundary))
                eligible = group_frame.loc[observable]
                eligible_boundary = eligible["idx"] + pd.to_timedelta(
                    window["days"], unit="D"
                )
                starter = (eligible["setup_date"].notna()
                           & (eligible["setup_date"] > eligible["idx"])
                           & (eligible["setup_date"] <= eligible_boundary))

            starters = eligible.loc[starter]
            n_observable = int(len(eligible))
            n_record = int(eligible["cpap_record_present"].sum())
            n_setup = int(len(starters))
            n_adherence = int(starters["adherence_flag"].notna().sum())
            n_adherence_missing = n_setup - n_adherence
            n_usage = int(starters["mean_pap_usage"].notna().sum())
            rows.append({
                "octant": group,
                "n_octant": n_group,
                "initiation_window_days": window["days"],
                "n_observable_for_window": n_observable,
                "n_cpap_record_present": n_record,
                "cpap_record_coverage_pct": _pct(n_record, n_observable),
                "n_documented_setup": n_setup,
                "documented_setup_pct": _pct(n_setup, n_observable),
                "initiation_label": window["label"],
                "n_started_with_adherence_flag": n_adherence,
                "n_adherent": pd.NA,
                "adherent_pct_among_evaluable_starters": np.nan,
                "adherence_flag_missing": n_adherence_missing,
                "adherence_coverage_pct": _pct(n_adherence, n_setup),
                "n_started_with_usage": n_usage,
                "usage_coverage_pct": _pct(n_usage, n_setup),
                "mean_usage_hours_night": np.nan,
                "sd_usage_hours_night": np.nan,
                "median_usage_hours_night": np.nan,
                "q1_usage_hours_night": np.nan,
                "q3_usage_hours_night": np.nan,
                "usage_window_definition": pd.NA,
                "usage_unit_source": USAGE_UNIT_SOURCE,
                "usage_unit_confirmed": True,
                "adherence_definition": pd.NA,
                "adherence_definition_confirmed": False,
                "suppressed": False,
                "suppression_reason": pd.NA,
                "_window_key": window["key"],
            })

    table = pd.DataFrame(rows)
    del ph, aux, follow, cpap, joined, group_frame, eligible, starters
    if len(table) != len(GROUPS) * len(WINDOWS):
        raise RuntimeError("aggregate CPAP table has incomplete group/window coverage")
    reconciliation = {
        "phenotype_rows": EXPECTED_COHORT_N,
        "octant_sizes": EXPECTED_OCTANT_N,
        "octant_sum": sum(EXPECTED_OCTANT_N.values()),
        "group_count": len(GROUPS),
        "octant_count": len(ORDER),
        "window_count": len(WINDOWS),
        "rows": len(table),
        "all_members_have_sleep_study_index": True,
        "members_with_existing_positive_followup_boundary": followup_overlap,
        "existing_followup_indices_all_match": followup_index_mismatches == 0,
        "phenotype_sleep_study_index_used_for_all_windows": True,
        "adherence_flag_observed_values": sorted(int(value) for value in observed_codes),
    }
    return table, reconciliation


def _row_for(raw: pd.DataFrame, group: str, window_key: str) -> int:
    if "_window_key" in raw.columns:
        window_match = raw["_window_key"].eq(window_key)
    else:
        window_match = raw["initiation_label"].eq(WINDOW_BY_KEY[window_key]["label"])
    matches = raw.index[
        raw["octant"].eq(group) & window_match
    ].tolist()
    if len(matches) != 1:
        raise RuntimeError("aggregate row key is not unique")
    return int(matches[0])


def _implicit_complements(raw: pd.DataFrame) -> list[dict]:
    """Return exact, nonnegative count complements implied by serialized counts."""
    active = set(COUNT_COLUMNS)
    complements: list[dict] = []

    def add_complement(
        key,
        name: str,
        coefficients: dict[tuple[int, str], float],
        candidates: list[tuple[int, str]],
        kind: str,
    ) -> None:
        value = 0.0
        for (idx, column), coefficient in coefficients.items():
            observed = raw.loc[idx, column]
            if pd.isna(observed):
                return
            value += coefficient * float(observed)
        if not np.isfinite(value) or not float(value).is_integer():
            raise RuntimeError("an implicit disclosure complement is not integral")
        integer_value = int(value)
        if integer_value < 0:
            raise RuntimeError(
                f"the declared monotone disclosure complement {name!r} is negative"
            )
        complements.append({
            "key": key,
            "name": name,
            "kind": kind,
            "coefficients": coefficients,
            "candidate_keys": [
                candidate for candidate in candidates if candidate[1] in active
            ],
            "value": integer_value,
            "primary": integer_value < DISCLOSURE_MIN,
        })

    for idx in raw.index:
        for name, minuend, subtrahend in IMPLICIT_COMPLEMENT_SPECS:
            if minuend not in active or subtrahend not in active:
                continue
            row_index = int(idx)
            add_complement(
                (row_index, name),
                name,
                {
                    (row_index, minuend): 1.0,
                    (row_index, subtrahend): -1.0,
                },
                [(row_index, subtrahend), (row_index, minuend)],
                "within_row",
            )

    for group in GROUPS:
        structural_idx = _row_for(raw, group, "before_or_on_index")
        structural_total = raw.loc[structural_idx, "n_octant"]
        structural_observable = raw.loc[
            structural_idx, "n_observable_for_window"
        ]
        if (not pd.isna(structural_total) and not pd.isna(structural_observable)
                and int(structural_total) != int(structural_observable)):
            raise RuntimeError(
                "the declared before/on-index structural-zero complement is not zero"
            )

        for column in ("n_observable_for_window", "n_cpap_record_present"):
            for larger_window, smaller_window in NESTED_OBSERVATION_WINDOW_PAIRS:
                larger_idx = _row_for(raw, group, larger_window)
                smaller_idx = _row_for(raw, group, smaller_window)
                name = f"{column}:{larger_window}_minus_{smaller_window}"
                add_complement(
                    (group, name),
                    name,
                    {
                        (larger_idx, column): 1.0,
                        (smaller_idx, column): -1.0,
                    },
                    [(smaller_idx, column), (larger_idx, column)],
                    "adjacent_nested_window",
                )

        for window_key in NONSTRUCTURAL_OBSERVATION_WINDOWS:
            idx = _row_for(raw, group, window_key)
            name = f"n_octant_minus_n_observable:{window_key}"
            add_complement(
                (group, name),
                name,
                {
                    (idx, "n_octant"): 1.0,
                    (idx, "n_observable_for_window"): -1.0,
                },
                [(idx, "n_observable_for_window")],
                "public_total_minus_observable",
            )
    return complements


def _exact_closure(raw: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    """Ensure every released exact equation has zero or at least two hidden cells."""
    hidden = hidden.copy()
    changed = True
    while changed:
        changed = False
        for window_key in WINDOW_KEYS:
            indices = [_row_for(raw, group, window_key) for group in GROUPS]
            for column in COUNT_COLUMNS:
                masked = [idx for idx in indices if bool(hidden.loc[idx, column])]
                if len(masked) != 1:
                    continue
                overall_idx = _row_for(raw, OVERALL, window_key)
                if not bool(hidden.loc[overall_idx, column]):
                    hidden.loc[overall_idx, column] = True
                else:
                    companion = next(
                        idx for idx in indices[:-1] if not bool(hidden.loc[idx, column])
                    )
                    hidden.loc[companion, column] = True
                changed = True
        for idx in raw.index:
            masked = [column for column in ADHERENCE_IDENTITY
                      if bool(hidden.loc[idx, column])]
            if len(masked) != 1:
                continue
            only = masked[0]
            if only == "adherence_flag_missing":
                companion = "n_started_with_adherence_flag"
            elif only == "n_started_with_adherence_flag":
                companion = "n_documented_setup"
            else:
                companion = "n_started_with_adherence_flag"
            hidden.loc[idx, companion] = True
            changed = True
    return hidden


def _relations(
    raw: pd.DataFrame,
    complements: list[dict] | None = None,
    strong: bool = False,
) -> list[tuple[dict[tuple[int, str], float], float, float]]:
    """Linear equations/inequalities available to a reconstruction attacker."""
    relations: list[tuple[dict[tuple[int, str], float], float, float]] = []

    def add(coefficients, lower=-np.inf, upper=np.inf):
        relations.append((coefficients, float(lower), float(upper)))

    for window_key in WINDOW_KEYS:
        overall_idx = _row_for(raw, OVERALL, window_key)
        octant_indices = [_row_for(raw, group, window_key) for group in ORDER]
        for column in COUNT_COLUMNS:
            coeff = {(overall_idx, column): 1.0}
            coeff.update({(idx, column): -1.0 for idx in octant_indices})
            add(coeff, 0.0, 0.0)

    for idx, row in raw.iterrows():
        add({
            (idx, "n_documented_setup"): 1.0,
            (idx, "n_started_with_adherence_flag"): -1.0,
            (idx, "adherence_flag_missing"): -1.0,
        }, 0.0, 0.0)
        n_group = int(row["n_octant"])
        add({(idx, "n_observable_for_window"): 1.0}, upper=n_group)
        subset_pairs = [
            ("n_cpap_record_present", "n_observable_for_window"),
            ("n_documented_setup", "n_observable_for_window"),
            ("n_documented_setup", "n_cpap_record_present"),
            ("n_started_with_adherence_flag", "n_documented_setup"),
            ("adherence_flag_missing", "n_documented_setup"),
            ("n_started_with_usage", "n_documented_setup"),
        ]
        if "n_adherent" in COUNT_COLUMNS:
            subset_pairs.append(("n_adherent", "n_started_with_adherence_flag"))
        for left, right in subset_pairs:
            add({(idx, left): 1.0, (idx, right): -1.0}, upper=0.0)

    nested_observation = ["within_365_days", "within_180_days",
                          "within_90_days", "ever_after_index"]
    for group in GROUPS:
        for column in ("n_observable_for_window", "n_cpap_record_present"):
            for smaller, larger in zip(nested_observation, nested_observation[1:]):
                add({
                    (_row_for(raw, group, smaller), column): 1.0,
                    (_row_for(raw, group, larger), column): -1.0,
                }, upper=0.0)

    if strong:
        for complement in complements or _implicit_complements(raw):
            if complement["primary"]:
                add(
                    complement["coefficients"],
                    lower=0.0,
                    upper=DISCLOSURE_MIN - 1,
                )
            else:
                add(complement["coefficients"], lower=DISCLOSURE_MIN)
    return relations


def _reconstruction_unique(
    raw: pd.DataFrame,
    primary: pd.DataFrame,
    hidden: pd.DataFrame,
    strong: bool,
    complements: list[dict],
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return uniquely recoverable explicit counts and primary implicit complements."""
    keys = [(int(idx), column) for idx in raw.index for column in COUNT_COLUMNS
            if bool(hidden.loc[idx, column])]
    if not keys:
        return [], [item["key"] for item in complements if item["primary"]]
    variable_index = {key: pos for pos, key in enumerate(keys)}
    lower = np.zeros(len(keys), dtype=float)
    upper = np.array([float(raw.loc[idx, "n_octant"]) for idx, _ in keys])
    if strong:
        for pos, (idx, column) in enumerate(keys):
            if bool(primary.loc[idx, column]):
                upper[pos] = min(upper[pos], DISCLOSURE_MIN - 1)
            else:
                lower[pos] = DISCLOSURE_MIN
    if np.any(lower > upper):
        raise RuntimeError("disclosure reconstruction bounds are infeasible")

    matrix_rows: list[np.ndarray] = []
    lower_rows: list[float] = []
    upper_rows: list[float] = []
    for relation_index, (coefficients, rel_lower, rel_upper) in enumerate(
        _relations(raw, complements=complements, strong=strong)
    ):
        row = np.zeros(len(keys), dtype=float)
        visible = 0.0
        for key, coefficient in coefficients.items():
            if key in variable_index:
                row[variable_index[key]] += coefficient
            else:
                visible += coefficient * float(raw.loc[key[0], key[1]])
        target_lower = rel_lower - visible
        target_upper = rel_upper - visible
        if not np.any(row):
            if target_lower > 1e-8 or target_upper < -1e-8:
                raise RuntimeError(
                    f"released aggregate values violate disclosure relation {relation_index}"
                )
            continue
        matrix_rows.append(row)
        lower_rows.append(target_lower)
        upper_rows.append(target_upper)

    constraints = LinearConstraint(
        np.vstack(matrix_rows), np.asarray(lower_rows), np.asarray(upper_rows)
    )
    bounds = Bounds(lower, upper)
    integrality = np.ones(len(keys), dtype=int)

    def solve(objective: np.ndarray) -> np.ndarray:
        result = milp(
            objective, integrality=integrality, bounds=bounds,
            constraints=constraints, options={"disp": False},
        )
        if not result.success or result.x is None:
            raise RuntimeError("disclosure reconstruction model is infeasible")
        integer = np.rint(result.x).astype(np.int64)
        if np.max(np.abs(result.x - integer)) > 1e-6:
            raise RuntimeError("disclosure reconstruction model returned a noninteger value")
        return integer

    unique_counts: list[tuple[int, str]] = []
    for pos, key in enumerate(keys):
        objective = np.zeros(len(keys), dtype=float)
        objective[pos] = 1.0
        minimum = int(solve(objective)[pos])
        maximum = int(solve(-objective)[pos])
        if minimum >= maximum:
            unique_counts.append(key)

    unique_complements: list[tuple[int, str]] = []
    for complement in complements:
        if not complement["primary"]:
            continue
        objective = np.zeros(len(keys), dtype=float)
        visible = 0.0
        for key, coefficient in complement["coefficients"].items():
            if key in variable_index:
                objective[variable_index[key]] += coefficient
            else:
                visible += coefficient * float(raw.loc[key[0], key[1]])
        if not np.any(objective):
            unique_complements.append(complement["key"])
            continue
        minimum = int(round(visible + objective @ solve(objective)))
        maximum = int(round(visible + objective @ solve(-objective)))
        if minimum >= maximum:
            unique_complements.append(complement["key"])
    return unique_counts, unique_complements


def _neighbors(raw: pd.DataFrame, key: tuple[int, str]) -> list[tuple[int, str]]:
    idx, column = key
    group = str(raw.loc[idx, "octant"])
    window_key = str(raw.loc[idx, "_window_key"])
    candidates: list[tuple[int, str]] = []

    overall_idx = _row_for(raw, OVERALL, window_key)
    if idx != overall_idx:
        candidates.append((overall_idx, column))
    for candidate_group in ORDER:
        candidate_idx = _row_for(raw, candidate_group, window_key)
        if candidate_idx != idx:
            candidates.append((candidate_idx, column))

    if column in ADHERENCE_IDENTITY:
        candidates.extend((idx, other) for other in ADHERENCE_IDENTITY if other != column)
    row_links = {
        "n_observable_for_window": ["n_cpap_record_present", "n_documented_setup"],
        "n_cpap_record_present": ["n_observable_for_window", "n_documented_setup"],
        "n_documented_setup": [
            "n_observable_for_window", "n_cpap_record_present",
            "n_started_with_adherence_flag", "adherence_flag_missing",
            "n_started_with_usage",
        ],
        "n_started_with_adherence_flag": ["n_documented_setup", "n_adherent"],
        "adherence_flag_missing": ["n_documented_setup"],
        "n_started_with_usage": ["n_documented_setup"],
        "n_adherent": ["n_started_with_adherence_flag"],
    }
    candidates.extend(
        (idx, other) for other in row_links[column] if other in COUNT_COLUMNS
    )
    if column in ("n_observable_for_window", "n_cpap_record_present"):
        for other_window in WINDOW_KEYS:
            if other_window != window_key and other_window != "before_or_on_index":
                candidates.append((_row_for(raw, group, other_window), column))

    seen = set()
    return [
        item for item in candidates
        if item[1] in COUNT_COLUMNS and not (item in seen or seen.add(item))
    ]


def _complement_neighbors(raw: pd.DataFrame, complement: dict) -> list[tuple[int, str]]:
    """Return deterministic secondary-suppression candidates for a complement."""
    candidates = list(complement["candidate_keys"])
    for candidate in complement["candidate_keys"]:
        candidates.extend(_neighbors(raw, candidate))
    seen = set()
    return [item for item in candidates if not (item in seen or seen.add(item))]


def _suppression(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    primary = raw[COUNT_COLUMNS].lt(DISCLOSURE_MIN)
    complements = _implicit_complements(raw)
    primary_complements = [item for item in complements if item["primary"]]

    # A primary implicit complement is already exposed if both source counts remain
    # visible.  Seed suppression with its subset count, then let exact closure and both
    # attacker models select any additional complementary cells that are required.
    hidden = primary.copy()
    for complement in primary_complements:
        if not complement["candidate_keys"]:
            raise RuntimeError("a primary implicit complement has no suppressible dependency")
        seed_idx, seed_column = complement["candidate_keys"][0]
        hidden.loc[seed_idx, seed_column] = True
    hidden = _exact_closure(raw, hidden)

    complement_by_key = {item["key"]: item for item in complements}
    max_iterations = len(raw) * len(COUNT_COLUMNS) + len(complements)
    for _ in range(max_iterations):
        fresh_counts, fresh_complements = _reconstruction_unique(
            raw, primary, hidden, strong=False, complements=complements
        )
        strong_counts, strong_complements = _reconstruction_unique(
            raw, primary, hidden, strong=True, complements=complements
        )
        unique_counts = list(dict.fromkeys(fresh_counts + strong_counts))
        unique_complements = list(dict.fromkeys(
            fresh_complements + strong_complements
        ))
        if not unique_counts and not unique_complements:
            break
        changed = False
        for key in unique_counts:
            for candidate in _neighbors(raw, key):
                if not bool(hidden.loc[candidate[0], candidate[1]]):
                    hidden.loc[candidate[0], candidate[1]] = True
                    changed = True
                    break
        for key in unique_complements:
            for candidate in _complement_neighbors(raw, complement_by_key[key]):
                if not bool(hidden.loc[candidate[0], candidate[1]]):
                    hidden.loc[candidate[0], candidate[1]] = True
                    changed = True
                    break
        if not changed:
            raise RuntimeError("suppressed count remains uniquely reconstructible")
        hidden = _exact_closure(raw, hidden)
    else:
        raise RuntimeError("complementary suppression did not converge")

    fresh_counts, fresh_complements = _reconstruction_unique(
        raw, primary, hidden, strong=False, complements=complements
    )
    strong_counts, strong_complements = _reconstruction_unique(
        raw, primary, hidden, strong=True, complements=complements
    )
    if (fresh_counts or strong_counts or fresh_complements or strong_complements
            or bool((primary & ~hidden).to_numpy().any())):
        raise RuntimeError("suppression reconstruction audit failed closed")

    released = raw.copy()
    for column in COUNT_COLUMNS:
        released.loc[hidden[column], column] = pd.NA
    derived_suppressed = 0
    for derived, dependencies in DERIVED_DEPENDENCIES.items():
        dependency_hidden = hidden[dependencies].any(axis=1)
        derived_suppressed += int(dependency_hidden.sum())
        released.loc[dependency_hidden, derived] = np.nan
    released["suppressed"] = hidden.any(axis=1)
    released["suppression_reason"] = np.where(
        released["suppressed"], GENERIC_SUPPRESSION_REASON, pd.NA
    )
    for column in SCIENTIFIC_NULL_COLUMNS:
        released[column] = pd.NA if column in (
            "n_adherent", "usage_window_definition", "adherence_definition"
        ) else np.nan

    for column in ["n_octant", "initiation_window_days", *COUNT_COLUMNS, "n_adherent"]:
        released[column] = released[column].astype("Int64")
    stats = {
        "primary_count_cells": int(primary.to_numpy().sum()),
        "primary_implicit_complement_counts": len(primary_complements),
        "protected_implicit_complement_counts": len(complements),
        "protected_adjacent_window_complements": sum(
            item["kind"] == "adjacent_nested_window" for item in complements
        ),
        "complementary_count_cells": int((hidden & ~primary).to_numpy().sum()),
        "suppressed_count_cells": int(hidden.to_numpy().sum()),
        "suppressed_rows": int(hidden.any(axis=1).sum()),
        "derived_cells_suppressed": derived_suppressed,
        "fresh_serialized_attacker_view": "PASS",
        "strong_primary_secondary_bound_view": "PASS",
        "all_suppressed_integer_counts_non_unique": True,
        "all_primary_implicit_complements_non_unique": True,
        "structural_zero_complement_count": len(GROUPS),
        "structural_zero_complement_rule": STRUCTURAL_ZERO_COMPLEMENT_RULE,
        "non_nested_initiation_series_excluded_from_increment_audit": (
            NON_NESTED_INITIATION_SERIES
        ),
        "non_nested_initiation_increment_rule": NON_NESTED_INITIATION_INCREMENT_RULE,
    }
    return released, hidden, stats


def _csv_bytes(table: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    table.to_csv(
        stream, index=False, columns=SCHEMA, na_rep="", lineterminator="\n",
        float_format=f"%.{ROUND_DIGITS}f",
    )
    return stream.getvalue().encode("utf-8")


def _parse_bool(series: pd.Series) -> pd.Series:
    parsed = series.astype(str).str.strip().str.casefold().map({"true": True, "false": False})
    if parsed.isna().any():
        raise RuntimeError("serialized boolean field is invalid")
    return parsed.astype(bool)


def _validate_csv_payload(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload))
    if frame.columns.tolist() != SCHEMA or len(frame) != len(GROUPS) * len(WINDOWS):
        raise RuntimeError("staged CPAP CSV schema or row count is invalid")
    if frame.duplicated(["octant", "initiation_label"]).any():
        raise RuntimeError("staged CPAP CSV contains duplicate logical keys")
    if set(frame["octant"]) != set(GROUPS):
        raise RuntimeError("staged CPAP CSV group coverage is invalid")
    expected_labels = {item["label"] for item in WINDOWS}
    if any(set(group["initiation_label"]) != expected_labels
           for _, group in frame.groupby("octant")):
        raise RuntimeError("staged CPAP CSV window coverage is invalid")

    flags = _parse_bool(frame["suppressed"])
    unit_confirmed = _parse_bool(frame["usage_unit_confirmed"])
    adherence_confirmed = _parse_bool(frame["adherence_definition_confirmed"])
    if (not unit_confirmed.all() or adherence_confirmed.any()
            or not frame["usage_unit_source"].eq(USAGE_UNIT_SOURCE).all()):
        raise RuntimeError("serialized source-definition confirmations are invalid")
    count_null = frame[COUNT_COLUMNS].isna().any(axis=1)
    if not flags.equals(count_null):
        raise RuntimeError("serialized suppression flags disagree with hidden counts")
    if not frame.loc[flags, "suppression_reason"].eq(GENERIC_SUPPRESSION_REASON).all():
        raise RuntimeError("suppressed rows do not use the generic reason")
    if frame.loc[~flags, "suppression_reason"].notna().any():
        raise RuntimeError("an unsuppressed row has a suppression reason")

    for column in ["n_octant", *COUNT_COLUMNS]:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if ((values < DISCLOSURE_MIN).any() or (values % 1 != 0).any()):
            raise RuntimeError("serialized CSV releases an exact count below threshold")
    visible_complements = _implicit_complements(frame)
    if any(item["primary"] for item in visible_complements):
        raise RuntimeError(
            "serialized CSV reconstructs an implicit exact count below threshold"
        )
    if frame[SCIENTIFIC_NULL_COLUMNS].notna().to_numpy().any():
        raise RuntimeError("an estimate with an unresolved source definition was released")

    for _, row in frame.iterrows():
        for derived, dependencies in DERIVED_DEPENDENCIES.items():
            values = [row[column] for column in dependencies]
            if any(pd.isna(value) for value in values):
                if not pd.isna(row[derived]):
                    raise RuntimeError("derived percentage exposes a hidden count")
                continue
            numerator, denominator = float(values[0]), float(values[1])
            expected = np.nan if denominator <= 0 else round(
                100.0 * numerator / denominator, ROUND_DIGITS
            )
            if pd.isna(expected):
                if not pd.isna(row[derived]):
                    raise RuntimeError("undefined percentage was serialized")
            elif pd.isna(row[derived]) or abs(float(row[derived]) - expected) > 5.1e-7:
                raise RuntimeError("serialized percentage is not redundant with released counts")

    expected_n = {**EXPECTED_OCTANT_N, OVERALL: EXPECTED_COHORT_N}
    for group, expected in expected_n.items():
        values = pd.to_numeric(frame.loc[frame["octant"].eq(group), "n_octant"])
        if not values.eq(expected).all():
            raise RuntimeError("serialized octant denominator does not reconcile")
    return frame


def _coverage_summary(frame: pd.DataFrame) -> list[dict]:
    columns = [
        "octant", "initiation_label", "n_observable_for_window",
        "n_cpap_record_present", "cpap_record_coverage_pct",
        "n_documented_setup", "documented_setup_pct",
        "n_started_with_adherence_flag", "adherence_flag_missing",
        "adherence_coverage_pct", "n_started_with_usage", "usage_coverage_pct",
        "suppressed",
    ]
    safe = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    return safe.to_dict(orient="records")


def _measure_approval() -> dict:
    return {
        "n_octant": {
            "fields": ["n_octant"],
            "approved_for_publication": True,
            "reason": "Established public octant denominator; not a CPAP treatment measure.",
        },
        "observable_denominators": {
            "fields": ["n_observable_for_window"],
            "approved_for_publication": False,
            "reason": (
                "The available diagnosis/insurance censor is not confirmed as the PAP "
                "data-capture boundary."
            ),
        },
        "cpap_record_coverage": {
            "fields": ["n_cpap_record_present", "cpap_record_coverage_pct"],
            "approved_for_publication": False,
            "reason": "An absent extract row cannot be classified as no initiation or no capture.",
        },
        "documented_setup": {
            "fields": ["n_documented_setup", "documented_setup_pct"],
            "approved_for_publication": False,
            "reason": (
                "The extract confirms only the earliest documented source setup value, "
                "not earliest lifetime PAP initiation or complete capture; no governance "
                "approval was found for the neutral documented-signal label."
            ),
        },
        "adherence_data_coverage": {
            "fields": [
                "n_started_with_adherence_flag", "adherence_flag_missing",
                "adherence_coverage_pct",
            ],
            "approved_for_publication": False,
            "reason": "Coverage is descriptive, but the enclosing treatment export fails its definition gate.",
        },
        "adherence_outcomes": {
            "fields": [
                "n_adherent", "adherent_pct_among_evaluable_starters",
                "adherence_definition", "adherence_definition_confirmed",
            ],
            "approved_for_publication": False,
            "released": False,
            "reason": (
                "The CMS criterion is authoritative, but the source field's exact "
                "implementation and numeric 0/1 mapping are not confirmed."
            ),
        },
        "usage_data_coverage": {
            "fields": ["n_started_with_usage", "usage_coverage_pct"],
            "approved_for_publication": False,
            "reason": "Coverage is descriptive, but the enclosing treatment export fails its definition gate.",
        },
        "usage_distributions": {
            "fields": [
                *USAGE_DISTRIBUTION_COLUMNS, "usage_window_definition",
                "usage_unit_source", "usage_unit_confirmed",
            ],
            "approved_for_publication": False,
            "released": False,
            "reason": (
                "Source metadata confirms an average PAP-usage quantity in minutes, but "
                "not its averaging denominator, missing-night handling, or setup-based "
                "first-90-day window."
            ),
        },
    }


def _audit_bytes(
    serialized: pd.DataFrame,
    reconciliation: dict,
    suppression: dict,
) -> tuple[bytes, dict]:
    confirmations = {
        "setup_date_is_minimum_documented_source_value_per_person": True,
        "setup_date_is_earliest_lifetime_pap_setup": False,
        "absent_cpap_row_means_no_pap_initiation": False,
        "absent_cpap_row_means_pap_not_captured": False,
        "protocol_specified_initiation_horizon_found": False,
        "pap_capture_observation_end_confirmed": False,
        "existing_followup_index_matches_all_phenotype_indices": (
            reconciliation["existing_followup_indices_all_match"]
        ),
        "adherence_flag_coding_confirmed_by_project_dictionary": True,
        "adherence_flag_observed_codes_match_dictionary": True,
        "adherence_flag_numeric_coding_authoritatively_confirmed": False,
        "adherence_source_label_cms_confirmed": True,
        "intended_cms_criterion_confirmed": True,
        "adherence_definition_confirmed": False,
        "adherence_exact_cms_implementation_confirmed": False,
        "mean_pap_usage_first_90_days_after_setup_confirmed": False,
        "mean_pap_usage_quantity_and_unit_confirmed": True,
        "mean_pap_usage_averaging_denominator_confirmed": False,
        "mean_pap_usage_missing_night_handling_confirmed": False,
        "signed_or_ticketed_governance_approval_found": False,
    }
    unresolved = [
        {
            "field": "setup_date earliest-lifetime meaning",
            "reason": (
                "The deterministic extract takes the minimum documented source value; "
                "no upstream specification establishes complete lifetime setup history."
            ),
        },
        {
            "field": "absent CPAP extract row",
            "reason": (
                "Absence means no retained setup/usage signal under the extract rule and "
                "cannot distinguish no initiation from noncapture."
            ),
        },
        {
            "field": "protocol initiation horizon",
            "reason": "No protocol-designated treatment-initiation horizon was found.",
        },
        {
            "field": "PAP-specific observation end",
            "reason": (
                "The available censor combines diagnosis follow-up, insurance coverage, "
                "and death; it is not confirmed as the PAP data-capture end."
            ),
        },
        {
            "field": "adherence authoritative definition",
            "reason": (
                "CMS policy and the source label are documented, but the source field's "
                "exact algorithm and numeric 0/1 mapping are not authoritatively specified."
            ),
        },
        {
            "field": "mean_pap_usage window",
            "reason": (
                "Source metadata assigns a first-90-day period to total_days_usage, not "
                "explicitly to mean_pap_usage or to a setup-date anchor."
            ),
        },
        {
            "field": "mean_pap_usage averaging denominator and missing-night handling",
            "reason": (
                "Source metadata confirms an average quantity in minutes but does not "
                "state whether the denominator is all nights, recorded nights, usage "
                "days, or another set, or how nights without recordings are handled."
            ),
        },
        {
            "field": "CPAP-specific data-governance approval",
            "reason": (
                "No signed or ticketed approval record was found for the public audience, "
                "neutral documented-setup label, primary horizon, source-coded adherence "
                "label, usage definition, or final measure set."
            ),
        },
    ]
    if not reconciliation["existing_followup_indices_all_match"]:
        unresolved.append({
            "field": "follow-up-frame index alignment",
            "reason": (
                "The follow-up frame was built from a separate sleep-study spine and "
                "does not match every phenotype sleep-study index; phenotype index is "
                "retained and only the absolute censor boundary is reused."
            ),
        })
    audit = {
        "confirmation_sources": {
            "source_metadata_workbook": {
                "file_name": SOURCE_METADATA_FILE,
                "sha256": SOURCE_METADATA_SHA256,
                "worksheet": SOURCE_METADATA_SHEET,
                "range": SOURCE_METADATA_RANGE,
                "confirmed_scope": (
                    "total_days_usage has a first-90-day source label; mean_pap_usage "
                    "is a source-reported average quantity in minutes; percentage_over_4h "
                    "is a percent of days with at least four hours; adherence_flag carries "
                    "the source label CMS adherence and a Y/N format."
                ),
                "patient_level_rows_accessed": False,
            },
            "cms_policy": {
                "document_id": CMS_LCD_ID,
                "title": (
                    "Positive Airway Pressure Devices for the Treatment of "
                    "Obstructive Sleep Apnea"
                ),
                "url": CMS_LCD_URL,
                "verbatim_excerpt": CMS_DEFINITION_VERBATIM_EXCERPT,
                "definition_paraphrase": CMS_DEFINITION_PARAPHRASE,
                "source_field_equivalence_confirmed": False,
            },
            "governance_record_search": {
                "required_decision_record": "website/OSA_CELL_DISCLOSURE_DECISIONS.md",
                "status_record": "website/OSA_CELL_RESOURCE_STATUS.md",
                "signed_or_ticketed_cpap_approval_found": False,
            },
        },
        "cohort_and_octant_reconciliation": reconciliation,
        "initiation_denominator_and_time_windows": {
            "primary_horizon_selected": False,
            "windows": [
                {
                    "initiation_label": item["label"],
                    "initiation_window_days": item["days"],
                }
                for item in WINDOWS
            ],
            "before_or_on_index_denominator": (
                "All members of the established phenotype group with a documented "
                "sleep-study index."
            ),
            "fixed_window_denominator": (
                "Members whose existing diagnosis/insurance/death censor remains on or "
                "after the full index-plus-window boundary."
            ),
            "ever_after_index_denominator": (
                "Members with positive existing follow-up; setup must be after index and "
                "on or before that censor."
            ),
            "fixed_window_percentages_require_full_window_observation": True,
            "pap_capture_observation_end_confirmed": False,
        },
        "missing_row_interpretation": {
            "cpap_record_present_definition": (
                "A retained row in the deterministic CPAP extract, which requires at "
                "least one documented setup or usage signal under its retention rule."
            ),
            "absent_row_interpretation": (
                "No retained extract signal; no-initiation versus noncapture is unresolved."
            ),
            "no_pap_initiation_confirmed": False,
            "pap_data_not_captured_confirmed": False,
        },
        "setup_date": {
            "operational_definition": (
                "Minimum parseable documented setup value per person in the existing extract."
            ),
            "earliest_documented_source_value_confirmed": True,
            "earliest_lifetime_pap_setup_confirmed": False,
            "approved_public_label_if_authorized": "documented PAP setup signal",
            "neutral_label_governance_approved": False,
        },
        "adherence_flag": {
            "source_field": "adherence_flag",
            "coding": "1 = source-coded adherent; 0 = source-coded nonadherent",
            "coding_source": "Project data dictionary",
            "coding_source_statement": "1 = adherent; 0 = nonadherent",
            "coding_internally_documented": True,
            "coding_authoritatively_confirmed": False,
            "observed_codes": reconciliation["adherence_flag_observed_values"],
            "observed_codes_match_documented_coding": True,
            "source_metadata_label": "CMS adherence",
            "source_metadata_format": "$YN.",
            "source_metadata_label_confirmed": True,
            "intended_cms_definition_verbatim_excerpt": (
                CMS_DEFINITION_VERBATIM_EXCERPT
            ),
            "intended_cms_definition_paraphrase": CMS_DEFINITION_PARAPHRASE,
            "intended_cms_definition_source": CMS_LCD_URL,
            "authoritative_definition_verbatim": None,
            "authoritative_definition_confirmed": False,
            "exact_cms_implementation_confirmed": False,
            "denominator_definition": (
                "Documented starters with nonmissing adherence_flag."
            ),
            "missing_flag_treatment": (
                "Reported separately and never classified as nonadherent."
            ),
            "outcome_estimates_released": False,
        },
        "mean_pap_usage": {
            "first_90_days_after_setup_confirmed": False,
            "quantity_type": "source-reported average PAP-usage duration",
            "unit": "minutes",
            "unit_confirmed": True,
            "source_label": "Average minutes for PAP usage",
            "averaging_denominator": None,
            "averaging_denominator_confirmed": False,
            "nights_without_recorded_usage_handling": None,
            "nights_without_recorded_usage_handling_confirmed": False,
            "source_block_first_90_day_label_applies_explicitly_to": (
                "total_days_usage"
            ),
            "converted_to_hours_per_night": False,
            "denominator_definition": (
                "Documented starters with nonmissing mean_pap_usage."
            ),
            "coverage_present_in_candidate_export": True,
            "coverage_approved_for_publication": False,
            "distribution_estimates_released": False,
        },
        "source_definition_confirmations": confirmations,
        "unresolved_fields": unresolved,
        "missingness_and_coverage_by_octant": _coverage_summary(serialized),
        "disclosure": {
            "threshold": DISCLOSURE_MIN,
            "primary_suppression_rule": (
                "Every exact count below 11, including zero, is null."
            ),
            "complementary_suppression_rule": (
                "Additional cells are nulled until partition, within-row identity, "
                "nested-window, and bound-aware integer audits cannot uniquely recover "
                "any hidden count."
            ),
            **suppression,
            "suppressed_values_cannot_be_reconstructed": True,
            "browser_delivered_values_contain_no_hidden_counts": True,
            "status": "PASS",
        },
        "measure_publication_approval": _measure_approval(),
        "release_approved": False,
        "release_approval_reasons": [item["reason"] for item in unresolved],
    }
    payload = (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return payload, audit


def _json_schema(audit: dict) -> dict:
    property_types = {
        "confirmation_sources": "object",
        "cohort_and_octant_reconciliation": "object",
        "initiation_denominator_and_time_windows": "object",
        "missing_row_interpretation": "object",
        "setup_date": "object",
        "adherence_flag": "object",
        "mean_pap_usage": "object",
        "source_definition_confirmations": "object",
        "unresolved_fields": "array",
        "missingness_and_coverage_by_octant": "array",
        "disclosure": "object",
        "measure_publication_approval": "object",
        "release_approved": "boolean",
        "release_approval_reasons": "array",
    }
    return {
        "type": "object",
        "required": list(audit.keys()),
        "properties": {key: {"type": property_types[key]} for key in audit},
        "additionalProperties": False,
    }


def _validate_audit_payload(payload: bytes, suppression: dict) -> dict:
    audit = json.loads(payload.decode("utf-8"))
    disclosure = audit.get("disclosure", {})
    confirmations = audit.get("source_definition_confirmations", {})
    if (audit.get("release_approved") is not False
            or disclosure.get("status") != "PASS"
            or disclosure.get("suppressed_count_cells") != suppression["suppressed_count_cells"]
            or disclosure.get("suppressed_rows") != suppression["suppressed_rows"]
            or disclosure.get("primary_implicit_complement_counts")
            != suppression["primary_implicit_complement_counts"]
            or disclosure.get("protected_implicit_complement_counts")
            != suppression["protected_implicit_complement_counts"]
            or disclosure.get("protected_adjacent_window_complements")
            != len(GROUPS) * 2 * len(NESTED_OBSERVATION_WINDOW_PAIRS)
            or disclosure.get("all_primary_implicit_complements_non_unique") is not True
            or disclosure.get("structural_zero_complement_count") != len(GROUPS)
            or disclosure.get("structural_zero_complement_rule")
            != STRUCTURAL_ZERO_COMPLEMENT_RULE
            or disclosure.get(
                "non_nested_initiation_series_excluded_from_increment_audit"
            ) != NON_NESTED_INITIATION_SERIES
            or disclosure.get("non_nested_initiation_increment_rule")
            != NON_NESTED_INITIATION_INCREMENT_RULE
            or disclosure.get("suppressed_values_cannot_be_reconstructed") is not True
            or disclosure.get("browser_delivered_values_contain_no_hidden_counts") is not True
            or confirmations.get("adherence_definition_confirmed") is not False
            or confirmations.get(
                "adherence_flag_coding_confirmed_by_project_dictionary"
            ) is not True
            or audit.get("adherence_flag", {}).get("observed_codes") != [0, 1]
            or confirmations.get("mean_pap_usage_quantity_and_unit_confirmed") is not True
            or confirmations.get(
                "mean_pap_usage_averaging_denominator_confirmed"
            ) is not False
            or confirmations.get(
                "mean_pap_usage_missing_night_handling_confirmed"
            ) is not False
            or confirmations.get("adherence_source_label_cms_confirmed") is not True
            or confirmations.get("intended_cms_criterion_confirmed") is not True
            or confirmations.get(
                "adherence_flag_numeric_coding_authoritatively_confirmed"
            ) is not False
            or confirmations.get("signed_or_ticketed_governance_approval_found") is not False
            or audit.get("mean_pap_usage", {}).get("unit_confirmed") is not True
            or audit.get("adherence_flag", {}).get(
                "exact_cms_implementation_confirmed"
            ) is not False
            or len(audit.get("missingness_and_coverage_by_octant", [])) != 45
            or not audit.get("unresolved_fields")):
        raise RuntimeError("staged CPAP audit payload failed validation")
    return audit


def _manifest_bytes(
    original: dict,
    csv_payload: bytes,
    audit_payload: bytes,
    audit: dict,
    suppression: dict,
) -> bytes:
    manifest = copy.deepcopy(original)
    csv_entry = {
        "name": CSV_NAME,
        "bytes": len(csv_payload),
        "sha256": _sha256_bytes(csv_payload),
        "internal": False,
        "rows": 45,
        "schema": SCHEMA,
        "disclosure_audit_status": "PASS",
        "public_website_release_approved": False,
    }
    audit_entry = {
        "name": AUDIT_NAME,
        "bytes": len(audit_payload),
        "sha256": _sha256_bytes(audit_payload),
        "internal": False,
        "records": 1,
        "rows": 1,
        "schema": _json_schema(audit),
        "disclosure_audit_status": "PASS",
        "public_website_release_approved": False,
    }
    retained = [entry for entry in manifest["files"] if entry.get("name") not in TARGET_NAMES]
    manifest["files"] = sorted(retained + [csv_entry, audit_entry], key=lambda x: x["name"])

    initiation_windows = [
        {"initiation_label": item["label"], "initiation_window_days": item["days"]}
        for item in WINDOWS
    ]
    manifest["cpap_treatment_summary"] = {
        "exporter": "analyses/incwas/31_octant_cpap_summary.py",
        "shared_phenotype_cohort": EXPECTED_COHORT_N,
        "groups": GROUPS,
        "octants": len(ORDER),
        "initiation_windows": initiation_windows,
        "window_count": len(WINDOWS),
        "rows": 45,
        "files": {"summary": CSV_NAME, "audit": AUDIT_NAME},
        "confirmation_sources": audit["confirmation_sources"],
        "source_definition_confirmations": audit["source_definition_confirmations"],
        "initiation_denominator_definition": audit[
            "initiation_denominator_and_time_windows"
        ],
        "missing_row_interpretation": audit["missing_row_interpretation"],
        "adherence_denominator_definition": audit["adherence_flag"][
            "denominator_definition"
        ],
        "usage_denominator_definition": audit["mean_pap_usage"][
            "denominator_definition"
        ],
        "suppressed_rows": suppression["suppressed_rows"],
        "suppressed_cells": suppression["suppressed_count_cells"],
        "disclosure_threshold": DISCLOSURE_MIN,
        "primary_suppression_rule": audit["disclosure"]["primary_suppression_rule"],
        "complementary_suppression_rule": audit["disclosure"][
            "complementary_suppression_rule"
        ],
        "reconstruction_audit_result": "PASS",
        "browser_delivery_contains_hidden_counts": False,
        "disclosure_audit_status": "PASS",
        "validation": {
            "status": "PASS",
            "expected_rows": 45,
            "rows": 45,
            "expected_groups": 9,
            "groups": 9,
            "expected_windows": 5,
            "windows": 5,
        },
        "measure_publication_approval": audit["measure_publication_approval"],
        "release_approved": False,
        "public_website_release_approved": False,
    }

    disclosure = manifest.setdefault("disclosure", {})
    disclosure.setdefault("suppressed_cells_by_file", {})[CSV_NAME] = suppression[
        "suppressed_count_cells"
    ] + suppression["derived_cells_suppressed"]
    disclosure["suppressed_cells_by_file"][AUDIT_NAME] = (
        suppression["suppressed_count_cells"] + suppression["derived_cells_suppressed"]
    )
    disclosure.setdefault("suppressed_rows_by_file", {})[CSV_NAME] = suppression[
        "suppressed_rows"
    ]
    disclosure["suppressed_rows_by_file"][AUDIT_NAME] = suppression["suppressed_rows"]

    updated_reason = (
        "A disclosure-audited aggregate documented-setup and field-coverage file is "
        "available but not approved for public release: absent rows are ambiguous; "
        "PAP-specific capture end, setup-history completeness, the primary initiation "
        "horizon, exact adherence implementation, usage averaging denominator/window, "
        "missing-night handling, and CPAP-specific governance approval remain unresolved. "
        "Source metadata confirms only an average usage quantity in minutes and the source "
        "label CMS adherence. Adherence outcomes and usage distributions are withheld."
    )
    unavailable = manifest.setdefault("unavailable_or_excluded", [])
    found = False
    for item in unavailable:
        if "cpap" in str(item.get("field", "")).casefold():
            item["reason"] = updated_reason
            found = True
    if not found:
        unavailable.append({"field": "CPAP treatment measures", "reason": updated_reason})
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _validate_manifest_payload(
    payload: bytes,
    original: dict,
    csv_payload: bytes,
    audit_payload: bytes,
    suppression: dict,
) -> None:
    manifest = json.loads(payload.decode("utf-8"))
    files = manifest.get("files", [])
    names = [entry.get("name") for entry in files]
    if len(names) != len(set(names)):
        raise RuntimeError("staged manifest contains duplicate file entries")
    by_name = {entry["name"]: entry for entry in files}
    original_base = {
        entry["name"]: entry for entry in original["files"]
        if entry.get("name") not in TARGET_NAMES
    }
    staged_base = {name: by_name[name] for name in original_base if name in by_name}
    if staged_base != original_base or len(staged_base) != len(original_base):
        raise RuntimeError("staged manifest changed a pre-existing artifact entry")

    csv_entry = by_name.get(CSV_NAME, {})
    audit_entry = by_name.get(AUDIT_NAME, {})
    audit = _validate_audit_payload(audit_payload, suppression)
    if (csv_entry.get("sha256") != _sha256_bytes(csv_payload)
            or csv_entry.get("bytes") != len(csv_payload)
            or csv_entry.get("rows") != 45
            or csv_entry.get("schema") != SCHEMA
            or csv_entry.get("disclosure_audit_status") != "PASS"
            or csv_entry.get("public_website_release_approved") is not False
            or audit_entry.get("sha256") != _sha256_bytes(audit_payload)
            or audit_entry.get("bytes") != len(audit_payload)
            or audit_entry.get("records") != 1
            or audit_entry.get("schema") != _json_schema(audit)
            or audit_entry.get("disclosure_audit_status") != "PASS"
            or audit_entry.get("public_website_release_approved") is not False):
        raise RuntimeError("staged manifest CPAP file entries are invalid")

    block = manifest.get("cpap_treatment_summary", {})
    validation = block.get("validation", {})
    if (block.get("rows") != 45 or block.get("octants") != 8
            or block.get("window_count") != 5
            or block.get("disclosure_threshold") != DISCLOSURE_MIN
            or block.get("reconstruction_audit_result") != "PASS"
            or block.get("confirmation_sources") != audit.get("confirmation_sources")
            or block.get("source_definition_confirmations")
            != audit.get("source_definition_confirmations")
            or block.get("measure_publication_approval")
            != audit.get("measure_publication_approval")
            or block.get("release_approved") is not False
            or block.get("public_website_release_approved") is not False
            or validation.get("status") != "PASS"
            or manifest.get("validation", {}).get("status") != "PASS"):
        raise RuntimeError("staged manifest CPAP block is invalid")
    disclosure = manifest.get("disclosure", {})
    expected_suppressed_cells = (
        suppression["suppressed_count_cells"] + suppression["derived_cells_suppressed"]
    )
    if (disclosure.get("suppressed_cells_by_file", {}).get(CSV_NAME)
            != expected_suppressed_cells
            or disclosure.get("suppressed_cells_by_file", {}).get(AUDIT_NAME)
            != expected_suppressed_cells
            or disclosure.get("suppressed_rows_by_file", {}).get(CSV_NAME)
            != suppression["suppressed_rows"]
            or disclosure.get("suppressed_rows_by_file", {}).get(AUDIT_NAME)
            != suppression["suppressed_rows"]):
        raise RuntimeError("staged manifest CPAP suppression registry is inconsistent")


def _validate_browser_payloads(
    csv_payload: bytes,
    audit_payload: bytes,
    manifest_payload: bytes,
    suppression: dict,
) -> None:
    _validate_csv_payload(csv_payload)
    audit = _validate_audit_payload(audit_payload, suppression)
    manifest = json.loads(manifest_payload.decode("utf-8"))
    targets = {
        entry["name"]: entry for entry in manifest.get("files", [])
        if entry.get("name") in TARGET_NAMES
    }
    scoped = {
        "cpap_treatment_summary": manifest.get("cpap_treatment_summary", {}),
        "files": targets,
    }
    browser_bytes = b"\n".join([
        csv_payload,
        audit_payload,
        json.dumps(scoped, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    ])
    text = browser_bytes.decode("utf-8")
    lowered = text.casefold()
    forbidden_tokens = (
        "mrn", "pat_id", "patient_id", "date_of_birth", "cross_domain_phenotypes",
        "cohort.parquet", "followup_frame", "_identified", "c:\\users\\", "/users/",
    )
    if (any(token in lowered for token in forbidden_tokens)
            or re.search(r"(?<![a-z])[a-z]:[\\/]", text, flags=re.IGNORECASE)
            or re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)):
        raise RuntimeError("browser-delivered CPAP payload contains a forbidden token")


def _verify_protected(expected: dict[str, str]) -> None:
    for name, recorded in expected.items():
        if name in TARGET_NAMES:
            continue
        path = OUT / name
        if not path.is_file() or _sha256_file(path) != recorded:
            raise RuntimeError("a pre-existing website artifact changed during export")


def _stage_payload(destination: Path, payload: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=OUT, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        staged = Path(handle.name)
    if staged.read_bytes() != payload:
        staged.unlink(missing_ok=True)
        raise RuntimeError("staged disclosure artifact failed byte verification")
    return staged


def main() -> None:
    _verify_definition_sources()
    manifest, original_manifest_payload, old_hashes = _load_verified_manifest()
    raw, reconciliation = _aggregate()
    released, _, suppression = _suppression(raw)
    csv_payload = _csv_bytes(released)
    serialized = _validate_csv_payload(csv_payload)
    audit_payload, audit = _audit_bytes(serialized, reconciliation, suppression)
    _validate_audit_payload(audit_payload, suppression)
    manifest_payload = _manifest_bytes(
        manifest, csv_payload, audit_payload, audit, suppression
    )
    _validate_manifest_payload(
        manifest_payload, manifest, csv_payload, audit_payload, suppression
    )
    _validate_browser_payloads(
        csv_payload, audit_payload, manifest_payload, suppression
    )

    if MANIFEST_PATH.read_bytes() != original_manifest_payload:
        raise RuntimeError("manifest changed concurrently during CPAP aggregation")
    _verify_manifest_files(manifest)
    staged_paths: list[Path] = []
    try:
        staged_csv = _stage_payload(CSV_PATH, csv_payload)
        staged_paths.append(staged_csv)
        staged_audit = _stage_payload(AUDIT_PATH, audit_payload)
        staged_paths.append(staged_audit)
        staged_manifest = _stage_payload(MANIFEST_PATH, manifest_payload)
        staged_paths.append(staged_manifest)

        os.replace(staged_csv, CSV_PATH)
        os.replace(staged_audit, AUDIT_PATH)
        _verify_protected(old_hashes)
        if MANIFEST_PATH.read_bytes() != original_manifest_payload:
            raise RuntimeError("manifest changed concurrently before final replacement")
        os.replace(staged_manifest, MANIFEST_PATH)
    finally:
        for staged in staged_paths:
            staged.unlink(missing_ok=True)

    _verify_protected(old_hashes)
    if (CSV_PATH.read_bytes() != csv_payload
            or AUDIT_PATH.read_bytes() != audit_payload
            or MANIFEST_PATH.read_bytes() != manifest_payload
            or _sha256_file(CSV_PATH) != _sha256_bytes(csv_payload)
            or _sha256_file(AUDIT_PATH) != _sha256_bytes(audit_payload)):
        raise RuntimeError("written CPAP disclosure artifacts failed hash verification")
    _validate_manifest_payload(
        MANIFEST_PATH.read_bytes(), manifest, CSV_PATH.read_bytes(),
        AUDIT_PATH.read_bytes(), suppression,
    )
    _validate_browser_payloads(
        CSV_PATH.read_bytes(), AUDIT_PATH.read_bytes(), MANIFEST_PATH.read_bytes(),
        suppression,
    )

    print(json.dumps({
        "aggregate_rows": 45,
        "groups": 9,
        "octants": 8,
        "windows": 5,
        "primary_suppressed_count_cells": suppression["primary_count_cells"],
        "primary_implicit_complement_counts": suppression[
            "primary_implicit_complement_counts"
        ],
        "complementary_suppressed_count_cells": suppression[
            "complementary_count_cells"
        ],
        "suppressed_rows": suppression["suppressed_rows"],
        "disclosure_audit": "PASS",
        "release_approved": False,
    }, indent=2))


if __name__ == "__main__":
    main()
