"""Disclosure-controlled aggregate race/ethnicity export for phenotype octants.

The patient-level join exists only in memory, is reduced immediately to a fixed
octant-by-category count table, and is never printed, sampled, or written.  Release is
fail-closed: both integer reconstruction audits and all serialization checks must pass
before any output file is touched.
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
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


PROJECT = Path(__file__).resolve().parents[2]
PHENOTYPES = (PROJECT / "latent-class-analysis" / "results" / "phenotypes"
              / "cross_domain_phenotypes.csv")
SPINE = (PROJECT / "results" / "icd_analysis_v2_results" / "cohort"
         / "cohort.parquet")
OUT = (PROJECT / "results" / "incwas_results" / "phenotype_exposure"
       / "website_exports")
CSV_NAME = "octant_race_ethnicity_distributions.csv"
AUDIT_NAME = "octant_race_ethnicity_audit.json"
MANIFEST_NAME = "phenotype_website_export_manifest.json"
CSV_PATH = OUT / CSV_NAME
AUDIT_PATH = OUT / AUDIT_NAME
MANIFEST_PATH = OUT / MANIFEST_NAME

DISCLOSURE_MIN = 11
EXPECTED_COHORT_N = 70_880
ROUND_DIGITS = 10
OVERALL = "overall"
EXPECTED_BASE_FILE_COUNT = 12

# Fixed before data are read. Labels are the exact established RACE_ETH source labels.
TAXONOMY = (
    ("white", "White"),
    ("hispanic", "Hispanic"),
    ("asian", "Asian"),
    ("black", "Black"),
    ("unknown", "Unknown"),
    ("other", "Other"),
    ("pacific_islander", "Pacific Islander"),
    ("multiple", "Multiple"),
    ("native_am_alaskan", "Native Am Alaskan"),
)
CATEGORY_IDS = [x[0] for x in TAXONOMY]
CATEGORY_LABELS = [x[1] for x in TAXONOMY]
ID_TO_LABEL = dict(TAXONOMY)
SOURCE_LOOKUP = {label.casefold(): label for label in CATEGORY_LABELS}

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
GROUPS = [OVERALL] + ORDER

SCHEMA = [
    "octant",
    "category_id",
    "category_label",
    "numerator",
    "denominator",
    "proportion",
    "coverage_pct",
    "cohort_proportion",
    "standardized_difference",
    "suppressed",
    "suppression_reason",
]
MASKED_VALUE_COLUMNS = [
    "numerator", "proportion", "cohort_proportion", "standardized_difference"
]
TARGET_NAMES = {CSV_NAME, AUDIT_NAME}
AUDIT_TOP_LEVEL_KEYS = {
    "taxonomy", "denominator_definition", "coverage_pct_definition",
    "cohort_proportion_definition", "standardized_difference_definition",
    "disclosure_threshold", "primary_suppression_rule", "secondary_suppression_rule",
    "row_count", "expected_group_category_coverage", "number_of_suppressed_cells",
    "reconstruction_audit", "browser_delivery_validation",
    "browser_delivered_values_contain_no_hidden_counts", "categories_not_released",
    "release_approved",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest_files(manifest: dict) -> dict[str, str]:
    """Verify every artifact already registered by the manifest."""
    expected: dict[str, str] = {}
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError("manifest files registry is missing")
    for entry in files:
        name = entry.get("name")
        recorded = entry.get("sha256")
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(recorded, str)):
            raise RuntimeError("manifest contains an invalid file entry")
        if name in expected:
            raise RuntimeError("manifest contains duplicate file names")
        path = OUT / name
        if not path.is_file() or _sha256_file(path) != recorded:
            raise RuntimeError("an existing manifest artifact failed hash verification")
        expected[name] = recorded
    return expected


def _load_verified_manifest() -> tuple[dict, dict[str, str]]:
    if not OUT.is_dir() or not MANIFEST_PATH.is_file():
        raise RuntimeError("website export directory or manifest is missing")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("validation", {}).get("status") != "PASS":
        raise RuntimeError("existing manifest validation status is not PASS")
    expected = _verify_manifest_files(manifest)
    if len([name for name in expected if name not in TARGET_NAMES]) != EXPECTED_BASE_FILE_COUNT:
        raise RuntimeError("existing manifest base artifact registry has drifted")
    return manifest, expected


def _aggregate_counts() -> pd.DataFrame:
    """Return only the 8 x 9 aggregate table; no joined rows escape this function."""
    ph = pd.read_csv(PHENOTYPES, usecols=["MRN", "octant"], dtype={"MRN": str})
    if len(ph) != EXPECTED_COHORT_N or ph["MRN"].isna().any() or ph["MRN"].duplicated().any():
        raise RuntimeError("phenotype cohort identity or size check failed")
    observed_n = ph.groupby("octant", observed=True).size().reindex(ORDER)
    if (observed_n.isna().any()
            or observed_n.astype(int).to_dict() != EXPECTED_OCTANT_N
            or set(ph["octant"].dropna()) != set(ORDER)):
        raise RuntimeError("phenotype octant coverage check failed")

    spine = pd.read_parquet(SPINE, columns=["id", "RACE_ETH"])
    if spine["id"].isna().any() or spine["id"].duplicated().any():
        raise RuntimeError("approved spine identity check failed")
    spine["id"] = spine["id"].astype(str)

    joined = ph.merge(
        spine, left_on="MRN", right_on="id", how="left", validate="one_to_one"
    )
    source_key = (joined["RACE_ETH"].astype("string").str.strip()
                  .str.replace(r"\s+", " ", regex=True).str.casefold())
    source_key = source_key.fillna("unknown").replace("", "unknown")
    fixed = source_key.map(SOURCE_LOOKUP)
    if fixed.isna().any():
        raise RuntimeError("RACE_ETH contains a value outside the fixed taxonomy")
    joined["race_category"] = fixed

    counts = pd.crosstab(joined["octant"], joined["race_category"], dropna=False)
    counts = counts.reindex(index=ORDER, columns=CATEGORY_LABELS, fill_value=0).astype(int)
    del joined, ph, spine, source_key, fixed

    if (counts.sum(axis=1).to_dict() != EXPECTED_OCTANT_N
            or int(counts.to_numpy().sum()) != EXPECTED_COHORT_N):
        raise RuntimeError("aggregate race/ethnicity table failed cohort reconciliation")
    return counts


def _suppression_mask(primary: pd.DataFrame) -> pd.DataFrame:
    """Union deterministic 2 x 2 complementary-suppression cycles."""
    mask = pd.DataFrame(False, index=ORDER, columns=CATEGORY_LABELS)
    for row_index, octant in enumerate(ORDER):
        companion_octant = ORDER[(row_index + 1) % len(ORDER)]
        for category in CATEGORY_LABELS:
            if not bool(primary.loc[octant, category]):
                continue
            companion_category = "Multiple" if category == "Other" else "Other"
            for row in (octant, companion_octant):
                for column in (category, companion_category):
                    mask.loc[row, column] = True
    return mask


def _rounded(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def _build_table(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = counts.sum(axis=0).astype(int)
    if bool((overall < DISCLOSURE_MIN).any()):
        raise RuntimeError("an overall taxonomy total fails the disclosure threshold")

    primary = counts < DISCLOSURE_MIN
    mask = _suppression_mask(primary)
    if bool((primary & ~mask).to_numpy().any()):
        raise RuntimeError("complementary suppression did not cover every primary cell")

    rows: list[dict] = []
    for octant in GROUPS:
        values = overall if octant == OVERALL else counts.loc[octant]
        denominator = EXPECTED_COHORT_N if octant == OVERALL else EXPECTED_OCTANT_N[octant]
        coverage = 100.0 * (denominator - int(values["Unknown"])) / denominator
        for category_id, category_label in TAXONOMY:
            numerator = int(values[category_label])
            proportion = numerator / denominator
            cohort_proportion = int(overall[category_label]) / EXPECTED_COHORT_N
            scale = np.sqrt(cohort_proportion * (1.0 - cohort_proportion))
            if not np.isfinite(scale) or scale <= 0:
                raise RuntimeError("a cohort category has no finite binary variance")
            standardized = 0.0 if octant == OVERALL else (
                proportion - cohort_proportion
            ) / scale
            suppressed = octant != OVERALL and bool(mask.loc[octant, category_label])
            rows.append({
                "octant": octant,
                "category_id": category_id,
                "category_label": category_label,
                "numerator": numerator,
                "denominator": denominator,
                "proportion": _rounded(proportion),
                "coverage_pct": _rounded(coverage),
                "cohort_proportion": _rounded(cohort_proportion),
                "standardized_difference": _rounded(standardized),
                "suppressed": suppressed,
                "suppression_reason": "disclosure_control" if suppressed else pd.NA,
            })

    table = pd.DataFrame(rows, columns=SCHEMA)
    suppressed_rows = table["suppressed"]
    table.loc[suppressed_rows, MASKED_VALUE_COLUMNS] = pd.NA

    unknown_suppressed_groups = [
        octant for octant in ORDER if bool(mask.loc[octant, "Unknown"])
    ]
    if unknown_suppressed_groups:
        table.loc[table["octant"].isin(unknown_suppressed_groups), "coverage_pct"] = pd.NA

    table["numerator"] = table["numerator"].astype("Int64")
    table["denominator"] = table["denominator"].astype("Int64")
    if len(table) != len(GROUPS) * len(TAXONOMY):
        raise RuntimeError("aggregate export row coverage is incomplete")
    return table, primary, mask


def _csv_bytes(table: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    table.to_csv(
        stream, index=False, columns=SCHEMA, na_rep="", lineterminator="\n",
        float_format=f"%.{ROUND_DIGITS}f",
    )
    return stream.getvalue().encode("utf-8")


def _exact_integer(value: float) -> int:
    if not np.isfinite(value) or value < 0 or not float(value).is_integer():
        raise RuntimeError("serialized count is not a nonnegative integer")
    return int(value)


def _close_to_serialized(value: float, expected: float) -> bool:
    return bool(np.isfinite(value) and abs(float(value) - _rounded(expected)) <= 5.1e-11)


def _parse_and_validate_staged(csv_payload: bytes) -> pd.DataFrame:
    """Prove every released derived field is redundant with released exact counts."""
    frame = pd.read_csv(io.BytesIO(csv_payload))
    if frame.columns.tolist() != SCHEMA or len(frame) != 81:
        raise RuntimeError("staged CSV schema or row count is invalid")
    if frame.duplicated(["octant", "category_id"]).any():
        raise RuntimeError("staged CSV contains duplicate logical keys")
    if set(frame["octant"]) != set(GROUPS) or set(frame["category_id"]) != set(CATEGORY_IDS):
        raise RuntimeError("staged CSV group or taxonomy coverage is invalid")
    expected_labels = frame["category_id"].map(ID_TO_LABEL)
    if not expected_labels.equals(frame["category_label"]):
        raise RuntimeError("staged CSV category IDs and labels disagree")

    flag_text = frame["suppressed"].astype(str).str.casefold()
    if not flag_text.isin(["true", "false"]).all():
        raise RuntimeError("staged CSV suppression flags are invalid")
    frame["_suppressed"] = flag_text.eq("true")
    suppressed = frame["_suppressed"]
    if frame.loc[suppressed, MASKED_VALUE_COLUMNS].notna().to_numpy().any():
        raise RuntimeError("a suppressed staged value was retained")
    if not frame.loc[suppressed, "suppression_reason"].eq("disclosure_control").all():
        raise RuntimeError("suppressed staged rows do not use the generic reason")
    if frame.loc[~suppressed, "suppression_reason"].notna().any():
        raise RuntimeError("an unsuppressed staged row has a suppression reason")
    if frame.loc[frame["octant"].eq(OVERALL), "_suppressed"].any():
        raise RuntimeError("an overall column total was suppressed")

    denominators: dict[str, int] = {}
    for group in GROUPS:
        values = pd.to_numeric(
            frame.loc[frame["octant"].eq(group), "denominator"], errors="coerce"
        ).unique()
        if len(values) != 1:
            raise RuntimeError("staged group denominators are inconsistent")
        denominators[group] = _exact_integer(float(values[0]))
    if (denominators[OVERALL] != EXPECTED_COHORT_N
            or any(denominators[x] != EXPECTED_OCTANT_N[x] for x in ORDER)
            or sum(denominators[x] for x in ORDER) != EXPECTED_COHORT_N):
        raise RuntimeError("staged denominators do not reconcile")

    overall_rows = frame[frame["octant"].eq(OVERALL)].set_index("category_label")
    overall_counts = {
        category: _exact_integer(float(overall_rows.loc[category, "numerator"]))
        for category in CATEGORY_LABELS
    }
    if sum(overall_counts.values()) != EXPECTED_COHORT_N:
        raise RuntimeError("staged overall category totals do not reconcile")

    for _, row in frame.iterrows():
        if bool(row["_suppressed"]):
            continue
        numerator = _exact_integer(float(row["numerator"]))
        if numerator < DISCLOSURE_MIN:
            raise RuntimeError("staged CSV releases a count below threshold")
        denominator = denominators[row["octant"]]
        cohort_proportion = overall_counts[row["category_label"]] / EXPECTED_COHORT_N
        expected_smd = 0.0 if row["octant"] == OVERALL else (
            numerator / denominator - cohort_proportion
        ) / np.sqrt(cohort_proportion * (1.0 - cohort_proportion))
        if (not _close_to_serialized(row["proportion"], numerator / denominator)
                or not _close_to_serialized(row["cohort_proportion"], cohort_proportion)
                or not _close_to_serialized(row["standardized_difference"], expected_smd)):
            raise RuntimeError("a released derived field is not redundant with released counts")

    # Coverage is released only when its sole count dependency (Unknown) is released.
    for group in GROUPS:
        group_rows = frame[frame["octant"].eq(group)]
        unknown = group_rows[group_rows["category_label"].eq("Unknown")].iloc[0]
        if bool(unknown["_suppressed"]):
            if group_rows["coverage_pct"].notna().any():
                raise RuntimeError("released coverage discloses a suppressed Unknown count")
            continue
        unknown_count = _exact_integer(float(unknown["numerator"]))
        expected = 100.0 * (denominators[group] - unknown_count) / denominators[group]
        if (group_rows["coverage_pct"].isna().any()
                or not group_rows["coverage_pct"].map(
                    lambda value: _close_to_serialized(value, expected)
                ).all()):
            raise RuntimeError("released coverage is not redundant with a released count")
    return frame


def _milp_nonunique(csv_payload: bytes, primary: pd.DataFrame | None = None) -> bool:
    """Audit every hidden octant cell using only the staged attacker view."""
    frame = _parse_and_validate_staged(csv_payload)
    octants = frame[frame["octant"].isin(ORDER)].copy()
    hidden = octants[octants["_suppressed"]]
    variable_keys = [
        (octant, category)
        for octant in ORDER
        for category in CATEGORY_LABELS
        if bool(((hidden["octant"] == octant)
                 & (hidden["category_label"] == category)).any())
    ]
    variable_index = {key: index for index, key in enumerate(variable_keys)}

    denominator = {
        group: _exact_integer(float(
            frame.loc[frame["octant"].eq(group), "denominator"].iloc[0]
        ))
        for group in GROUPS
    }
    overall = {
        category: _exact_integer(float(frame.loc[
            frame["octant"].eq(OVERALL) & frame["category_label"].eq(category),
            "numerator",
        ].iloc[0]))
        for category in CATEGORY_LABELS
    }

    equations: list[np.ndarray] = []
    targets: list[int] = []
    for octant in ORDER:
        coefficient = np.zeros(len(variable_keys), dtype=float)
        for category in CATEGORY_LABELS:
            key = (octant, category)
            if key in variable_index:
                coefficient[variable_index[key]] = 1.0
        released = octants[
            octants["octant"].eq(octant) & ~octants["_suppressed"]
        ]["numerator"].sum()
        equations.append(coefficient)
        targets.append(denominator[octant] - _exact_integer(float(released)))

    for category in CATEGORY_LABELS:
        coefficient = np.zeros(len(variable_keys), dtype=float)
        for octant in ORDER:
            key = (octant, category)
            if key in variable_index:
                coefficient[variable_index[key]] = 1.0
        released = octants[
            octants["category_label"].eq(category) & ~octants["_suppressed"]
        ]["numerator"].sum()
        equations.append(coefficient)
        targets.append(overall[category] - _exact_integer(float(released)))

    if not variable_keys:
        return all(
            np.count_nonzero(coefficient) == 0 and target == 0
            for coefficient, target in zip(equations, targets)
        )

    lower = np.zeros(len(variable_keys), dtype=float)
    upper = np.array([
        min(denominator[octant], overall[category])
        for octant, category in variable_keys
    ], dtype=float)
    if primary is not None:
        for index, (octant, category) in enumerate(variable_keys):
            if bool(primary.loc[octant, category]):
                upper[index] = min(upper[index], DISCLOSURE_MIN - 1)
            else:
                lower[index] = DISCLOSURE_MIN
    if np.any(lower > upper):
        return False

    matrix = np.vstack(equations)
    target_array = np.asarray(targets, dtype=float)
    constraints = LinearConstraint(matrix, target_array, target_array)
    bounds = Bounds(lower, upper)
    integrality = np.ones(len(variable_keys), dtype=int)

    def checked_value(result, variable: int) -> int | None:
        if not result.success or result.x is None:
            return None
        integer_solution = np.rint(result.x).astype(np.int64)
        if (np.max(np.abs(result.x - integer_solution)) > 1e-6
                or np.any(integer_solution < lower)
                or np.any(integer_solution > upper)
                or not np.array_equal(
                    matrix.astype(np.int64) @ integer_solution,
                    target_array.astype(np.int64),
                )):
            return None
        return int(integer_solution[variable])

    for index in range(len(variable_keys)):
        objective = np.zeros(len(variable_keys), dtype=float)
        objective[index] = 1.0
        result_min = milp(
            objective, integrality=integrality, bounds=bounds,
            constraints=constraints, options={"disp": False},
        )
        result_max = milp(
            -objective, integrality=integrality, bounds=bounds,
            constraints=constraints, options={"disp": False},
        )
        minimum = checked_value(result_min, index)
        maximum = checked_value(result_max, index)
        if minimum is None or maximum is None:
            return False
        if minimum >= maximum:
            return False
    return True


def _category_definitions() -> list[dict[str, str]]:
    definitions = []
    for category_id, category_label in TAXONOMY:
        if category_label == "Unknown":
            definition = (
                "Source RACE_ETH Unknown/UNKNOWN (case-insensitive), blank or null, "
                "or phenotype member absent from the approved v2 spine."
            )
        else:
            definition = (
                f"Exact source RACE_ETH label '{category_label}' after whitespace "
                "normalization and case-insensitive matching."
            )
        definitions.append({
            "category_id": category_id,
            "category_label": category_label,
            "definition": definition,
        })
    return definitions


def _audit_bytes(suppressed_cells: int) -> tuple[bytes, dict]:
    audit = {
        "taxonomy": {
            "source_field": "RACE_ETH",
            "prespecified_before_aggregation": True,
            "categories": _category_definitions(),
            "unexpected_source_values": "fail_closed",
        },
        "denominator_definition": (
            "Full established phenotype group size, including Unknown; overall is the "
            "70,880-person shared phenotype cohort and each octant denominator is its "
            "published full-cohort size."
        ),
        "coverage_pct_definition": (
            "100 * (denominator - Unknown numerator) / denominator; null for every row "
            "of a group whenever that group's Unknown numerator is suppressed."
        ),
        "cohort_proportion_definition": (
            "Overall category numerator divided by 70,880."
        ),
        "standardized_difference_definition": (
            "Signed (group proportion - cohort proportion) / "
            "sqrt(cohort proportion * (1 - cohort proportion)); overall is zero."
        ),
        "disclosure_threshold": DISCLOSURE_MIN,
        "primary_suppression_rule": (
            "Every octant count below 11, including zero, is suppressed. An overall "
            "category count below 11 fails the release closed."
        ),
        "secondary_suppression_rule": (
            "For each primary octant/category cell, suppress the full 2x2 block formed "
            "with the next octant cyclically and category Other; when the primary "
            "category is Other, use Multiple. Union overlapping blocks. Overall category "
            "totals remain released. All suppressed rows use one generic reason."
        ),
        "row_count": 81,
        "expected_group_category_coverage": {
            "groups": GROUPS,
            "group_count": len(GROUPS),
            "category_ids": CATEGORY_IDS,
            "category_count": len(TAXONOMY),
            "rows_per_group": len(TAXONOMY),
            "complete": True,
        },
        "number_of_suppressed_cells": suppressed_cells,
        "reconstruction_audit": {
            "result": "PASS",
            "fresh_serialized_attacker_view": "PASS",
            "strong_internal_bound_aware_view": "PASS",
            "suppressed_cells_audited": suppressed_cells,
            "all_suppressed_integer_counts_non_unique": True,
        },
        "browser_delivery_validation": {
            "schema_allowlist": "PASS",
            "suppressed_value_dependency_scan": "PASS",
            "identifier_and_path_leak_scan": "PASS",
        },
        "browser_delivered_values_contain_no_hidden_counts": True,
        "categories_not_released": [],
        "release_approved": True,
    }
    payload = (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return payload, audit


def _audit_json_schema(audit: dict) -> dict:
    property_types = {
        "taxonomy": "object",
        "denominator_definition": "string",
        "coverage_pct_definition": "string",
        "cohort_proportion_definition": "string",
        "standardized_difference_definition": "string",
        "disclosure_threshold": "integer",
        "primary_suppression_rule": "string",
        "secondary_suppression_rule": "string",
        "row_count": "integer",
        "expected_group_category_coverage": "object",
        "number_of_suppressed_cells": "integer",
        "reconstruction_audit": "object",
        "browser_delivery_validation": "object",
        "browser_delivered_values_contain_no_hidden_counts": "boolean",
        "categories_not_released": "array",
        "release_approved": "boolean",
    }
    return {
        "type": "object",
        "required": list(audit.keys()),
        "properties": {
            key: {"type": property_types[key]}
            for key in audit
        },
        "additionalProperties": False,
    }


def _manifest_bytes(
    original: dict,
    csv_payload: bytes,
    audit_payload: bytes,
    audit: dict,
    suppressed_cells: int,
) -> bytes:
    manifest = copy.deepcopy(original)
    csv_entry = {
        "name": CSV_NAME,
        "bytes": len(csv_payload),
        "sha256": _sha256_bytes(csv_payload),
        "internal": False,
        "rows": 81,
        "schema": SCHEMA,
        "disclosure_audit_status": "PASS",
        "public_website_release_approved": True,
    }
    audit_entry = {
        "name": AUDIT_NAME,
        "bytes": len(audit_payload),
        "sha256": _sha256_bytes(audit_payload),
        "internal": False,
        "records": 1,
        "rows": 1,
        "schema": _audit_json_schema(audit),
        "disclosure_audit_status": "PASS",
        "public_website_release_approved": True,
    }
    retained = [
        entry for entry in manifest["files"] if entry.get("name") not in TARGET_NAMES
    ]
    manifest["files"] = sorted(retained + [csv_entry, audit_entry], key=lambda x: x["name"])

    manifest["race_ethnicity"] = {
        "exporter": "analyses/incwas/30_octant_race_ethnicity_export.py",
        "source_field": "RACE_ETH",
        "shared_phenotype_cohort": EXPECTED_COHORT_N,
        "groups": GROUPS,
        "octants": len(ORDER),
        "categories": _category_definitions(),
        "category_count": len(TAXONOMY),
        "rows": 81,
        "files": {
            "distribution": CSV_NAME,
            "audit": AUDIT_NAME,
        },
        "denominator_definition": audit["denominator_definition"],
        "coverage_pct_definition": audit["coverage_pct_definition"],
        "standardized_difference_definition": audit["standardized_difference_definition"],
        "suppressed_cells": suppressed_cells,
        "disclosure_threshold": DISCLOSURE_MIN,
        "reconstruction_audit_result": "PASS",
        "browser_delivery_contains_hidden_counts": False,
        "disclosure_audit_status": "PASS",
        "validation": {
            "status": "PASS",
            "expected_rows": 81,
            "rows": 81,
            "expected_groups": len(GROUPS),
            "groups": len(GROUPS),
            "expected_categories": len(TAXONOMY),
            "categories": len(TAXONOMY),
        },
        "release_approved": True,
        "public_website_release_approved": True,
    }

    disclosure = manifest.setdefault("disclosure", {})
    disclosure.setdefault("suppressed_cells_by_file", {})[CSV_NAME] = suppressed_cells
    disclosure["suppressed_cells_by_file"][AUDIT_NAME] = 0
    disclosure.setdefault("suppressed_rows_by_file", {})[CSV_NAME] = suppressed_cells
    disclosure["suppressed_rows_by_file"][AUDIT_NAME] = 0
    old_note = str(disclosure.get("note", ""))
    old_note = old_note.replace("Race/ethnicity is excluded entirely.", "").strip()
    release_note = (
        "Race/ethnicity is released only through the disclosure-audited aggregate "
        "distribution; suppressed values and every direct derivative are null."
    )
    disclosure["note"] = (
        old_note if release_note in old_note else f"{old_note} {release_note}".strip()
    )

    unavailable = manifest.get("unavailable_or_excluded", [])
    manifest["unavailable_or_excluded"] = [
        item for item in unavailable
        if str(item.get("field", "")).strip().casefold() != "race/ethnicity breakdown"
    ]
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _validate_audit_payload(payload: bytes, suppressed_cells: int) -> dict:
    audit = json.loads(payload.decode("utf-8"))
    taxonomy = audit.get("taxonomy", {}).get("categories", [])
    reconstruction = audit.get("reconstruction_audit", {})
    browser = audit.get("browser_delivery_validation", {})
    if ([item.get("category_id") for item in taxonomy] != CATEGORY_IDS
            or [item.get("category_label") for item in taxonomy] != CATEGORY_LABELS
            or audit.get("row_count") != 81
            or audit.get("number_of_suppressed_cells") != suppressed_cells
            or reconstruction.get("result") != "PASS"
            or reconstruction.get("fresh_serialized_attacker_view") != "PASS"
            or reconstruction.get("strong_internal_bound_aware_view") != "PASS"
            or reconstruction.get("suppressed_cells_audited") != suppressed_cells
            or reconstruction.get("all_suppressed_integer_counts_non_unique") is not True
            or set(browser.values()) != {"PASS"}
            or audit.get("browser_delivered_values_contain_no_hidden_counts") is not True
            or audit.get("release_approved") is not True):
        raise RuntimeError("staged disclosure audit payload failed validation")
    return audit


def _validate_manifest_payload(
    payload: bytes,
    original: dict,
    csv_payload: bytes,
    audit_payload: bytes,
    suppressed_cells: int,
) -> None:
    manifest = json.loads(payload.decode("utf-8"))
    files = manifest.get("files", [])
    names = [entry.get("name") for entry in files]
    if (len(files) != EXPECTED_BASE_FILE_COUNT + len(TARGET_NAMES)
            or len(names) != len(set(names))):
        raise RuntimeError("staged manifest file registry is invalid")
    by_name = {entry["name"]: entry for entry in files}
    original_base = {
        entry["name"]: entry for entry in original["files"]
        if entry.get("name") not in TARGET_NAMES
    }
    staged_base = {name: by_name[name] for name in original_base if name in by_name}
    if staged_base != original_base or len(staged_base) != EXPECTED_BASE_FILE_COUNT:
        raise RuntimeError("staged manifest changed a pre-existing artifact entry")

    audit = _validate_audit_payload(audit_payload, suppressed_cells)
    csv_entry = by_name.get(CSV_NAME, {})
    audit_entry = by_name.get(AUDIT_NAME, {})
    if (csv_entry.get("sha256") != _sha256_bytes(csv_payload)
            or csv_entry.get("bytes") != len(csv_payload)
            or csv_entry.get("rows") != 81
            or csv_entry.get("schema") != SCHEMA
            or csv_entry.get("disclosure_audit_status") != "PASS"
            or csv_entry.get("public_website_release_approved") is not True
            or audit_entry.get("sha256") != _sha256_bytes(audit_payload)
            or audit_entry.get("bytes") != len(audit_payload)
            or audit_entry.get("records") != 1
            or audit_entry.get("rows") != 1
            or audit_entry.get("schema") != _audit_json_schema(audit)
            or audit_entry.get("disclosure_audit_status") != "PASS"
            or audit_entry.get("public_website_release_approved") is not True):
        raise RuntimeError("staged manifest target entries are invalid")

    race = manifest.get("race_ethnicity", {})
    validation = race.get("validation", {})
    if (race.get("octants") != len(ORDER)
            or race.get("category_count") != len(TAXONOMY)
            or race.get("rows") != 81
            or race.get("disclosure_threshold") != DISCLOSURE_MIN
            or race.get("suppressed_cells") != suppressed_cells
            or race.get("reconstruction_audit_result") != "PASS"
            or race.get("release_approved") is not True
            or race.get("public_website_release_approved") is not True
            or validation.get("status") != "PASS"
            or manifest.get("validation", {}).get("status") != "PASS"):
        raise RuntimeError("staged manifest race/ethnicity release block is invalid")
    if any(
        str(item.get("field", "")).strip().casefold() == "race/ethnicity breakdown"
        for item in manifest.get("unavailable_or_excluded", [])
    ):
        raise RuntimeError("staged manifest still marks race/ethnicity unavailable")


def _validate_browser_payloads(
    csv_payload: bytes,
    audit_payload: bytes,
    manifest_payload: bytes,
    suppressed_cells: int,
) -> None:
    """Allowlist browser-facing fields and reject identifier/path leakage."""
    _parse_and_validate_staged(csv_payload)
    audit = _validate_audit_payload(audit_payload, suppressed_cells)
    manifest = json.loads(manifest_payload.decode("utf-8"))
    race = manifest.get("race_ethnicity", {})
    targets = {
        entry["name"]: entry for entry in manifest.get("files", [])
        if entry.get("name") in TARGET_NAMES
    }

    if (set(audit) != AUDIT_TOP_LEVEL_KEYS
            or set(audit["taxonomy"]) != {
                "source_field", "prespecified_before_aggregation", "categories",
                "unexpected_source_values",
            }
            or any(set(item) != {"category_id", "category_label", "definition"}
                   for item in audit["taxonomy"]["categories"])
            or set(audit["expected_group_category_coverage"]) != {
                "groups", "group_count", "category_ids", "category_count",
                "rows_per_group", "complete",
            }
            or set(audit["reconstruction_audit"]) != {
                "result", "fresh_serialized_attacker_view",
                "strong_internal_bound_aware_view", "suppressed_cells_audited",
                "all_suppressed_integer_counts_non_unique",
            }
            or set(audit["browser_delivery_validation"]) != {
                "schema_allowlist", "suppressed_value_dependency_scan",
                "identifier_and_path_leak_scan",
            }):
        raise RuntimeError("browser audit payload violates the release allowlist")

    race_keys = {
        "exporter", "source_field", "shared_phenotype_cohort", "groups", "octants",
        "categories", "category_count", "rows", "files", "denominator_definition",
        "coverage_pct_definition", "standardized_difference_definition",
        "suppressed_cells", "disclosure_threshold", "reconstruction_audit_result",
        "browser_delivery_contains_hidden_counts", "disclosure_audit_status",
        "validation", "release_approved", "public_website_release_approved",
    }
    if (set(race) != race_keys
            or any(set(item) != {"category_id", "category_label", "definition"}
                   for item in race.get("categories", []))
            or set(race.get("validation", {})) != {
                "status", "expected_rows", "rows", "expected_groups", "groups",
                "expected_categories", "categories",
            }
            or set(targets) != TARGET_NAMES):
        raise RuntimeError("browser manifest payload violates the release allowlist")
    if (set(targets[CSV_NAME]) != {
            "name", "bytes", "sha256", "internal", "rows", "schema",
            "disclosure_audit_status", "public_website_release_approved",
        } or set(targets[AUDIT_NAME]) != {
            "name", "bytes", "sha256", "internal", "records", "rows", "schema",
            "disclosure_audit_status", "public_website_release_approved",
        }):
        raise RuntimeError("browser manifest file entry violates the release allowlist")

    scoped_manifest = {
        "race_ethnicity": race,
        "files": targets,
    }
    browser_bytes = b"\n".join([
        csv_payload,
        audit_payload,
        json.dumps(scoped_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    ])
    browser_text = browser_bytes.decode("utf-8").casefold()
    forbidden = (
        "mrn", "pat_id", "patient_id", "date_of_birth", "cohort.parquet",
        "cross_domain_phenotypes", "_identified", "c:\\users\\", "/users/",
    )
    if any(token in browser_text for token in forbidden) or re.search(
        r"[a-z]:[\\/]", browser_text
    ):
        raise RuntimeError("browser payload contains an identifier or absolute-path token")


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


def main(log: Callable[[str], None] | None = None) -> None:
    manifest, old_hashes = _load_verified_manifest()
    counts = _aggregate_counts()
    table, primary, mask = _build_table(counts)
    csv_payload = _csv_bytes(table)

    # First model knows only browser-delivered values. The second additionally knows
    # which hidden cells are primary (0..10) versus secondary (>=11).
    serialized_contract_pass = False
    fresh_pass = False
    strong_pass = False
    try:
        _parse_and_validate_staged(csv_payload)
        serialized_contract_pass = True
        fresh_pass = _milp_nonunique(csv_payload)
        strong_pass = _milp_nonunique(csv_payload, primary=primary)
    except Exception as exc:
        raise RuntimeError("race/ethnicity disclosure audit failed closed") from exc

    suppressed_cells = int(mask.to_numpy().sum())
    checks = {
        "serialized_browser_contract": serialized_contract_pass,
        "fresh_attacker_integer_reconstruction": fresh_pass,
        "strong_bound_aware_integer_reconstruction": strong_pass,
        "all_primary_cells_masked": not bool((primary & ~mask).to_numpy().any()),
        "no_overall_cell_suppressed": not bool(
            table.loc[table["octant"].eq(OVERALL), "suppressed"].any()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("race/ethnicity disclosure audit failed closed")

    audit_payload, audit = _audit_bytes(suppressed_cells)
    _validate_audit_payload(audit_payload, suppressed_cells)
    manifest_payload = _manifest_bytes(
        manifest, csv_payload, audit_payload, audit, suppressed_cells
    )
    _validate_manifest_payload(
        manifest_payload, manifest, csv_payload, audit_payload, suppressed_cells
    )
    _validate_browser_payloads(
        csv_payload, audit_payload, manifest_payload, suppressed_cells
    )

    # Recheck the entire old registry immediately before the first write.
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
        raise RuntimeError("written disclosure artifacts failed hash verification")
    _validate_manifest_payload(
        MANIFEST_PATH.read_bytes(), manifest, CSV_PATH.read_bytes(),
        AUDIT_PATH.read_bytes(), suppressed_cells,
    )
    _validate_browser_payloads(
        CSV_PATH.read_bytes(), AUDIT_PATH.read_bytes(), MANIFEST_PATH.read_bytes(),
        suppressed_cells,
    )

    if log is not None:
        log("race/ethnicity aggregate export disclosure audit: PASS")


if __name__ == "__main__":
    main()
