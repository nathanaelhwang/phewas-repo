"""Produce aggregate-only Table 1 additions; never print/persist patient rows.

Run inside the local approved data environment. Missing sources produce explicit
unavailable results, never invented zeroes. The source weighted Charlson score
is descriptive only until its ascertainment window is verified as baseline.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {"Overall": 222109, "OSA": 161491, "No OSA": 60618}
BANDS = ["0%-50%", "51%-75%", "76%-100%"]


def guard(path):
    path = Path(path).absolute()
    try:
        path = path.resolve()
    except OSError:
        pass  # Locked/missing source drives are reported unavailable below.
    normalized = str(path).replace("\\", "/").lower()
    forbidden = ["/data/raw/", "/phi/", "/patient_records/", "_identified", "/.ssh/", "/.aws/"]
    if any(x in normalized for x in forbidden) or path.name.startswith(".env") or path.suffix.lower() in {".pem", ".key"}:
        raise ValueError("Prohibited data source")
    return path


def available(path):
    try:
        return path.is_file()
    except OSError:
        return False


def key(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(12)


def groups(frame):
    return {"Overall": frame, "OSA": frame.loc[frame.AHI.ge(5)], "No OSA": frame.loc[frame.AHI.lt(5)]}


def dedupe(frame):
    """Collapse exact duplicates; omit conflicting keys without arbitrary selection."""
    frame = frame.drop_duplicates()
    conflicts = frame.id.duplicated(keep=False)
    n_conflicting = int(frame.loc[conflicts, "id"].nunique())
    return frame.loc[~conflicts], n_conflicting


def numeric_summary(frame, column):
    result = {}
    for name, group in groups(frame).items():
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        result[name] = {
            "cohort_n": len(group), "usable_n": len(values),
            "missing_n": len(group) - len(values),
            "mean": float(values.mean()) if len(values) else None,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "median": float(values.median()) if len(values) else None,
            "q1": float(values.quantile(.25)) if len(values) else None,
            "q3": float(values.quantile(.75)) if len(values) else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=ROOT / "results/icd_analysis_v2_results/cohort/cohort.parquet")
    parser.add_argument("--geocode", default="D:/phewas_data/geocode.csv")
    parser.add_argument("--charlson", default="D:/phewas_data/temporal/charlson.csv")
    parser.add_argument("--output", default=Path(__file__).with_name("table1_additions.json"))
    args = parser.parse_args()
    cohort_path, geocode_path, charlson_path = map(guard, [args.cohort, args.geocode, args.charlson])
    # Load only the exposure to verify denominators if no source is available.
    c = pd.read_parquet(cohort_path, columns=["AHI"])
    counts = {name: len(group) for name, group in groups(c).items()}
    if counts != EXPECTED or c.AHI.isna().any():
        raise ValueError("Cohort denominator verification failed")
    out = {
        "cohort_counts_verified": counts,
        "group_definition": "OSA: AHI >=5; No OSA: AHI <5",
        "cohort_source": str(cohort_path),
        "education": {
            "status": "unavailable_source_inaccessible", "publication_ready": False,
            "source": str(geocode_path),
            "definition": "Neighborhood share of residents with 12 or more years of education; not individual educational attainment",
            "bands": BANDS, "group_summaries": None,
        },
        "charlson": {
            "status": "unavailable_source_inaccessible", "publication_ready": False,
            "source": str(charlson_path), "weighted_column": "Charlson_Wt",
            "count_column": "Charlson", "group_summaries": None,
            "limitation": "Authoritative temporal source describes a recent ~49,012-person extract, not the full 222,109-person cohort. Baseline ascertainment must be established before using its score as baseline CCI. Missing scores must not be set to zero.",
        },
        "provenance": [
            "analyses/icd_analysis/v2_01_build_cohort.py",
            "results/data_preparation/data_dictionary.md",
            "results/data_preparation/phase2_data_structure_synthesis.md",
        ],
    }
    if available(geocode_path) or available(charlson_path):
        c = pd.read_parquet(cohort_path, columns=["id", "AHI", "STUDY_DATE"])
        c["id"] = key(c.id)
        if c.id.isna().any() or c.id.duplicated().any():
            raise ValueError("Cohort key verification failed")
    if available(geocode_path):
        education = pd.read_csv(geocode_path, usecols=["MRN", "education_12_up_gp"], dtype="string")
        education["id"] = key(education.pop("MRN"))
        education["education_12_up_gp"] = education.education_12_up_gp.str.strip()
        invalid = education.education_12_up_gp.notna() & ~education.education_12_up_gp.isin(BANDS + ["", "None"])
        if invalid.any():
            raise ValueError("Unexpected education category; no category values exposed")
        education.loc[~education.education_12_up_gp.isin(BANDS), "education_12_up_gp"] = pd.NA
        education, conflicts = dedupe(education)
        merged = c.merge(education, how="left", on="id", validate="one_to_one")
        summaries = {}
        for name, group in groups(merged).items():
            summaries[name] = {
                "cohort_n": len(group),
                "usable_n": int(group.education_12_up_gp.notna().sum()),
                "missing_n": int(group.education_12_up_gp.isna().sum()),
                "strata": {band: {"n": int(group.education_12_up_gp.eq(band).sum()), "pct_full_group": float(100 * group.education_12_up_gp.eq(band).sum() / len(group))} for band in BANDS},
            }
        out["education"].update(status="computed", publication_ready=True, conflicting_source_keys_omitted=conflicts, group_summaries=summaries)
    if available(charlson_path):
        # Header inspection only; project structured scores and date columns.
        headers = pd.read_csv(charlson_path, nrows=0).columns
        cols = [x for x in headers if x in {"MRN", "Charlson", "Charlson_Wt"} or x.startswith("icdx_date_")]
        if not {"MRN", "Charlson_Wt"}.issubset(cols):
            raise ValueError("Required Charlson columns are missing")
        scores = pd.read_csv(charlson_path, usecols=cols, dtype="string")
        scores["id"] = key(scores.pop("MRN"))
        scores, conflicts = dedupe(scores)
        merged = c.merge(scores, how="left", on="id", validate="one_to_one")
        dates = [x for x in cols if x.startswith("icdx_date_")]
        post_index = pd.Series(False, index=merged.index)
        index_date = pd.to_datetime(merged.STUDY_DATE, errors="coerce")
        for column in dates:
            parsed = pd.to_datetime(merged[column], format="mixed", errors="coerce")
            post_index |= parsed.gt(index_date)
        summaries = numeric_summary(merged, "Charlson_Wt")
        out["charlson"].update(
            status="available_subset_descriptive_only", publication_ready=False,
            conflicting_source_keys_omitted=conflicts, group_summaries=summaries,
            matched_people_with_any_condition_date_after_index=int(post_index.sum()),
            baseline_validity="unverified; source scores are not automatically restricted to cohort index date",
        )
        if "Charlson" in merged:
            out["charlson"]["unweighted_condition_count_summaries"] = numeric_summary(merged, "Charlson")
    output_path = guard(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"cohort_counts_verified": counts, "education_status": out["education"]["status"], "charlson_status": out["charlson"]["status"], "aggregate_output": str(output_path)}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never echo an exception that could embed a source data value.
        print("Aggregation stopped safely. Verify schemas, approved source paths, and cohort integrity locally; no patient data were emitted.")
        raise SystemExit(1)
