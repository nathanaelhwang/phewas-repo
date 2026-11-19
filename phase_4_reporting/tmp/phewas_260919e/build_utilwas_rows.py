"""Normalize reviewed aggregate UtilWAS model results for manuscript figures."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {
    "phecode": ROOT / "results/utilwas_results/analysis/results_phecode_5yr.csv",
    "specialty": ROOT / "results/utilwas_results/analysis/results_specialty_5yr.csv",
}
SPECS = {
    "ed_primarypresence": ("phecode", "combo", "ED_primary", "presence", "ED primary diagnoses"),
    "ed_secondarypresence": ("phecode", "combo", "ED_secondary", "presence", "ED secondary diagnoses"),
    "physician_primarypresence": ("phecode", "combo", "physician_primary", "presence", "Outpatient primary diagnoses"),
    "physicianspecialtyrates": ("specialty", "family", "physician_level", "rate", "Physician specialties"),
    "alliedspecialtyrates": ("specialty", "family", "allied_health", "rate", "Allied-health specialties"),
}


def read_aggregate(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        assert not set(columns) & {"id", "MRN", "mrn", "name", "DOB", "dob", "patient_id", "UTIL_ID"}
        return list(reader), columns


def normalize(raw, key, title):
    result = dict(raw)
    for column in ("effect", "ci_low", "ci_high", "p"):
        result[column] = float(raw[column])
        assert math.isfinite(result[column]), (key, raw["feature"], column)
    assert min(result[c] for c in ("effect", "ci_low", "ci_high")) > 0
    for flag in ("sig_bon", "sig_fdr"):
        assert raw[flag].lower() in ("true", "false")
        result[flag] = raw[flag].lower() == "true"
    result["n"] = int(raw["n"])
    result["n_value"] = result["n"]
    result["analysis_key"] = key
    result["label"] = raw.get("phenotype") or raw["feature"]
    result["group"] = raw.get("category") or title
    result["detail"] = result["group"]
    result["code"] = raw["feature"]
    result["signed"] = math.log2(result["effect"])
    result["signed_low"] = math.log2(result["ci_low"])
    result["signed_high"] = math.log2(result["ci_high"])
    result["magnitude"] = abs(result["signed"])
    result["plot_y"] = min(50.0, -math.log10(max(result["p"], 1e-300)))
    return result


def main():
    loaded = {}
    sources = {}
    for source, path in SOURCES.items():
        loaded[source], columns = read_aggregate(path)
        sources[source] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "columns": columns,
            "aggregate_rows_total": len(loaded[source]),
            "generator": str(ROOT / ("analyses/utilwas/08_run_phecode_was.py" if source == "phecode" else "analyses/utilwas/07_run_specialty.py")),
        }
    groups = {}
    audit = {}
    for key, (source, subgroup_column, subgroup, part, title) in SPECS.items():
        rows = [normalize(r, key, title) for r in loaded[source]
                if r["window"] == "5yr" and r["contrast"] == "ahi_ge5" and r["model"] == "m4"
                and r[subgroup_column] == subgroup and r["part"] == part]
        assert rows and len({r["feature"] for r in rows}) == len(rows)
        expected = "OR" if part == "presence" else "IRR"
        assert all(r["effect_type"] == expected and r["n"] == 222109 for r in rows)
        groups[key] = rows
        by_group = defaultdict(list)
        for row in rows:
            if row["sig_bon"]:
                by_group[row["group"]].append(row)
        forest = []
        for group in sorted(by_group):
            forest.extend(sorted(by_group[group], key=lambda r: (-r["magnitude"], r["p"], r["label"]))[:3])
        labels = sorted([r for r in rows if r["sig_bon"] and r["signed"] > 0],
                        key=lambda r: (r["p"], -r["magnitude"], r["label"]))[:10]
        audit[key] = {
            "title": title, "n": len(rows),
            "bonferroni": sum(r["sig_bon"] for r in rows),
            "positive_bonferroni": sum(r["sig_bon"] and r["signed"] > 0 for r in rows),
            "fdr": sum(r["sig_fdr"] for r in rows),
            "positive_fdr": sum(r["sig_fdr"] and r["signed"] > 0 for r in rows),
            "grouping": "Source Phecode category" if source == "phecode" else "Source encounter family; no authoritative body-system mapping exists",
            "scatter_positive_bonf_labels": [r["feature"] for r in labels],
            "forest_selection": [r["feature"] for r in forest],
            "forest_selected_details": [{c: r[c] for c in ("feature", "label", "group", "effect", "ci_low", "ci_high", "p", "sig_bon", "signed", "magnitude")} for r in forest],
        }
    payload = {
        "schema_version": 1,
        "sources": sources,
        "selection": {"window": "5yr", "contrast": "ahi_ge5", "model": "m4"},
        "provenance": {
            "population_n": 222109,
            "window": "Five years before index, index-exclusive; actual observation time may be shorter",
            "presence": "Logistic OR for any qualifying diagnosis, entire cohort including zero encounters",
            "specialty_rate": "Poisson GLM IRR, robust HC0 SE, count offset log(min(obs_window,5)), floor 1/365 year; entire cohort including zero encounters",
            "m4_covariates": ["age", "sex", "race/ethnicity", "study year", "smoking", "income", "BMI"],
            "observation_adjustment": "Presence includes standardized observability window as covariate; rates use observation time as offset",
            "multiplicity": "Original flags retained: Bonferroni and BH within combo/part/contrast/model or family/contrast/model, separately by window",
            "eligibility": "At least 500 patients with the qualifying feature",
            "outpatient_definition": "CLINIC or VIRTUAL care; OFFICE VISIT, TAV, or VIDEO subtype",
            "outpatient_primary_diagnosis": "All outpatient primary diagnoses (PRIMARY_DX_IND=Y); not restricted to primary-care specialties. User approved using this existing source.",
            "specialty_family_definition": "Physician: outpatient encounter with ICD9/ICD10 diagnosis. Allied-health: outpatient encounter with at least one non-ICD row. Families overlap and must not be summed.",
            "scatter_labels": "Up to 10 positive Bonferroni-significant findings by ascending P; do not label inverse findings",
            "forest_selection": "Existing policy preserved: up to three Bonferroni-significant features per established group, descending absolute log2 effect, selected across both directions before splitting",
            "specialty_forest_note": "Only an encounter-family grouping is established. Current selection yields three physician and three allied-health specialties. Selecting all Bonferroni findings would require an explicit change of forest selection policy.",
            "excluded": "All ED conditional-rate results",
        },
        "groups": groups,
        "audit": audit,
    }
    output = OUT / "utilwas_rows.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for key, info in audit.items():
        print(key, info["n"], "tests;", info["positive_bonferroni"], "positive Bonferroni; forest:", info["forest_selection"])
    print(output)


if __name__ == "__main__":
    main()
