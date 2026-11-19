"""Locally derive baseline CCI and emit only group-level aggregates.

Public upstream mapping source is parsed as data; no downloaded code is executed.
Patient records remain in process memory. No row-level extracts are saved.
"""
from pathlib import Path
import hashlib
import json
import re
import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = HERE / "charlson_reference"
ICD = ROOT / "results/icd_analysis_v2_results/pyphewas/icds_full.csv"
COHORT = ROOT / "results/icd_analysis_v2_results/cohort/cohort.parquet"
HIERARCHY = {"mld": "msld", "diab": "diabwc", "canc": "metacanc"}
EXPECTED = {"Overall": 222109, "OSA": 161491, "No OSA": 60618}


def parse_reference():
    source = (REF / "make-mapping.R").read_text()
    maps = {}
    for system in (9, 10):
        block = source.split(f'.tmpn <- "charlson_icd{system}_quan"')[1].split('.tmpn <- ')[0]
        maps[system] = {name: tuple(re.findall(r'"([A-Z0-9]+)"', items)) for name, items in re.findall(r'\[\["([a-z]+)"\]\] <- c\(([^\n]+)\)', block)}
        assert len(maps[system]) == 17
    wsource = (REF / "make-weights.R").read_text().split('.weights[[w]][["charlson"]] <- c(')[1].split(')')[0]
    weights = {k: int(v) for k, v in re.findall(r'([a-z]+)\s*=\s*(\d+)', wsource)}
    assert set(weights) == set(maps[9]) == set(maps[10]) and sum(weights.values()) == 33
    return maps, weights


def score_synthetic(codes, maps, weights):
    present = {name: any(re.sub('[^A-Z0-9]', '', code.upper()).startswith(maps[system][name]) for system, code in codes) for name in weights}
    for mild, severe in HIERARCHY.items():
        if present[severe]:
            present[mild] = False
    return sum(weights[name] * present[name] for name in weights)


def main():
    maps, weights = parse_reference()
    cases = [([], 0), ([(9, '428.0')], 1), ([(10, 'I50.9')], 1), ([(9, '250.00'), (10, 'E11.21')], 2),
             ([(10, 'K74.6'), (10, 'K72.9')], 3), ([(10, 'C50.9'), (10, 'C78.7')], 6),
             ([(9, '042'), (10, 'B20')], 6), ([(9, '585.6'), (10, 'N18.6')], 2), ([(10, 'G47.33')], 0)]
    assert all(score_synthetic(codes, maps, weights) == expected for codes, expected in cases)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    con.execute("SET enable_progress_bar=false")
    con.execute(f"CREATE TEMP TABLE cohort AS SELECT id, AHI, age FROM read_parquet('{COHORT.as_posix()}')")
    n, distinct_n = con.execute("SELECT count(*),count(DISTINCT id) FROM cohort").fetchone()
    assert n == distinct_n == EXPECTED['Overall']
    schema = "columns={'id':'VARCHAR','ICD_CODE':'VARCHAR','ICD_TYPE':'INTEGER','AgeAtICD':'DOUBLE'}, header=true"
    source = f"read_csv('{ICD.as_posix()}', {schema})"
    unique_codes = con.execute(f"SELECT DISTINCT ICD_CODE, ICD_TYPE FROM {source}").df()
    assert unique_codes.ICD_TYPE.isin([9, 10]).all() and unique_codes.ICD_CODE.notna().all()
    normalized = unique_codes.ICD_CODE.str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    assert normalized.ne('').all()
    for name in weights:
        unique_codes[name] = False
        for system in (9, 10):
            mask = unique_codes.ICD_TYPE.eq(system)
            unique_codes.loc[mask, name] = normalized.loc[mask].str.startswith(maps[system][name])
    con.register("code_map", unique_codes)
    expressions = ", ".join(f"bool_or(m.{name}) AS {name}" for name in weights)
    con.execute(f"""CREATE TEMP TABLE person AS
        SELECT a.id, count(*) AS source_rows, {expressions},
            count(*) FILTER (WHERE a.AgeAtICD IS NULL) AS missing_event_age_n,
            count(*) FILTER (WHERE a.AgeAtICD*365.2425 > c.age*365.25+0.00001) AS after_index_age_n
        FROM {source} a
        JOIN cohort c ON a.id=c.id
        JOIN code_map m ON a.ICD_CODE=m.ICD_CODE AND a.ICD_TYPE=m.ICD_TYPE
        GROUP BY a.id""")
    audit = con.execute(f"""SELECT
      (SELECT count(*) FROM {source}) AS input_rows,
      (SELECT sum(source_rows) FROM person) AS matched_source_rows,
      (SELECT count(*) FROM person) AS people_with_source_coverage,
      (SELECT sum(missing_event_age_n) FROM person) AS missing_event_age_rows,
      (SELECT sum(after_index_age_n) FROM person) AS age_implied_post_index_rows""").df().iloc[0].to_dict()
    audit = {k:int(v) for k,v in audit.items()}
    assert audit['input_rows'] == audit['matched_source_rows']
    assert audit['people_with_source_coverage'] == EXPECTED['Overall']
    assert audit['missing_event_age_rows'] == audit['age_implied_post_index_rows'] == 0
    # Bring only needed local fields into local Python; never output records.
    frame = con.execute("SELECT c.AHI, p.* EXCLUDE (id) FROM cohort c LEFT JOIN person p USING(id)").df()
    assert frame[list(weights)].notna().all().all()
    before = frame[list(weights)].astype(bool).copy()
    for mild, severe in HIERARCHY.items():
        frame.loc[frame[severe], mild] = False
    frame['cci'] = sum(frame[name].astype(int) * weight for name, weight in weights.items())
    # Independently check every score via a SQL expression over the unmodified flags.
    score_terms = []
    for name, weight in weights.items():
        indicator = f"({name} AND NOT {HIERARCHY[name]})" if name in HIERARCHY else name
        score_terms.append(f"{weight}*CAST({indicator} AS INTEGER)")
    sql_histogram = dict(con.execute(f"SELECT {'+'.join(score_terms)} AS cci, count(*) FROM person GROUP BY cci ORDER BY cci").fetchall())
    python_histogram = {int(k):int(v) for k,v in frame.cci.value_counts().sort_index().items()}
    assert sql_histogram == python_histogram
    output = {
        'status':'newly_derived_baseline_cci_verified', 'publication_ready':True,
        'group_definition':'OSA AHI >=5; No OSA AHI <5',
        'method': {
            'index':'Non-age-adjusted weighted Charlson Comorbidity Index',
            'mapping':'Quan et al. (2005) ICD-9-CM and ICD-10 algorithms',
            'weights':'Original Charlson et al. (1987), not revised Quan 2011 weights',
            'component_weights':weights, 'hierarchy_zero_mild_if_severe':HIERARCHY,
            'window':'All available ICD-coded encounter history at or before the index sleep-study date; no additional fixed lookback restriction',
            'ascertainment':'One or more matching diagnosis codes defines a component; repeated codes and cross-system duplicates collapse to one component per person',
            'age_adjustment':False,
            'zero_interpretation':'Observed pre-index diagnosis history but no matching Charlson component; not an imputed score for absent source coverage',
            'baseline_filter_provenance':'03_extract_phenome.py restricts ADATE <= STUDY_DATE; v2_02_build_group_files.py subsets this artifact to v2 cohort without adding post-index records',
            'independent_temporal_check':'Every AgeAtICD converted to days using extraction divisor 365.2425 is <= cohort age in days using build divisor 365.25',
        },
        'input_audit':audit,
        'sources': {'cohort':str(COHORT), 'pre_index_icd':str(ICD)},
        'reference': {
            'repository':'https://github.com/ellessenne/comorbidity',
            'commit':(REF/'commit.txt').read_text().strip(),
            'mapping_documentation':'https://ellessenne.github.io/comorbidity/articles/B-comorbidity-scores.html',
            'hierarchy_documentation':'https://ellessenne.github.io/comorbidity/reference/score.html',
            'quan_2005_doi':'10.1097/01.mlr.0000182534.19832.83',
            'charlson_1987_doi':'10.1016/0021-9681(87)90171-8',
            'reference_file_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in REF.glob('*.R')},
        },
        'verification': {'synthetic_boundary_hierarchy_cases_passed':len(cases), 'sql_and_python_score_histograms_identical':True},
        'limitations': [
            'New descriptive baseline measure engineered for Table 1; not the stored temporal Charlson_Wt score and not used in existing model adjustment.',
            'Captures recorded ICD comorbidity, not chart-adjudicated disease. Zero means no matching recorded component.',
            'Uses all available pre-index history, whose duration varies between cohort members.',
            'Existing extraction used ignore_errors=true; upstream malformed-record handling is inherited rather than independently re-audited from 104 GB utilization source.',
        ],
        'groups': {},
    }
    for name, mask in {'Overall':pd.Series(True,index=frame.index),'OSA':frame.AHI.ge(5),'No OSA':frame.AHI.lt(5)}.items():
        values = frame.loc[mask,'cci']
        assert len(values) == EXPECTED[name]
        output['groups'][name] = {
            'cohort_n':len(values), 'usable_n':len(values), 'missing_source_n':0,
            'people_with_any_mapped_component_n':int(before.loc[mask].any(axis=1).sum()),
            'zero_score_n':int(values.eq(0).sum()),
            'mean':float(values.mean()),'sd':float(values.std(ddof=1)),
            'median':float(values.median()),'q1':float(values.quantile(.25)),'q3':float(values.quantile(.75)),
            'score_histogram':{str(k):int(v) for k,v in values.value_counts().sort_index().items()},
            'component_counts_before_hierarchy':{k:int(before.loc[mask,k].sum()) for k in weights},
            'component_counts_after_hierarchy':{k:int(frame.loc[mask,k].sum()) for k in weights},
            'mild_components_suppressed_by_hierarchy':{mild:int((before.loc[mask,mild]&before.loc[mask,severe]).sum()) for mild,severe in HIERARCHY.items()},
        }
    destination = HERE/'table1_charlson_baseline.json'
    destination.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':output['status'],'audit':audit,'group_summaries':{name:{k:v for k,v in g.items() if k in ['cohort_n','usable_n','missing_source_n','zero_score_n','mean','sd','median','q1','q3']} for name,g in output['groups'].items()},'output':str(destination)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Baseline CCI aggregation stopped safely ({type(exc).__name__}); no source values or records emitted.')
        raise SystemExit(1)
