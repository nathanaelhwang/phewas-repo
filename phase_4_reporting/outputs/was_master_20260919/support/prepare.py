import csv, json, math, re, hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sources = []
def add(was, pattern, analysis, window=''):
    for p in sorted(ROOT.glob(pattern)):
        assert not any(x in p.as_posix().lower() for x in ['data/raw/', '/phi/', '/patient_records/', '_identified', '/.ssh/', '/.aws/'])
        with p.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert not any(re.search(r'(^|_)(mrn|ssn|dob|patient_id|patient_name|address)($|_)', h, re.I) for h in headers), p
            rows = list(reader)
        sources.append(dict(was=was,path=p.relative_to(ROOT).as_posix(),analysis=analysis,window=window,headers=headers,rows=rows))

add('PheWAS','results/icd_analysis_v2_results/summary/phewas_results_combined.csv','Primary L1 logistic','lifetime')
add('PheWAS','results/icd_analysis_v2_results/sensitivity_ipw/ipw_severe_vs_none_m4.csv','IPW sensitivity','lifetime')
add('IncWAS','results/incwas_results/summary/cox_results_combined.csv','OSA severity Cox','follow-up')
for days in [90,180]:
    add('IncWAS',f'results/incwas_results/analysis/results_cpap_landmark_{days}.csv','CPAP landmark',f'{days}-day landmark')
add('IncWAS','results/incwas_results/analysis/results_cpap_landmark_death.csv','CPAP landmark mortality')
for suffix in ['tierA','tierB','system_anynew','system_strict','octant_phecode','octant_system']:
    add('IncWAS',f'results/incwas_results/phenotype_exposure/results_{suffix}.csv',f'Phenotype {suffix}','follow-up')
for window in ['1yr','5yr']:
    add('LabWAS',f'results/labwas_results/analysis/results_{window}.csv','Lab values and order rate',window)
    add('LabWAS',f'results/labwas_results/analysis/results_propensity_{window}.csv','Lab order propensity',window)
    add('LabWAS',f'results/labwas_results/analysis/twopart_{window}.csv','Two-part combined',window)
    add('MedWAS',f'results/medwas_results/analysis/results_{window}.csv','Medication',window)
    add('ProcWAS',f'results/procwas_results/analysis/results_{window}.csv','Procedure feature',window)
    add('ProcWAS',f'results/procwas_results/analysis/results_total_{window}.csv','Procedure total',window)
    add('UtilWAS',f'results/utilwas_results/analysis/results_phecode_{window}.csv','Phecode utilization',window)
    add('UtilWAS',f'results/utilwas_results/analysis/results_specialty_{window}.csv','Specialty utilization',window)
add('MeasWAS','results/procwas_results/measwas/analysis/results_5yr.csv','Cardiopulmonary measurements','5yr')
add('QWAS','results/qwas_results/analysis/results.csv','Questionnaire','index-anchored')
add('BehWAS','results/behwas_results/analysis/results_all.csv','Behavior')

def read_csv(path, skip=0, line_numbers=False):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:
        for _ in range(skip): next(f)
        reader=csv.DictReader(f)
        return [dict(r,_regression_row=reader.line_num+skip) if line_numbers else r for r in reader]

# Verify that the primary combined files cover their full component exports.
checks = []
primary = sources[0]
raw_phe = {}
for p in sorted(ROOT.glob('results/icd_analysis_v2_results/pyphewas/regressions_*.csv')):
    contrast, model = p.stem.removeprefix('regressions_').split('__')
    for r in read_csv(p.relative_to(ROOT),1,True):
        key = (r['PheWAS Code'],contrast,model)
        assert key not in raw_phe
        raw_phe[key] = (r,p.relative_to(ROOT).as_posix())
primary_keys={(r['PheWAS Code'],r['contrast'],r['model']) for r in primary['rows']}
print(json.dumps({'raw_phe_rows':len(raw_phe),'combined_phe_rows':len(primary['rows']),'omitted_from_combined':len(set(raw_phe)-primary_keys)}))
assert primary_keys <= set(raw_phe)
for r in primary['rows']:
    original, p = raw_phe[(r['PheWAS Code'],r['contrast'],r['model'])]
    for h,v in original.items():
        if h=='_regression_row': continue
        if h in r and r[h] != v:
            try: assert math.isclose(float(r[h]),float(v),rel_tol=1e-12,abs_tol=0)
            except (ValueError,AssertionError):
                assert h == 'note' or (not r[h] and not v), (h,p)
        elif h not in r:
            r[h] = v
            if h not in primary['headers']: primary['headers'].append(h)
    r['regression_source'] = p
primary['headers'].append('regression_source')
missing_by_file={}
for key in sorted(set(raw_phe)-primary_keys):
    r,p=raw_phe[key]
    r=dict(r,contrast=key[1],model=key[2])
    missing_by_file.setdefault(p,[]).append(r)
for p,rs in missing_by_file.items():
    sources.append(dict(was='PheWAS',path=p,analysis='Primary L1 omitted from summary',window='lifetime',headers=[h for h in rs[0] if h!='_regression_row'],rows=rs))
checks.append({'check':'All 20 PheWAS regressions covered, including 2,017 omitted records','rows':len(raw_phe)})
inc = next(s for s in sources if s['analysis']=='OSA severity Cox')
original_inc = read_csv('results/incwas_results/analysis/results_cox.csv')
def keys(rows, fields): return Counter(tuple(r.get(h,'') for h in fields) for r in rows)
assert keys(inc['rows'],['phecode','contrast','model']) == keys(original_inc,['phecode','contrast','model'])
checks.append({'check':'IncWAS combined versus full Cox export','rows':len(original_inc)})
beh = next(s for s in sources if s['was']=='BehWAS')
beh_parts = sum((read_csv(f'results/behwas_results/analysis/results_{w}.csv') for w in ['1yr','5yr','lifetime']),[])
assert keys(beh['rows'],beh['headers']) == keys(beh_parts,beh['headers'])
checks.append({'check':'BehWAS combined versus all time-window exports','rows':len(beh_parts)})

labs = {r['loinc_code']:r for r in read_csv('results/labwas_results/loinc_category.csv')}
meds = {r['code']:r for r in read_csv('results/medwas_results/gpi_category.csv')}
text_columns = {'feature_id','feature_name','phecode','PheWAS Code','PheWAS Name','phenotype','phecode_name','loinc_code','code','gpi2','feature','concept','outcome','target','category','category_string','domain','form','referral_item','family','combo','part','procedure_type','body_system','window','contrast','model','kind','effect_type','note','ICD-9','ICD-10','analysis_variant','source_file','regression_source','label_source','drug_class_name','drug_group_name','metadata_source','numeric_source_text','record_status','Conf-interval beta','rule'}
numeric_re = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?$')
def typed(h,v,exceptions):
    if v is None or v=='': return None
    if h in text_columns: return str(v)
    if str(v).lower() in ['true','false']: return str(v).lower()=='true'
    if numeric_re.match(str(v)):
        n = float(v)
        if not math.isfinite(n) or (n != 0 and abs(n) < 2.2250738585072014e-308):
            exceptions[h] = v
            return str(v)
        if n.is_integer() and abs(n)<2**53: return int(n)
        return n
    return str(v)

order=['PheWAS','IncWAS','LabWAS','MedWAS','ProcWAS','MeasWAS','QWAS','BehWAS','UtilWAS']
manifest=[]
for was in order:
    group=[s for s in sources if s['was']==was]
    rows=[]
    all_headers=[]
    for s in group:
        for h in s['headers']:
            if h not in all_headers: all_headers.append(h)
        for i,original in enumerate(s['rows'],2):
            r=dict(original)
            feature=next((r[k] for k in ['PheWAS Code','phecode','loinc_code','code','feature','outcome','body_system'] if r.get(k)), 'all_cause_mortality')
            name=next((r[k] for k in ['PheWAS Name','phenotype','phecode_name','concept'] if r.get(k)), '')
            if was=='LabWAS':
                name=labs.get(feature,{}).get('loinc_long_common_name','')
                r['metadata_source']='results/labwas_results/loinc_category.csv'
            if was=='MedWAS':
                name=meds.get(feature,{}).get('drug_class_name','')
                for h in ['drug_group_name','category','label_source','needs_review']:
                    r[h]=meds.get(feature,{}).get(h,'')
                r['metadata_source']='results/medwas_results/gpi_category.csv'
            source_row=r.pop('_regression_row',i)
            r.update(feature_id=feature,feature_name=name or feature,analysis_variant=s['analysis'],source_file=s['path'],source_row=source_row)
            r['window']=r.get('window') or s['window']
            if s['analysis']=='IPW sensitivity':
                r.update(contrast='severe_vs_none',model='m4_IPW',effect_type='OR',effect=r['OR_ipw'],ci_low=r['OR_ipw_low'],ci_high=r['OR_ipw_high'],p=r['p_ipw'])
            if s['analysis']=='Primary L1 logistic':
                r.update(effect_type='OR',effect=r['OR'],se=r['std_error'],ci_low=r['OR_low'],ci_high=r['OR_high'],p=r['p-val'],category=r['category_string'])
            if s['analysis']=='Primary L1 omitted from summary':
                r.update(effect_type='beta_log_odds',effect=r.get('beta',''),se=r.get('std_error',''),p=r.get('p-val',''),category=r.get('category_string',''))
            if s['analysis']=='CPAP landmark mortality':
                r.update(window=f"{r['window_days']}-day landmark",model='m4 + severity',contrast=r['level'],effect_type='HR',effect=r['HR'],feature_name='All-cause mortality')
            p=r.get('p','')
            r['record_status']='p unavailable' if p in ['',None,'nan','NaN'] else 'Reported'
            if s['analysis']=='Two-part combined': r['record_status']='Combined component summary'
            if str(r.get('unstable','')).lower() not in ['', 'false', '0', 'nan', 'none']: r['record_status']='Flagged unstable in source'
            if s['analysis']=='Primary L1 omitted from summary': r['record_status']='Omitted from combined summary; see note'
            exceptions={}
            r={h:typed(h,v,exceptions) for h,v in r.items()}
            r['numeric_source_text']='; '.join(f'{h}={v}' for h,v in exceptions.items()) or None
            rows.append(r)
    front=['feature_id','feature_name','contrast','model','window','analysis_variant']
    sub=[h for h in ['target','kind','form','part','combo','family','domain','category','category_string','effect_type','effect','se','ci_low','ci_high','p','sig_bon','sig_fdr','unstable','record_status','n','n_atrisk','n_events'] if any(h in r for r in rows)]
    tail=['numeric_source_text','metadata_source','source_file','source_row','regression_source']
    headers=front+sub+[h for h in all_headers if h not in front+sub+tail]
    for r in rows:
        for h in r:
            if h not in headers+tail: headers.append(h)
    headers += [h for h in tail if any(h in r for r in rows)]
    rows.sort(key=lambda r:tuple(str(r.get(h) or '') for h in ['feature_id','analysis_variant','window','target','kind','form','part','combo','family','contrast','model']))
    data=[[r.get(h) for h in headers] for r in rows]
    report=dict(was=was,rows=len(rows),columns=len(headers),features=len(set(r['feature_id'] for r in rows)),models=sorted(set(r.get('model') or '' for r in rows)),contrasts=sorted(set(r.get('contrast') or '' for r in rows)),numeric_exceptions=sum(bool(r['numeric_source_text']) for r in rows),sources=[{'path':s['path'],'rows':len(s['rows']),'sha256':hashlib.sha256((ROOT/s['path']).read_bytes()).hexdigest()} for s in group])
    (OUT/f'{was}.json').write_text(json.dumps({'headers':headers,'data':data,'report':report},allow_nan=False),encoding='utf-8')
    manifest.append(report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources','contrasts']}))
(OUT/'manifest.json').write_text(json.dumps({'sheets':manifest,'checks':checks},indent=2),encoding='utf-8')
