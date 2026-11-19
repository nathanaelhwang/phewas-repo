import csv,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
manifest=json.loads((HERE/'manifest.json').read_text())
total=0
for meta in manifest['sheets']:
    data=json.loads((HERE/(meta['was']+'.json')).read_text())
    headers=data['headers']
    lookup={}
    for row in data['data']:
        r=dict(zip(headers,row))
        lookup.setdefault(r['source_file'],{})[r['source_row']]=r
    for file,selected in lookup.items():
        with (ROOT/file).open(encoding='utf-8-sig',newline='') as f:
            is_regression='/pyphewas/regressions_' in file
            if is_regression: next(f)
            reader=csv.DictReader(f)
            for src in reader:
                source_row=reader.line_num + int(is_regression)
                if source_row not in selected:
                    assert is_regression
                    continue
                dest=selected[source_row]
                for h,s in src.items():
                    assert h in dest,(meta['was'],h,'missing column')
                    v=dest[h]
                    if s=='': assert v is None,(file,source_row,h)
                    elif isinstance(v,bool): assert s.lower()==str(v).lower()
                    elif isinstance(v,(int,float)):
                        assert float(s)==v or math.isclose(float(s),v,rel_tol=2e-14,abs_tol=0),(file,source_row,h)
                    else: assert str(v)==s,(file,source_row,h)
                total+=1
    print(json.dumps({'sheet':meta['was'],'source_columns_preserved':True}),flush=True)
assert total==sum(m['rows'] for m in manifest['sheets'])
print(json.dumps({'source_records_verified':total}))
