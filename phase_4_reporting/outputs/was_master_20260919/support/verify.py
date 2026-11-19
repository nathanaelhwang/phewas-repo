import json, math, zipfile, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import openpyxl

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
filename=HERE.parent/'WAS_master_comprehensive.xlsx'
wb=openpyxl.load_workbook(filename,read_only=True,data_only=False)
assert wb.sheetnames == [s['was'] for s in manifest['sheets']]
counts=[]
for m in manifest['sheets']:
    payload=json.loads((HERE/(m['was']+'.json')).read_text(encoding='utf-8'))
    sh=wb[m['was']]
    iterator=sh.iter_rows(min_row=8,max_col=m['columns'],values_only=True)
    assert list(next(iterator))==payload['headers'],m['was']
    count=0
    for actual,expected in zip(iterator,payload['data'],strict=True):
        count+=1
        for c,(a,e) in enumerate(zip(actual,expected,strict=True)):
            if isinstance(e,(int,float)) and not isinstance(e,bool):
                assert isinstance(a,(int,float)) and not isinstance(a,bool),(m['was'],count,c,'numeric type')
                assert a==e or math.isclose(a,e,rel_tol=2e-14,abs_tol=0),(m['was'],count,c,'numeric mismatch')
            else:
                assert a==e,(m['was'],count,c,'value mismatch')
    assert count==m['rows']
    for source in m['sources']:
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
    counts.append({'sheet':m['was'],'rows':count,'all_cells_match':True})
    print(json.dumps(counts[-1]),flush=True)
wb.close()
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(filename) as z:
    for i,m in enumerate(manifest['sheets'],1):
        tree=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
        pane=tree.find('s:sheetViews/s:sheetView/s:pane',ns)
        assert pane is not None
        assert pane.get('xSplit')=='2' and pane.get('ySplit')=='8',(m['was'],pane.attrib)
        assert tree.find('s:sheetViews/s:sheetView',ns).get('showGridLines')=='0'
        assert len(tree.findall('s:tableParts/s:tablePart',ns))==1
        assert not tree.findall('.//s:c[@t="e"]',ns),m['was']
    tables=[n for n in z.namelist() if n.startswith('xl/tables/') and n.endswith('.xml')]
    assert len(tables)==9
    for p in tables:
        tree=ET.fromstring(z.read(p))
        assert tree.find('s:autoFilter',ns) is not None
report={'total_rows':sum(c['rows'] for c in counts),'sheets':counts,'native_tables':9,'freeze_panes':'2 columns and 8 rows','source_hashes_unchanged':True,'file_bytes':filename.stat().st_size}
(HERE/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
