"""Rebuild Figure 7 with only safe five-year m4 AHI>=5 aggregate results."""
import ast
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(r'C:\Users\natha\Projects\work\phewas')
OUT=ROOT/'manuscript/figs-tables/20261001c'
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'results/behwas_results/analysis/results_all.csv'
with SOURCE.open(encoding='utf-8-sig',newline='') as handle:
    reader=csv.DictReader(handle)
    assert not {'id','MRN','DOB'}.intersection(reader.fieldnames)
    selected=[dict(r,source_line=i) for i,r in enumerate(reader,2)
              if r['window']=='5yr' and r['model']=='m4' and r['contrast']=='ahi_ge5']
assert len(selected)==8
rows=[]
labels={'adherent':'Medication adherence (PDC ≥0.80)',
        'adherence_pdc':'Medication PDC among users',
        'flu_received':'Influenza vaccination received',
        'flu_count':'Influenza vaccination count'}
for row in selected:
    if row['sig_bon']!='True':continue
    item=dict(row)
    for key in ['effect','ci_low','ci_high','p']:item[key]=float(row[key])
    item['group']='Medication adherence' if row['outcome'] in {'adherent','adherence_pdc'} else 'Vaccination'
    item['label']=labels[row['outcome']]
    transform=math.log2 if row['effect_type']=='OR' else lambda x:x
    item.update(signed=transform(item['effect']),signed_low=transform(item['ci_low']),
                signed_high=transform(item['ci_high']))
    item['magnitude']=abs(item['signed'])
    rows.append(item)
assert {r['outcome'] for r in rows}==set(labels)
assert all(r['window']=='5yr' for r in rows)

# Reuse the original figure's forest-layout function, without importing its pipeline.
prior=ROOT/'tmp/phewas_260919g/build_association_panels.py'
text=prior.read_text(encoding='utf-8')
node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='forest_axis')
function=ast.get_source_segment(text,node).replace("f'{2**v:.2g}'","f'{2**v:.3g}'")
colors={'Medication adherence':'#A89720','Vaccination':'#799E3A'}
b=SimpleNamespace(order_groups=lambda rs:sorted({r['group'] for r in rs}),
                  color=lambda group:colors[group],display=lambda group:group)
AUDIT={'forest_layout':[]}
exec(compile(function,str(prior),'exec'),globals())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,
                     'pdf.fonttype':42,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(13,2.6),dpi=150)
fig.text(.025,1-.25/2.6,'Figure 7  Five-year BehWAS',fontsize=16,fontweight='bold',va='top')
forest_h=1.1
for index,(kind,title,effect) in enumerate([('OR','Binary outcomes','OR'),('beta','Continuous outcomes','β')]):
    ax=fig.add_axes([(.4+index*6.45)/13,.70/2.6,5.95/13,forest_h/2.6])
    subset=[r for r in rows if r['effect_type']==kind]
    assert len(subset)==2
    forest_axis(ax,subset,effect,f'{chr(65+index)}  {title}',fontsize=8)
fig.text(.028,.05,'Medication adherence and PDC are among users. BehWAS remains preliminary.',fontsize=8)
for suffix in ['png','pdf']:
    fig.savefig(OUT/f'figure7_panel.{suffix}',dpi=300,facecolor='white')
plt.close(fig)
audit={'source':str(SOURCE),'window':'5yr','model':'m4','contrast':'ahi_ge5',
       'tested_features':8,'bonferroni_significant':4,'binary':2,'continuous':2,
       'lifetime_features':0,'rows':rows,'layout':AUDIT['forest_layout'],
       'width_inches':13,'height_inches':2.6,'colors':colors}
(OUT/'figure7_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('Generated Figure 7: four five-year m4 AHI>=5 features, zero lifetime features.')
