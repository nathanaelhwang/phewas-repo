"""Generate distribution figures from existing aggregate analysis exports."""
from pathlib import Path
import importlib.util
import csv
import json
import shutil
from collections import Counter
import textwrap
import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'manuscript/figs-tables/260922'
OUT.mkdir(parents=True, exist_ok=True)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def read(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

p = module('panels_20260922', 'tmp/phewas_260919g/build_association_panels.py')
p.OUT = OUT
# Narrow ratio ranges need three significant digits to avoid duplicate tick labels.
forest_source = inspect.getsource(p.forest_axis).replace("f'{2**v:.2g}'", "f'{2**v:.3g}'")
exec(forest_source, p.__dict__)
renumber = {4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
original_figure, original_save = p.figure, p.save
def figure(number, title, height):
    return original_figure(renumber.get(number, number), title, height)
def save(fig, number, caption):
    new = renumber.get(number, number)
    caption = caption.replace(f'Figure {number}.', f'Figure {new}.', 1)
    caption = caption.replace('supplemental atlas', 'Supplementary Atlas')
    return original_save(fig, new, caption)
p.figure, p.save = figure, save

prevalence, incidence = p.b.source.load_disease_data()
data = p.b.source.base.load_data()
med = [r for r in data['med'] if r['window'] == '5yr']
proc = [r for r in data['proc'] if r['window'] == '5yr']
qwas = data['qwas']
beh = [r for r in data['beh'] if r['window'] in {'5yr', 'lifetime'}]
for r in beh:
    o = r['outcome']
    r['group'] = ('Medication adherence' if o in {'adherence_pdc','adherent'} else
                  'Vaccination' if o.startswith('flu_') else 'Attendance' if o in {'any_noshow','no_show_pct'} else 'Work leave')
util = json.loads((ROOT / 'tmp/phewas_260919g/utilwas_rows.json').read_text(encoding='utf-8'))
labels = {r['loinc_code']: r['loinc_long_common_name'] for r in read('results/labwas_results/loinc_category.csv')}
lab_raw = read('results/labwas_results/analysis/results_5yr.csv')
prop_raw = read('results/labwas_results/analysis/results_propensity_5yr.csv')
labs = {}
for target, rows in [('median', lab_raw), ('order_rate', lab_raw), ('any_order', prop_raw)]:
    selected = []
    for r in rows:
        if r['model'] != 'm4' or r['contrast'] != 'ahi_ge5' or r.get('target', 'any_order') != target:
            continue
        x = p.b.source.base.enrich(r)
        if x:
            x.update(feature=r['loinc_code'], label=labels[r['loinc_code']], group=r['category'])
            selected.append(x)
    labs[target] = selected
assert {k:len(v) for k,v in labs.items()} == {'median':185,'order_rate':186,'any_order':143}
assert {k:sum(r['sig_fdr'] for r in v) for k,v in labs.items()} == {'median':99,'order_rate':65,'any_order':125}
p.b.init_colors(prevalence + incidence + med + proc + qwas + beh +
                [r for rs in util['groups'].values() for r in rs] + [r for rs in labs.values() for r in rs])

# Preserve the extra prevalence labels requested in the latest figure revision.
default_labels = p.chosen_labels
extra = ['401.1','402','278.1','272.1','627.2']
def prevalence_labels(rows, kind='manhattan'):
    chosen = default_labels(rows, kind)
    if kind == 'volcano':
        present = {r['code'] for r in chosen}
        chosen += [next(r for r in rows if r['code'] == code) for code in extra if code not in present]
    return chosen
p.chosen_labels = prevalence_labels
p.single_was(1, 'PheDAS Prevalence', prevalence, 'OR', 'prevalence',
    'Fully adjusted m4_bmi primary AHI ≥5 versus <5 estimates for 1,779 phecodes, excluding sleep apnea and subcodes. '
    'Panel B additionally labels essential hypertension, elevated blood pressure, obesity, hyperlipidemia, and symptomatic menopause. '
    'Complete results: Supplementary Atlas, PheDAS Prevalence tab.')
p.chosen_labels = default_labels
p.single_was(2, 'PheDAS Incidence', incidence, 'HR', 'incidence',
    'Fully adjusted m4 primary AHI ≥5 versus <5 estimates for 1,012 phecodes, excluding sleep apnea and subcodes. '
    'Complete results: Supplementary Atlas, PheDAS Incidence tab.')
print('Figures 1 and 2 generated.', flush=True)

lab_desc = {
    'median': ('4A', 'Five-year LabWAS median laboratory values', 'β',
        '185 analytes with estimable fully adjusted m4 rank-inverse-normal β estimates among measured patients.'),
    'order_rate': ('4B', 'Five-year LabWAS laboratory ordering rates', 'IRR',
        '186 analytes with estimable fully adjusted m4 ordering-rate IRRs among patients with laboratory observations.'),
    'any_order': ('4C', 'Five-year LabWAS any-order probability', 'OR',
        '143 analytes with estimable fully adjusted m4 odds ratios for any laboratory order; four non-estimable rows are omitted.'),
}
for target, (number,title,effect,description) in lab_desc.items():
    p.single_was(number,title,labs[target],effect,'lab_'+target,
        description + ' All contrasts compare AHI ≥5 versus <5. Colors identify laboratory categories. '
        'Laboratory values and ordering outcomes are separate estimands and retain their original multiplicity flags. '
        'Complete 1- and 5-year results: Supplementary Atlas, LabWAS tab.')
print('LabWAS Figure 4A–C generated.', flush=True)
p.single_was(4, 'Five-year MedWAS', med, 'OR', 'med',
    'All 218 medication classes; fully adjusted m4 AHI ≥5 versus <5. Complete results: Supplementary Atlas, MedWAS tab.')
p.single_was(5, 'Five-year ProcWAS', proc, 'IRR', 'proc',
    'All 67 procedures; fully adjusted m4 AHI ≥5 versus <5. Complete results: Supplementary Atlas, ProcWAS tab.')
p.utilization(util)
p.MANIFEST[-1]['caption'] += ' Complete results: Supplementary Atlas, UtilWAS tab.'
# Keep the latest QWAS label count and wording.
original_scatter = p.scatter_panel
def qscatter(fig, box, rows, kind, effect, title, key, **kwargs):
    if kind == 'manhattan' and key == 'qwas_continuous_outcomes':
        rows = [dict(r, label='Sitting, inactive, in a public place') if r['feature']=='ESS3' else r for r in rows]
    return original_scatter(fig,box,rows,kind,effect,title,key,**kwargs)
p.scatter_panel = qscatter
p.chosen_labels = lambda rows,kind='manhattan': default_labels(rows,kind)[:3] if kind=='volcano' else default_labels(rows,kind)
p.SCATTER_NOTE = p.SCATTER_NOTE.replace('Volcano plots label up to five','Volcano plots label up to three')
p.mixed_was(7,'QWAS',[
    ('Binary outcomes',[r for r in qwas if r['form']=='binary'],'OR'),
    ('Continuous outcomes',[r for r in qwas if r['form']=='continuous'],'β')],
    'All 83 binary and 49 continuous rows; fully adjusted m4 AHI ≥5 versus <5. Complete results: Supplementary Atlas, QWAS tab.',
    '* Referral-driving symptoms. Higher FOSQ scores indicate better function.')
p.behavior_forests([('Binary outcomes',[r for r in beh if r['effect_type']=='OR'],'OR'),
                    ('Continuous outcomes',[r for r in beh if r['effect_type']=='beta'],'β')])
p.MANIFEST[-1]['caption'] += ' Complete results: Supplementary Atlas, BehWAS tab.'
print('Association figures 5–9 generated.', flush=True)

# Keep Figure 3 exactly as supplied.
curve_manifest = json.loads((ROOT/'tmp/phewas_260919g/survival_manifest.json').read_text(encoding='utf-8'))
for rec in curve_manifest:
    src=Path(rec['file']); dest=OUT/src.name; shutil.copy2(src,dest); rec['file']=str(dest)
    p.MANIFEST.append(rec)

# Re-render the existing phenotype signature with the new figure number.
legacy = p.b.source
legacy.OUT = OUT
legacy.base.OUT = OUT
old_header = legacy.draw_header
legacy.draw_header = lambda image,kicker,*args,**kwargs: old_header(image,kicker.replace('Figure 11','Figure 10'),*args,**kwargs)
heatmap = legacy.build_figure11()
heatmap_path=OUT/'figure10_panel.png'
(OUT/heatmap['file']).replace(heatmap_path)
old_signature=json.loads((ROOT/'tmp/phewas_260919e/phenotype_manifest.json').read_text(encoding='utf-8'))[0]
old_signature.update(number=10,file=str(heatmap_path),caption=old_signature['caption'].replace('Figure 9.','Figure 10.'))
p.MANIFEST.append(old_signature)

# A complete forest of all 14 significant gated phenotype contrasts.
raw=read('results/incwas_results/phenotype_exposure/results_octant_phecode.csv')
m4=[r for r in raw if r['model']=='m4']
omnibus=[r for r in m4 if r['contrast']=='omnibus']
contrasts=[r for r in m4 if r['contrast'].startswith('one_vs_rest:')]
significant=[r for r in contrasts if r['sig_fdr']=='True']
assert (len(omnibus),sum(float(r['p'])<.05 for r in omnibus),len(contrasts),len(significant),sum(r['sig_bon']=='True' for r in significant))==(368,15,120,14,6)
colors={'high-all':'#A33E3E','symptom-predominant':'#2879A1','symptomatic-comorbid':'#7B5698'}
names={'high-all':'High All','symptom-predominant':'Symptom Predominant','symptomatic-comorbid':'Symptomatic Comorbid'}
fig=plt.figure(figsize=(13,8.8))
fig.text(.035,.96,'Figure 11  Phenotype-associated disease incidence',fontsize=17,weight='bold')
fig.text(.035,.925,'Fully adjusted m4 hazard ratios for all 14 FDR-significant one-versus-rest contrasts',fontsize=11)
ax=fig.add_axes([.57,.15,.25,.70]); ax.set_xscale('log'); ax.set_xlim(.85,3.1)
ax.set_ylim(-1,19); ax.set_yticks([]); ax.set_xticks([1,1.5,2,3],['1','1.5','2','3'])
ax.axvline(1,color='#8A8A8A',lw=1,ls='--');ax.set_xlabel('Hazard ratio (log scale)',fontsize=10)
ax.spines[['top','right','left']].set_visible(False)
fig.text(.04,.87,'Focal phenotype / incident outcome (phecode)',fontsize=10,weight='bold')
fig.text(.88,.87,'HR (95% CI)',fontsize=10,weight='bold',ha='center')
y=18.2
for key in colors:
    fy=.15+.70*(y+1)/20
    fig.text(.04,fy,names[key],color=colors[key],fontsize=11,weight='bold',va='center')
    y-=1.1
    rows=sorted([r for r in significant if r['contrast'].endswith(':'+key)],key=lambda r:-float(r['effect']))
    for r in rows:
        hr,lo,hi=[float(r[k]) for k in ['effect','ci_low','ci_high']]
        ax.errorbar(hr,y,xerr=[[hr-lo],[hi-hr]],fmt='D' if r['sig_bon']=='True' else 'o',
                    color=colors[key],mfc=colors[key] if r['sig_bon']=='True' else 'white',ms=6,capsize=3,lw=1.3)
        fy=.15+.70*(y+1)/20
        label=r['phenotype'].replace('Neuralgia, neuritis, and radiculitis NOS','Neuralgia / neuritis / radiculitis NOS')
        label+=f" ({r['outcome']})"+(' †' if 'epv<10' in r['unstable'] else '')
        fig.text(.052,fy,label,fontsize=9.5,va='center')
        fig.text(.88,fy,f'{hr:.2f} ({lo:.2f}–{hi:.2f})',fontsize=10,ha='center',va='center')
        y-=.92
    y-=.45
fig.legend(handles=[Line2D([],[],marker='o',mfc='white',color='#444444',ls='',label='FDR significant'),
                    Line2D([],[],marker='D',color='#444444',ls='',label='Also Bonferroni significant')],
           loc='lower left',bbox_to_anchor=(.035,.052),frameon=False,ncol=2,fontsize=10)
fig.text(.04,.035,'† Fewer than 10 events per model parameter. Each focal phenotype is compared with its pooled other seven phenotypes.',fontsize=9)
for ext in ['png','pdf']:fig.savefig(OUT/f'figure11_panel.{ext}',dpi=300,facecolor='white')
plt.close(fig)
p.MANIFEST.append({'number':11,'file':str(OUT/'figure11_panel.png'),'width_inches':13,'height_inches':8.8,
    'page_width_inches':14,'page_height_inches':11,
    'caption':'Figure 11. Phenotype-associated disease incidence. All 14 FDR-significant fully adjusted m4 cause-specific Cox contrasts are shown with 95% CIs. '
    'Of 368 eligible phecodes, 15 passed the nominal 7-df omnibus gate (P<0.05), generating 120 one-versus-rest contrasts. '
    'Each focal phenotype is compared with its pooled other seven phenotypes. Benjamini–Hochberg FDR and Bonferroni corrections were applied within each focal-phenotype family across the 15 gated outcomes. '
    'Filled diamonds identify the six contrasts also significant after Bonferroni correction; open circles identify the remaining FDR-significant contrasts. '
    'The dashed line marks HR=1; daggers identify fewer than ten events per model parameter. Estimates are observational and do not establish causal effects. '
    'All 120 contrasts are in the Supplementary Atlas, PheDAS Incidence tab (phenotype analyses).'})

# Re-render latest nine-panel heart-failure curves with all saved CIs intact.
heart=module('heart_20260922','tmp/phewas_260919i/build_figure10.py')
heart.OUT=OUT/'heart_source';heart.OUT.mkdir(exist_ok=True)
original_text=plt.Figure.text
def numbered_text(self,x,y,s,*args,**kwargs):
    return original_text(self,x,y,s.replace('FIGURE 10','FIGURE 12'),*args,**kwargs)
plt.Figure.text=numbered_text
heart.main()
plt.Figure.text=original_text
rec=json.loads((heart.OUT/'phenotype_manifest.json').read_text(encoding='utf-8'))[0]
for ext in ['png','pdf']:shutil.copy2(heart.OUT/f'figure10_panel.{ext}',OUT/f'figure12_panel.{ext}')
rec.update(number=12,file=str(OUT/'figure12_panel.png'),caption=rec['caption'].replace('Figure 10.','Figure 12.'))
p.MANIFEST.append(rec)
order=lambda r: float(str(r['number']).replace('A','.1').replace('B','.2').replace('C','.3'))
p.MANIFEST.sort(key=order)
(WORK/'figure_manifest.json').write_text(json.dumps(p.MANIFEST,indent=2,ensure_ascii=False),encoding='utf-8')
(WORK/'figure_audit.json').write_text(json.dumps({'plot_audit':p.AUDIT,'lab_counts':{k:len(v) for k,v in labs.items()},
    'phenotype_forest':significant,'counts':{'tested':368,'gated':15,'contrasts':120,'fdr':14,'bonferroni':6}},indent=2,ensure_ascii=False),encoding='utf-8')
print('All 12 figures generated; Figure 4 has three separate estimand pages.',flush=True)
