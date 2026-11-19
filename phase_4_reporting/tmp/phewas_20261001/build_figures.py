"""Dated figure revision; original scripts and source assets remain unchanged."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
OUT=ROOT/'manuscript/figs-tables/20261001'

# Materialize the existing generator's string transformations, then apply this revision.
prior=(ROOT/'tmp/phewas_260923/build_figures.py').read_text(encoding='utf-8-sig')
prefix=prior[:prior.index("exec(compile(code,str(__file__),'exec'),globals())")]
scope={'__file__':str(__file__)}
exec(prefix,scope)
code=scope['code'].replace('manuscript/figs-tables/260923','manuscript/figs-tables/20261001')
code=code[:code.index('curve_code=')]
anchor='original_figure, original_save = p.figure, p.save'
patch='''
original_scatter_panel = p.scatter_panel
def revised_scatter(fig, box, rows, kind, effect, title, key, **kwargs):
    old = p.b.scatter
    if kind == 'volcano':
        title += '\\nFDR significant: solid; others: faint'
        def fdr_scatter(ax, rows, xvals):
            import numpy as np
            scale=np.quantile([r['magnitude'] for r in rows],.9) if rows else 1
            for row,x in zip(rows,xvals):
                fdr=bool(row['sig_fdr']); bon=bool(row['sig_bon'])
                ax.scatter([x],[row['plot_y']],s=9+29*min(row['magnitude']/max(scale,.001),1),
                           marker='^' if row['signed']>=0 else 'v',c=p.b.color(row['group']),
                           alpha=1 if fdr else .18,edgecolors='#202020' if bon else 'none',
                           linewidths=.4,zorder=3 if fdr else 2)
        p.b.scatter=fdr_scatter
    try:
        result=original_scatter_panel(fig,box,rows,kind,effect,title,key,**kwargs)
        if kind == 'volcano':
            p.AUDIT['scatter'][f'{key}_{kind}']['fdr_significant']=sum(bool(r['sig_fdr']) for r in rows)
            p.AUDIT['scatter'][f'{key}_{kind}']['all_points_retained']=True
        return result
    finally: p.b.scatter=old
p.scatter_panel=revised_scatter
'''
code=code.replace(anchor,patch+'\n'+anchor)
code+="\n(WORK/'association_audit.json').write_text(json.dumps(p.AUDIT,indent=2),encoding='utf-8')\n"
exec(compile(code,str(__file__),'exec'),globals())

# Figure 9: preserve all curve coordinates, add pairwise tables alongside every plot.
curve_code=(ROOT/'tmp/phewas_260919g/build_figure3.py').read_text(encoding='utf-8-sig')
curve_code=curve_code.replace('Figure 3','Figure 9').replace('figure3_panel','figure9_panel')
curve_code=curve_code.replace('OUT = Path(__file__).resolve().parent',"OUT = ROOT / 'manuscript/figs-tables/20261001'")
curve_code=curve_code.replace('ylim=(0, None)','ylim=(0, 10)')
curve_code=curve_code.replace('ax.set_xticks([0, 1, 2, 3])','ax.set_xticks([0, 1, 2, 3])\n    ax.set_yticks([0, 2, 4, 6, 8, 10])')
# Increase canvas width to retain the original physical plot widths alongside the statistics.
curve_code=curve_code.replace('figsize=(13, 18.4)','figsize=(18, 18.4)')
curve_code=curve_code.replace('.415, .16]', '.285, .16]').replace('.415, .145]', '.285, .145]')
curve_code=curve_code.replace('.07 + (i % 2) * .49', '.05 + (i % 2) * .50')
insert='''
tests=json.loads((ROOT/'tmp/phewas_20261001/pairwise_pvalues.json').read_text())
short={'No OSA':'None','Mild':'Mild','Moderate':'Mod','Severe':'Sev',
       '4+ hr/night':'4+ h','2-4 hr/night':'2–4 h','0-2 hr/night':'0–2 h',
       'never-started':'Never','unknown':'Unk'}
def annotate_pairwise(ax,panel):
    selected=[r for r in tests if r['panel']==panel]
    label=lambda p: f'{p:.1e}' if p<.001 else f'{p:.3f}'
    lines=['Pairwise p values']+[f"{short[r['group1']]} vs {short[r['group2']]}: {label(r['p'])}" for r in selected]
    ax.text(1.04,.99,'\\n'.join(lines),transform=ax.transAxes,ha='left',va='top',
            fontsize=8,linespacing=1.55)
'''
curve_code=curve_code.replace('audit = []',insert+'\naudit = []')
curve_code=curve_code.replace('    omitted = [entry', '    annotate_pairwise(ax,panel)\n    omitted = [entry')
curve_code=curve_code.replace('    table = ax.table(', '    annotate_pairwise(ax,panel)\n    table = ax.table(')
exec(compile(curve_code,'figure9_generator','exec'),{'__file__':str(__file__),'__name__':'figure9_generator'})

# Figure 12: color and associated color wording only.
heart_code=(ROOT/'tmp/phewas_260919i/build_figure10.py').read_text(encoding='utf-8-sig')
heart_code=heart_code.replace("NO_OSA = '#171717'", "NO_OSA = '#0072B2'")
heart_code=heart_code.replace('black dotted','blue dotted').replace('same black ', 'same blue ')
heart_code=heart_code.replace('FIGURE 10','FIGURE 12').replace('Figure 10','Figure 12').replace('figure10_panel','figure12_panel')
# Read the existing figure-10 caption before numbering its copied text.
heart_code=heart_code.replace("text.startswith('Figure 12.')", "text.startswith('Figure 10.')")
heart_code=heart_code.replace('    return caption', "    return caption.replace('Figure 10.', 'Figure 12.').replace('black dotted','blue dotted').replace('same black ', 'same blue ')")
heart_code=heart_code.replace('OUT = Path(__file__).resolve().parent', "OUT = ROOT / 'manuscript/figs-tables/20261001'")
exec(compile(heart_code,'figure12_generator','exec'),{'__file__':str(__file__),'__name__':'__main__'})
print('Requested figure revisions complete.')
