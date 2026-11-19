"""Regenerate the dated figure set with requested order and Bonferroni emphasis."""
from pathlib import Path
import inspect

ROOT=Path(__file__).resolve().parents[2]
prior=ROOT/'tmp/phewas_260922/build_figures.py'
code=prior.read_text(encoding='utf-8-sig')
code=code.replace("'manuscript/figs-tables/260922'", "'manuscript/figs-tables/260923'")
code=code.replace('renumber = {4: 5, 5: 6, 6: 7, 7: 8, 8: 9}',
                  'renumber = {2: 8, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}')
code=code.replace("('4A', 'Five-year LabWAS", "('2A', 'Five-year LabWAS")
code=code.replace("('4B', 'Five-year LabWAS", "('2B', 'Five-year LabWAS")
code=code.replace("('4C', 'Five-year LabWAS", "('2C', 'Five-year LabWAS")
# Remove only the two specialty Manhattan plots; retain their exact volcano and forest data.
override='''
util_source=inspect.getsource(p.utilization)
util_source=util_source.replace("    specialty_h=4.2", "    specialty_h=3.2")
util_source=util_source.replace("        scatter_panel(fig,(left,bottom+2.20,5.65,1.05),rows,'manhattan',effect,'Manhattan',key,small=True)\\n", "")
util_source=util_source.replace("(left,bottom,2.40,1.55)", "(left,bottom,2.40,2.1)")
util_source=util_source.replace("(left+2.82,bottom,2.83,1.55)", "(left+2.82,bottom,2.83,2.1)")
exec(util_source,p.__dict__)
'''
code=code.replace('p.utilization(util)',override+'\np.utilization(util)')
# Generate the existing AJ plots from their saved aggregate curves, with Figure 9 numbering.
start=code.index('# Keep Figure 3 exactly as supplied.')
end=code.index('# Re-render the existing phenotype signature',start)
code=code[:start]+'''
curve_code=(ROOT/'tmp/phewas_260919g/build_figure3.py').read_text(encoding='utf-8')
curve_code=curve_code.replace('Figure 3','Figure 9').replace('figure3_panel','figure9_panel')
curve_code=curve_code.replace('"number": 3','"number": 9')
curve_code=curve_code.replace('OUT = Path(__file__).resolve().parent', 'OUT = ROOT / "manuscript/figs-tables/260923"')
exec(compile(curve_code,'figure9_generator','exec'),{'__file__':__file__,'__name__':'figure9_generator'})
p.MANIFEST.extend(json.loads((OUT/'survival_manifest.json').read_text(encoding='utf-8')))
'''+code[end:]
code=code.replace("significant=[r for r in contrasts if r['sig_fdr']=='True']", "significant=[r for r in contrasts if r['sig_bon']=='True']")
code=code.replace('==(368,15,120,14,6)', '==(368,15,120,6,6)')
code=code.replace('figsize=(13,8.8)', 'figsize=(13,7.2)')
code=code.replace('all 14 FDR-significant one-versus-rest contrasts', 'all six Bonferroni-significant one-versus-rest contrasts')
code=code.replace('ax.set_ylim(-1,19)', 'ax.set_ylim(-1,12)').replace('y=18.2', 'y=11.2').replace('*(y+1)/20', '*(y+1)/13')
old="fig.legend(handles=[Line2D([],[],marker='o',mfc='white',color='#444444',ls='',label='FDR significant'),\n                    Line2D([],[],marker='D',color='#444444',ls='',label='Also Bonferroni significant')],"
new="fig.legend(handles=[Line2D([],[],marker='D',color='#444444',ls='',label='Bonferroni significant')],"
assert old in code
code=code.replace(old,new)
code=code.replace("'height_inches':8.8", "'height_inches':7.2")
code=code.replace('All 14 FDR-significant fully adjusted m4 cause-specific Cox contrasts', 'All six Bonferroni-significant fully adjusted m4 cause-specific Cox contrasts')
code=code.replace('Filled diamonds identify the six contrasts also significant after Bonferroni correction; open circles identify the remaining FDR-significant contrasts.',
                  'Filled diamonds identify Bonferroni-significant contrasts. Fourteen contrasts met the secondary FDR criterion and remain available in the Supplementary Atlas.')
code=code.replace("'Bonferroni and low-events-per-parameter indicators.'", "'Bonferroni and low-events-per-parameter indicators.'")
exec(compile(code,str(__file__),'exec'),globals())

# Captions describe the actual panel selection and primary multiplicity criterion.
import json
work=Path(__file__).resolve().parent
manifest=json.loads((work/'figure_manifest.json').read_text(encoding='utf-8'))
for r in manifest:
    if r['number']==1:
        r['caption']+=' C1 contains seven of the 29 positive Bonferroni-significant primary associations. The per-system cap is applied across both directions, so omitted positive findings can be significant; the forests are selected summaries, not exhaustive lists.'
    if r['number']==5:
        r['caption']=r['caption'].replace('Each row combines Manhattan, volcano, and forest plots.',
            'A–C combine Manhattan, volcano, and forest plots; D and E show volcano and forest plots only.')
        r['caption']+=' Specialty Manhattan plots were omitted because the source exports did not include a validated surgical, obstetric/gynecologic, and nonsurgical domain classification.'
    if r['number']==9:
        r['caption']=r['caption'].replace('For essential hypertension, the fully adjusted PAP-use comparisons were FDR-null.',
            'The fully adjusted PAP-use comparisons for these four outcomes were not significant after Bonferroni correction or the secondary FDR correction.')
(work/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Final order: 1 prevalence; 2 LabWAS; 3 MedWAS; 4 ProcWAS; 5 UtilWAS; 6 QWAS; 7 BehWAS; 8 incidence; 9 AJ; 10–12 phenotypes.')
