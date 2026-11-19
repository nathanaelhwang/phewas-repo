import json, math
from pathlib import Path
import xlsxwriter

HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'manifest.json').read_text())
wb=xlsxwriter.Workbook(HERE.parent/'WAS_master_comprehensive.xlsx',{'strings_to_formulas':False,'strings_to_urls':False})
base={'font_name':'Arial','font_size':10,'font_color':'#182B3A','valign':'vcenter'}
formats={}
def fmt(number='@',wrap=False,shade=False,boolean=False):
    key=(number,wrap,shade,boolean)
    if key not in formats:
        formats[key]=wb.add_format(base|{'num_format':number,'text_wrap':wrap,'bg_color':'#EDF3F8' if shade else '#FFFFFF','align':'center' if boolean else 'right' if number!='@' else 'left'})
    return formats[key]
title=wb.add_format(base|{'font_size':15,'bold':True})
context=wb.add_format(base|{'font_color':'#506272'})
header=wb.add_format(base|{'bold':True,'font_color':'#FFFFFF','bg_color':'#294765','text_wrap':True,'align':'center'})
definitions={
'PheWAS':'m1 crude; m2 + age, sex, race, year; m3 + smoking, income, observation window; m4 + BMI. IPW sensitivity is labeled separately.',
'IncWAS':'Severity: m1 crude; m2 + demographics; m3 + smoking, income, observation window, encounter rate; m4 + BMI. CPAP models also adjust for severity.',
'LabWAS':'m1 crude; m3 + demographics, year, smoking, income; m4 + BMI. Value/propensity models also include observation window; order-rate models use an offset.',
'MedWAS':'m1 crude; m3 + demographics, year, smoking, income, observation window; m4 + BMI. Unadjusted screening tests retained separately.',
'ProcWAS':'m1 crude; m2 + demographics and year; m3 + smoking and income; m4 + BMI. Procedure rates use the observation-time offset.',
'MeasWAS':'m1 crude; m2 + demographics, year, observation window; m3 + smoking and income; m4 + BMI. Adjusted beta is on the rank-normal scale.',
'QWAS':'m1 crude; m2 + demographics and year; m3 + smoking, income, observation window; m4 + BMI. Binary and continuous forms are separate records.',
'BehWAS':'m1 crude; m3 + demographics, year, smoking, income, observation window; m4 + BMI. Binary and continuous forms are separate records.',
'UtilWAS':'m1 crude; m3 + demographics, year, smoking, income; m4 + BMI. Binary models add observation window; rate models use an observation-time offset.'}
caveats={
'PheWAS':'Includes 2,017 records omitted from the combined summary. See record_status and note. Their original beta and beta interval are retained; ORs are not invented.',
'IncWAS':'Phenotype ladder: m3a excludes encounter rate; m3b adds it; m4 adds BMI. Tier, body-system, octant, and CPAP analyses remain distinct.',
'LabWAS':'Two-part combined rows repeat selected component statistics for comparison; they are not additional independent tests. Filter analysis_variant before counting tests.',
'MeasWAS':'Complete-case estimates among measured patients. PFT results are exploratory. Echo and separate EF-source measures remain distinct.',
'QWAS':'referral_item retains the source flag for questionnaire items related to referral. Each feature and form retains its own sample size.'}
for m in manifest['sheets']:
    p=json.loads((HERE/(m['was']+'.json')).read_text())
    headers,data=p['headers'],p['data']
    sh=wb.add_worksheet(m['was'])
    sh.hide_gridlines(2)
    sh.set_tab_color('#294765')
    sh.freeze_panes(8,2)
    sh.set_default_row(18)
    sh.set_row(0,27)
    notes=[m['was']+' — complete available results',f"{m['rows']:,} records · {m['features']:,} feature IDs · {len(m['sources'])} source exports · Current results snapshot: September 19, 2026",'Filter by feature_id, contrast (exposure), model (adjustment), window, and analysis_variant. All source result columns and reported rows are retained.',definitions[m['was']],caveats.get(m['was'],'Only observed model combinations are included. Missing values and unrun adjustment levels are unavailable; no results are imputed.'),'Original blanks stay blank; source zero p-values stay zero. Nonzero numbers below Excel’s numeric range remain text and are listed in numeric_source_text.','Row provenance is in source_file and source_row. Significance and instability flags are copied from their source analysis; source_row is the CSV line number.']
    for i,note in enumerate(notes): sh.write_string(i,0,note,title if i==0 else context)
    widths=[]; wraps=[]; number_formats=[]
    for j,h in enumerate(headers):
        width=max(14,min(25,len(h)+3))
        if h=='feature_id': width=23
        if h in ['feature_name','phenotype','PheWAS Name','phecode_name']: width=54
        if h=='contrast': width=48
        if h=='model': width=22
        if h=='window': width=23
        if h in ['analysis_variant','record_status']: width=44
        if h in ['source_file','regression_source','metadata_source']: width=100
        if h in ['note','ICD-9','ICD-10']: width=150
        if h in ['numeric_source_text','Conf-interval beta']: width=80
        if h in ['feature','outcome','concept','drug_group_name']: width=45
        sh.set_column(j,j,width)
        widths.append(width)
        wraps.append(h in ['feature_id','feature_name','contrast','note','record_status','PheWAS Name','ICD-9','ICD-10'])
        number_formats.append('#,##0' if h=='n' or h.startswith('n_') or h in ['source_row','window_days','df'] else '0.000E+00' if h=='p' or h.startswith(('p-','p_')) or h=='ph_p' else '0.0000')
    sh.set_row(7,36)
    for i,row in enumerate(data,8):
        lines=1
        for j,v in enumerate(row):
            number=number_formats[j] if isinstance(v,(int,float)) and not isinstance(v,bool) else '@'
            if number=='0.0000' and v and abs(v)<0.0001: number='0.000E+00'
            f=fmt(number,wraps[j],i%2==0,isinstance(v,bool))
            if v is None: sh.write_blank(i,j,None,f)
            elif isinstance(v,bool): sh.write_boolean(i,j,v,f)
            elif isinstance(v,(int,float)): sh.write_number(i,j,v,f)
            else: sh.write_string(i,j,v,f)
            if wraps[j] and v is not None: lines=max(lines,math.ceil(len(str(v))/(widths[j]-3)))
        if lines>1: sh.set_row(i,min(409,lines*14+4))
    sh.add_table(7,0,len(data)+7,len(headers)-1,{'name':m['was']+'Results','style':'Table Style Medium 2','columns':[{'header':h,'header_format':header} for h in headers]})
    status=headers.index('record_status')
    sh.conditional_format(8,status,len(data)+7,status,{'type':'text','criteria':'containing','value':'unstable','format':wb.add_format({'bg_color':'#FCE4D6','font_color':'#9C2C22'})})
    print(json.dumps({'written':m['was'],'rows':len(data)}),flush=True)
wb.close()
print('EXPORTED',flush=True)
