import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile, FileBlob } from '@oai/artifact-tool';

const dir=path.dirname(fileURLToPath(import.meta.url));
const manifest=JSON.parse(await fs.readFile(path.join(dir,'manifest.json'),'utf8'));
const col=(n)=>{let s=''; for(n++;n;n=Math.floor((n-1)/26)) s=String.fromCharCode(65+(n-1)%26)+s; return s;};
if(process.argv.includes('--verify-render')){
  const filename=path.join(dir,'preview_excerpt.xlsx');
  const book=await SpreadsheetFile.importXlsx(await FileBlob.load(filename));
  console.log('Imported exact final-workbook excerpts for visual verification');
  book.recalculate();
  for(const m of manifest.sheets){
    const sh=book.worksheets.getItem(m.was);
    const data=JSON.parse(await fs.readFile(path.join(dir,m.was+'.json'),'utf8'));
    const j=data.headers.indexOf('effect_type');
    console.log((await book.inspect({kind:'table',range:`${m.was}!A8:F10`,include:'values',tableMaxRows:3,tableMaxCols:6,maxChars:250})).ndjson);
    for(const [tag,range] of [['view','A1:F13'],['stats',`${col(j)}8:${col(Math.min(j+7,data.headers.length-1))}14`]]){
      const preview=await book.render({sheetName:m.was,range,scale:1,format:'png'});
      await fs.writeFile(path.join(dir,`${m.was}_${tag}.png`),new Uint8Array(await preview.arrayBuffer()));
    }
    console.log('Verified layout '+m.was);
  }
  console.log('VISUAL VERIFICATION RENDERS COMPLETE');
  process.exit(0);
}
const wb=Workbook.create();
const definitions={
  PheWAS:'m1 crude; m2 + age, sex, race, year; m3 + smoking, income, observation window; m4 + BMI. IPW sensitivity is labeled separately.',
  IncWAS:'Severity: m1 crude; m2 + demographics; m3 + smoking, income, observation window, encounter rate; m4 + BMI. CPAP models also adjust for severity.',
  LabWAS:'m1 crude; m3 + demographics, year, smoking, income; m4 + BMI. Value/propensity models also include observation window; order-rate models use an offset.',
  MedWAS:'m1 crude; m3 + demographics, year, smoking, income, observation window; m4 + BMI. Unadjusted screening tests retained separately.',
  ProcWAS:'m1 crude; m2 + demographics and year; m3 + smoking and income; m4 + BMI. Procedure rates use the observation-time offset.',
  MeasWAS:'m1 crude; m2 + demographics, year, observation window; m3 + smoking and income; m4 + BMI. Adjusted beta is on the rank-normal scale.',
  QWAS:'m1 crude; m2 + demographics and year; m3 + smoking, income, observation window; m4 + BMI. Binary and continuous forms are separate records.',
  BehWAS:'m1 crude; m3 + demographics, year, smoking, income, observation window; m4 + BMI. Binary and continuous forms are separate records.',
  UtilWAS:'m1 crude; m3 + demographics, year, smoking, income; m4 + BMI. Binary models add observation window; rate models use an observation-time offset.'
};
const caveats={
  PheWAS:'Includes 2,017 records omitted from the combined summary. See record_status and note. Their original beta and beta interval are retained; ORs are not invented.',
  IncWAS:'Phenotype ladder: m3a excludes encounter rate; m3b adds it; m4 adds BMI. Tier, body-system, octant, and CPAP analyses remain distinct.',
  LabWAS:'Two-part combined rows repeat selected component statistics for comparison; they are not additional independent tests. Filter analysis_variant before counting tests.',
  MeasWAS:'Complete-case estimates among measured patients. PFT results are exploratory. Echo and separate EF-source measures remain distinct.',
  QWAS:'referral_item retains the source flag for questionnaire items related to referral. Each feature and form retains its own sample size.'
};
for (const meta of manifest.sheets) {
  const {headers,data}=JSON.parse(await fs.readFile(path.join(dir,meta.was+'.json'),'utf8'));
  const sh=wb.worksheets.add(meta.was);
  sh.showGridLines=false;
  sh.tabColor='#294765';
  const end=data.length+8, last=col(headers.length-1);
  const full=sh.getRange(`A1:${last}${end}`);
  full.format.font={name:'Arial',size:10,color:'#182B3A'};
  full.format.rowHeight=18;
  full.format.verticalAlignment='center';
  sh.getRange('A1').values=[[meta.was+' — complete available results']];
  sh.getRange('A1').format.font={name:'Arial',size:15,bold:true,color:'#182B3A'};
  sh.getRange('A1').format.rowHeight=27;
  sh.getRange('A2').values=[[`${meta.rows.toLocaleString('en-US')} records · ${meta.features.toLocaleString('en-US')} feature IDs · ${meta.sources.length} source exports · Current results snapshot: September 19, 2026`]];
  sh.getRange('A3').values=[['Filter by feature_id, contrast (exposure), model (adjustment), window, and analysis_variant. All source result columns and reported rows are retained.']];
  sh.getRange('A4').values=[[definitions[meta.was]]];
  sh.getRange('A5').values=[[caveats[meta.was] || 'Only observed model combinations are included. Missing values and unrun adjustment levels are unavailable; no results are imputed.']];
  sh.getRange('A6').values=[['Original blanks stay blank; source zero p-values stay zero. Nonzero numbers below Excel’s numeric range remain text and are listed in numeric_source_text.']];
  sh.getRange('A7').values=[['Row provenance is in source_file and source_row. Significance and instability flags are copied from their source analysis; source_row is the CSV line number.']];
  sh.getRange('A2:A7').format.font={name:'Arial',size:10,color:'#506272'};
  sh.getRange(`A8:${last}8`).values=[headers];
  for(let start=0;start<data.length;start+=2500){
    const block=data.slice(start,start+2500);
    sh.getRangeByIndexes(start+8,0,block.length,headers.length).values=block;
  }
  const table=sh.tables.add(`A8:${last}${end}`,true,meta.was+'Results');
  table.style='TableStyleMedium2';
  table.showFilterButton=true;
  sh.getRange(`A8:${last}8`).format={fill:'#294765',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:36};
  for(let j=0;j<headers.length;j++){
    const h=headers[j], letter=col(j), rg=sh.getRange(`${letter}9:${letter}${end}`);
    const nums=data.some(r=>typeof r[j]==='number');
    let width=Math.max(14,Math.min(25,h.length+3));
    if(h==='feature_id') width=23;
    if(h==='feature_name'||h==='phenotype'||h==='PheWAS Name'||h==='phecode_name') width=54;
    if(h==='contrast') width=48;
    if(h==='model') width=22;
    if(h==='window') width=23;
    if(h==='analysis_variant'||h==='record_status') width=44;
    if(h==='source_file'||h==='regression_source'||h==='metadata_source') width=100;
    if(h==='note'||h==='ICD-9'||h==='ICD-10'||h==='numeric_source_text'||h==='Conf-interval beta') width=80;
    if(['feature','outcome','concept','drug_group_name'].includes(h)) width=45;
    sh.getRange(`${letter}8:${letter}${end}`).format.columnWidth=width;
    rg.format.horizontalAlignment=nums?'right':'left';
    if(nums){
      const count=/^(n($|_)|source_row|window_days|df$)/.test(h);
      const prob=/^(p$|p-|p_|p_ipw|ph_p)/.test(h);
      rg.setNumberFormat(count?'#,##0':prob?'0.000E+00':'0.0000');
    } else rg.setNumberFormat('@');
    if(['feature_id','feature_name','contrast','note','record_status','PheWAS Name','ICD-9','ICD-10'].includes(h)) rg.format.wrapText=true;
    if(h==='unstable') rg.conditionalFormats.add('cellIs',{operator:'equal',formula:'TRUE',format:{fill:'#FCE4D6',font:{color:'#9C2C22',bold:true}}});
  }
  // Fit only columns that contain long textual data; retain readable body text.
  const wraps=headers.map((h,j)=>['feature_id','feature_name','contrast','note','record_status','PheWAS Name','ICD-9','ICD-10'].includes(h)?j:-1).filter(j=>j>=0);
  for(let i=0;i<data.length;i++){
    let lines=1;
    for(const j of wraps){
      const h=headers[j], chars= h==='feature_id'?23:h==='contrast'?48:h==='record_status'?44:(h==='note'||h.startsWith('ICD-'))?80:54;
      lines=Math.max(lines,Math.ceil(String(data[i][j]??'').length/(chars-3)));
    }
    if(lines>1) sh.getRange(`A${i+9}:${last}${i+9}`).format.rowHeight=Math.min(409,lines*14+4);
  }
  sh.freezePanes.freezeRows(8);
  sh.freezePanes.freezeColumns(2);
  console.log(JSON.stringify({built:meta.was,rows:data.length,cols:headers.length}));
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'PheWAS!A8:H11',include:'values,formulas',tableMaxRows:4,tableMaxCols:8,maxChars:1700})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},summary:'Final error scan',maxChars:1000})).ndjson);
for(const meta of manifest.sheets){
  const preview=await wb.render({sheetName:meta.was,range:'A1:F14',scale:1,format:'png'});
  await fs.writeFile(path.join(dir,meta.was+'.png'),new Uint8Array(await preview.arrayBuffer()));
  console.log('Rendered '+meta.was);
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(dir,'..','WAS_master_comprehensive.xlsx'));
console.log('EXPORTED WAS_master_comprehensive.xlsx');
