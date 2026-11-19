"""Rebuild manuscript association panels from aggregate result tables only.

Embedded Manhattan labels retain up to ten positive Bonferroni associations by P.
Volcano labels show up to five positive Bonferroni associations by effect size.
The original domain-based forest selection and fitted estimates are unchanged.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prior_panels', ROOT / 'tmp/phewas_260919c/build_panels.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

MANIFEST = []
AUDIT = {'scatter': {}, 'forests': {}, 'layout': {}, 'forest_layout': []}
SCATTER_NOTE = ('Colors identify body systems or established study domains. Upward/downward triangles indicate positive/inverse effects; '
                'size reflects absolute effect magnitude, and dark outlines mark Bonferroni-significant results. '
                'Manhattan plots label up to ten positive Bonferroni-significant associations ranked by smallest P value. '
                'Volcano plots label up to five positive Bonferroni-significant associations ranked by largest effect size '
                '(OR, HR, IRR, or positive β), with ties resolved by smallest P value and then label. '
                'Inverse associations are not labeled. All qualifying labels are shown when fewer than the limit exist. −log10(P) is capped at 50.')
FOREST_NOTE = b.FOREST_NOTE


def record(r):
    return {k: r.get(k) for k in ('code', 'feature', 'outcome', 'analysis_key', 'label', 'group', 'effect', 'ci_low', 'ci_high', 'p', 'sig_bon', 'signed')}


def axes(fig, box):
    width, height = fig.get_size_inches()
    x, y, w, h = box
    return fig.add_axes([x/width, y/height, w/width, h/height])


def figure(number, title, height):
    fig = plt.figure(figsize=(13, height), dpi=150)
    fig.text(.025, 1-.25/height, f'Figure {number}  {title}', fontsize=16, fontweight='bold', va='top')
    return fig


def legend(fig, rows, top, ncol=6, fontsize=8):
    groups = b.order_groups(rows)
    handles = [Line2D([], [], color=b.color(g), marker='o', linestyle='', markersize=4.5,
                      label=f'{i+1}  {b.display(g)}') for i,g in enumerate(groups)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.025, top/fig.get_figheight()),
               frameon=False, ncol=ncol, fontsize=fontsize, handletextpad=.25,
               columnspacing=1.1, labelspacing=.35)


def chosen_labels(rows, kind='manhattan'):
    eligible = [r for r in rows if r['sig_bon'] and r['signed'] > 0]
    if kind == 'volcano':
        return sorted(eligible, key=lambda r: (-r['effect'], r['p'], r['label']))[:5]
    return sorted(eligible, key=lambda r: (r['p'], -r['magnitude'], r['label']))[:10]


def scatter_panel(fig, box, rows, kind, effect, title, key, small=False, common_groups=None):
    """Use the entire panel width and keep every annotation inside the axes."""
    x, y, w, h = box
    ax = axes(fig, box)
    ax.set_title(title, loc='left', fontsize=9 if small else 11, fontweight='bold', pad=9)
    groups = b.order_groups(rows)
    rows = sorted(rows, key=lambda r:(groups.index(r['group']), str(r.get('code',r.get('feature',r['label'])))))
    if kind == 'manhattan':
        xvals = list(range(len(rows)))
        ticks = []
        offset = 0
        for n,g in enumerate(groups):
            size = sum(r['group']==g for r in rows)
            ax.axvspan(offset-.5,offset+size-.5,color='#F0F2F4' if n%2==0 else 'white',zorder=0)
            ticks.append(offset+(size-1)/2)
            offset += size
        labels = [str((common_groups or groups).index(g)+1) for g in groups]
        ax.set_xticks(ticks,labels)
        ax.tick_params(axis='x',labelrotation=90 if len(groups)>8 else 0,labelsize=7)
        pad = max(1,len(rows)*.015)
        ax.set_xlim(-pad,len(rows)-1+pad)
        ax.set_xlabel('Body system / domain (legend numbers)',fontsize=7 if small else 8)
        if key in {'physicianspecialtyrates','alliedspecialtyrates'}:
            ax.set_xticks([])
            ax.set_xlabel('Physician specialties' if key=='physicianspecialtyrates' else 'Allied-health specialties',fontsize=8)
    else:
        xvals = [r['signed'] for r in rows]
        extent = max(max(abs(v) for v in xvals)*1.12,.05)
        ax.set_xlim(-extent,extent)
        ax.axvline(0,color='#919191',linewidth=.7,zorder=1)
        ax.set_xlabel('Adjusted β' if effect=='β' else f'log₂({effect})',fontsize=8)
    b.scatter(ax,rows,xvals)
    ax.set_ylim(-.5,54)
    ax.set_yticks([0,10,20,30,40,50],['0','10','20','30','40','≥50'])
    ax.set_ylabel('−log₁₀(P)',fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y',color='#E6E6E6',linewidth=.5,zorder=0)
    selected = chosen_labels(rows, kind)
    eligible = [r for r in rows if r['sig_bon'] and r['signed']>0]
    AUDIT['scatter'][f'{key}_{kind}'] = {'rows':len(rows),'positive_bonferroni_eligible':len(eligible),
        'selection_rule':('largest positive effect, then smallest p, then label' if kind=='volcano' else
                          'smallest p, then largest effect magnitude, then label'),
        'label_limit':5 if kind=='volcano' else 10,
        'selected':[record(r) for r in selected]}
    if kind == 'volcano':
        AUDIT['scatter'][f'{key}_{kind}']['eligible_candidates']=[record(r) for r in eligible]
    fontsize = 7.25 if small else 7.75
    if not selected:
        ax.text(.5,.82,'No positive Bonferroni-significant associations',transform=ax.transAxes,
                ha='center',va='center',fontsize=fontsize,color='#666666')
        AUDIT['layout'][f'{key}_{kind}']={'full_labels':0,'label_overlap_pairs':0,
            'labels_outside_axes':0,'label_font_pt':fontsize,'axes_box_inches':list(box)}
        return
    wrap = 24 if w < 4.5 else 38
    labels = [textwrap.fill(r['label'],wrap,break_long_words=False,break_on_hyphens=False) for r in selected]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    samples = [ax.text(0,0,label,fontsize=fontsize,linespacing=1.1) for label in labels]
    dimensions=np.array([[t.get_window_extent(renderer).width,t.get_window_extent(renderer).height] for t in samples])
    for t in samples:t.remove()
    targets=np.array([ax.transData.transform((xvals[rows.index(r)],r['plot_y'])) for r in selected])
    all_points=ax.transData.transform(np.column_stack((xvals,[r['plot_y'] for r in rows])))
    bounds=ax.bbox
    pad=4.5
    candidates=[]
    scores=[]
    # Measure all labels and choose short nearby leaders from an interior lattice.
    # Hard constraints prevent any label overlap or covering a labeled marker.
    for (tw,th),target in zip(dimensions,targets):
        cx=np.linspace(bounds.x0+tw/2+pad,bounds.x1-tw/2-pad,57)
        cy=np.linspace(bounds.y0+th/2+pad,bounds.y1-th/2-pad,43)
        centers=np.array(np.meshgrid(cx,cy)).reshape(2,-1).T
        close=np.array([(target[0]+dx*(tw/2+gap),target[1]+dy*(th/2+gap))
                        for gap in [9,17,29,47,74] for dx,dy in
                        [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]])
        centers=np.vstack([close,centers])
        rects=np.column_stack((centers[:,0]-tw/2-pad/2,centers[:,1]-th/2-pad/2,
                               centers[:,0]+tw/2+pad/2,centers[:,1]+th/2+pad/2))
        valid=(rects[:,0]>=bounds.x0+pad/2)&(rects[:,2]<=bounds.x1-pad/2)&(rects[:,1]>=bounds.y0+pad/2)&(rects[:,3]<=bounds.y1-pad/2)
        near_selected=((targets[None,:,0]>=rects[:,None,0]-5)&(targets[None,:,0]<=rects[:,None,2]+5)&
                       (targets[None,:,1]>=rects[:,None,1]-5)&(targets[None,:,1]<=rects[:,None,3]+5)).any(axis=1)
        valid &= ~near_selected
        rects=rects[valid];centers=centers[valid]
        inside=((all_points[None,:,0]>=rects[:,None,0]-3)&(all_points[None,:,0]<=rects[:,None,2]+3)&
                (all_points[None,:,1]>=rects[:,None,1]-3)&(all_points[None,:,1]<=rects[:,None,3]+3)).sum(axis=1)
        distance=np.linalg.norm(centers-target,axis=1)
        candidates.append((centers,rects))
        scores.append(distance+inside*24)
    rng=np.random.default_rng(260919)
    orders=[list(range(len(labels))),list(np.argsort(-np.prod(dimensions,axis=1))),
            list(np.argsort(targets[:,0])),list(np.argsort(-targets[:,0]))]
    orders += [list(rng.permutation(len(labels))) for _ in range(8)]
    best=None
    for order in orders:
        occupied=[];placements={};total=0
        for i in order:
            centers,rects=candidates[i]
            valid=np.ones(len(rects),dtype=bool)
            for old in occupied:
                valid &= ~((rects[:,0]<old[2]+3)&(rects[:,2]>old[0]-3)&(rects[:,1]<old[3]+3)&(rects[:,3]>old[1]-3))
            if not valid.any():break
            cost=np.where(valid,scores[i],np.inf)
            j=int(np.argmin(cost));placements[i]=centers[j];occupied.append(rects[j]);total+=cost[j]
        if len(placements)==len(labels) and (best is None or total<best[0]):best=(total,placements)
    assert best is not None,(key,kind,'cannot place complete label set')
    texts=[]
    for i,(r,label) in enumerate(zip(selected,labels)):
        loc=ax.transData.inverted().transform(best[1][i])
        point=(xvals[rows.index(r)],r['plot_y'])
        texts.append(ax.annotate(label,xy=point,xytext=loc,ha='center',va='center',
            fontsize=fontsize,linespacing=1.1,zorder=6,
            bbox={'facecolor':'white','edgecolor':'none','pad':.5,'alpha':.92},
            arrowprops={'arrowstyle':'-','lw':.6,'color':'#555555','shrinkA':2,'shrinkB':3}))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # Text bboxes only (Annotation extent includes the connecting line).
    from matplotlib.text import Text
    boxes=[Text.get_window_extent(t,renderer) for t in texts]
    overlap=int(sum(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i+1,len(boxes))))
    assert not overlap,(key,kind,'overlapping label text')
    outside=int(sum(box.x0<bounds.x0 or box.x1>bounds.x1 or box.y0<bounds.y0 or box.y1>bounds.y1 for box in boxes))
    assert not outside,(key,kind,'label exceeds axes')
    AUDIT['layout'][f'{key}_{kind}']={'full_labels':len(texts),'label_overlap_pairs':overlap,
        'labels_outside_axes':outside,'label_font_pt':fontsize,'axes_box_inches':list(box),
        'label_bboxes_inches':[[v/fig.dpi for v in bb.extents] for bb in boxes]}


def selected(key,rows):
    chosen=b.select(key,rows)
    AUDIT['forests'][key]={'rows':len(rows),'bonferroni':sum(r['sig_bon'] for r in rows),
                          'selected':[record(r) for r in chosen]}
    return chosen


def forest_axis(ax,rows,effect,title,fontsize=8,scale_rows=None,measure_only=False):
    """Lay out every forest text line in physical units, with measured clearance."""
    ax.set_title(title,loc='left',fontweight='bold',fontsize=10,pad=8)
    ax.set_axis_off()
    if not rows:
        if measure_only:return .85
        ax.text(.5,.55,'No associations selected\nin this direction',transform=ax.transAxes,
                ha='center',va='center',color='#666666',fontsize=9)
        return
    fig=ax.figure
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    width=ax.bbox.width/fig.dpi
    height=ax.bbox.height/fig.dpi
    label_width=width*.60
    def wrapped(label):
        words=label.split()
        lines=[]; line=''
        for word in words:
            candidate=(line+' '+word).strip()
            sample=ax.text(0,0,candidate,fontsize=fontsize)
            fits=sample.get_window_extent(renderer).width <= label_width*fig.dpi
            sample.remove()
            if line and not fits:lines.append(line);line=word
            else:line=candidate
        if line:lines.append(line)
        return '\n'.join(lines)
    items=[]
    for group in b.order_groups(rows):
        items.append(('group',b.display(group),group,None))
        for r in sorted([r for r in rows if r['group']==group],key=lambda r:(-r['magnitude'],r['p'],r['label'])):
            items.append(('row',wrapped(r['label']),group,r))
    heights=[]
    for kind,label,_,_ in items:
        sample=ax.text(0,0,label,fontsize=fontsize,fontweight='bold' if kind=='group' else 'normal',linespacing=1.1)
        heights.append(sample.get_window_extent(renderer).height/fig.dpi)
        sample.remove()
    required=sum(heights)+.025*(len(items)-1)+.035*sum(kind=='group' for kind,_,_,_ in items)+.35
    if measure_only:return required
    if required>height:
        raise ValueError(f'Forest {title}: needs {required:.2f}in, available {height:.2f}in, {len(rows)} rows')
    ax.set_xlim(0,1);ax.set_ylim(0,height)
    scale_rows=scale_rows or rows
    lo=min(0,min(r['signed_low'] for r in scale_rows))
    hi=max(0,max(r['signed_high'] for r in scale_rows))
    span=max(hi-lo,.1);lo-=span*.05;hi+=span*.06
    left,right=.65,.98
    def xp(v):return left+(v-lo)/(hi-lo)*(right-left)
    cursor=height-.03
    plotted=[]
    text_objs=[]
    for (kind,label,group,r),ht in zip(items,heights):
        if kind=='group':cursor-=.035
        center=cursor-ht/2
        text_objs.append(ax.text(0 if kind=='group' else .007,center,label,fontsize=fontsize,
                       fontweight='bold' if kind=='group' else 'normal',color=b.color(group) if kind=='group' else '#202020',
                       va='center',linespacing=1.1))
        if r:
            ax.plot([xp(r['signed_low']),xp(r['signed_high'])],[center,center],color=b.color(group),lw=1)
            for v in (r['signed_low'],r['signed_high']):ax.plot([xp(v),xp(v)],[center-.026,center+.026],color=b.color(group),lw=.7)
            ax.scatter([xp(r['signed'])],[center],marker='^' if r['signed']>0 else 'v',s=21,c=b.color(group),edgecolors='#333333',linewidth=.35,zorder=3)
            plotted.append(center)
        cursor-=ht+.025
    axis_y=max(.24,cursor-.045)
    ax.plot([xp(0),xp(0)],[axis_y,height-.02],color='#999999',lw=.65,zorder=0)
    ax.plot([left,right],[axis_y,axis_y],color='#555555',lw=.6)
    if effect=='β':
        ticks=list(np.linspace(lo+.05*span,hi-.06*span,3));labels=[f'{v:.2f}' for v in ticks]
    else:
        ticks=[math.log2(v) for v in [.1,.2,.33,.5,.67,1,1.5,2,3,4,6,8,12,16,32] if lo<=math.log2(v)<=hi]
        while len(ticks)>4:ticks=[v for n,v in enumerate(ticks) if n%2==0 or v==0]
        if len(ticks)<2:ticks=list(np.linspace(lo+.05*span,hi-.06*span,3))
        labels=[f'{2**v:.2g}' for v in ticks]
    for v,label in zip(ticks,labels):
        ax.plot([xp(v)]*2,[axis_y,axis_y-.028],color='#555555',lw=.5)
        ax.text(xp(v),axis_y-.05,label,fontsize=7,ha='center',va='top')
    ax.text((left+right)/2,axis_y-.20,f'{effect} (95% CI)'+(' · log scale' if effect!='β' else ''),fontsize=7.2,ha='center',va='top')
    fig.canvas.draw()
    boxes=[t.get_window_extent(renderer) for t in text_objs]
    assert not any(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i+1,len(boxes))),title
    AUDIT['forest_layout'].append({'figure':fig.texts[0].get_text(),'title':title,'rows':len(rows),
                                  'text_font_pt':fontsize,'label_overlap_pairs':0,
                                  'required_height_inches':required,'available_height_inches':height})


def split_groups(rows, parts):
    groups=b.order_groups(rows)
    sets=[]
    left=groups[:]
    for n in range(parts-1):
        desired=sum(r['group'] in left for r in rows)/(parts-n)
        cut=min(range(1,len(left)),key=lambda k:abs(sum(r['group'] in left[:k] for r in rows)-desired))
        sets.append(left[:cut]); left=left[cut:]
    sets.append(left)
    return [[r for r in rows if r['group'] in group_set] for group_set in sets]


def forest_specs(chosen):
    pos=[r for r in chosen if r['signed']>0]
    inv=[r for r in chosen if r['signed']<0]
    if len(inv)>22:
        invsplit=split_groups(inv,2)
        return [('C1  Positive',pos,pos),('C2  Inverse',invsplit[0],inv),('C2  Inverse (continued)',invsplit[1],inv)]
    else:
        return [('C1  Positive associations',pos,pos),('C2  Inverse associations',inv,inv)]


def forest_height(columns,width,effect,fontsize=8,gap=.26):
    cw=(width-gap*(len(columns)-1))/len(columns)
    fig=plt.figure(figsize=(width,1),dpi=150)
    required=max(forest_axis(axes(fig,(0,0,cw,1)),subset,effect,'',fontsize=fontsize,
                            scale_rows=scale,measure_only=True) for _,subset,scale in columns)
    plt.close(fig)
    return required+.08


def forest_columns(fig,box,columns,effect,fontsize=8,gap=.26):
    x,y,w,h=box
    cw=(w-gap*(len(columns)-1))/len(columns)
    for n,(title,subset,scale) in enumerate(columns):
        ax=axes(fig,(x+n*(cw+gap),y,cw,h))
        forest_axis(ax,subset,effect,title,fontsize=fontsize,scale_rows=scale)


def save(fig,number,caption):
    OUT.mkdir(parents=True,exist_ok=True)
    png=OUT/f'figure{number}_panel.png'
    fig.savefig(png,dpi=300,facecolor='white')
    fig.savefig(OUT/f'figure{number}_panel.pdf',facecolor='white')
    height=fig.get_figheight()
    assert height+1.8 <= 22, (number, 'page exceeds Word height limit')
    MANIFEST.append({'number':number,'file':str(png),'caption':caption,
                     'width_inches':13,'height_inches':height,
                     'page_width_inches':14,'page_height_inches':height+1.8})
    plt.close(fig)


def single_was(number,title,rows,effect,key,description):
    chosen=selected(key,rows)
    columns=forest_specs(chosen)
    volcano_w=3.55 if len(columns)==3 else 4.1
    forest_x=volcano_w+1.03
    forest_w=12.7-forest_x
    font=7.6 if len(columns)==3 else 8
    lower_h=max(forest_height(columns,forest_w,effect,font),3.75 if len(chosen_labels(rows))>5 else 2.6)
    manhattan_h=2.45
    height=1.7+manhattan_h+.85+lower_h+.85
    fig=figure(number,title,height)
    legend(fig,rows,height-.75)
    scatter_panel(fig,(.55,.85+lower_h+.85,12.15,manhattan_h),rows,'manhattan',effect,'A  Manhattan plot',key)
    scatter_panel(fig,(.55,.85,volcano_w,lower_h),rows,'volcano',effect,'B  Volcano plot',key)
    forest_columns(fig,(forest_x,.85,forest_w,lower_h),columns,effect,fontsize=font)
    fig.text(.027,.02,'Forest selection: up to 3 per domain by absolute effect size after Bonferroni correction, across both directions.',fontsize=8)
    caption=f'Figure {number}. {title}. A, Manhattan plot; B, volcano plot; C1/C2, positive/inverse forests. {description} {SCATTER_NOTE} {FOREST_NOTE}'
    save(fig,number,caption)


def mixed_was(number,title,groups,description,notes=''):
    # Separate outcome scales receive their own Manhattan, volcano, and forest.
    plans=[]
    for form,rows,effect in groups:
        key=f'{title.lower()}_{form.lower().replace(" ","_")}'
        chosen=selected(key,rows)
        split=split_groups(chosen,2) if len(b.order_groups(chosen))>1 else [chosen]
        columns=[('Forest plot' if n==0 else 'Forest plot (continued)',subset,chosen) for n,subset in enumerate(split)]
        lower_h=max(forest_height(columns,7.7,effect,8),3.0 if len(chosen_labels(rows))>5 else 1.5)
        manhattan_h=1.9 if len(rows)>30 else 1.45
        block_h=1.0+manhattan_h+.8+lower_h+.5
        plans.append((key,columns,lower_h,manhattan_h,block_h))
    height=sum(p[4] for p in plans)+1.15
    fig=figure(number,title,height)
    top=height-.8
    for index,(form,rows,effect) in enumerate(groups):
        key,columns,lower_h,manhattan_h,rh=plans[index]
        section_top=top
        fig.text(.028,section_top/height,f'{chr(65+index)}  {form}',fontsize=12,fontweight='bold',va='top')
        legend(fig,rows,section_top-.30,ncol=5,fontsize=7.7)
        bottom=top-rh+.5
        scatter_panel(fig,(.55,bottom+lower_h+.8,12.15,manhattan_h),rows,'manhattan',effect,'Manhattan plot',key)
        scatter_panel(fig,(.55,bottom,4.1,lower_h),rows,'volcano',effect,'Volcano plot',key)
        forest_columns(fig,(5.0,bottom,7.7,lower_h),columns,effect,fontsize=8)
        top-=rh
    if notes:fig.text(.028,.017,notes,fontsize=8)
    save(fig,number,f'Figure {number}. {title}. A, {groups[0][0].lower()}; B, {groups[1][0].lower()}. Each section combines Manhattan, volcano, and forest plots. {description} {SCATTER_NOTE} {FOREST_NOTE} {notes}')


def behavior_forests(groups):
    """Retain exactly the prior BehWAS forest records in two forest-only panels."""
    plans=[]
    width=5.95
    for form,rows,effect in groups:
        key=f'behwas_{form.lower().replace(" ","_")}'
        chosen=selected(key,rows)
        required=forest_height([('',chosen,chosen)],width,effect,8)
        plans.append((form,chosen,effect,required))
    forest_h=max(plan[3] for plan in plans)
    height=forest_h+1.65
    fig=figure(8,'BehWAS',height)
    for index,(form,chosen,effect,_) in enumerate(plans):
        ax=axes(fig,(.40+index*6.45,.7,width,forest_h))
        forest_axis(ax,chosen,effect,f'{chr(65+index)}  {form}',fontsize=8)
    fig.text(.028,.05,'Medication adherence and PDC are among users. BehWAS remains preliminary.',fontsize=8)
    description=('Fully adjusted m4 estimates compare AHI ≥5 versus <5. Five-year estimates are used for windowed outcomes '
                 'and lifetime estimates for undated outcomes. Colors identify medication adherence, vaccination, attendance, '
                 'and work-leave domains. Forests show selected Bonferroni-significant associations. '
                 'Medication adherence and PDC are among users. BehWAS remains preliminary.')
    save(fig,8,f'Figure 8. BehWAS. A, binary-outcome forest plot (OR); B, continuous-outcome forest plot (β). {description} {FOREST_NOTE}')


def verify_against_prior():
    prior=json.loads((ROOT/'tmp/phewas_260919f/association_audit.json').read_text(encoding='utf-8'))
    assert AUDIT['forests']==prior['forests'], 'Forest records changed'
    unchanged=[]
    for key,current in AUDIT['scatter'].items():
        assert current['rows']==prior['scatter'][key]['rows'], key
        assert current['positive_bonferroni_eligible']==prior['scatter'][key]['positive_bonferroni_eligible'], key
        if key.endswith('_manhattan'):
            assert current['selected']==prior['scatter'][key]['selected'], key
            unchanged.append(key)
        else:
            selected=current['selected']
            assert len(selected)==min(5,current['positive_bonferroni_eligible']), key
            assert all(r['sig_bon'] and r['signed']>0 for r in selected), key
            assert selected==sorted(selected,key=lambda r:(-r['effect'],r['p'],r['label'])), key
        layout=AUDIT['layout'][key]
        assert layout['labels_outside_axes']==0 and layout['label_overlap_pairs']==0, key
        old_font=prior['layout'][key]['label_font_pt']
        if current['selected']:
            assert abs(old_font-layout['label_font_pt']-.75)<1e-9, key
        layout['previous_label_font_pt']=old_font
        layout['font_reduction_pt']=old_font-layout['label_font_pt']
    assert not any(key.startswith('behwas_') for key in AUDIT['scatter'])
    AUDIT['verification']={'forest_records_exact_match_to_f':True,
                           'forest_panels_compared':len(AUDIT['forests']),
                           'unchanged_manhattan_selection':unchanged,
                           'behwas_forest_only':True,
                           'page_heights_within_word_limit':all(r['page_height_inches']<=22 for r in MANIFEST)}


def utilization(payload):
    specs=[('ed_primarypresence','ED primary diagnosis presence','OR'),
           ('ed_secondarypresence','ED secondary diagnosis presence','OR'),
           ('physician_primarypresence','Outpatient primary diagnosis presence','OR'),
           ('physicianspecialtyrates','Physician encounters by specialty','IRR'),
           ('alliedspecialtyrates','Allied-health encounters by specialty','IRR')]
    groups=payload['groups']
    all_diagnosis=[r for k,_,_ in specs[:3] for r in groups[k]]
    plans=[]
    for n,(key,title,effect) in enumerate(specs[:3]):
        rows=groups[key]
        chosen=selected(key,rows)
        split=split_groups(chosen,2 if n==0 else 3)
        columns=[('Forest plot' if j==0 else 'Forest (continued)',subset,chosen) for j,subset in enumerate(split)]
        lower_h=max(forest_height(columns,7.7,effect,7.6),1.3)
        manhattan_h=1.0
        row_h=.36+manhattan_h+.60+lower_h+.6
        plans.append((columns,lower_h,manhattan_h,row_h))
    specialty_h=4.2
    height=sum(p[3] for p in plans)+specialty_h+1.4
    fig=figure(6,'Five-year UtilWAS',height)
    legend(fig,all_diagnosis,height-.65,ncol=6,fontsize=7.5)
    top=height-1.4
    for n,(key,title,effect) in enumerate(specs[:3]):
        rows=groups[key]
        columns,lower_h,manhattan_h,row_h=plans[n]
        fig.text(.027,top/height,f'{chr(65+n)}  {title} · {len(rows)} features',fontsize=10.5,fontweight='bold')
        bottom=top-row_h+.6
        common=b.order_groups(all_diagnosis)
        scatter_panel(fig,(.55,bottom+lower_h+.60,12.15,manhattan_h),rows,'manhattan',effect,'Manhattan',key,small=True,common_groups=common)
        scatter_panel(fig,(.55,bottom,4.1,lower_h),rows,'volcano',effect,'Volcano',key,small=True)
        forest_columns(fig,(5.0,bottom,7.7,lower_h),columns,effect,fontsize=7.6)
        top-=row_h
    for j,(key,title,effect) in enumerate(specs[3:]):
        left=.55+6.5*j
        rows=groups[key]
        bottom=.65
        fig.text((left-.2)/13,top/height,f'{chr(68+j)}  {title}',fontsize=10.5,fontweight='bold')
        scatter_panel(fig,(left,bottom+2.20,5.65,1.05),rows,'manhattan',effect,'Manhattan',key,small=True)
        scatter_panel(fig,(left,bottom,2.40,1.55),rows,'volcano',effect,'Volcano',key,small=True)
        chosen=selected(key,rows)
        columns=[('Forest plot',chosen,chosen)]
        forest_columns(fig,(left+2.82,bottom,2.83,1.55),columns,effect,fontsize=7.6)
    description=('Five-year fully adjusted m4 estimates compare AHI ≥5 versus <5. A/B, ED-primary/secondary diagnosis presence; '
                 'C, primary diagnoses associated with outpatient physician encounters; D/E, physician/allied-health encounter rates by specialty. '
                 'Diagnosis-presence odds ratios use the full cohort. Specialty IRRs describe encounters per person-time and are not diagnosis-conditional rates. '
                 'Outpatient physician encounters are not restricted to Family Practice, Internal Medicine, or Pediatrics. '
                 'Diagnosis colors use Phecode body systems; specialty panels retain the source physician and allied-health groups. '
                 'ED conditional rates are omitted. Original analysis-specific multiplicity flags are retained.')
    save(fig,6,f'Figure 6. Five-year UtilWAS. Each row combines Manhattan, volcano, and forest plots. {description} {SCATTER_NOTE} {FOREST_NOTE}')


def main():
    prevalence,incidence=b.source.load_disease_data()
    data=b.source.base.load_data()
    med=[r for r in data['med'] if r['window']=='5yr']
    proc=[r for r in data['proc'] if r['window']=='5yr']
    qwas=data['qwas']
    beh=[r for r in data['beh'] if r['window'] in {'5yr','lifetime'}]
    for r in beh:
        outcome=r['outcome']
        r['group']=('Medication adherence' if outcome in {'adherence_pdc','adherent'} else
                    'Vaccination' if outcome.startswith('flu_') else
                    'Attendance' if outcome in {'any_noshow','no_show_pct'} else 'Work leave')
    util=json.loads((OUT/'utilwas_rows.json').read_text(encoding='utf-8'))
    utilrows=[r for rows in util['groups'].values() for r in rows]
    b.init_colors(prevalence+incidence+med+proc+qwas+beh+utilrows)
    single_was(1,'Prevalence disease PheWAS',prevalence,'OR','prevalence',
        'All 1,779 primary prevalence rows compare AHI ≥5 versus <5 using m4_bmi estimates (OR), excluding sleep apnea (Phecode 327.3 and subcodes).')
    single_was(2,'Incidence disease PheWAS',incidence,'HR','incidence',
        'All 1,012 primary incidence rows compare AHI ≥5 versus <5 using m4 estimates (HR), excluding sleep apnea (Phecode 327.3 and subcodes).')
    single_was(4,'Five-year MedWAS',med,'OR','med',
        'All 218 medication-class rows compare AHI ≥5 versus <5 using fully adjusted m4 estimates (OR). Colors identify established therapeutic categories. Both 1- and 5-year estimates remain in the supplemental atlas.')
    single_was(5,'Five-year ProcWAS',proc,'IRR','proc',
        'All 67 procedure rows compare AHI ≥5 versus <5 using fully adjusted m4 estimates (IRR). Colors identify body systems. Both 1- and 5-year estimates remain in the supplemental atlas.')
    mixed_was(7,'QWAS',[('Binary outcomes',[r for r in qwas if r['form']=='binary'],'OR'),
                         ('Continuous outcomes',[r for r in qwas if r['form']=='continuous'],'β')],
        'All 83 binary and 49 continuous questionnaire rows compare AHI ≥5 versus <5 using fully adjusted m4 estimates (OR and rank-inverse-normal β). Colors identify established questionnaire domains.',
        '* Referral-driving symptoms. Higher FOSQ scores indicate better function.')
    behavior_forests([('Binary outcomes',[r for r in beh if r['effect_type']=='OR'],'OR'),
                      ('Continuous outcomes',[r for r in beh if r['effect_type']=='beta'],'β')])
    utilization(util)
    MANIFEST.sort(key=lambda r:r['number'])
    verify_against_prior()
    (OUT/'association_manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'association_audit.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'figures':[r['number'] for r in MANIFEST],'scatter_panels':len(AUDIT['scatter']),
                      'label_counts':{k:len(v['selected']) for k,v in AUDIT['scatter'].items()}},indent=2))


if __name__=='__main__':main()
