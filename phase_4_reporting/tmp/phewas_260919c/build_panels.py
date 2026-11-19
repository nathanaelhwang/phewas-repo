"""Rebuild manuscript panels from allow-listed aggregate association results only."""
from __future__ import annotations

import json
import math
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from adjustText import adjust_text

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tmp'))
import build_phewas_figures_260906 as source

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.titlesize': 10, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})

ORDER = ['infectious diseases', 'neoplasms', 'endocrine/metabolic', 'hematopoietic',
         'mental disorders', 'neurological', 'sense organs', 'circulatory system',
         'respiratory', 'digestive', 'genitourinary', 'pregnancy complications',
         'dermatologic', 'musculoskeletal', 'congenital anomalies',
         'symptoms', 'injuries & poisonings', 'other']
PALETTE = ['#4E79A7', '#A44386', '#E08718', '#AA3939', '#8064A2', '#4A548F',
           '#8A6E44', '#D1495B', '#1C8B9D', '#4F8A47', '#8660A8', '#CC79A7',
           '#A97525', '#527E7F', '#7A8450', '#555555', '#B55B31', '#939393']
ALIASES = {'cardiovascular': 'circulatory system',
           'cardiovascular / lymphatic': 'circulatory system',
           'endocrine & metabolic': 'endocrine/metabolic', 'endocrine': 'endocrine/metabolic',
           'gastrointestinal': 'digestive', 'hematological': 'hematopoietic',
           'cns': 'neurological', 'nervous system': 'neurological',
           'urinary / renal': 'genitourinary', 'eye / ophthalmic': 'sense organs',
           'integumentary / skin': 'dermatologic', 'maternity / obstetric': 'pregnancy complications',
           'anti-infective agents': 'infectious diseases', 'antineoplastic agents': 'neoplasms',
           'general / not system-specific': 'other', 'miscellaneous': 'other'}
EXTRA = ['#A86F9B', '#526D38', '#3C97C7', '#AC5016', '#A89720', '#2B5D57',
         '#7770B3', '#DA8D6C', '#7A4F36', '#14786C', '#B74E73', '#799E3A']
COLORS = dict(zip(ORDER, PALETTE))
AUDIT = {}
MANIFEST = []


def canonical(group):
    return ALIASES.get(group.lower(), group.lower())


def order_groups(rows):
    groups = set(r['group'] for r in rows)
    return sorted(groups, key=lambda g: (ORDER.index(canonical(g)) if canonical(g) in ORDER else 100, g.lower()))


def init_colors(all_rows):
    unknown = sorted({canonical(r['group']) for r in all_rows} - set(COLORS))
    # Additional non-anatomical study domains retain their established names.
    for i, group in enumerate(unknown):
        COLORS[group] = EXTRA[i % len(EXTRA)]
    COLORS.update(dict(zip(
        ['chief complaint','function','insomnia','mood','osa symptoms','parasomnia','restless legs','sleep habits','sleepiness','somatic'],
        ['#A64C14','#347645','#65519B','#C04468','#8C5A32','#338A96','#727A26','#CD741C','#A38700','#4674A6'])))
    COLORS.update({'analgesics & anesthetics':'#A86F9B','biologicals':'#526D38','misc psych/neuro':'#7A4F36',
                   'neuromuscular':'#14786C','nutritional':'#799E3A','stimulants/anti-obesity':'#7770B3',
                   'topical':'#C76D43','female reproductive':'#B74E73'})


def color(group):
    return COLORS[canonical(group)]


def display(group):
    return group[0].upper() + group[1:]


def select(key, rows):
    groups = defaultdict(list)
    for r in rows:
        if r['sig_bon']:
            groups[r['group']].append(r)
    chosen = []
    for group in order_groups(rows):
        chosen.extend(sorted(groups[group], key=lambda r: (-r['magnitude'], r['p'], r['label']))[:3])
    assert all(r['sig_bon'] for r in chosen)
    assert all(sum(r['group'] == g for r in chosen) <= 3 for g in groups)
    AUDIT[key] = {'available': len(rows), 'bonferroni_significant': sum(r['sig_bon'] for r in rows),
                  'selected': [{'key': r.get('code', r.get('feature', r.get('outcome'))),
                                'label': r['label'], 'group': r['group'],
                                'effect': r['effect'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'],
                                'p': r['p'], 'sig_bon': r['sig_bon'], 'signed': r['signed'],
                                'magnitude': r['magnitude']} for r in chosen]}
    return chosen


def legend(fig, groups, y, ncol=6, fontsize=8):
    handles = [Line2D([], [], color=color(g), marker='o', linestyle='', markersize=5,
                      label=f'{i + 1}  {display(g)}') for i, g in enumerate(groups)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.515, y),
               frameon=False, ncol=ncol, fontsize=fontsize, handletextpad=.25,
               columnspacing=1.0, labelspacing=.4)


def scatter(ax, rows, xvals):
    scale = np.quantile([r['magnitude'] for r in rows], .9) if rows else 1
    for group in order_groups(rows):
        for direction, marker in [(1, '^'), (-1, 'v')]:
            ids = [i for i, r in enumerate(rows) if r['group'] == group and (1 if r['signed'] >= 0 else -1) == direction]
            for sig in [False, True]:
                subset = [i for i in ids if bool(rows[i]['sig_bon']) == sig]
                if not subset:
                    continue
                ax.scatter([xvals[i] for i in subset], [rows[i]['plot_y'] for i in subset],
                           s=[9 + 29 * min(rows[i]['magnitude'] / max(scale, .001), 1) for i in subset],
                           marker=marker, c=color(group), alpha=1 if sig else .35,
                           edgecolors='#202020' if sig else 'none', linewidths=.4,
                           zorder=3 if sig else 2)


def scatter_axis(ax, rows, kind, effect, title, label_count=5):
    ax.set_title(title, loc='left', fontweight='bold', pad=7)
    groups = order_groups(rows)
    if kind == 'manhattan':
        rows = sorted(rows, key=lambda r: (groups.index(r['group']), str(r.get('code', r.get('feature', r['label'])))))
        xvals = list(range(len(rows)))
        ticks = []
        offset = 0
        for j, g in enumerate(groups):
            n = sum(r['group'] == g for r in rows)
            ax.axvspan(offset - .5, offset + n - .5, color='#F0F2F4' if j % 2 == 0 else 'white', zorder=0)
            ticks.append(offset + (n - 1) / 2)
            offset += n
        ax.set_xticks(ticks, [str(i + 1) for i in range(len(groups))])
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlim(-max(2, len(rows) * .006), len(rows) + max(2, len(rows) * .006))
        ax.set_xlabel('Body system / domain (legend numbers)')
    else:
        xvals = [r['signed'] for r in rows]
        extent = max(abs(v) for v in xvals) * 1.13
        ax.set_xlim(-extent, extent)
        ax.axvline(0, color='#919191', linewidth=.7, zorder=1)
        ax.set_xlabel('Adjusted β' if effect == 'β' else f'log₂({effect})')
    scatter(ax, rows, xvals)
    ax.set_ylim(-.5, 54)
    ax.set_yticks([0, 10, 20, 30, 40, 50], ['0', '10', '20', '30', '40', '≥50'])
    ax.set_ylabel('−log₁₀(P)')
    ax.grid(axis='y', color='#E6E6E6', linewidth=.5, zorder=0)
    eligible = sorted([i for i, r in enumerate(rows) if r['sig_bon'] or r['sig_fdr']],
                      key=lambda i: (rows[i]['p'], -rows[i]['magnitude']))
    # Retain a few visible labels without crowding the smaller panel format.
    selected = []
    for i in eligible:
        if all(abs((xvals[i] - xvals[j]) / (ax.get_xlim()[1] - ax.get_xlim()[0])) > .09
               or abs(rows[i]['plot_y'] - rows[j]['plot_y']) > 9 for j in selected):
            selected.append(i)
        if len(selected) >= label_count:
            break
    texts = []
    for n, i in enumerate(selected):
        x, y = xvals[i], rows[i]['plot_y']
        label = textwrap.fill(rows[i]['label'], 24)
        if len(label.splitlines()) > 2:
            label = '\n'.join(label.splitlines()[:2]) + '…'
        right = x < np.mean(ax.get_xlim())
        offset_y = -20 - 10 * (n % 2) if y > 38 else 12 + 12 * (n % 2)
        texts.append(ax.text(x, y - 6 if y > 38 else y + 6, label, fontsize=6.5,
                    ha='center', va='center',
                    bbox={'facecolor':'white', 'alpha':.88, 'edgecolor':'none', 'pad':1},
                    zorder=5))
    if texts:
        adjust_text(texts, x=xvals, y=[r['plot_y'] for r in rows],
                    target_x=[xvals[i] for i in selected], target_y=[rows[i]['plot_y'] for i in selected],
                    ax=ax, expand=(1.12, 1.25), force_static=(.3,.5), ensure_inside_axes=True,
                    iter_lim=300, arrowprops={'arrowstyle':'-', 'color':'#666666', 'lw':.45})
        renderer = ax.figure.canvas.get_renderer()
        for text in texts:
            box = text.get_window_extent(renderer)
            inset = 7 * ax.figure.dpi / 72
            correction = max(0, ax.bbox.x0 + inset - box.x0) - max(0, box.x1 - ax.bbox.x1 + inset)
            if correction:
                px, py = ax.transData.transform(text.get_position())
                text.set_position(ax.transData.inverted().transform((px + correction, py)))


def forest_axis(ax, rows, effect, title, direction=None, fontsize=7.1, scale_rows=None):
    ax.set_title(title, loc='left', fontweight='bold', pad=8)
    if direction is not None:
        rows = [r for r in rows if (r['signed'] > 0 if direction == 1 else r['signed'] < 0)]
    if not rows:
        ax.set_axis_off()
        ax.text(.5, .55, 'No Bonferroni-significant\nfeatures in this direction',
                transform=ax.transAxes, ha='center', va='center', color='#666666', fontsize=9)
        return
    ax.set_xlim(0, 1)
    groups = order_groups(rows)
    y = 0
    positions = []
    group_positions = []
    for g in groups:
        group_positions.append((y, g))
        y += 1.2
        for r in sorted([r for r in rows if r['group'] == g], key=lambda r: (-r['magnitude'], r['p'], r['label'])):
            label = textwrap.fill(r['label'], 31)
            lines = len(label.splitlines())
            positions.append((y + .4 * (lines - 1), r, label))
            y += max(1.13, lines * .85) + .16
        y += .45
    ax.set_ylim(y + 1.2, -1)
    scale_rows = scale_rows or rows
    lo = min(r['signed_low'] for r in scale_rows)
    hi = max(r['signed_high'] for r in scale_rows)
    lo, hi = min(lo, 0), max(hi, 0)
    span = max(hi - lo, .1)
    lo -= .04 * span
    hi += .07 * span
    left, right = .59, .98
    def xp(v):
        return left + (v - lo) / (hi - lo) * (right - left)
    ax.plot([xp(0), xp(0)], [-.6, y - .3], color='#888888', linewidth=.75, zorder=0)
    if effect == 'β':
        ticks = np.linspace(lo + .04 * span, hi - .07 * span, 4)
        labels = [f'{v:.2f}' for v in ticks]
    else:
        candidates = [.1, .2, .33, .5, .67, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 32]
        ticks = [math.log2(v) for v in candidates if lo <= math.log2(v) <= hi]
        while len(ticks) > 5:
            ticks = [v for i, v in enumerate(ticks) if i % 2 == 0 or v == 0]
        labels = [f'{2**v:g}' for v in ticks]
        if len(ticks) < 3:
            ticks = list(np.linspace(lo + .04 * span, hi - .07 * span, 3))
            labels = [f'{2**v:.2f}' for v in ticks]
    for tick, label in zip(ticks, labels):
        ax.plot([xp(tick), xp(tick)], [y - .1, y + .13], color='#555555', linewidth=.5)
        ax.text(xp(tick), y + .28, label, ha='center', va='top', fontsize=6.5)
    ax.plot([left, right], [y - .1, y - .1], color='#555555', linewidth=.6)
    for gy, g in group_positions:
        ax.text(0, gy, display(g), fontsize=fontsize, fontweight='bold', color=color(g), va='center')
    for ry, r, label in positions:
        ax.text(.01, ry, label, fontsize=fontsize, va='center', linespacing=1.12)
        ax.plot([xp(r['signed_low']), xp(r['signed_high'])], [ry, ry], color=color(r['group']), linewidth=1)
        ax.plot([xp(r['signed_low'])] * 2, [ry - .17, ry + .17], color=color(r['group']), linewidth=.7)
        ax.plot([xp(r['signed_high'])] * 2, [ry - .17, ry + .17], color=color(r['group']), linewidth=.7)
        ax.scatter([xp(r['signed'])], [ry], marker='^' if r['signed'] > 0 else 'v',
                   c=color(r['group']), s=22, edgecolors='#333333', linewidth=.35, zorder=3)
    ax.text((left + right) / 2, y + 1.4, f'{effect} (95% CI)' + (' · log scale' if effect != 'β' else ''),
            fontsize=7, ha='center', va='top')
    ax.set_axis_off()


def save(fig, number, caption):
    path = OUT / f'figure{number}_panel.png'
    fig.savefig(path, dpi=300)
    # Keep editable vector master locally for future manuscript production.
    fig.savefig(OUT / f'figure{number}_panel.pdf')
    plt.close(fig)
    height = float(fig.get_size_inches()[1])
    MANIFEST.append({'number':number, 'file':str(path), 'caption':caption,
                     'width_inches':13, 'height_inches':height,
                     'page_width_inches':14,'page_height_inches':height+2})


def newfig(title, height=9):
    fig = plt.figure(figsize=(13, height))
    fig.text(.035, .98, title, fontsize=15, fontweight='bold', va='top')
    return fig


SCATTER_NOTE = ('Colors identify body systems or established study domains. Upward/downward triangles indicate positive/inverse effects; '
                'size reflects absolute effect magnitude, and dark outlines mark Bonferroni-significant results. −log10(P) is capped at 50. '
                'Labels, when present, are restricted to FDR- or Bonferroni-significant results.')
FOREST_NOTE = ('Forests show up to three Bonferroni-significant features per body system or established domain, selected across both directions '
               'by descending absolute log2 ratio (OR, HR or IRR), or absolute β for continuous outcomes. Ranking is within each domain; '
               'selection precedes splitting into positive/inverse panels where shown. '
               'Whiskers are 95% CIs; ratio axes use a null of 1 and β axes a null of 0. Original multiplicity flags are retained.')


def disease(number, rows, effect, name):
    selected = select(name, rows)
    fig = newfig(f'Figure {number}  {name.capitalize()} disease PheWAS', 10 if number == 1 else 9)
    legend(fig, order_groups(rows), .945, 6, 7.5)
    scatter_axis(fig.add_axes([.055, .625, .925, .21]), rows, 'manhattan', effect, 'A  Manhattan plot', 8)
    if number == 1:
        scatter_axis(fig.add_axes([.06, .16, .19, .34]), rows, 'volcano', effect, 'B  Volcano plot', 3)
        forest_axis(fig.add_axes([.285, .075, .21, .47]), selected, effect, 'C1  Positive', 1)
        inverse = [r for r in selected if r['signed'] < 0]
        groups = order_groups(inverse)
        cutoff = min(range(1,len(groups)),key=lambda k: abs(sum(r['group'] in groups[:k] for r in inverse) - len(inverse)/2))
        for col, group_set in enumerate([groups[:cutoff], groups[cutoff:]]):
            subset = [r for r in inverse if r['group'] in group_set]
            forest_axis(fig.add_axes([.54 + .235 * col, .075, .21, .47]), subset, effect,
                        'C2  Inverse' if col == 0 else 'C2  Inverse (cont.)', fontsize=6.6, scale_rows=inverse)
    else:
        scatter_axis(fig.add_axes([.06, .12, .25, .39]), rows, 'volcano', effect, 'B  Volcano plot', 4)
        forest_axis(fig.add_axes([.36, .075, .29, .475]), selected, effect, 'C1  Positive associations', 1)
        forest_axis(fig.add_axes([.695, .075, .29, .475]), selected, effect, 'C2  Inverse associations', -1)
    fig.text(.035, .027, 'Forest selection: top 3 per body system by |log₂ effect|, among Bonferroni-significant results.', fontsize=8)
    cap = (f'Figure {number}. {name.capitalize()} disease PheWAS. A, Manhattan plot; B, volcano plot; C1, positive forest; C2, inverse forest. '
           f'All {len(rows):,} primary {name} rows compare AHI ≥5 versus <5 using ' + ('m4_bmi' if number == 1 else 'm4') +
           f' estimates ({effect}), excluding sleep apnea (Phecode 327.3 and subcodes). {SCATTER_NOTE} {FOREST_NOTE}')
    save(fig, number, cap)


def paired_scatter(number, kind, med, proc):
    fig = newfig(f'Figure {number}  Five-year medication and procedure associations')
    for rows, effect, letter, name, top in [(med, 'OR', 'A', 'MedWAS', .92), (proc, 'IRR', 'B', 'ProcWAS', .46)]:
        groups = order_groups(rows)
        legend(fig, groups, top, 6, 7.2)
        ax = fig.add_axes([.065, top - .33, .91, .22])
        scatter_axis(ax, rows, kind, effect, f'{letter}  {name} · {len(rows)} features', 6)
    cap = (f'Figure {number}. Five-year MedWAS and ProcWAS {kind} plots. A, all 218 medication-class rows (OR); '
           f'B, all 67 procedure rows (IRR). Fully adjusted m4 estimates compare AHI ≥5 versus <5. '
           f'MedWAS colors identify established therapeutic categories; ProcWAS colors identify body systems. {SCATTER_NOTE} '
           'Both 1- and 5-year estimates remain in the supplemental atlas.')
    save(fig, number, cap)


def multi_scatter(number, kind, util, qwas):
    fig = newfig(f'Figure {number}  Five-year emergency-department and questionnaire associations')
    all_util = order_groups(util)
    legend(fig, all_util, .948, 6, 7.3)
    specs = [('presence', 'OR', 'A', .62), ('count_among_present', 'IRR', 'B', .36)]
    for part, effect, letter, bottom in specs:
        for col, role in enumerate(['ED_primary', 'ED_secondary']):
            rows = [r for r in util if r['part'] == part and r['combo'] == role]
            ax = fig.add_axes([.06 + .50 * col, bottom, .415, .17])
            scatter_axis(ax, rows, kind, effect, f'{letter}{col + 1}  ED {"primary" if col == 0 else "secondary"} · {"presence" if part == "presence" else "conditional rate"}', 3)
            if kind == 'manhattan':
                # Use a common legend numbering across roles and estimands.
                groups = order_groups(rows)
                ax.set_xticklabels([str(all_util.index(g) + 1) for g in groups])
    qgroups = order_groups(qwas)
    legend(fig, qgroups, .298, 5, 7.0)
    for col, form in enumerate(['binary', 'continuous']):
        rows = [r for r in qwas if r['form'] == form]
        ax = fig.add_axes([.06 + .50 * col, .075, .415, .13])
        scatter_axis(ax, rows, kind, 'OR' if col == 0 else 'β', f'C{col + 1}  QWAS · {form}', 2)
        if kind == 'manhattan':
            ax.set_xticklabels([str(qgroups.index(g) + 1) for g in order_groups(rows)])
    cap = (f'Figure {number}. ED-UtilWAS and QWAS {kind} plots. A1/A2, five-year ED-primary/secondary diagnosis presence (OR); '
           'B1/B2, five-year ED-primary/secondary encounter rates conditional on ≥1 qualifying encounter for that diagnosis (IRR); '
           'C1/C2, 83 binary and 49 continuous QWAS rows (OR and rank-inverse-normal β). All primary m4 rows are displayed. '
           f'Colors identify Phecode body systems for ED outcomes and questionnaire domains for QWAS. {SCATTER_NOTE} '
           'Both 1- and 5-year ED estimates remain in the supplemental atlas; higher FOSQ scores indicate better function.')
    save(fig, number, cap)


def forests9(med, proc, util):
    fig = newfig('Figure 9  Five-year medication, procedure and ED forest plots', 11)
    medsel = select('med', med)
    procsel = select('proc', proc)
    primary = select('ed_primary_presence', [r for r in util if r['part'] == 'presence' and r['combo'] == 'ED_primary'])
    secondary = select('ed_secondary_presence', [r for r in util if r['part'] == 'presence' and r['combo'] == 'ED_secondary'])
    cond1 = select('ed_primary_conditional', [r for r in util if r['part'] == 'count_among_present' and r['combo'] == 'ED_primary'])
    cond2 = select('ed_secondary_conditional', [r for r in util if r['part'] == 'count_among_present' and r['combo'] == 'ED_secondary'])
    inverse = [r for r in medsel if r['signed'] < 0]
    groups = order_groups(inverse)
    cutoff = min(range(1,len(groups)),key=lambda k: abs(sum(r['group'] in groups[:k] for r in inverse) - len(inverse)/2))
    forest_axis(fig.add_axes([.04,.49,.28,.39]),medsel,'OR','A1  MedWAS · positive',1,7.0)
    for col, group_set in enumerate([groups[:cutoff],groups[cutoff:]]):
        forest_axis(fig.add_axes([.37+.33*col,.49,.28,.39]),[r for r in inverse if r['group'] in group_set],
                    'OR','A2  MedWAS · inverse' if col == 0 else 'A2  MedWAS · inverse (cont.)',fontsize=7.0,scale_rows=inverse)
    forest_axis(fig.add_axes([.04,.19,.28,.215]),procsel,'IRR','B  ProcWAS',fontsize=7.0)
    forest_axis(fig.add_axes([.37,.045,.28,.36]),primary,'OR','C1  ED-primary presence',fontsize=7.0)
    forest_axis(fig.add_axes([.70,.045,.28,.36]),secondary,'OR','C2  ED-secondary presence',fontsize=7.0)
    forest_axis(fig.add_axes([.04,.045,.28,.075]),cond1+cond2,'IRR','C3  ED conditional · secondary',fontsize=7.0)
    fig.text(.035,.93,'Within each analysis: top 3 per body system / therapeutic category by effect size, after Bonferroni correction.',fontsize=9)
    if not cond1 and not cond2:
        ax = fig.axes[-1]
        for txt in list(ax.texts):
            txt.remove()
        ax.text(.5,.55,'No conditional-rate features survive\nBonferroni correction\n(primary or secondary ED diagnoses)',
                transform=ax.transAxes,ha='center',va='center',fontsize=9,color='#666666')
    cap = ('Figure 9. Five-year MedWAS, ProcWAS and ED-UtilWAS forests. A1/A2, positive/inverse medication-class associations (OR); '
           'B, procedure rates (IRR); C1/C2, ED-primary/secondary diagnosis presence in the full cohort (OR); '
           'C3, conditional ED rates (IRR). All estimates use m4 and AHI ≥5 versus <5. '
           f'{FOREST_NOTE} Selection is separate for each ED diagnosis role and estimand; medication groups use established therapeutic categories. '
           'C3 shows the single Bonferroni-significant secondary-diagnosis conditional-rate result; no primary-diagnosis result qualifies. '
           'Both 1- and 5-year estimates remain in the supplemental atlas.')
    save(fig, 9, cap)


def forests10(qwas, beh):
    fig = newfig('Figure 10  Questionnaire and behavior forest plots')
    qb = select('qwas_binary',[r for r in qwas if r['form']=='binary'])
    qc = select('qwas_continuous',[r for r in qwas if r['form']=='continuous'])
    bb = select('beh_binary',[r for r in beh if r['effect_type']=='OR'])
    bc = select('beh_continuous',[r for r in beh if r['effect_type']=='beta'])
    specs = [(qb,'OR','A1  QWAS · binary',.04,.19,.44,.64),
             (qc,'β','A2  QWAS · continuous',.53,.44,.43,.39),
             (bb,'OR','B1  BehWAS · binary',.53,.225,.205,.14),
             (bc,'β','B2  BehWAS · continuous',.78,.225,.205,.14)]
    # Give each behavioral panel enough width for full feature labels.
    specs = [(qb,'OR','A1  QWAS · binary',.035,.33,.445,.54),
             (qc,'β','A2  QWAS · continuous',.53,.33,.445,.54),
             (bb,'OR','B1  BehWAS · binary',.035,.07,.445,.19),
             (bc,'β','B2  BehWAS · continuous',.53,.07,.445,.19)]
    for rows,effect,title,x,y,w,h in specs:
        forest_axis(fig.add_axes([x,y,w,h]),rows,effect,title,fontsize=7.4)
    fig.text(.035,.93,'Top 3 per established questionnaire / behavior domain by effect size, after Bonferroni correction.',fontsize=9)
    fig.text(.035,.017,'* Referral-driving symptom. Higher FOSQ scores indicate better function. Medication adherence / PDC are among users.',fontsize=8)
    cap = ('Figure 10. QWAS and BehWAS forests. A1, binary questionnaire outcomes (OR); A2, continuous questionnaire outcomes '
           '(rank-inverse-normal β); B1/B2, binary/continuous behavior outcomes (OR/β). All estimates use m4 and AHI ≥5 versus <5. '
           f'{FOREST_NOTE} QWAS uses established questionnaire domains. BehWAS domains are medication adherence, vaccination, attendance, '
           'and work leave; five-year estimates are used for windowed outcomes and lifetime estimates for undated outcomes. '
           'Asterisks mark referral-driving symptoms; higher FOSQ scores indicate better function. Medication adherence and PDC are among users. '
           'BehWAS remains preliminary; results that fail Bonferroni correction are omitted.')
    save(fig,10,cap)


def main():
    prevalence, incidence = source.load_disease_data()
    data = source.base.load_data()
    med = [r for r in data['med'] if r['window']=='5yr']
    proc = [r for r in data['proc'] if r['window']=='5yr']
    util = [dict(r,group=r['detail']) for r in data['util'] if r['window']=='5yr']
    qwas = data['qwas']
    beh = [r for r in data['beh'] if r['window'] in {'5yr','lifetime'}]
    for r in beh:
        outcome = r['outcome']
        r['group'] = ('Medication adherence' if outcome in {'adherence_pdc','adherent'} else
                      'Vaccination' if outcome.startswith('flu_') else
                      'Attendance' if outcome in {'any_noshow','no_show_pct'} else 'Work leave')
    init_colors(prevalence+incidence+med+proc+util+qwas+beh)
    disease(1,prevalence,'OR','prevalence')
    disease(2,incidence,'HR','incidence')
    paired_scatter(5,'manhattan',med,proc)
    multi_scatter(6,'manhattan',util,qwas)
    paired_scatter(7,'volcano',med,proc)
    multi_scatter(8,'volcano',util,qwas)
    forests9(med,proc,util)
    forests10(qwas,beh)
    (OUT/'manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'selection_audit.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'color_key.json').write_text(json.dumps(COLORS,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{'rows':v['available'],'bon':v['bonferroni_significant'],'selected':len(v['selected'])} for k,v in AUDIT.items()},indent=2))


if __name__ == '__main__':
    main()
