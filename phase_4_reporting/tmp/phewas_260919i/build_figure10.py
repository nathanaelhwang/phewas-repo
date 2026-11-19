"""Add the saved pointwise No OSA confidence intervals to the phenotype CIF panels."""
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'tmp/phewas_260919d'
ORDER = [
    ('high-all', 'High All'), ('mild-all', 'Mild All'), ('no-osa', 'No OSA'),
    ('hypoxemic-comorbid', 'Hypoxemic Comorbid'),
    ('hypoxemic-symptomatic', 'Hypoxemic Symptomatic'),
    ('hypoxemia-predominant', 'Hypoxemia Predominant'),
    ('symptomatic-comorbid', 'Symptomatic Comorbid'),
    ('symptom-predominant', 'Symptom Predominant'),
    ('comorbidity-predominant', 'Comorbidity Predominant'),
]
ORANGE, GRAY, INK = '#D26930', '#758581', '#203E3B'
NO_OSA = '#171717'


def p_label(p):
    assert p is not None and 0 <= p <= 1
    return f'Log-rank p = {p:.1e}' if p < .001 else f'Log-rank p = {p:.3f}'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_curve(curve):
    t, y, lo, hi = [np.asarray(curve[k], float) for k in
                    ('time_years', 'cif_pct', 'ci_low_pct', 'ci_high_pct')]
    assert len(t) == len(y) == len(lo) == len(hi)
    assert np.isfinite(np.r_[t, y, lo, hi]).all()
    assert np.all(lo <= y + 1e-9) and np.all(y <= hi + 1e-9)
    assert np.all(np.diff(t) >= 0)
    assert t[0] == 0 and t[-1] == 3


def draw_curve(ax, curve, color, linestyle, label, zorder, shade,
               shade_alpha=.15, line_width=None):
    if shade:
        ax.fill_between(curve['time_years'], curve['ci_low_pct'], curve['ci_high_pct'],
                        color=color, alpha=shade_alpha, step='post', linewidth=0, zorder=1)
    line, = ax.step(curve['time_years'], curve['cif_pct'], where='post', color=color,
                    linestyle=linestyle,
                    lw=(1.4 if not shade else 1.55) if line_width is None else line_width,
                    label=label, zorder=zorder)
    # Validate the rendered artist uses the complete source event-time arrays.
    assert np.array_equal(line.get_xdata(), curve['time_years'])
    assert np.array_equal(line.get_ydata(), curve['cif_pct'])
    assert line.get_drawstyle() == 'steps-post'
    return line


def figure_caption():
    source_docx = ROOT / 'manuscript/drafts/PheWAS Tables and Figures 260919h.docx'
    with ZipFile(source_docx) as package:
        document = ElementTree.fromstring(package.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = [''.join(p.itertext()) for p in document.findall('w:body/w:p', ns)]
    captions = [text for text in paragraphs if text.startswith('Figure 10.')]
    assert len(captions) == 1
    original = captions[0]
    old_note = 'panel, without additional confidence shading.'
    new_note = 'panel, with shading for its pointwise 95% confidence interval.'
    assert original.count(old_note) == 1
    caption = original.replace(old_note, new_note)
    assert caption.replace(new_note, old_note) == original
    return caption


def main():
    curve_source = SOURCE / 'figure12_curves.json'
    annotation_source = SOURCE / 'figure12_annotations.json'
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (curve_source, annotation_source)}
    data = json.loads(curve_source.read_text())
    assert data['curve_granularity'] == 'event_time'
    data_before = digest(data)
    panels = {p['octant']: p for p in data['panels']}
    annotation_data = json.loads(annotation_source.read_text())
    annotations_before = digest(annotation_data)
    annotations = {p['octant']: p for p in annotation_data['annotations']}
    assert set(annotations) == {key for key, _ in ORDER}
    no_osa = data['no_osa']['curve']
    validate_curve(no_osa)
    groups = []
    for key, label in ORDER:
        if key == 'no-osa':
            panel = data['no_osa']
            focal, rest = panel['curve'], data['overall']['curve']
            reference = 'Pooled all eight OSA'
        else:
            panel = panels[key]
            focal, rest = panel['curves']['focal'], panel['curves']['other_seven']
            reference = 'Pooled other seven OSA'
        assert panel['curve_status'] == 'available'
        for curve in (focal, rest):
            validate_curve(curve)
        groups.append((key, label, focal, rest, reference, panel['logrank_p']))
    y_max = math.ceil(max(max(c['ci_high_pct']) for g in groups for c in g[2:4])) + 1
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'axes.titlesize': 11,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'path.simplify': False})
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.064, right=.985, top=.846, bottom=.128,
                        wspace=.18, hspace=.43)
    fig.text(.018, .975, 'FIGURE 10  ·  PHASE 3 CROSS-DOMAIN PHENOTYPES',
             fontsize=10, color=INK, weight='bold', va='top')
    fig.text(.018, .938, 'Heart failure incidence across all phenotypes and No OSA',
             fontsize=17, color=INK, weight='bold', va='top')
    fig.text(.018, .899, 'Heart failure NOS (Phecode 428.2)  |  Aalen–Johansen cumulative incidence with pointwise 95% confidence intervals',
             fontsize=10, color=INK, va='top')
    audit_panels = []
    no_osa_reference_hashes = []
    no_osa_ci_polygon_hashes = []
    for i, (ax, group) in enumerate(zip(axes.flat, groups)):
        key, label, focal, rest, reference, p = group
        ax.set_title(f'{chr(65+i)}  {label}', loc='left', fontweight='bold', pad=9, color=INK)
        rest_line = draw_curve(ax, rest, GRAY, '--', reference, 2, True)
        focal_line = draw_curve(ax, focal, NO_OSA if key == 'no-osa' else ORANGE,
                                ':' if key == 'no-osa' else '-', label, 3, True)
        handles = [focal_line, rest_line]
        if key != 'no-osa':
            no_osa_line = draw_curve(ax, no_osa, NO_OSA, ':', 'No OSA reference', 4, True,
                                     shade_alpha=.10, line_width=1.4)
            handles.append(no_osa_line)
            no_osa_reference_hashes.append(digest({'time_years': list(no_osa_line.get_xdata()),
                                                   'cif_pct': list(no_osa_line.get_ydata())}))
            assert no_osa_line.get_linewidth() == 1.4
        # The final band is No OSA in all panels, including its own panel.
        no_osa_ci_polygon_hashes.append(
            digest([path.vertices.tolist() for path in ax.collections[-1].get_paths()]))
        assert len(ax.lines) == (2 if key == 'no-osa' else 3)
        assert len(ax.collections) == (2 if key == 'no-osa' else 3)
        ax.text(.035, .94, p_label(p), transform=ax.transAxes, va='top', fontsize=9, color=INK)
        note = annotations[key]
        assert note['median_ahi'] is not None and math.isfinite(note['median_ahi'])
        hr_note = ('M4 HR: not estimated' if note['m4_hr'] is None else
                   f"M4 HR: {note['m4_hr']:.2f} ({note['m4_ci_low']:.2f}–{note['m4_ci_high']:.2f})")
        ax.text(.035, .835, f"Median AHI: {note['median_ahi']:.2f} events/h\n{hr_note}",
                transform=ax.transAxes, va='top', fontsize=8, linespacing=1.4, color=INK)
        ax.legend(handles, [line.get_label() for line in handles], loc='upper left',
                  bbox_to_anchor=(.012, .625), fontsize=7.5, frameon=False,
                  handlelength=2.3, borderaxespad=0, labelspacing=.3)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, y_max)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks(np.arange(0, y_max + .1, 1 if y_max <= 7 else 2))
        ax.grid(axis='y', color='#DDE5E2', lw=.6)
        ax.tick_params(colors=INK, length=3)
        ax.spines['left'].set_color('#91A29C')
        ax.spines['bottom'].set_color('#91A29C')
        ax.tick_params(labelbottom=True)
        if i % 3 == 0:
            ax.set_ylabel('Cumulative incidence (%)', color=INK)
        if i >= 6:
            ax.set_xlabel('Years since index', color=INK)
        audit_panels.append({
            'group': key, 'comparator': reference, 'logrank_p': p,
            'median_ahi': note['median_ahi'], 'm4_hr': note['m4_hr'],
            'm4_ci_low': note['m4_ci_low'], 'm4_ci_high': note['m4_ci_high'],
            'focal_points': len(focal['time_years']), 'pooled_points': len(rest['time_years']),
            'focal_curve_and_ci_sha256': digest(focal), 'pooled_curve_and_ci_sha256': digest(rest),
            'no_osa_reference_added': key != 'no-osa',
            'no_osa_ci_band_count': 1,
            'ci_band_count': len(ax.collections),
            'cif3_focal_pct': focal['cif_pct'][-1], 'cif3_reference_pct': rest['cif_pct'][-1],
        })
    fig.text(.018, .045,
             'Each OSA phenotype retains its comparison with the other seven OSA phenotypes; the dotted No OSA curve is an added visual reference.\n'
             'No OSA (AHI <5; black dotted) is compared with all eight OSA phenotypes pooled. No OSA is excluded from every pooled-seven group.\n'
             'Shading shows focal, pooled, and No OSA reference pointwise 95% CIs. Log-rank tests retain their original comparators and censor competing deaths.\n'
             'Median AHI describes each focal risk set. M4 HRs (95% CIs) retain their original adjusted estimates and model-cohort comparisons.',
             fontsize=8, linespacing=1.4, color=INK, va='center')
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'figure10_panel.{extension}', dpi=300, facecolor='white')
    plt.close(fig)
    assert len(no_osa_reference_hashes) == 8
    expected_no_osa_hash = digest({k: no_osa[k] for k in ('time_years', 'cif_pct')})
    assert set(no_osa_reference_hashes) == {expected_no_osa_hash}
    assert len(no_osa_ci_polygon_hashes) == 9 and len(set(no_osa_ci_polygon_hashes)) == 1
    expected_no_osa_ci_hash = digest({k: no_osa[k] for k in
                                    ('time_years', 'ci_low_pct', 'ci_high_pct')})
    assert digest(data) == data_before and digest(annotation_data) == annotations_before
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
               for p, h in source_hashes.items())
    original_summary = json.loads((ROOT / 'manuscript/figs-tables/260919d/Figure12_analysis_summary.json').read_text())
    original_panels = {p['group']: p for p in original_summary['panels']}
    for panel in audit_panels:
        assert all(panel[key] == value for key, value in original_panels[panel['group']].items())
    audit = {'figure': 10, 'curve_granularity': 'event_time', 'outcome': '428.2',
             'horizon_years': 3, 'source_files_sha256': source_hashes,
             'source_data_and_annotations_unchanged': True,
             'all_statistics_and_endpoint_estimates_match_previous_figure': True,
             'complete_source_arrays_verified_against_rendered_artists': True,
             'focal_and_pooled_ci_bands_retained_in_all_nine_panels': True,
             'identical_no_osa_reference_in_all_eight_osa_panels': True,
             'identical_no_osa_ci_geometry_in_all_nine_panels': True,
             'no_osa_pointwise_95_ci_added_to_all_eight_osa_panels': True,
             'no_osa_reference_ci_time_and_bounds_sha256': expected_no_osa_ci_hash,
             'no_osa_ci_polygon_sha256': no_osa_ci_polygon_hashes[0],
             'no_osa_reference_points': len(no_osa['time_years']),
             'no_osa_reference_time_and_cif_sha256': expected_no_osa_hash,
             'no_osa_panel_duplicate_reference': False, 'panels': audit_panels}
    (OUT / 'phenotype_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    caption = figure_caption()
    manifest = [{'number': 10, 'file': str(OUT / 'figure10_panel.png'), 'caption': caption,
                 'width_inches': 13, 'height_inches': 9,
                 'page_width_inches': 14, 'page_height_inches': 12.4}]
    (OUT / 'phenotype_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Figure 10: saved No OSA pointwise 95% CIs shown in all nine panels; all event-time curves and statistics retained.')


if __name__ == '__main__':
    main()
