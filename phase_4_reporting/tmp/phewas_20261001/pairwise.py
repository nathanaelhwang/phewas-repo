"""Local computation on existing no-identifier exports; emit aggregate tests only."""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CODES = ['401.1', '250.2', '428.2', '427.2']
SPECS = [
    ('A', 'survival_curves/survival_long.parquet', 'severity',
     ['No OSA', 'Mild', 'Moderate', 'Severe'], 'survival_curves/curve_severity.csv'),
    ('B', 'survival_curves_landmark/survival_long_landmark.parquet', 'adherence',
     ['4+ hr/night', '2-4 hr/night', '0-2 hr/night', 'never-started', 'unknown'],
     'survival_curves_landmark/curve_adherence.csv'),
]
records = []
for letter, path, group_col, groups, curve_path in SPECS:
    columns = ['phecode', 'time_years', 'etype', group_col]
    if letter == 'B': columns.append('window_days')
    data = pd.read_parquet(ROOT/'results/incwas_results'/path, columns=columns)
    data = data[data.phecode.astype(str).isin(CODES)].copy()
    if letter == 'B': data = data[data.window_days == 180].copy()
    days = data.time_years.to_numpy(float) * 365.25
    assert np.all(np.abs(days - np.rint(days)) <= .001827)
    data['time_years'] = np.rint(days)/365.25
    curves = pd.read_csv(ROOT/'results/incwas_results'/curve_path, dtype={'phecode':str})
    if letter == 'B': curves = curves[curves.window_days == 180]
    for i, code in enumerate(CODES, 1):
        sub = data[data.phecode.astype(str) == code]
        # Verify each tested cohort against all annual risk/event counts in the plot export.
        for group in groups:
            g = sub[sub[group_col] == group]
            source = curves[(curves.phecode == code) & (curves.group == group)]
            for year in range(4):
                row = source[source.time_years == year].iloc[0]
                assert (g.time_years >= year).sum() == row.n_at_risk
                assert ((g.time_years <= year) & (g.etype == 1)).sum() == row.n_events_cum
        for g1, g2 in combinations(groups, 2):
            a, b = [sub[sub[group_col] == g] for g in (g1,g2)]
            ta,tb = [np.minimum(g.time_years.to_numpy(),3) for g in (a,b)]
            ea,eb = [((g.etype.to_numpy() == 1) & (g.time_years.to_numpy() <= 3)) for g in (a,b)]
            result = logrank_test(ta,tb,ea,eb)
            # Independent risk-set / hypergeometric-variance calculation, including ties.
            times = np.unique(np.r_[ta[ea],tb[eb]])
            y1 = len(ta)-np.searchsorted(np.sort(ta),times,side='left')
            y2 = len(tb)-np.searchsorted(np.sort(tb),times,side='left')
            d1 = np.searchsorted(np.sort(ta[ea]),times,side='right')-np.searchsorted(np.sort(ta[ea]),times,side='left')
            d2 = np.searchsorted(np.sort(tb[eb]),times,side='right')-np.searchsorted(np.sort(tb[eb]),times,side='left')
            y,d = y1+y2,d1+d2
            score = np.sum(d1-d*y1/y)
            valid = y>1
            variance = np.sum(y1[valid]*y2[valid]*d[valid]*(y[valid]-d[valid])/(y[valid]**2*(y[valid]-1)))
            stat = score**2/variance
            assert np.isclose(stat,result.test_statistic,rtol=1e-9,atol=1e-9)
            records.append(dict(panel=f'{letter}{i}',phecode=code,group1=g1,group2=g2,
                                p=float(result.p_value),chi2=float(stat)))
assert len(records) == 64
(OUT/'pairwise_pvalues.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Verified 64 pairwise tests, three-year horizon, all annual risk/event counts match existing curves.')
