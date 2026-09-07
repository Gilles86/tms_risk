"""Full-grid verdict for the 24-cell lfx2 spline/hyperprior comparison.

Reads notes/data/lfxgrid_{loo,draws}.tsv + looi_*.npy (pointwise LOO).
Prints: the ladder with paired dSE vs best, per-config TMS gains (b - null,
paired), and factor summaries. Purely tabular; the figure comes from
plot_lfxgrid_progress.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[3] / 'notes' / 'data'

loo = pd.read_csv(DATA / 'lfxgrid_loo.tsv', sep='\t')
parts = loo['label'].str.extract(
    r'lfx2-(?P<basis>\w+)-(?P<mem>\w+)-(?P<hp>\w+)-(?P<tms>\w+)')
loo = pd.concat([loo, parts], axis=1).set_index('label')
looi = {l: np.load(DATA / f'looi_{l}.npy') for l in loo.index}
n = len(next(iter(looi.values())))

best = loo['elpd_loo'].idxmax()
loo['delta_best'] = loo['elpd_loo'] - loo.loc[best, 'elpd_loo']
loo['dse_best'] = [np.sqrt(n * np.var(looi[l] - looi[best])) for l in loo.index]

print('=== full ladder (best first) ===')
t = loo.sort_values('elpd_loo', ascending=False)
print(t[['basis', 'mem', 'hp', 'tms', 'elpd_loo', 'delta_best', 'dse_best',
         'p_loo', 'divergences', 'max_rhat']].round(2).to_string())

print('\n=== TMS gain per configuration (b - null, paired dSE) ===')
rows = []
for (basis, mem, hp), _ in loo.groupby(['basis', 'mem', 'hp']):
    lb = f'lfx2-{basis}-{mem}-{hp}-b'
    ln = f'lfx2-{basis}-{mem}-{hp}-null'
    if lb in looi and ln in looi:
        d = looi[lb] - looi[ln]
        rows.append(dict(basis=basis, mem=mem, hp=hp, gain=d.sum(),
                         dse=np.sqrt(n * np.var(d))))
g = pd.DataFrame(rows).sort_values('gain', ascending=False)
g['z'] = g['gain'] / g['dse']
print(g.round(2).to_string(index=False))

print('\n=== factor effects on ELPD (mean over matched cells) ===')
for factor, levels in [('basis', ['bs3', 'bs2', 'cr3']),
                       ('mem', ['fm', 'sm']), ('hp', ['dp', 'tp'])]:
    for tms in ['null', 'b']:
        sub = loo[loo.tms == tms]
        means = sub.groupby(factor)['elpd_loo'].mean()
        ref = means.max()
        s = '  '.join(f'{lv}: {means.get(lv, np.nan) - ref:+.1f}'
                      for lv in levels)
        print(f'{factor:6s} ({tms:4s}):  {s}   (relative to best level)')

print('\n=== diagnostics summary ===')
print(loo.groupby(['hp'])[['divergences', 'max_rhat']].agg(['mean', 'max']).round(3).to_string())
