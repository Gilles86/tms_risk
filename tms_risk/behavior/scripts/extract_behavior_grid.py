"""Observed cTBS effect on choice, on the same (safe payoff x ratio) grid as the model.

Figure 5's first three columns are all model quantities. This produces the fourth: what
participants actually did, binned the same way, so the model's predicted effect can be
read against the data without leaving the figure.

`chose_risky` is used directly, so there is no option-1/option-2 convention to get
wrong -- the column already refers to the risky option regardless of presentation
order. Choice proportions are averaged within subject first, then across subjects, so
each participant contributes equally regardless of how many trials landed in a cell.

    python -m tms_risk.behavior.scripts.extract_behavior_grid

Writes notes/data/behavior_effect_grid.tsv with one row per
(order, safe payoff, ratio bin): the vertex and IPS choice proportions, their
difference, a between-subject SEM for that difference, and the trial and subject counts
behind it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(bids_folder, out_dir, n_ratio_bins):
    from tms_risk.utils.data import get_all_behavior

    df = get_all_behavior(bids_folder=bids_folder, exclude_outliers=True)
    df = df[df.index.get_level_values('session').astype(str).str.startswith(('2', '3'))]
    d = df.reset_index()
    d['order'] = np.where(d['p1'] == 0.55, 'Risky first', 'Risky second')
    d['ratio'] = d['n_risky'] / d['n_safe']
    # quantile bins so every bin carries a comparable number of trials
    d['ratio_bin'] = pd.qcut(d['ratio'], n_ratio_bins, labels=False)
    d['ratio_mid'] = d.groupby('ratio_bin')['ratio'].transform('mean')
    d['y'] = d['chose_risky'].astype(float)

    keys = ['order', 'n_safe', 'ratio_bin', 'ratio_mid']
    # subject means first, so each participant weighs the same
    per_sub = (d.groupby(keys + ['subject', 'stimulation_condition'])['y']
                 .mean().unstack('stimulation_condition'))
    per_sub = per_sub.dropna()                      # subjects seen in both conditions
    per_sub['delta'] = per_sub['ips'] - per_sub['vertex']

    g = per_sub.groupby(keys)
    out = pd.DataFrame({
        'vertex': g['vertex'].mean(),
        'ips': g['ips'].mean(),
        'delta': g['delta'].mean(),
        'sem': g['delta'].sem(),
        'n_subjects': g['delta'].size(),
    }).reset_index()
    counts = d.groupby(keys).size().rename('n_trials').reset_index()
    out = out.merge(counts, on=keys)

    # invariant: the aggregate must reproduce the direction seen in the raw choices
    for order, gg in out.groupby('order'):
        rho = np.corrcoef(gg.ratio_mid.rank(), gg.vertex.rank())[0, 1]
        assert rho > 0, (f'P(chose risky) must rise with the payoff ratio, '
                         f'got rho = {rho:.2f} for {order}')

    # Collapsed over the ratio bins: 35 subjects per cell instead of ~25, so this is
    # the version that can carry a meaningful error bar. The 2D grid above is too
    # thin per cell to read as evidence on its own.
    keys2 = ['order', 'n_safe']
    ps2 = (d.groupby(keys2 + ['subject', 'stimulation_condition'])['y']
             .mean().unstack('stimulation_condition').dropna())
    ps2['delta'] = ps2['ips'] - ps2['vertex']
    g2 = ps2.groupby(keys2)
    out2 = pd.DataFrame({
        'vertex': g2['vertex'].mean(), 'ips': g2['ips'].mean(),
        'delta': g2['delta'].mean(), 'sem': g2['delta'].sem(),
        'n_subjects': g2['delta'].size(),
    }).reset_index()
    p2 = Path(out_dir) / 'behavior_effect_by_safe.tsv'
    out2.to_csv(p2, sep='\t', index=False)
    print(f'wrote {p2}  ({len(out2)} cells, '
          f'{out2.n_subjects.min()}-{out2.n_subjects.max()} subjects per cell)')
    print(out2.round(3).to_string(index=False))

    p = Path(out_dir) / 'behavior_effect_grid.tsv'
    out.to_csv(p, sep='\t', index=False)
    print(f'wrote {p}  ({len(out)} cells, {out.n_trials.sum()} trials, '
          f'{out.n_subjects.min()}-{out.n_subjects.max()} subjects per cell)')
    print('\nobserved Δ P(chose risky), IPS − vertex:')
    piv = out.pivot_table(index='n_safe', columns=['order', 'ratio_bin'], values='delta')
    print(piv.round(3).to_string())
    print('\nmean over cells, by order:')
    print(out.groupby('order').delta.mean().round(4).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--out_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--n_ratio_bins', default=4, type=int)
    a = parser.parse_args()
    main(a.bids_folder, a.out_dir, a.n_ratio_bins)
