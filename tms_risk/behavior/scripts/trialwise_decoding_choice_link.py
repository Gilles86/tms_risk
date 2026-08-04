"""Within-subject test: does trial-by-trial decoding quality predict the choice?

The across-subject correlations are difference scores over two sessions, so they can
only ever be as clean as the session pairing. This test is immune to that: it asks,
*inside* a session, whether the trials on which the first option's numerosity was
decoded more precisely are also the trials on which the choice was more consistent.

The decoded stimulus is `n1`, the **first-presented** option. Under the PMC account the
first option carries both noise components, so a noisier representation of it should
(a) flatten the psychometric slope and (b) pull that option's percept toward the prior,
which on risky-second trials (safe option first) means *more* risky choices.

Decoding quality is residualised on log(n1) within session, because the decoder's error
grows with distance from the grid centre and would otherwise just re-encode the payoff.

    python -m tms_risk.behavior.scripts.trialwise_decoding_choice_link
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / 'notes' / 'data'


def residualise(y, x):
    X = np.column_stack([np.ones(len(x)), x, x ** 2])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def fit_logit(y, X, ridge=1e-3):
    X = np.column_stack([np.ones(len(y))] + list(X.T))
    y = np.asarray(y, dtype=float)

    def nll(b):
        z = np.clip(X @ b, -30, 30)
        return -np.sum(y * z - np.logaddexp(0, z)) + ridge * np.sum(b[1:] ** 2)

    return minimize(nll, np.zeros(X.shape[1]), method='BFGS').x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', default='NPCr2cm-cluster')
    ap.add_argument('--quality', default='post_spread',
                    choices=['post_spread', 'log_abs_err'])
    args = ap.parse_args()

    t = pd.read_csv(DATA / 'bb_decoding_trials.tsv', sep='\t')
    t = t[(t['mask'] == args.mask) & t['chose_risky'].notna()].copy()
    t['lr'] = np.log(t['n_risky'] / t['n_safe'])

    # higher = better decoding, residualised on the presented numerosity
    rows = []
    for (sub, ses), g in t.groupby(['subject', 'session']):
        g = g.copy()
        q = -g[args.quality].values
        g['q'] = residualise(q, np.log(g['n1'].values))
        g['q'] = (g['q'] - g['q'].mean()) / g['q'].std()
        rows.append(g)
    t = pd.concat(rows)

    print(f'=== Mask {args.mask}, decoding quality = -{args.quality} '
          f'(residualised on log n1)\n')

    by_order = []
    for order_label, sel in [('risky second (safe first)', ~t['risky_first']),
                             ('risky first', t['risky_first'])]:
        d = t[sel]
        res = []
        for sub, g in d.groupby('subject'):
            if len(g) < 60:
                continue
            b = fit_logit(g['chose_risky'].astype(float),
                          np.column_stack([g['lr'], g['q'], g['lr'] * g['q']]))
            res.append({'subject': sub, 'b_lr': b[1], 'b_q': b[2], 'b_lr_x_q': b[3]})
        res = pd.DataFrame(res)
        by_order.append(res.assign(order=order_label))
        print(f'-- {order_label}: n = {len(res)} subjects')
        for c, meaning in [('b_lr', 'psychometric slope'),
                           ('b_q', 'better decoding -> chose risky'),
                           ('b_lr_x_q', 'better decoding -> steeper slope')]:
            v = res[c].dropna()
            tt, p = stats.ttest_1samp(v, 0)
            w = stats.wilcoxon(v)[1]
            print(f'   {c:10s} {meaning:36s} mean {v.mean():+.4f}  '
                  f't({len(v)-1}) = {tt:+5.2f}  p = {p:.4f}  (Wilcoxon p = {w:.4f})')
        print()

    pd.concat(by_order).to_csv(DATA / f'bb_trialwise_byorder_{args.quality}.tsv',
                               sep='\t', index=False)

    # does the coupling itself change with cTBS?
    print('-- Same, split by stimulation condition (risky-second trials only):')
    d = t[~t['risky_first'] & t['stimulation_condition'].isin(['ips', 'vertex'])]
    res = []
    for (sub, cond), g in d.groupby(['subject', 'stimulation_condition']):
        if len(g) < 40:
            continue
        b = fit_logit(g['chose_risky'].astype(float),
                      np.column_stack([g['lr'], g['q'], g['lr'] * g['q']]))
        res.append({'subject': sub, 'cond': cond, 'b_lr': b[1], 'b_q': b[2],
                    'b_lr_x_q': b[3]})
    res = pd.DataFrame(res)
    for c in ['b_lr', 'b_q', 'b_lr_x_q']:
        w = res.pivot_table(index='subject', columns='cond', values=c)
        diff = (w['ips'] - w['vertex']).dropna()
        tt, p = stats.ttest_1samp(diff, 0)
        print(f'   {c:10s} vertex {w["vertex"].mean():+.4f}  ips {w["ips"].mean():+.4f}'
              f'  delta {diff.mean():+.4f}  t({len(diff)-1}) = {tt:+5.2f}  p = {p:.4f}')

    res.to_csv(DATA / f'bb_trialwise_{args.quality}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
