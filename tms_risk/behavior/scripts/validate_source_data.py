"""Consistency gate for everything under notes/data/.

Every figure in this paper is rebuilt from a TSV rather than from a trace, which makes
the figures cheap but means a silent error in an extraction propagates into every plot
that reads it and is invisible at plotting time. This script is the gate: it asserts
the invariants those files must satisfy, including cross-file ones, and exits non-zero
if any fail.

It exists because of a real failure. `decision_space.<label>.tsv` stored
`norm.cdf((EV2-EV1)/s)`, i.e. P(choose the SECOND option), under the column name
`p_vertex` and was plotted as "P(chose risky)". On risky-first trials the second
option is the SAFE one, so that column -- and the `effect` column derived from it --
carried the wrong sign across half the design, and the figure looked plausible.
Check `decision_space_p_rises_with_ratio` below is the one that catches it: observed
P(chose risky) rises with the payoff ratio in both presentation orders, so any model
quantity claiming to be P(chose risky) must rise too.

    python -m tms_risk.behavior.scripts.validate_source_data
    python -m tms_risk.behavior.scripts.validate_source_data --label flexible2nf

Add a check whenever a new derived file gains an invariant worth trusting.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9


class Report:
    def __init__(self):
        self.rows = []

    def add(self, fname, check, ok, detail=''):
        self.rows.append((fname, check, bool(ok), detail))

    def skip(self, fname, check, why):
        self.rows.append((fname, check, None, why))

    def summary(self):
        run = [r for r in self.rows if r[2] is not None]
        bad = [r for r in run if not r[2]]
        width = max((len(r[0]) for r in self.rows), default=10)
        for fname, check, ok, detail in self.rows:
            mark = '  --' if ok is None else ('  ok' if ok else 'FAIL')
            line = f'{mark}  {fname:<{width}}  {check}'
            if detail:
                line += f'   [{detail}]'
            print(line)
        print(f'\n{len(run) - len(bad)}/{len(run)} checks passed'
              + (f', {len(self.rows) - len(run)} skipped' if len(self.rows) - len(run) else ''))
        return len(bad)


def spearman(x, y):
    """Rank correlation without a scipy dependency."""
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def check_decision_space(data, label, rep):
    f = data / f'decision_space.{label}.tsv'
    name = f.name
    if not f.exists():
        rep.skip(name, 'decision_space', 'not present')
        return
    d = pd.read_csv(f, sep='\t')

    rep.add(name, 'p_vertex within [0, 1]',
            d.p_vertex.between(0, 1).all(),
            f'range {d.p_vertex.min():.3f}-{d.p_vertex.max():.3f}')

    # The one that catches the P(option 2) bug.
    for order, g in d.groupby('order'):
        rho = spearman(g.ratio, g.p_vertex)
        rep.add(name, f'P(chose risky) rises with ratio [{order}]', rho > .5,
                f'rho = {rho:+.3f}')

    if 'leverage' in d:
        rep.add(name, 'leverage is non-negative', (d.leverage >= 0).all(),
                f'min {d.leverage.min():.4f}')
    if 'cause' in d:
        rep.add(name, 'cause (a ratio) is centred near 1',
                bool(0.5 < d.cause.mean() < 2.0), f'mean {d.cause.mean():.3f}')
    if {'noise_vertex', 'noise_ips'} <= set(d):
        rep.add(name, 'total noise is positive in both conditions',
                bool((d.noise_vertex > 0).all() and (d.noise_ips > 0).all()))
    if {'ev_safe_vertex', 'ev_risky_vertex'} <= set(d):
        rep.add(name, 'perceived EVs are positive',
                bool((d.ev_safe_vertex > 0).all() and (d.ev_risky_vertex > 0).all()))


def check_percepts_by_order(data, label, rep):
    f = data / f'pmc_percepts_by_order.{label}.tsv'
    name = f.name
    if not f.exists():
        rep.skip(name, 'percepts_by_order', 'not present')
        return
    d = pd.read_csv(f, sep='\t')
    err = float((d.delta - (d.ips - d.vertex)).abs().max())
    rep.add(name, 'delta == ips - vertex', err < 1e-6, f'max err {err:.2e}')
    rep.add(name, 'delta lies inside its own credible interval',
            bool(((d.lo <= d.delta + TOL) & (d.delta - TOL <= d.hi)).all()))
    # position must be consistent with order: the safe option is first exactly when
    # the risky option is second
    exp = np.where((d.option == 'safe') == (d.order == 'Risky second'), 'first', 'second')
    rep.add(name, 'position is consistent with presentation order',
            bool((d.position.values == exp).all()))
    rep.add(name, 'percepts are positive',
            bool((d.vertex > 0).all() and (d.ips > 0).all()))


def check_curves(data, label, rep):
    f = data / f'pmcpars_curves.{label}.tsv'
    name = f.name
    if not f.exists():
        rep.skip(name, 'pmcpars_curves', 'not present')
        return
    d = pd.read_csv(f, sep='\t')
    pos = d[d.stimulation.isin(['ips', 'vertex'])]
    rep.add(name, 'noise is strictly positive', bool((pos.nu > 0).all()),
            f'min {pos.nu.min():.4f}')
    rep.add(name, 'nu lies inside its own credible interval',
            bool(((pos.lo <= pos.nu + TOL) & (pos.nu - TOL <= pos.hi)).all()))
    # the stored contrast must equal the difference of the stored means
    worst, worst_term = 0., ''
    for term, g in d.groupby('term'):
        piv = g.pivot_table(index='payoff', columns='stimulation', values='nu')
        if not {'ips', 'vertex', 'ips - vertex'} <= set(piv.columns):
            continue
        e = float((piv['ips - vertex'] - (piv['ips'] - piv['vertex'])).abs().max())
        if e > worst:
            worst, worst_term = e, term
    rep.add(name, "'ips - vertex' equals ips minus vertex", worst < 1e-6,
            f'max err {worst:.2e}{" on " + worst_term if worst_term else ""}')


def check_ppc(data, label, rep):
    f = data / f'ppc_fig3a.{label}.tsv'
    name = f.name
    if not f.exists():
        rep.skip(name, 'ppc_fig3a', 'not present')
        return
    d = pd.read_csv(f, sep='\t')
    for col in ['mean', 'observed']:
        for order, g in d.groupby('order'):
            gg = g.groupby('frac')[col].mean().sort_index()
            rho = spearman(gg.index.values, gg.values)
            rep.add(name, f'{col} P(chose risky) rises with ratio [{order}]', rho > .5,
                    f'rho = {rho:+.3f}')
    rep.add(name, 'model mean lies inside its own predictive interval',
            bool(((d.lo <= d['mean'] + TOL) & (d['mean'] - TOL <= d.hi)).all()))
    rep.add(name, 'probabilities within [0, 1]',
            bool(d['mean'].between(0, 1).all() and d.observed.between(0, 1).all()))


def check_cross(data, label, rep):
    """decision_space and the PPC must agree about which way choices go."""
    fa, fb = data / f'decision_space.{label}.tsv', data / f'ppc_fig3a.{label}.tsv'
    if not (fa.exists() and fb.exists()):
        rep.skip('cross-file', 'decision_space vs ppc_fig3a', 'need both files')
        return
    a, b = pd.read_csv(fa, sep='\t'), pd.read_csv(fb, sep='\t')
    for order in a.order.unique():
        ga = a[a.order == order]
        gb = b[b.order == order]
        if not len(gb):
            continue
        ra = spearman(ga.ratio, ga.p_vertex)
        rb = spearman(gb.frac, gb.observed)
        rep.add('cross-file', f'model and data agree in direction [{order}]',
                np.sign(ra) == np.sign(rb), f'model {ra:+.2f} vs observed {rb:+.2f}')


def main(data_dir, labels):
    data = Path(data_dir)
    if not labels:
        labels = sorted({p.name.split('.', 1)[1].rsplit('.tsv', 1)[0]
                         for p in data.glob('decision_space.*.tsv')}
                        | {p.name.split('.', 1)[1].rsplit('.tsv', 1)[0]
                           for p in data.glob('pmcpars_curves.*.tsv')})
    rep = Report()
    for label in labels:
        print(f'\n=== {label} ===')
        n0 = len(rep.rows)
        check_decision_space(data, label, rep)
        check_percepts_by_order(data, label, rep)
        check_curves(data, label, rep)
        check_ppc(data, label, rep)
        check_cross(data, label, rep)
        for fname, check, ok, detail in rep.rows[n0:]:
            mark = '  --' if ok is None else ('  ok' if ok else 'FAIL')
            print(f'{mark}  {check}' + (f'   [{detail}]' if detail else ''))
    print('\n' + '=' * 70)
    n_bad = sum(1 for r in rep.rows if r[2] is False)
    n_run = sum(1 for r in rep.rows if r[2] is not None)
    print(f'{n_run - n_bad}/{n_run} checks passed across {len(labels)} label(s)')
    if n_bad:
        print('\nFAILING:')
        for fname, check, ok, detail in rep.rows:
            if ok is False:
                print(f'  {fname}: {check}   [{detail}]')
    return 1 if n_bad else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', nargs='*', default=None)
    a = parser.parse_args()
    sys.exit(main(a.data_dir, a.label))
