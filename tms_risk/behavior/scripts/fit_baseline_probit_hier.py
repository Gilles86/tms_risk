"""Hierarchical Bayesian probit on the BASELINE session, and its PPC.

Figure 4 asks whether choice consistency depends on stake before any
stimulation. That needs the same estimator the rest of the paper uses -- a
hierarchical probit with partial pooling, per-subject intercepts and slopes --
not a maximum-likelihood fit with bootstrap intervals. Two reasons:

* pooling subjects within a cell FLATTENS the psychometric function, because
  participants sit at different indifference points, so the observed number has
  to be a per-subject quantity averaged over subjects, exactly like the model's;
* with a posterior in hand the uncertainty band on the curve is a genuine
  POSTERIOR PREDICTIVE interval -- simulate choices for the trials people
  actually saw, aggregate them the same way as the data -- so a point outside
  the band is a real misfit rather than a small error bar.

    y ~ Bernoulli(Phi(a[c] + sd_a * za[g] + exp(log_b[c] + sd_b * zb[g]) * log frac))

with c = (order, stake half) -- the four cells Figure 4 plots -- and
g = (subject, order), the level the nuisance heterogeneity lives at. Slopes are
positive by construction (log-normal offsets) so rnp = exp(-a/b) stays finite.

    python -m tms_risk.behavior.scripts.fit_baseline_probit_hier
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

REPO = Path(__file__).resolve().parents[3]


def main(bids_folder, out_stem, draws, tune, chains, seed, n_ppc):
    from tms_risk.utils.data import get_all_behavior
    d = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=False,
                         exclude_outliers=True)
    d = d.xs(1, 0, 'session', drop_level=False).reset_index()
    # `get_all_behavior` already supplies `chose_risky`. Do NOT recompute it as
    # `choice == 2`: option 2 is the risky one only when the risky option came
    # SECOND, so that formula silently inverts every risky-first trial (it makes
    # P(risky) DECREASE with the payoff ratio and drives the fitted slope to
    # ~0.02). `choice == 2` is bauer's dependent variable, not this one.
    d = d.dropna(subset=['chose_risky', 'frac', 'n_safe', 'n_risky'])
    d['order'] = d['risky_first'].map({True: 'Risky first',
                                       False: 'Risky second'})
    d['stake_chf'] = (d['n_safe'] + d['n_risky']) / 2
    # groupby.apply returning an ndarray per group yields a Series of arrays,
    # which then fails to concatenate as a string key -- use transform
    _med = d.groupby('subject')['stake_chf'].transform('median')
    d['stake'] = np.where(d['stake_chf'] > _med, 'high', 'low')
    # the ladder rung, so the PPC can be aggregated on each participant's own
    # calibrated ratios rather than on raw payoff ratios
    d['rung'] = (d.groupby(['subject', 'n_safe'], group_keys=False)['frac']
                 .rank(method='dense').astype(int))
    d['cell'] = d['order'] + ' | ' + d['stake']
    d['grp'] = d['subject'].astype(str) + ' | ' + d['order']
    cells, groups = sorted(d.cell.unique()), sorted(d.grp.unique())
    ci = d.cell.map({c: i for i, c in enumerate(cells)}).values
    gi = d.grp.map({g: i for i, g in enumerate(groups)}).values
    x = np.log(d['frac'].values)
    y = d['chose_risky'].astype(int).values
    print(f'{len(d)} trials, {d.subject.nunique()} subjects, '
          f'{len(cells)} cells, {len(groups)} offset groups')

    with pm.Model() as m:
        a = pm.Normal('a', 0., 2.5, shape=len(cells))
        log_b = pm.Normal('log_b', np.log(1.5), 1., shape=len(cells))
        pm.Deterministic('b', pm.math.exp(log_b))
        sd_a = pm.HalfNormal('sd_a', 1.5)
        sd_b = pm.HalfNormal('sd_b', 0.5)
        za = pm.Normal('za', 0., 1., shape=len(groups))
        zb = pm.Normal('zb', 0., 1., shape=len(groups))
        eta = (a[ci] + sd_a * za[gi]
               + pm.math.exp(log_b[ci] + sd_b * zb[gi]) * x)
        p = pm.Deterministic('p', pm.math.invprobit(eta))
        pm.Bernoulli('obs', p=p, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=.95, random_seed=seed,
                          progressbar=False)

    su = az.summary(idata, var_names=['a', 'b', 'sd_a', 'sd_b'])
    print(f"max r-hat {su['r_hat'].max():.3f}  min ess_bulk {su['ess_bulk'].min():.0f}"
          f"  {int(idata.sample_stats.diverging.values.sum())} divergences")

    po = idata.posterior
    A = po['a'].stack(s=('chain', 'draw')).values
    B = po['b'].stack(s=('chain', 'draw')).values
    rows = []
    for i, c in enumerate(cells):
        order, stake = c.split(' | ')
        row = dict(order=order, stake=stake, n_trials=int((ci == i).sum()),
                   n_sub=int(d.loc[ci == i, 'subject'].nunique()))
        for nm, v in (('slope', B[i]), ('rnp', np.exp(-A[i] / B[i])),
                      ('intercept', A[i])):
            row[nm] = float(np.median(v))
            row[f'{nm}_lo'], row[f'{nm}_hi'] = np.percentile(v, [2.5, 97.5])
        rows.append(row)
    slopes = pd.DataFrame(rows).sort_values(['order', 'stake'])
    # the low-minus-high contrast, within order, per draw
    for order in slopes.order.unique():
        il = cells.index(f'{order} | low')
        ih = cells.index(f'{order} | high')
        dd_ = B[ih] / B[il] - 1
        print(f'  {order}: slope high/low - 1 = {100 * np.median(dd_):+.1f}% '
              f'[{100 * np.percentile(dd_, 2.5):+.1f}, '
              f'{100 * np.percentile(dd_, 97.5):+.1f}]  '
              f'p(<0) = {float((dd_ < 0).mean()):.3f}')
    slopes.to_csv(f'{out_stem}.slopes.tsv', sep='\t', index=False)

    # -- posterior predictive: simulate CHOICES, aggregate as the data --------
    P = po['p'].stack(s=('chain', 'draw')).values                # (nTrial, nS)
    keep = np.linspace(0, P.shape[1] - 1, min(n_ppc, P.shape[1])).astype(int)
    P = P[:, keep]
    rng = np.random.default_rng(seed)
    sim = (rng.random(P.shape) < P).astype(float)
    keys = ['order', 'stake', 'rung']
    idx = pd.MultiIndex.from_frame(d[keys + ['subject']])
    per = pd.DataFrame(sim, index=idx).groupby(level=keys + ['subject']).mean()
    cellwise = per.groupby(level=keys).mean()
    obs = (d.assign(v=y).groupby(keys + ['subject'])['v'].mean()
             .groupby(level=keys).mean().reindex(cellwise.index))
    ppc = pd.DataFrame({
        'observed': obs.values,
        'model': cellwise.mean(axis=1).values,
        'lo': cellwise.quantile(.025, axis=1).values,
        'hi': cellwise.quantile(.975, axis=1).values,
    }, index=cellwise.index).reset_index()
    ppc = ppc.merge(d.groupby(keys)['frac'].mean().rename('frac').reset_index(),
                    on=keys)
    cov = float(((ppc.lo <= ppc.observed) & (ppc.observed <= ppc.hi)).mean())
    print(f'  PPC coverage of the 95% band: {cov:.0%} of {len(ppc)} cells')
    ppc.to_csv(f'{out_stem}.ppc.tsv', sep='\t', index=False)
    print(f'wrote {out_stem}.slopes.tsv and .ppc.tsv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/data/weber_baseline_hier.ses1'))
    ap.add_argument('--draws', default=1000, type=int)
    ap.add_argument('--tune', default=1000, type=int)
    ap.add_argument('--chains', default=4, type=int)
    ap.add_argument('--seed', default=0, type=int)
    ap.add_argument('--n_ppc', default=400, type=int)
    a = ap.parse_args()
    main(a.bids_folder, a.out_stem, a.draws, a.tune, a.chains, a.seed, a.n_ppc)
