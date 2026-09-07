"""Observed probit slope and indifference point per design cell, hierarchically.

Neither simpler estimator works at this resolution:

* Per subject, per cell (`fit_observed_probit_subject --by n_safe`) fits ZERO of
  the 700 cells -- each holds a median of 12 trials, so every one is separable
  or under-powered.
* Pooling subjects within a cell (`fit_observed_probit_cells`) fits fine but
  measures the wrong thing. Participants have different indifference points, and
  averaging psychometric functions that sit at different places FLATTENS the
  aggregate. Measured here: the pooled slope comes out around 0.4-1.8 where the
  model's per-subject slope averages 2.5-3.5. Most of that gap is the artefact,
  not a misfit.

The model's number is a per-subject quantity averaged over subjects, so the
observed number has to be too. A hierarchical probit gives that: each subject
gets their own intercept and slope in every cell, partially pooled toward the
cell mean, and the cell mean is what the figure plots. Partial pooling also
makes the comparison fair in the other direction -- the model's per-subject
estimates are shrunk by its own hierarchy.

    y ~ Bernoulli(Phi(a[c] + da[g] + (b[c] * exp(db[g])) * log frac))

with c = (order, safe payoff, stimulation) -- the 20 cells the figure plots --
and g = (subject, order, stimulation), the level the nuisance heterogeneity
actually lives at. Giving each subject an offset in each CELL instead (1400
offsets on 12 trials apiece) is what the first version did, and it produced
3342 divergences: the funnel is unidentifiable at that resolution. Safe payoff
is the within-subject axis being resolved, so it does not need its own offset.

b is positive by construction (log-normal offsets), which keeps
rnp = exp(-a/b) finite -- a free-sign slope puts mass near b = 0 and sends the
derived indifference point to 10^3.

    python -m tms_risk.behavior.scripts.fit_observed_probit_hier --by n_safe
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def main(bids_folder, out_tsv, by, draws, tune, chains, seed):
    from tms_risk.behavior.fit_model import get_data
    d = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    d['stake2'] = (d.groupby('subject', group_keys=False)['stake']
                   .apply(lambda v: (v > v.median()).astype(int)))
    d['cell'] = (d['order'] + ' | ' + d[by].astype(str) + ' | '
                 + d['stimulation_condition'])
    cells = sorted(d.cell.unique())
    subjects = sorted(d.subject.unique())
    d['grp'] = (d['subject'].astype(str) + ' | ' + d['order'] + ' | '
                + d['stimulation_condition'])
    groups = sorted(d.grp.unique())
    ci = d.cell.map({c: i for i, c in enumerate(cells)}).values
    gi = d.grp.map({g: i for i, g in enumerate(groups)}).values
    x = np.log(d['frac'].values)
    y = d['chose_risky'].astype(int).values
    nC, nS, nG = len(cells), len(subjects), len(groups)
    print(f'{len(d)} trials, {nC} cells, {nS} subjects, {nG} offset groups')

    with pm.Model() as m:
        a = pm.Normal('a', 0., 2.5, shape=nC)
        log_b = pm.Normal('log_b', np.log(1.5), 1., shape=nC)
        b = pm.Deterministic('b', pm.math.exp(log_b))
        sd_a = pm.HalfNormal('sd_a', 1.5)
        sd_b = pm.HalfNormal('sd_b', 0.5)
        za = pm.Normal('za', 0., 1., shape=nG)
        zb = pm.Normal('zb', 0., 1., shape=nG)
        eta = (a[ci] + sd_a * za[gi]
               + pm.math.exp(log_b[ci] + sd_b * zb[gi]) * x)
        pm.Bernoulli('obs', p=pm.math.invprobit(eta), observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=.95, random_seed=seed,
                          progressbar=False)

    su = az.summary(idata, var_names=['a', 'b', 'sd_a', 'sd_b'])
    print(f"max r-hat {su['r_hat'].max():.3f}   min ess_bulk {su['ess_bulk'].min():.0f}")
    div = int(idata.sample_stats.diverging.values.sum())
    print(f'{div} divergences')

    po = idata.posterior
    A = po['a'].stack(d=('chain', 'draw')).values          # (nC, ndraw)
    B = po['b'].stack(d=('chain', 'draw')).values
    RNP = np.exp(-A / B)
    rows = []
    for i, c in enumerate(cells):
        order, lvl, stim = c.split(' | ')
        row = {'order': order, by: float(lvl) if by == 'n_safe' else int(lvl),
               'stimulation_condition': stim,
               'n_trials': int((ci == i).sum())}
        for nm, v in [('slope', B[i]), ('rnp', RNP[i]), ('intercept', A[i])]:
            row[nm] = float(np.median(v))
            row[f'{nm}_lo'], row[f'{nm}_hi'] = np.percentile(v, [2.5, 97.5])
        rows.append(row)
    out = (pd.DataFrame(rows)
             .sort_values(['order', by, 'stimulation_condition']))
    print(out[['order', by, 'stimulation_condition', 'n_trials',
               'slope', 'rnp']].to_string(index=False,
                                          float_format=lambda v: f'{v:.3f}'))
    out.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--by', default='n_safe', choices=['stake2', 'n_safe'])
    ap.add_argument('--out_tsv', default=None)
    ap.add_argument('--draws', default=1000, type=int)
    ap.add_argument('--tune', default=1000, type=int)
    ap.add_argument('--chains', default=4, type=int)
    ap.add_argument('--seed', default=0, type=int)
    a = ap.parse_args()
    out = a.out_tsv or str(REPO / f'notes/data/probit_observed_hier.{a.by}.tsv')
    main(a.bids_folder, out, a.by, a.draws, a.tune, a.chains, a.seed)
