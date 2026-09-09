"""The nPRF-amplitude x choice-consistency correlation, with a posterior.

The manuscript reports this as r = 0.53 with a BOOTSTRAP 95% CI. That is the
last maximum-likelihood interval in the paper and it violates the project's own
rule: a bootstrap resamples 35 point estimates of choice consistency as though
each were known, when each is itself estimated from ~120 trials and carries real
uncertainty.

Here consistency is a parameter of a hierarchical probit fitted to the choices,
so every participant's slope is a posterior rather than a number. The
correlation with the (fixed) nPRF amplitude change is then computed ONCE PER
DRAW, giving a posterior for r whose width already contains the measurement
error in consistency. Nothing is bootstrapped and nothing is estimated by
maximum likelihood.

The model is the one in `fit_observed_probit_hier`, with the cells that figure
uses (order x safe payoff x stimulation) and per-participant offsets on both
intercept and slope within (participant x order x stimulation). The
participant's consistency in a condition is the slope of that offset group.

    python -m tms_risk.behavior.scripts.brain_behavior_consistency_posterior
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]
MASK, SELECTION, AMP = 'NPCr2cm-cluster', 'cvr2pos', 'd_amp_median'


def _rank(a):
    o = np.argsort(a, axis=-1, kind='stable')
    r = np.empty_like(o, dtype=float)
    np.put_along_axis(r, o, np.broadcast_to(
        np.arange(a.shape[-1], dtype=float), a.shape).copy(), axis=-1)
    return r


def _corr(x, y):
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean()
    den = np.sqrt((x ** 2).sum(-1) * (y ** 2).sum())
    return (x * y).sum(-1) / np.where(den > 0, den, np.nan)


def main(bids_folder, neural_tsv, out_tsv, draws, tune, chains, seed):
    from tms_risk.behavior.fit_model import get_data
    d = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    # Cells are order x stimulation only. The safe-payoff dimension that
    # Figure 3 uses is not needed here and made the model 20 cells over 140
    # offset groups, which would not sample (r-hat 2.7, ESS 5, 2963
    # divergences). "Consistency" is one slope per participant per condition,
    # and that is exactly what this parameterisation estimates.
    d['cell'] = d['order'] + ' | ' + d['stimulation_condition']
    d['grp'] = (d['subject'].astype(str) + ' | ' + d['order'] + ' | '
                + d['stimulation_condition'])
    cells, groups = sorted(d.cell.unique()), sorted(d.grp.unique())
    ci = d.cell.map({c: i for i, c in enumerate(cells)}).values
    gi = d.grp.map({g: i for i, g in enumerate(groups)}).values
    x = np.log(d['frac'].values)
    y = d['chose_risky'].astype(int).values
    print(f'{len(d)} trials, {len(cells)} cells, {len(groups)} offset groups')

    with pm.Model():
        a = pm.Normal('a', 0., 2.5, shape=len(cells))
        log_b = pm.Normal('log_b', np.log(1.5), 1., shape=len(cells))
        sd_a = pm.HalfNormal('sd_a', 1.5)
        sd_b = pm.HalfNormal('sd_b', 0.5)
        za = pm.Normal('za', 0., 1., shape=len(groups))
        zb = pm.Normal('zb', 0., 1., shape=len(groups))
        # The participant's consistency in a condition, on the log scale: the
        # cell slopes for that (order, stimulation) averaged over safe payoff,
        # plus the participant's own offset. Averaging over safe payoff is what
        # makes it one number per participant per condition rather than five.
        import collections
        by_cond = collections.defaultdict(list)
        for j, cc in enumerate(cells):
            o_, st_ = cc.split(' | ')
            by_cond[(o_, st_)].append(j)
        cond_of_grp = [tuple(g.split(' | ')[1:]) for g in groups]
        M = np.zeros((len(groups), len(cells)))
        for k, cnd in enumerate(cond_of_grp):
            idxs = by_cond[cnd]
            M[k, idxs] = 1.0 / len(idxs)
        pm.Deterministic('log_slope',
                         pm.math.dot(pm.math.constant(M), log_b) + sd_b * zb)
        eta = a[ci] + sd_a * za[gi] + pm.math.exp(log_b[ci] + sd_b * zb[gi]) * x
        pm.Bernoulli('obs', p=pm.math.invprobit(eta), observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=.98, random_seed=seed, progressbar=False)
    su = az.summary(idata, var_names=['a', 'b'] if 'b' in idata.posterior
                    else ['a', 'log_b'])
    print(f"max r-hat {su['r_hat'].max():.3f}  min ess {su['ess_bulk'].min():.0f}"
          f"  {int(idata.sample_stats.diverging.values.sum())} divergences")

    ls = idata.posterior['log_slope'].stack(s=('chain', 'draw')).values  # (nG, nD)
    key = pd.DataFrame([g.split(' | ') for g in groups],
                       columns=['subject', 'order', 'stim'])
    neu = pd.read_csv(neural_tsv, **READ)
    neu = neu[(neu['mask'] == MASK) & (neu['selection'] == SELECTION)]
    neu = neu.set_index('subject')[AMP].dropna()

    # Sanity gate: the posterior-mean contrast must agree with the raw
    # per-participant consistency contrast the manuscript used. If it does not,
    # the alignment is wrong and no correlation below means anything.
    chk = pd.read_csv(REPO / 'notes/data/bb_behavior.tsv', **READ)
    w = chk.pivot_table(index='subject', columns='stimulation_condition',
                        values='consistency_rsecond')
    raw = (w['ips'] - w['vertex'])
    _m_i = ((key.order == 'Risky second') & (key.stim == 'ips')).values
    _m_v = ((key.order == 'Risky second') & (key.stim == 'vertex')).values
    _si = key.subject[_m_i].values
    _pos = {s_: j for j, s_ in enumerate(key.subject[_m_v].values)}
    _d = (ls[_m_i] - ls[_m_v][[_pos[s_] for s_ in _si]]).mean(1)
    _r = np.corrcoef(_d, raw.loc[[int(s_) for s_ in _si]].values)[0, 1]
    print(f'ALIGNMENT CHECK: hierarchical vs raw consistency contrast r = {_r:+.3f}')
    if _r < .5:
        raise SystemExit('alignment check failed -- refusing to report a correlation')

    rows = []
    for order in ('Risky second', 'Risky first'):
        m_i = ((key.order == order) & (key.stim == 'ips')).values
        m_v = ((key.order == order) & (key.stim == 'vertex')).values
        s_i = key.subject[m_i].values
        # align vertex rows to the ips subject order
        pos = {s: j for j, s in enumerate(key.subject[m_v].values)}
        v = ls[m_v][[pos[s] for s in s_i]]
        # consistency contrast per participant per draw, on the log scale
        dcons = ls[m_i] - v
        keep = np.array([int(s) in neu.index for s in s_i])
        amp = neu.loc[[int(s) for s in s_i[keep]]].values
        for kind, X in (('pearson', dcons[keep].T),
                        ('spearman', _rank(dcons[keep].T))):
            r = _corr(X, amp if kind == 'pearson' else _rank(amp[None])[0])
            rows.append(dict(order=order, kind=kind, n=int(keep.sum()),
                             r=float(np.median(r)),
                             lo=float(np.quantile(r, .025)),
                             hi=float(np.quantile(r, .975)),
                             p_gt0=float((r > 0).mean())))
    out = pd.DataFrame(rows)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep='\t', index=False)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nwrote {out_tsv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--neural_tsv', default=str(REPO / 'notes/data/bb_neural.tsv'))
    ap.add_argument('--out_tsv',
                    default=str(REPO / 'notes/data/bb_consistency_posterior.tsv'))
    ap.add_argument('--draws', default=2000, type=int)
    ap.add_argument('--tune', default=3000, type=int)
    ap.add_argument('--chains', default=4, type=int)
    ap.add_argument('--seed', default=0, type=int)
    a = ap.parse_args()
    main(a.bids_folder, a.neural_tsv, a.out_tsv, a.draws, a.tune, a.chains, a.seed)
