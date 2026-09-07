"""The Figure 3 probit, refit separately at low and high stakes.

Figure 3 asks what cTBS did to the psychometric function. This script asks the same
question once per stake band, because the cognitive-model account predicts the effect
should be concentrated where the payoffs are small: the nPRF population at the
stimulation site prefers numerosities well below the presented range, so the noise it
adds should bite hardest at the low end.

"Stake" is the magnitude the two options share, (n_safe + n_risky) / 2, split at each
participant's own median. That is the repo's existing convention -- `probit_average_n_full`
in `fit_probit.build_model` and the stake terciles in `plot_ppc_fig3a.py` both use it.
Splitting within subject rather than globally keeps all 35 subjects in all four cells
(2 orders x 2 stakes, ~2080 trials each) instead of dropping whoever's own payoffs
happened to sit high or low.

Each cell gets its own hierarchical probit, the same structure as the paper's model:

    P(risky) = Phi(b0_s + d0_s*IPS + (b1_s + d1_s*IPS) * log(risky/safe))

Fitting the four cells separately, rather than adding a stake interaction to one
model, means no partial pooling across cells -- deliberately conservative, and it
keeps each cell's IPS - vertex difference a within-cell, within-draw quantity.

Outputs (into notes/data/):
  probit_stake_group_posterior.tsv   group slope + RNP draws per (order, stake, stim)
  probit_stake_by_ratio.tsv          observed and predicted P(risky) per ratio bin

    python -m tms_risk.behavior.scripts.analyze_probit_by_stake

Then plot with plot_fig3_probit_stake.py, which needs only those two TSVs.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats as ss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_probit import get_data  # noqa: E402

ORDERS = ['Risky first', 'Risky second']
STAKES = ['Low stake', 'High stake']
RATIO_BINS = ['20%', '32%', '44%', '56%', '68%', '80%']


def prepare(bids_folder):
    df = get_data('probit_order', bids_folder)
    d = df.reset_index()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['stim'] = d['stimulation_condition']
    d['bin'] = d['bin(risky/safe)'].astype(str)
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    # rank first so qcut never trips over duplicate payoff values at the median
    d['stake_bin'] = (d.groupby('subject', group_keys=False)['stake']
                      .apply(lambda v: pd.qcut(v.rank(method='first'), 2,
                                               labels=STAKES)))
    return d


def fit_cell(sub, draws, tune, seed, cores, random_effects='full'):
    """Hierarchical probit for one cell.

    Same model as `fit_hier_probit` in analyze_localized_noise, but the per-subject
    parameters are kept as well: predicted choice proportions have to be aggregated
    over subjects exactly the way the observed ones are, and Phi at the group mean
    parameters is not that -- by Jensen it is systematically too steep.

    `random_effects='full'` lets the intercept, the slope and both cTBS shifts vary
    by subject. `'intercept'` varies only the intercept, which is the structure the
    published `probit_average_n_full` uses (`... + (1|subject)`); it estimates every
    slope, including the stake x stimulation interaction, as though all trials were
    independent, so its intervals on slope contrasts are too narrow. Provided so the
    two can be compared on identical data -- see notes/checks_20260803.md 1(a-ter).
    """
    import pymc as pm

    sub = sub.reset_index(drop=True)
    s_idx, subjects = pd.factorize(sub['subject'])
    ips = (sub['stim'].values == 'ips').astype(float)
    x, y = sub['x'].values, sub['chose_risky'].astype(float).values
    ns = len(subjects)
    varying = ['b0', 'b1', 'd0', 'd1'] if random_effects == 'full' else ['b0']

    with pm.Model():
        names = ['b0', 'b1', 'd0', 'd1']
        mu = {n: pm.Normal(f'mu_{n}', 0., 2.5) for n in names}
        par = {}
        for n in names:
            if n in varying:
                sd = pm.HalfNormal(f'sd_{n}', 1.)
                off = pm.Normal(f'off_{n}', 0., 1., shape=ns)
                par[n] = pm.Deterministic(n, mu[n] + sd * off)
            else:
                par[n] = pm.Deterministic(n, mu[n] * 1.)

        def tr(n):
            """Per-trial value: index by subject only where the term varies."""
            return par[n][s_idx] if n in varying else par[n]

        eta = ((tr('b0') + tr('d0') * ips) + (tr('b1') + tr('d1') * ips) * x)
        pm.Bernoulli('obs', p=pm.math.invprobit(eta), observed=y)

        pm.Deterministic('slope_vertex', mu['b1'])
        pm.Deterministic('slope_ips', mu['b1'] + mu['d1'])
        # indifference sits at x0 = -b0/b1 in log(risky/safe), so the indifference
        # ratio is exp(-b0/b1) and the risk-neutral probability its reciprocal
        pm.Deterministic('rnp_vertex', pm.math.exp(mu['b0'] / mu['b1']))
        pm.Deterministic('rnp_ips',
                         pm.math.exp((mu['b0'] + mu['d0']) / (mu['b1'] + mu['d1'])))

        idata = pm.sample(draws=draws, tune=tune, chains=4, cores=cores,
                          target_accept=.9, random_seed=seed, progressbar=False)
    return idata, s_idx


def predicted_by_bin(idata, sub, s_idx, n_keep=400, seed=1):
    """P(risky) per ratio bin, aggregated over subjects exactly like the data."""
    post = idata.posterior
    # a term that varies by subject arrives as (chain, draw, subject), a fixed one as
    # (chain, draw); flatten the sampling axes and keep the distinction
    flat = {n: post[n].values.reshape(-1, *post[n].values.shape[2:])
            for n in ['b0', 'b1', 'd0', 'd1']}
    n_draw = flat['b0'].shape[0]
    keep = np.random.default_rng(seed).choice(n_draw, min(n_keep, n_draw),
                                              replace=False)
    ips = (sub['stim'].values == 'ips').astype(float)
    x = sub['x'].values

    def tr(n):
        a = flat[n][keep]
        return a[:, s_idx] if a.ndim == 2 else a[:, None]

    eta = (tr('b0') + tr('d0') * ips) + (tr('b1') + tr('d1') * ips) * x
    p = ss.norm.cdf(eta)                                   # (draw, trial)

    key = sub[['subject', 'bin', 'stim']]
    out = {}
    for stim in ['vertex', 'ips']:
        cols, rows = [], []
        for (b,), g in key[key.stim == stim].groupby(['bin']):
            per_sub = pd.DataFrame(p[:, g.index.values].T).groupby(
                g['subject'].values).mean()          # (subject, draw)
            rows.append(per_sub.mean(0).values)      # mean over subjects, per draw
            cols.append(b)
        arr = np.vstack(rows)
        out[f'probit_{stim}'] = pd.Series(arr.mean(1), index=cols)
        out[f'probit_{stim}_lo'] = pd.Series(np.quantile(arr, .025, axis=1), index=cols)
        out[f'probit_{stim}_hi'] = pd.Series(np.quantile(arr, .975, axis=1), index=cols)
    return pd.DataFrame(out)


def observed_by_bin(sub):
    """Mean over subjects of each subject's own choice proportion, per bin."""
    g = (sub.groupby(['subject', 'bin', 'stim'], observed=True)['chose_risky']
           .mean().unstack('stim'))
    out = g.groupby('bin', observed=True).mean()
    out.columns = [f'observed_{c}' for c in out.columns]
    out['frac'] = (sub.groupby(['subject', 'bin'], observed=True)['frac'].mean()
                      .groupby('bin', observed=True).mean())
    return out


def main(bids_folder, out_dir, draws, tune, cores, seed, random_effects, tag):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'.{tag}' if tag else ''
    d = prepare(bids_folder)
    print(f'{d.subject.nunique()} subjects, {len(d)} trials, '
          f'random effects: {random_effects}')

    posts, ratios = [], []
    for order in ORDERS:
        for stake in STAKES:
            sub = d[(d.order == order) & (d.stake_bin == stake)].copy()
            print(f'\n=== {order} / {stake}: {len(sub)} trials, '
                  f'{sub.subject.nunique()} subjects, '
                  f'stake {sub.stake.min():.0f}-{sub.stake.max():.0f} CHF')
            idata, s_idx = fit_cell(sub, draws, tune, seed, cores, random_effects)

            import arviz as az
            summ = az.summary(idata, var_names=['mu_b0', 'mu_b1', 'mu_d0', 'mu_d1'])
            print(f'  max r-hat {summ["r_hat"].max():.3f}, '
                  f'min ESS {summ["ess_bulk"].min():.0f}')

            post = idata.posterior
            for par, stim in [('slope', 'vertex'), ('slope', 'ips'),
                              ('rnp', 'vertex'), ('rnp', 'ips')]:
                v = post[f'{par}_{stim}'].values.ravel()
                posts.append(pd.DataFrame({
                    'parameter': par, 'order': order, 'stake': stake,
                    'stimulation_condition': stim, 'draw': np.arange(len(v)),
                    'value': v, 'random_effects': random_effects}))

            sub = sub.reset_index(drop=True)
            r = observed_by_bin(sub).join(predicted_by_bin(idata, sub, s_idx,
                                                           seed=seed))
            r = r.reindex(RATIO_BINS)
            ratios.append(r.assign(order=order, stake=stake).reset_index())

    pd.concat(posts, ignore_index=True).to_csv(
        out_dir / f'probit_stake_group_posterior{suffix}.tsv', sep='\t', index=False)
    ratio = pd.concat(ratios, ignore_index=True)
    ratio.to_csv(out_dir / f'probit_stake_by_ratio{suffix}.tsv', sep='\t', index=False)

    # the numbers the figure will show, so a mismatch is caught here and not by eye
    print('\n=== cTBS effect per cell (IPS - vertex, 95% CrI) ===')
    p = pd.concat(posts, ignore_index=True)
    for par in ['slope', 'rnp']:
        for order in ORDERS:
            for stake in STAKES:
                s = p[(p.parameter == par) & (p.order == order)
                      & (p.stake == stake)]
                w = s.pivot_table(index='draw', columns='stimulation_condition',
                                  values='value')
                delta = (w['ips'] - w['vertex']).values
                lo, hi = np.quantile(delta, [.025, .975])
                pb = min((delta > 0).mean(), (delta < 0).mean())
                print(f'  {par:<6}{order:<14}{stake:<12}{delta.mean():+.4f}  '
                      f'[{lo:+.4f}, {hi:+.4f}]  p = {pb:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--out_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--draws', default=1000, type=int)
    parser.add_argument('--tune', default=1000, type=int)
    parser.add_argument('--cores', default=4, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--random_effects', default='full',
                        choices=['full', 'intercept'],
                        help="'full' = intercept, slope and both cTBS shifts vary by "
                             "subject; 'intercept' = the published (1|subject) structure")
    parser.add_argument('--tag', default='',
                        help='suffix for the output TSVs, e.g. --tag ri')
    a = parser.parse_args()
    main(a.bids_folder, a.out_dir, a.draws, a.tune, a.cores, a.seed,
         a.random_effects, a.tag)
