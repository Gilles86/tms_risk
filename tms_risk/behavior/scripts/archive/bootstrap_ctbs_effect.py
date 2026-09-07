"""How stable are the observed cTBS effects, before any model is involved?

Every model in the anchor grid underpredicts two observed quantities: the
order-specific shift in P(risky) and the order-specific FLATTENING of the
psychometric function. Before adding more mechanism to close that gap, ask how
firmly the data pin the targets down. Both statistics are computed exactly as
the posterior-predictive checks compute them, then bootstrapped over
participants and recomputed leaving each participant out.

    python -m tms_risk.behavior.scripts.bootstrap_ctbs_effect
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(bids_folder):
    from tms_risk.behavior.fit_model import get_data
    d = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    d['order'] = d['risky_first'].map({True: 'first', False: 'second'})
    d['y'] = d['chose_risky'].astype(float)
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    d['rung'] = (d.groupby(['subject', 'n_safe'], group_keys=False)['frac']
                 .rank(method='dense').astype(int))
    d['stake_bin'] = (d.groupby('subject', group_keys=False)['stake']
                      .apply(lambda v: pd.qcut(v.rank(method='first'), 3,
                                               labels=False)))
    d['logfrac'] = np.log(d['frac'])
    return d


def per_subject(d):
    """One row per subject with every statistic already reduced.

    Reducing to subject level FIRST is what makes the bootstrap correct: the
    resampling unit is the participant, and every statistic below is a mean over
    participants of a within-participant quantity.
    """
    cell = (d.groupby(['subject', 'order', 'stake_bin',
                       'stimulation_condition'])['y'].mean()
            .unstack('stimulation_condition'))
    dp = (cell['ips'] - cell['vertex']).rename('dp').reset_index()
    dp = dp.pivot_table(index='subject', columns='order', values='dp')
    dp.columns = [f'dp_{c}' for c in dp.columns]

    def slope(g):
        x = g['logfrac'].values - g['logfrac'].values.mean()
        return (x @ g['y'].values) / (x @ x) if (x @ x) > 0 else np.nan

    sl = (d.groupby(['subject', 'order', 'stimulation_condition'])
            .apply(slope, include_groups=False)
            .unstack('stimulation_condition'))
    ds = (sl['ips'] - sl['vertex']).rename('dslope').reset_index()
    ds = ds.pivot_table(index='subject', columns='order', values='dslope')
    ds.columns = [f'dslope_{c}' for c in ds.columns]
    return dp.join(ds)


STATS = {
    'order_contrast':      lambda s: s.dp_second.mean() - s.dp_first.mean(),
    'dp_second':           lambda s: s.dp_second.mean(),
    'slope_second_ctbs':   lambda s: s.dslope_second.mean(),
    'slope_contrast':      lambda s: s.dslope_second.mean() - s.dslope_first.mean(),
}


def main(bids_folder, out_tsv, n_boot, seed):
    d = prepare(bids_folder)
    s = per_subject(d).dropna()
    subs = s.index.values
    print(f'{len(subs)} subjects')

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(subs), size=(n_boot, len(subs)))
    rows = []
    for name, fn in STATS.items():
        obs = fn(s)
        boot = np.array([fn(s.iloc[i]) for i in idx])
        # leave-one-out: how much does dropping ONE participant move it
        loo = np.array([fn(s.drop(index=subj)) for subj in subs])
        worst = subs[np.argmax(np.abs(loo - obs))]
        rows.append(dict(
            statistic=name, observed=obs,
            boot_lo=np.quantile(boot, .025), boot_hi=np.quantile(boot, .975),
            boot_sd=boot.std(),
            p_same_sign=float((np.sign(boot) == np.sign(obs)).mean()),
            loo_min=loo.min(), loo_max=loo.max(),
            most_influential=int(worst),
            loo_shift=float(loo[np.argmax(np.abs(loo - obs))] - obs),
            n_subj_same_sign=int((np.sign(
                s.dp_second if name.startswith('dp') else s.dslope_second)
                == np.sign(obs)).sum())))
        r = rows[-1]
        print(f"{name:20s} {obs:+.4f}  95% CI [{r['boot_lo']:+.4f}, "
              f"{r['boot_hi']:+.4f}]  sign-stable {r['p_same_sign']:.1%}  "
              f"LOO range [{r['loo_min']:+.4f}, {r['loo_max']:+.4f}]  "
              f"worst sub-{worst} ({r['loo_shift']:+.4f})")
    out = pd.DataFrame(rows)
    out.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}')
    s.to_csv(str(out_tsv).replace('.tsv', '_subject.tsv'), sep='\t')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--out_tsv',
                    default=str(REPO / 'notes/data/ctbs_effect_bootstrap.tsv'))
    ap.add_argument('--n_boot', default=10000, type=int)
    ap.add_argument('--seed', default=1, type=int)
    a = ap.parse_args()
    main(a.bids_folder, a.out_tsv, a.n_boot, a.seed)
