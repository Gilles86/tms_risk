"""Observed probit slope and risk-neutral probability, one fit per design cell.

The model side of this comparison is derived in closed form by
`extract_anchor_probit --by n_safe`; this is its empirical counterpart.

A per-SUBJECT probit is not available at this resolution: each
(subject, order, safe payoff, stimulation) cell holds a median of 12 trials,
so every one of the 700 cells is either separable or under-powered (checked --
`fit_observed_probit_subject --by n_safe` fits exactly zero of them). Pooling
subjects gives ~415 trials per cell, which is plenty, and the uncertainty that
matters for a group-level PPC is between-subject anyway. So: one probit per
cell over the pooled trials, with a subject-cluster bootstrap for the interval
(resample the 35 participants with replacement, refit, repeat).

    python -m tms_risk.behavior.scripts.fit_observed_probit_cells --by n_safe
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore')

RNP_LO, RNP_HI, SLOPE_MIN = 0.05, 5.0, 0.3


def probit(x, y):
    """Return (slope, rnp) for one pooled cell, or (nan, nan) if unusable."""
    if len(np.unique(y)) < 2 or len(y) < 40:
        return np.nan, np.nan
    try:
        r = sm.GLM(y, sm.add_constant(x),
                   family=sm.families.Binomial(sm.families.links.Probit())).fit()
        b0, b1 = r.params
    except Exception:                                       # noqa: BLE001
        return np.nan, np.nan
    if not np.isfinite([b0, b1]).all() or b1 <= SLOPE_MIN:
        return np.nan, np.nan
    # indifference point: index = 0 at log frac* = -b0/b1, so the risk-neutral
    # probability the participant behaves as if holding is p_R * frac*.
    rnp = float(np.exp(-b0 / b1))
    return (float(b1), rnp) if RNP_LO <= rnp <= RNP_HI else (float(b1), np.nan)


def main(bids_folder, out_tsv, by, n_boot, seed):
    from tms_risk.behavior.fit_model import get_data
    d = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['y'] = d['chose_risky'].astype(float)
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    d['stake2'] = (d.groupby('subject', group_keys=False)['stake']
                   .apply(lambda v: (v > v.median()).astype(int)))
    d['x'] = np.log(d['frac'])

    keys = ['order', by, 'stimulation_condition']
    subjects = np.sort(d.subject.unique())
    rng = np.random.default_rng(seed)
    boot_idx = [rng.choice(len(subjects), len(subjects), replace=True)
                for _ in range(n_boot)]
    # a subject's trials, once, so each bootstrap replicate is a concat of rows
    by_subj = {s: g for s, g in d.groupby('subject')}

    # point estimate per cell
    point = {k: probit(g['x'].values, g['y'].values) for k, g in d.groupby(keys)}

    # one resampled frame per replicate, then all cells at once -- resampling
    # inside the cell loop instead rebuilds the whole dataset 20x per replicate
    boot = {k: {'slope': [], 'rnp': []} for k in point}
    for idx in boot_idx:
        r = pd.concat([by_subj[subjects[i]] for i in idx], ignore_index=True)
        for k, g in r.groupby(keys):
            if k not in boot:
                continue
            s_, p_ = probit(g['x'].values, g['y'].values)
            boot[k]['slope'].append(s_)
            boot[k]['rnp'].append(p_)

    rows = []
    for k, (slope, rnp) in point.items():
        row = dict(zip(keys, k)) | dict(
            n_trials=int((d.groupby(keys).size())[k]), slope=slope, rnp=rnp)
        for nm in ('slope', 'rnp'):
            v = np.asarray(boot[k][nm], float)
            v = v[np.isfinite(v)]
            row[f'{nm}_lo'], row[f'{nm}_hi'] = (
                np.percentile(v, [2.5, 97.5]) if len(v) > 10 else (np.nan, np.nan))
            row[f'{nm}_sem'] = float(np.std(v)) if len(v) > 10 else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(keys)
    print(out[keys + ['n_trials', 'slope', 'slope_sem', 'rnp', 'rnp_sem']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    out.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}  ({n_boot} subject-cluster bootstrap replicates)')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--by', default='n_safe', choices=['stake2', 'n_safe'])
    ap.add_argument('--out_tsv', default=None)
    ap.add_argument('--n_boot', default=400, type=int)
    ap.add_argument('--seed', default=0, type=int)
    a = ap.parse_args()
    out = a.out_tsv or str(REPO / f'notes/data/probit_observed_cells.{a.by}.tsv')
    main(a.bids_folder, out, a.by, a.n_boot, a.seed)
