"""Is the subject-wise probit PPC measuring anything, or just noise?

A correlation between a model's per-subject prediction and an observed
per-subject estimate is bounded above by the square root of the product of the
two reliabilities. Before reading anything into r = -0.35 on Delta-rnp, measure
that ceiling: split each participant's trials in half, fit the same probit to
each half, and correlate across participants (Spearman-Brown corrected). If the
observed measure has near-zero reliability, no correlation of either sign means
anything.

Also reports Spearman alongside Pearson -- per-subject probit slopes have a long
tail and one participant at -24.7 can set a Pearson r on its own -- and the
leave-one-out range, which says whether a marginal p is carried by one person.

    python -m tms_risk.behavior.scripts.probit_subject_reliability
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings('ignore')
READ = dict(sep='\t', keep_default_na=False, na_values=[''])
CELL = ('Risky second', 0)


def probit(g):
    if g['y'].nunique() < 2 or len(g) < 15:
        return None
    try:
        r = sm.GLM(g['y'].values, sm.add_constant(g['x'].values),
                   family=sm.families.Binomial(sm.families.links.Probit())).fit()
        b0, b1 = r.params
    except Exception:                                        # noqa: BLE001
        return None
    if not np.isfinite([b0, b1]).all() or b1 <= 0.5:
        return None
    rnp = float(np.exp(b0 / b1))
    return None if not (0.05 <= rnp <= 5.0) else (float(b1), rnp)


def cell_effects(d, half=None):
    """Per-subject cTBS effect on both probit parameters, in the key cell."""
    out = {}
    sel = d if half is None else d[d['half'] == half]
    for subj, g in sel.groupby('subject'):
        fits = {}
        for cond in ('ips', 'vertex'):
            f = probit(g[g.stimulation_condition == cond])
            if f is None:
                break
            fits[cond] = f
        if len(fits) == 2:
            out[subj] = (fits['ips'][0] - fits['vertex'][0],
                         fits['ips'][1] - fits['vertex'][1])
    return pd.DataFrame(out, index=['slope', 'rnp']).T


def main(bids_folder, data_dir, label, seed):
    from tms_risk.behavior.fit_model import get_data
    d = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['y'] = d['chose_risky'].astype(float)
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    d['stake2'] = (d.groupby('subject', group_keys=False)['stake']
                   .apply(lambda v: (v > v.median()).astype(int)))
    d['x'] = np.log(d['frac'])
    d = d[(d.order == CELL[0]) & (d.stake2 == CELL[1])].copy()
    rng = np.random.default_rng(seed)
    full = cell_effects(d)
    print(f'{len(full)} subjects with a full-data estimate')

    # Split-half fails here: half a cell is ~60 trials and the probit stops
    # being estimable for most participants. Estimate reliability instead by
    # bootstrapping TRIALS within participant, which uses all the data:
    #     reliability = (var_between - mean var_within) / var_between
    # i.e. the share of the observed between-subject variance that is not
    # measurement error.
    nb = 60
    boot = {p: [] for p in ('slope', 'rnp')}
    for subj, g in d.groupby('subject'):
        vals = {p: [] for p in ('slope', 'rnp')}
        for _ in range(nb):
            gb = g.sample(len(g), replace=True, random_state=rng.integers(1 << 31))
            fits = {}
            for cond in ('ips', 'vertex'):
                f = probit(gb[gb.stimulation_condition == cond])
                if f is None:
                    break
                fits[cond] = f
            if len(fits) == 2:
                vals['slope'].append(fits['ips'][0] - fits['vertex'][0])
                vals['rnp'].append(fits['ips'][1] - fits['vertex'][1])
        for p in ('slope', 'rnp'):
            if len(vals[p]) >= 10:
                boot[p].append((subj, np.var(vals[p], ddof=1)))

    rel = {}
    for par in ('slope', 'rnp'):
        w = pd.Series(dict(boot[par]))
        obs = full[par].reindex(w.index).dropna()
        w = w.reindex(obs.index)
        var_between = obs.var(ddof=1)
        var_within = w.median()          # median: bootstrap vars are skewed
        r = max((var_between - var_within) / var_between, 0.0)
        rel[par] = r
        print(f'{par:6s} between-subject var {var_between:.4f}  '
              f'median within-subject (bootstrap) var {var_within:.4f}  '
              f'-> reliability {r:.3f}')

    sub = pd.read_csv(Path(data_dir) /
                      f'probit_derived/probit_subject.{label}.tsv', **READ)
    s = sub[(sub.order == CELL[0]) & (sub.stake2 == CELL[1])]
    m = s.pivot_table(index=['subject', 'parameter'],
                      columns='stimulation_condition', values='median')
    mod = (m['ips'] - m['vertex']).unstack('parameter')

    print()
    for par in ('slope', 'rnp'):
        j = pd.DataFrame({'model': mod[par], 'data': full[par]}).dropna()
        pr = stats.pearsonr(j.model, j.data)
        sp = stats.spearmanr(j.model, j.data)
        loo = [stats.spearmanr(j.drop(index=i).model,
                               j.drop(index=i).data).statistic for i in j.index]
        ceiling = np.sqrt(rel[par]) if rel[par] > 0 else np.nan
        print(f'{par:6s} n={len(j):2d}  pearson {pr.statistic:+.3f} '
              f'(p={pr.pvalue:.3f})  spearman {sp.statistic:+.3f} '
              f'(p={sp.pvalue:.3f})')
        print(f'{"":6s}   LOO spearman range [{min(loo):+.3f}, {max(loo):+.3f}]'
              f'   attenuation ceiling |r| <= {ceiling:.2f}')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--seed', default=1, type=int)
    a = ap.parse_args()
    main(a.bids_folder, a.data_dir, a.model_label, a.seed)
