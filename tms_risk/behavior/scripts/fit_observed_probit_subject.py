"""Per-subject probit fits of the observed choices, one per stake x order cell.

The model's counterpart is derived in closed form by `extract_anchor_probit`.
Fitting the same two parameters to each participant's own data gives the
subject-wise PPC: does the model predict WHICH people show the biggest cTBS
effect, not merely the group mean?

A per-subject-per-cell-per-condition probit rests on ~6 ladder rungs and ~120
trials, so the individual estimates are noisy -- that is expected and is why the
comparison below is a correlation across participants, not a per-participant
test. Cells where a participant's choices are separable (all 0 or all 1 at every
rung) have no finite MLE and are dropped, with the count reported.

    python -m tms_risk.behavior.scripts.fit_observed_probit_subject
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore')


def main(bids_folder, out_tsv, by='stake2'):
    from tms_risk.behavior.fit_model import get_data
    d = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['y'] = d['chose_risky'].astype(float)
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    d['stake2'] = (d.groupby('subject', group_keys=False)['stake']
                   .apply(lambda v: (v > v.median()).astype(int)))
    d['x'] = np.log(d['frac'])

    rows, dropped = [], 0
    keys = ['subject', 'order', by, 'stimulation_condition']
    for k, g in d.groupby(keys):
        if g['y'].nunique() < 2 or len(g) < 20:
            dropped += 1
            continue
        X = sm.add_constant(g['x'].values)
        try:
            r = sm.GLM(g['y'].values, X,
                       family=sm.families.Binomial(sm.families.links.Probit())
                       ).fit()
            b0, b1 = r.params
        except Exception:                                   # noqa: BLE001
            dropped += 1
            continue
        # rnp = exp(b0/b1) explodes when the fitted slope is near zero -- one
        # such cell put the group MEAN at -2160 while the median sat at +0.06.
        # Require a slope the data can actually support and an rnp inside the
        # range a probability can occupy.
        # a probit slope above ~20 per log-ratio is quasi-separation, not a
        # very consistent participant: one such cell put the group mean paired
        # slope difference at +41.6 (SEM 43.1) against a median near zero.
        if not np.isfinite([b0, b1]).all() or not (0.5 < b1 < 20.):
            dropped += 1
            continue
        # indifference is at log frac* = -b0/b1, and the risk-neutral
        # probability is the p that equates the two EVs there:
        # p n_risky* = n_safe  =>  p = 1/frac* = exp(b0/b1).
        # (extract_anchor_probit reports the reciprocal-side quantity
        # p_R * frac*, so the two are related by RNP = p_R / rnp_model.)
        rnp = float(np.exp(b0 / b1))
        if not (0.05 <= rnp <= 5.0):
            dropped += 1
            continue
        rows.append(dict(zip(keys, k))
                    | {'slope': b1, 'rnp': rnp, 'n_trials': len(g)})
    out = pd.DataFrame(rows)
    print(f'{len(out)} cells fitted, {dropped} dropped (separable or too few '
          f'trials)')
    out.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}')

    w = out.pivot_table(index=['subject', 'order', by],
                        columns='stimulation_condition',
                        values=['slope', 'rnp'])
    for par in ('slope', 'rnp'):
        dd = (w[(par, 'ips')] - w[(par, 'vertex')]).dropna()
        lo = sorted(out[by].unique())[0]
        g = dd.xs(('Risky second', lo), level=('order', by))
        print(f'  observed per-subject Δ{par}, {by}={lo} / risky second: '
              f'n={len(g)}  mean {g.mean():+.3f}  median {g.median():+.3f}')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--out_tsv',
                    default=str(REPO / 'notes/data/probit_observed_subject.tsv'))
    ap.add_argument('--by', default='stake2', choices=['stake2', 'n_safe'],
                    help='cell definition: stake median split, or safe payoff')
    a = ap.parse_args()
    out = a.out_tsv
    if a.by != 'stake2' and out.endswith('.tsv'):
        out = out[:-4] + f'.{a.by}.tsv'
    main(a.bids_folder, out, a.by)
