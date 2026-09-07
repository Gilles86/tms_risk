"""Where does one model beat another? Decompose an ELPD gap trial by trial.

LOO returns a per-trial log predictive density, so the difference between two models
is not a single number but a value for each of the 8,335 choices. Summing it inside
cells of the design says exactly which trials one model predicts better -- i.e. what
pattern in the data the loser gets wrong. No refitting, no PPC, no bauer: it reads the
pointwise vectors already extracted to `notes/data/ploo/*.npz`.

Used here for Log-Flexible vs Log-Weber: the 46-nat gap is the whole content of the
claim that noise depends on magnitude, so it matters a great deal whether that gap is
spread evenly over trials or concentrated at particular payoffs.

    python -m tms_risk.behavior.scripts.decompose_elpd_gap \\
        --model lfx2-bs3-m2-dp-bm --baseline lfx2-bs3-w-dp-bm
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_pointwise(ploo_dir, label):
    z = np.load(Path(ploo_dir) / f'{label}.npz')
    return z['elpd_i'], z['obs'], json.loads(str(z['meta']))


def cell_table(df, d, by, min_n=100):
    """Summed and per-trial ELPD advantage within cells of `by`."""
    g = df.assign(d=d).groupby(by, observed=True)['d']
    out = pd.DataFrame({'n': g.size(), 'total': g.sum(), 'per_trial': g.mean()})
    return out[out.n >= min_n].reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ploo_dir', default='notes/data/ploo')
    ap.add_argument('--model', default='lfx2-bs3-m2-dp-bm')
    ap.add_argument('--baseline', default='lfx2-bs3-w-dp-bm')
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--out_tsv', default='notes/data/elpd_gap_decomposition.tsv')
    args = ap.parse_args()

    from tms_risk.behavior.fit_model import get_data
    df = get_data(args.bids_folder, model_label=args.model).reset_index()

    e_m, obs_m, meta_m = load_pointwise(args.ploo_dir, args.model)
    e_b, obs_b, meta_b = load_pointwise(args.ploo_dir, args.baseline)
    if meta_m['obs_hash'] != meta_b['obs_hash']:
        raise SystemExit('the two traces scored different observations')
    if len(df) != len(e_m):
        raise SystemExit(f'design has {len(df)} rows, pointwise has {len(e_m)}')
    # The trace observes `choice` (chose the second-presented option), NOT
    # `chose_risky` -- the risky-referenced version is derived downstream by
    # flipping on `risky_first`. Checking against the wrong one silently passes a
    # misaligned merge, so verify against the column the model actually saw.
    if not np.allclose(df['choice'].astype(float).values, obs_m):
        raise SystemExit('observed choices do not match the pointwise ordering')

    d = e_m - e_b
    print(f'{args.model}  minus  {args.baseline}')
    print(f'total gap {d.sum():+.1f} nats over {len(d)} trials '
          f'({d.mean():+.4f} per trial)\n')

    df['order'] = df['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    df['stake'] = (df['n_safe'] + df['n_risky']) / 2
    df['safe_bin'] = pd.cut(df['n_safe'], [0, 9, 13, 19, 27, 200],
                            labels=['7-9', '10-13', '14-19', '20-27', '28+'])
    df['risky_bin'] = pd.qcut(df['n_risky'], 5, duplicates='drop')
    df['stake_bin'] = pd.qcut(df['stake'], 5, duplicates='drop')
    df['ratio_bin'] = pd.qcut(df['log(risky/safe)'], 5, duplicates='drop')

    frames = []
    for by, nice in [(['safe_bin'], 'Safe payoff (CHF)'),
                     (['stake_bin'], 'Stake quintile'),
                     (['ratio_bin'], 'Risky/safe ratio quintile'),
                     (['order'], 'Presentation order'),
                     (['stimulation_condition'], 'Stimulation'),
                     (['chose_risky'], 'Chose risky'),
                     (['order', 'safe_bin'], 'Order x safe payoff')]:
        t = cell_table(df, d, by)
        t.insert(0, 'grouping', nice)
        t['cell'] = t[by].astype(str).agg(' | '.join, axis=1)
        frames.append(t[['grouping', 'cell', 'n', 'total', 'per_trial']])
        print(f'--- {nice} ---')
        for _, r in t.iterrows():
            bar = '#' * int(round(max(r.per_trial, 0) * 400))
            print(f'  {r.cell:26s} n={r.n:5d}  total {r.total:+8.1f}  '
                  f'per-trial {r.per_trial:+.4f}  {bar}')
        print()

    out = pd.concat(frames, ignore_index=True)
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_tsv, sep='\t', index=False)
    print(f'wrote {args.out_tsv}')


if __name__ == '__main__':
    main()
