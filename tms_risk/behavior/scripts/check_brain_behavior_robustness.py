"""Follow-up checks on the nPRF-gain / choice-consistency link.

Everything the headline correlation needs before it can be believed:

  1. the group-level effects it is a covariation of (does consistency drop at all?)
  2. split-half reliability of the per-subject difference scores
  3. stability across mask, voxel selection and amplitude summary
  4. whether the voxels' preferred numerosity matters (the "specific region of
     numbers" question)
  5. whether it is a psychometric-slope artifact (slope/intercept entanglement)

    python -m tms_risk.behavior.scripts.check_brain_behavior_robustness
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tms_risk.modeling.scripts.extract_brain_behavior_table import _logit_fit
from tms_risk.behavior.scripts.analyze_brain_behavior_link import build_master

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / 'notes' / 'data'


def group_effects(bids_folder):
    beh = pd.read_csv(DATA / 'bb_behavior.tsv', sep='\t')
    print('=== 1. Group-level cTBS effects the correlation is built on')
    for c in ['p_risky_rfirst', 'p_risky_rsecond', 'consistency',
              'consistency_rfirst', 'consistency_rsecond',
              'indifference_rfirst', 'indifference_rsecond']:
        w = beh.pivot_table(index='subject', columns='stimulation_condition', values=c)
        d = (w['ips'] - w['vertex']).dropna()
        t, p = stats.ttest_1samp(d, 0)
        print(f'  {c:22s} vertex {w["vertex"].mean():7.3f}  ips {w["ips"].mean():7.3f}'
              f'  delta {d.mean():+7.3f}  t({len(d)-1}) = {t:+5.2f}  p = {p:.4f}')

    neu = pd.read_csv(DATA / 'bb_neural.tsv', sep='\t')
    print('\n  nPRF gain change (IPS - vertex), per mask/selection:')
    for (m, s), g in neu.groupby(['mask', 'selection']):
        d = g['d_amp_median'].dropna()
        t, p = stats.ttest_1samp(d, 0)
        print(f'  {m:16s} {s:8s} median delta amp {d.mean():+.4f}  '
              f'({(d < 0).sum()}/{len(d)} subjects negative)  '
              f't({len(d)-1}) = {t:+5.2f}  p = {p:.4f}')


def split_half(bids_folder, n_splits=200, seed=0):
    """Reliability of the per-subject IPS - vertex consistency difference score."""
    from tms_risk.utils.data import get_all_behavior

    print('\n=== 2. Split-half reliability of the behavioural difference scores')
    df = get_all_behavior(bids_folder=bids_folder).reset_index()
    df = df[df['session'].isin([2, 3])].copy()
    df['lr'] = df['log(risky/safe)']
    rng = np.random.default_rng(seed)

    res = {k: [] for k in ['consistency', 'consistency_rsecond', 'consistency_rfirst',
                           'indifference_rsecond', 'p_risky_rsecond']}
    for _ in range(n_splits):
        df['half'] = rng.integers(0, 2, len(df))
        est = {}
        for h in [0, 1]:
            rows = []
            for (sub, ses), g in df[df['half'] == h].groupby(['subject', 'session']):
                slope, indiff = _logit_fit(g['chose_risky'], g['lr'])
                m2 = ~g['risky_first']
                s2, i2 = _logit_fit(g.loc[m2, 'chose_risky'], g.loc[m2, 'lr'])
                m1 = g['risky_first']
                s1, _ = _logit_fit(g.loc[m1, 'chose_risky'], g.loc[m1, 'lr'])
                rows.append({'subject': sub, 'session': ses,
                             'stimulation_condition': g['stimulation_condition'].iloc[0],
                             'consistency': slope, 'consistency_rsecond': s2,
                             'consistency_rfirst': s1, 'indifference_rsecond': i2,
                             'p_risky_rsecond': g.loc[m2, 'chose_risky'].mean()})
            r = pd.DataFrame(rows)
            est[h] = {c: (r.pivot_table(index='subject', columns='stimulation_condition',
                                        values=c)
                          .pipe(lambda w: w['ips'] - w['vertex']))
                      for c in res}
        for c in res:
            a, b = est[0][c], est[1][c]
            d = pd.concat([a, b], axis=1).dropna()
            if len(d) > 10:
                res[c].append(stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])[0])

    for c, v in res.items():
        v = np.array(v)
        # Spearman-Brown steps the half-length reliability up to full length
        sb = 2 * v / (1 + v)
        print(f'  {c:22s} half-half r = {v.mean():+.3f}  '
              f'-> full-length reliability {sb.mean():+.3f}')


def rng_mask(rng, n, key):
    return rng.integers(0, 2, n)


def stability(pmc_label='flexible2nf'):
    print('\n=== 3. Stability of the gain x consistency(risky-second) link')
    rows = []
    for mask, sel, amp in product(['NPCr2cm-cluster', 'NPC12r'],
                                  ['cvr2pos', 'top100', 'all'],
                                  ['d_amp_median', 'd_amp_mean', 'd_amp_rel_median']):
        m = build_master(mask, sel)
        d = m[[amp, 'd_consistency_rsecond', 'd_consistency_rfirst']].dropna()
        r2, p2 = stats.pearsonr(d[amp], d['d_consistency_rsecond'])
        r1, p1 = stats.pearsonr(d[amp], d['d_consistency_rfirst'])
        rows.append({'mask': mask, 'selection': sel, 'amp': amp, 'n': len(d),
                     'r_rsecond': r2, 'p_rsecond': p2, 'r_rfirst': r1, 'p_rfirst': p1})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format='%.4f'))
    print(f'\n  risky-second: {(out.r_rsecond > 0).sum()}/{len(out)} positive, '
          f'{(out.p_rsecond < .05).sum()}/{len(out)} at p < .05, '
          f'range {out.r_rsecond.min():+.3f} to {out.r_rsecond.max():+.3f}')
    print(f'  risky-first : {(out.r_rfirst > 0).sum()}/{len(out)} positive, '
          f'{(out.p_rfirst < .05).sum()}/{len(out)} at p < .05, '
          f'range {out.r_rfirst.min():+.3f} to {out.r_rfirst.max():+.3f}')
    return out


def numerosity_specificity():
    """Does it matter which numerosity the disrupted voxels prefer?"""
    print('\n=== 4. Preferred-numerosity specificity of the gain change')
    for mask, sel in product(['NPC12r', 'NPCr2cm-cluster'], ['all', 'cvr2pos']):
        m = build_master(mask, sel)
        cols = ['d_amp_low', 'd_amp_high', 'd_amp_w7', 'd_amp_w14', 'd_amp_w28',
                'd_amp_wslope']
        cols = [c for c in cols if c in m and m[c].notna().sum() >= 25]
        print(f'  -- {mask} / {sel}')
        for c in cols:
            d = m[[c, 'd_consistency_rsecond']].dropna()
            r, p = stats.pearsonr(d[c], d['d_consistency_rsecond'])
            print(f'     {c:16s} n={len(d):3d}  r = {r:+.3f}  p = {p:.4f}')
        if 'd_amp_low' in cols and 'd_amp_high' in cols:
            d = m[['d_amp_low', 'd_amp_high', 'd_consistency_rsecond']].dropna()
            # which band survives the other?
            for a, b in [('d_amp_low', 'd_amp_high'), ('d_amp_high', 'd_amp_low')]:
                C = np.column_stack([np.ones(len(d)), d[b].values])
                rx = d[a].values - C @ np.linalg.lstsq(C, d[a].values, rcond=None)[0]
                ry = (d['d_consistency_rsecond'].values
                      - C @ np.linalg.lstsq(C, d['d_consistency_rsecond'].values,
                                            rcond=None)[0])
                r = np.corrcoef(rx, ry)[0, 1]
                dof = len(d) - 3
                t = r * np.sqrt(dof / (1 - r ** 2))
                print(f'     {a} | {b}: partial r = {r:+.3f}, '
                      f'p = {2 * stats.t.sf(abs(t), dof):.4f} (n={len(d)})')


def slope_artifact():
    """Is the consistency drop just a shift in choice level under a fixed slope?"""
    print('\n=== 5. Is it a slope/intercept artifact?')
    m = build_master('NPCr2cm-cluster', 'cvr2pos')
    for c in ['d_p_risky_rsecond', 'd_indifference_rsecond']:
        d = m[['d_amp_median', 'd_consistency_rsecond', c]].dropna()
        r, p = stats.pearsonr(d['d_consistency_rsecond'], d[c])
        print(f'  d_consistency_rsecond vs {c:24s}: r = {r:+.3f}, p = {p:.4f}')
    # partial the amplitude link on both
    d = m[['d_amp_median', 'd_consistency_rsecond', 'd_p_risky_rsecond',
           'd_indifference_rsecond']].dropna()
    C = np.column_stack([np.ones(len(d)), d['d_p_risky_rsecond'],
                         d['d_indifference_rsecond']])
    rx = d['d_amp_median'] - C @ np.linalg.lstsq(C, d['d_amp_median'], rcond=None)[0]
    ry = (d['d_consistency_rsecond']
          - C @ np.linalg.lstsq(C, d['d_consistency_rsecond'], rcond=None)[0])
    r = np.corrcoef(rx, ry)[0, 1]
    dof = len(d) - 4
    t = r * np.sqrt(dof / (1 - r ** 2))
    print(f'  gain x consistency, partialling out both choice-level measures: '
          f'r = {r:+.3f}, p = {2 * stats.t.sf(abs(t), dof):.4f} (n={len(d)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--n_splits', type=int, default=200)
    args = ap.parse_args()

    group_effects(args.bids_folder)
    split_half(args.bids_folder, args.n_splits)
    stability().to_csv(DATA / 'bb_link_stability.tsv', sep='\t', index=False)
    numerosity_specificity()
    slope_artifact()


if __name__ == '__main__':
    main()
