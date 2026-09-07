"""Does the size of the neural cTBS effect predict the size of the behavioural one?

Everything is a per-subject IPS - vertex contrast, so a correlation across the 35
subjects asks: do the people whose parietal numerosity code was disrupted most also
show the largest behavioural change?

Neural side (canonical m1 only, `notes/encoding_model_choice.md`):
  * nPRF gain loss    -- `notes/data/bb_neural.tsv`, per mask x voxel-selection,
                         overall, split by preferred numerosity, and as a
                         tuning-weighted profile over payoffs 7..28.
  * decoding loss     -- `notes/data/bb_decoding.tsv`, the m1-based trial-wise
                         decoder: accuracy (corr with the presented numerosity),
                         error, and posterior width.

Behavioural side:
  * model-free        -- `notes/data/bb_behavior.tsv`: P(chose risky), psychometric
                         slope (consistency), indifference point.
  * model-based       -- `notes/data/subject_noise_shift.<label>.tsv`: the per-subject
                         cTBS increase in perceptual noise from the latest PMC refits.

Two families of tests. A small **primary** set stated directionally in advance, and an
**exploratory grid** of every neural x behavioural pair whose family-wise error is
controlled by a max-|r| permutation over subject labels.

    python -m tms_risk.behavior.scripts.analyze_brain_behavior_link \
        --pmc_label flexible2nf --mask NPCr2cm-cluster --selection cvr2pos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / 'notes' / 'data'

PAYOFFS = [7, 10, 14, 20, 28]


# ------------------------------------------------------------------ master table


def _contrast(df, keys, value_cols):
    """IPS - vertex, per subject."""
    w = df.pivot_table(index=keys, columns='stimulation_condition', values=value_cols)
    out = {}
    for c in value_cols:
        if ('ips' in w[c].columns) and ('vertex' in w[c].columns):
            out[c] = w[(c, 'ips')] - w[(c, 'vertex')]
    return pd.DataFrame(out)


def build_master(mask, selection, pmc_labels=('flexible2nf', 'flexible1nf')):
    neu = pd.read_csv(DATA / 'bb_neural.tsv', sep='\t')
    neu = neu[(neu['mask'] == mask) & (neu['selection'] == selection)].set_index('subject')
    keep = [c for c in neu.columns if c.startswith('d_amp')] + ['n_voxels', 'pref_n_median']
    N = neu[keep].copy()

    dec = pd.read_csv(DATA / 'bb_decoding.tsv', sep='\t')
    dec = dec[(dec['mask'] == mask) & (dec['session'] != 1)]
    cols = ['r_En1', 'log_abs_err', 'post_spread',
            'log_abs_err_low', 'log_abs_err_high',
            'post_spread_low', 'post_spread_high']
    D = _contrast(dec, ['subject'], cols).add_prefix('d_')
    D['d_log_abs_err_lowminushigh'] = D['d_log_abs_err_low'] - D['d_log_abs_err_high']
    D['d_post_spread_lowminushigh'] = D['d_post_spread_low'] - D['d_post_spread_high']

    beh = pd.read_csv(DATA / 'bb_behavior.tsv', sep='\t')
    bcols = ['p_risky', 'p_risky_rfirst', 'p_risky_rsecond',
             'p_risky_lowsafe', 'p_risky_highsafe',
             'consistency', 'indifference',
             'consistency_rfirst', 'indifference_rfirst',
             'consistency_rsecond', 'indifference_rsecond']
    B = _contrast(beh, ['subject'], bcols).add_prefix('d_')
    for c in ['p_risky', 'consistency', 'indifference']:
        # risky-second minus risky-first: the order-specificity of the cTBS effect
        B[f'd_{c}_orderdiff'] = B[f'd_{c}_rsecond'] - B[f'd_{c}_rfirst']

    P = []
    for lab in pmc_labels:
        f = DATA / f'subject_noise_shift.{lab}.tsv'
        if not f.exists():
            continue
        s = pd.read_csv(f, sep='\t')
        s = s[s['term'] == 'perceptual'].pivot_table(index='subject', columns='payoff',
                                                     values='d_nu')
        s.columns = [f'dnu_{lab}_{int(c)}' if c > 0 else f'dnu_{lab}_slope'
                     for c in s.columns]
        P.append(s)
    P = pd.concat(P, axis=1) if P else pd.DataFrame()

    master = pd.concat([N, D, B, P], axis=1)
    master.index.name = 'subject'
    return master


NEURAL_PREFIXES = ('d_amp', 'd_r_En1', 'd_log_abs_err', 'd_post_spread')
BEHAV_PREFIXES = ('d_p_risky', 'd_consistency', 'd_indifference', 'dnu_')


def split_blocks(master, pmc_label):
    neural = [c for c in master.columns if c.startswith(NEURAL_PREFIXES)]
    behav = [c for c in master.columns
             if c.startswith(BEHAV_PREFIXES) and (
                 not c.startswith('dnu_') or pmc_label in c)]
    return neural, behav


# ----------------------------------------------------------------------- testing


def corr_grid(master, neural, behav, method='pearson'):
    rows = []
    for n in neural:
        for b in behav:
            d = master[[n, b]].dropna()
            if len(d) < 15 or d[n].std() == 0 or d[b].std() == 0:
                continue
            if method == 'pearson':
                r, p = stats.pearsonr(d[n], d[b])
            else:
                r, p = stats.spearmanr(d[n], d[b])
            rows.append({'neural': n, 'behav': b, 'n': len(d), 'r': r, 'p': p})
    return pd.DataFrame(rows)


def perm_fwer(master, neural, behav, n_perm=10000, method='pearson', seed=1):
    """Max-|r| permutation over subject labels: FWER across the whole grid.

    The max statistic needs one common set of subjects, so measures that are missing
    for more than a few subjects (the preferred-numerosity split needs >=10 voxels per
    band, which the small stimulation-site masks do not always have) are dropped from
    the grid rather than shrinking it to their intersection.
    """
    rng = np.random.default_rng(seed)
    cov = master[neural + behav].notna().mean()
    dropped = cov[cov < 0.9].index.tolist()
    if dropped:
        print(f'  [grid] dropping {len(dropped)} sparse measures: {dropped}')
    neural = [c for c in neural if c not in dropped]
    behav = [c for c in behav if c not in dropped]
    d = master[neural + behav].dropna()
    X = d[neural].values
    Y = d[behav].values
    if method == 'spearman':
        X = np.apply_along_axis(stats.rankdata, 0, X)
        Y = np.apply_along_axis(stats.rankdata, 0, Y)
    Xz = (X - X.mean(0)) / X.std(0)
    Yz = (Y - Y.mean(0)) / Y.std(0)
    n = len(d)
    obs = np.abs(Xz.T @ Yz / n)

    null = np.empty(n_perm)
    for i in range(n_perm):
        ix = rng.permutation(n)
        null[i] = np.abs(Xz.T @ Yz[ix] / n).max()

    grid = pd.DataFrame(obs, index=neural, columns=behav).stack().reset_index()
    grid.columns = ['neural', 'behav', 'abs_r']
    grid['p_fwer'] = [(1 + (null >= a).sum()) / (1 + n_perm) for a in grid['abs_r']]
    grid['n'] = n
    return grid.sort_values('abs_r', ascending=False), null


def zs(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def composites(master):
    """One neural and one behavioural disruption index, both signed 'more disrupted'."""
    m = master
    neural_parts = {
        'amp_loss': -zs(m['d_amp_rel_median']),
        'decode_acc_loss': -zs(m['d_r_En1']),
        'decode_err_gain': zs(m['d_log_abs_err']),
        'post_width_gain': zs(m['d_post_spread']),
    }
    behav_parts = {
        'risk_seeking_gain': zs(m['d_p_risky']),
        'consistency_loss': -zs(m['d_consistency']),
    }
    out = pd.DataFrame(neural_parts | behav_parts, index=m.index)
    out['neural_index'] = out[list(neural_parts)].mean(axis=1)
    out['behav_index'] = out[list(behav_parts)].mean(axis=1)
    return out


def focused_family(master, neural, behav, n_perm=20000, seed=2):
    """FWER over a small, a-priori family, by the same max-|r| permutation."""
    grid, null = perm_fwer(master, neural, behav, n_perm=n_perm, seed=seed)
    raw = corr_grid(master, neural, behav)
    return grid.merge(raw[['neural', 'behav', 'r', 'p']], on=['neural', 'behav'],
                      how='left'), null


def robustness(master, n, b, n_boot=10000, seed=3):
    """Bootstrap CI, leave-one-subject-out range, and rank version of one pair."""
    rng = np.random.default_rng(seed)
    d = master[[n, b]].dropna()
    x, y = d[n].values, d[b].values
    r = stats.pearsonr(x, y)[0]

    boot = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, len(x), len(x))
        boot[i] = np.corrcoef(x[ix], y[ix])[0, 1] if np.std(x[ix]) > 0 else np.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])

    loo = np.array([np.corrcoef(np.delete(x, i), np.delete(y, i))[0, 1]
                    for i in range(len(x))])
    rho = stats.spearmanr(x, y)[0]
    return {'neural': n, 'behav': b, 'n': len(d), 'r': r, 'ci_lo': lo, 'ci_hi': hi,
            'loo_min': loo.min(), 'loo_max': loo.max(), 'rho': rho,
            'boot_frac_same_sign': float(np.mean(np.sign(boot) == np.sign(r)))}


def partial_corr(master, x, y, covars):
    """Pearson correlation of x and y after regressing both on `covars`."""
    d = master[[x, y] + list(covars)].dropna()
    C = np.column_stack([np.ones(len(d))] + [d[c].values for c in covars])
    rx = d[x].values - C @ np.linalg.lstsq(C, d[x].values, rcond=None)[0]
    ry = d[y].values - C @ np.linalg.lstsq(C, d[y].values, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    dof = len(d) - 2 - len(covars)
    t = r * np.sqrt(dof / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), dof), len(d)


def report_pair(master, n, b, alternative='two-sided', label=''):
    d = master[[n, b]].dropna()
    r, p = stats.pearsonr(d[n], d[b])
    if alternative != 'two-sided':
        p = p / 2 if ((r > 0) == (alternative == 'greater')) else 1 - p / 2
    rho, prho = stats.spearmanr(d[n], d[b])
    return {'label': label, 'neural': n, 'behav': b, 'n': len(d),
            'r': r, 'p': p, 'alternative': alternative, 'rho': rho, 'p_rho': prho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', default='NPCr2cm-cluster')
    ap.add_argument('--selection', default='cvr2pos')
    ap.add_argument('--pmc_label', default='flexible2nf')
    ap.add_argument('--n_perm', type=int, default=10000)
    ap.add_argument('--out_prefix', default=str(DATA / 'bb_link'))
    args = ap.parse_args()

    master = build_master(args.mask, args.selection)
    master.to_csv(f'{args.out_prefix}_master.tsv', sep='\t')
    neural, behav = split_blocks(master, args.pmc_label)

    print(f'\n=== mask {args.mask} / selection {args.selection} / PMC {args.pmc_label}')
    print(f'{len(neural)} neural x {len(behav)} behavioural measures, '
          f'{master.shape[0]} subjects\n')

    # ---- primary, directional -------------------------------------------
    amp = 'd_amp_rel_median'
    prim = [
        (amp, 'd_p_risky', 'less', 'gain loss -> more risk seeking'),
        (amp, 'd_consistency', 'greater', 'gain loss -> less consistent'),
        ('d_r_En1', 'd_p_risky', 'less', 'decoding loss -> more risk seeking'),
        ('d_r_En1', 'd_consistency', 'greater', 'decoding loss -> less consistent'),
        (amp, f'dnu_{args.pmc_label}_7', 'less', 'gain loss -> more model noise'),
        ('d_r_En1', f'dnu_{args.pmc_label}_7', 'less',
         'decoding loss -> more model noise'),
    ]
    rows = [report_pair(master, n, b, alt, lab) for n, b, alt, lab in prim
            if n in master and b in master]
    prim = pd.DataFrame(rows)
    # Holm within the primary family
    order = prim['p'].rank(method='first').astype(int)
    prim['p_holm'] = np.minimum(1, prim['p'] * (len(prim) - order + 1))
    prim['p_holm'] = prim.sort_values('p')['p_holm'].cummax().reindex(prim.index)
    print('--- Primary, directional (Holm over 6):')
    print(prim[['label', 'n', 'r', 'p', 'p_holm', 'rho']].to_string(index=False,
                                                                   float_format='%.4f'))

    # ---- focused a-priori family ----------------------------------------
    # Two canonical neural measures x three behavioural read-outs x the two
    # presentation orders. The order split is not a free choice: the group-level
    # cTBS effect is itself order-specific (risky-second +0.053 vs risky-first
    # +0.006), so a brain-behaviour link should live in the same cell.
    f_neural = ['d_amp_median', 'd_r_En1']
    f_behav = [f'd_{c}_{o}' for c in ['p_risky', 'consistency', 'indifference']
               for o in ['rsecond', 'rfirst']]
    ffam, fnull = focused_family(master, f_neural, f_behav)
    print(f'\n--- Focused family ({len(ffam)} pairs), max-|r| null 95th pct = '
          f'{np.quantile(fnull, .95):.3f}')
    print(ffam[['neural', 'behav', 'n', 'r', 'p', 'p_fwer']]
          .to_string(index=False, float_format='%.4f'))
    ffam.to_csv(f'{args.out_prefix}_focused.tsv', sep='\t', index=False)

    print('\n--- Order specificity (neural vs risky-second minus risky-first):')
    inter = [report_pair(master, n, f'd_{c}_orderdiff')
             for n in f_neural for c in ['p_risky', 'consistency', 'indifference']]
    print(pd.DataFrame(inter)[['neural', 'behav', 'n', 'r', 'p', 'rho']]
          .to_string(index=False, float_format='%.4f'))

    # ---- robustness of whatever the focused family turned up -------------
    hits = ffam[ffam['p'] < 0.05]
    if len(hits):
        print('\n--- Robustness of the focused-family hits:')
        rob = pd.DataFrame([robustness(master, r.neural, r.behav)
                            for r in hits.itertuples()])
        print(rob.to_string(index=False, float_format='%.3f'))
        rob.to_csv(f'{args.out_prefix}_robustness.tsv', sep='\t', index=False)

        print('\n--- Partialling out the same measure in the other order, '
              'and the overall choice level:')
        for r in hits.itertuples():
            if r.behav.endswith('rsecond'):
                other = r.behav.replace('rsecond', 'rfirst')
                pr, pp, nn = partial_corr(master, r.neural, r.behav, [other])
                pr2, pp2, _ = partial_corr(master, r.neural, r.behav,
                                           [other, 'd_p_risky'])
                print(f'  {r.neural:14s} x {r.behav:22s} '
                      f'| {other}: r = {pr:+.3f}, p = {pp:.4f} (n={nn}); '
                      f'+ d_p_risky: r = {pr2:+.3f}, p = {pp2:.4f}')

    # ---- composites ------------------------------------------------------
    comp = composites(master)
    d = comp[['neural_index', 'behav_index']].dropna()
    r, p = stats.pearsonr(d['neural_index'], d['behav_index'])
    print(f'\n--- Composite indices: r({len(d)-2}) = {r:.3f}, '
          f'p(two-sided) = {p:.4f}, one-sided(+) = {p/2 if r>0 else 1-p/2:.4f}')
    comp.to_csv(f'{args.out_prefix}_composite.tsv', sep='\t')

    # ---- exploratory grid ------------------------------------------------
    grid, null = perm_fwer(master, neural, behav, args.n_perm)
    raw = corr_grid(master, neural, behav)
    grid = grid.merge(raw[['neural', 'behav', 'r', 'p']], on=['neural', 'behav'],
                      how='left')
    grid.to_csv(f'{args.out_prefix}_grid.tsv', sep='\t', index=False)
    print(f'\n--- Exploratory grid: {len(grid)} pairs, '
          f'max-|r| null 95th pct = {np.quantile(null, .95):.3f}')
    print(grid.head(15)[['neural', 'behav', 'n', 'r', 'p', 'p_fwer']]
          .to_string(index=False, float_format='%.4f'))
    print(f'\nuncorrected p < .05: {(grid["p"] < .05).sum()} / {len(grid)} '
          f'(expected by chance {0.05*len(grid):.1f}); '
          f'FWER-significant: {(grid["p_fwer"] < .05).sum()}')


if __name__ == '__main__':
    main()
