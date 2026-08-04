"""Per-subject neural, decoding and behavioural measures for the brain-behaviour link.

Everything here is built from the **canonical m1 encoding model** (`amplitude` is the
only per-session regressor, so IPS - vertex is a gain change at fixed tuning) and from
the m1-based trial-wise decoder. See `notes/encoding_model_choice.md` for why m1 and
not m2.

Three stages, three outputs:

    neural     notes/data/bb_neural.tsv       one row per (subject, mask, weighting)
    decoding   notes/data/bb_decoding.tsv     one row per (subject, session, mask)
               notes/data/bb_decoding_runs.tsv    per (subject, session, run, mask)
               notes/data/bb_decoding_trials.tsv  per trial, for the within-subject test
    behavior   notes/data/bb_behavior.tsv     one row per (subject, session)

The amplitude stage reads `notes/data/prf_voxel_table.tsv` (written by
`extract_prf_voxel_table.py`) and needs no BIDS access; the decoding and behaviour
stages read the BIDS tree.

    python -m tms_risk.modeling.scripts.extract_brain_behavior_table all \
        --bids_folder /data/ds-tmsrisk
"""
from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / 'notes' / 'data'

# Payoff grid the behavioural noise function is evaluated on
# (`notes/data/subject_noise_shift.*.tsv` uses 7, 10, 14, 20, 28).
PAYOFFS = [7.0, 10.0, 14.0, 20.0, 28.0]

MASKS = ['NPC12r', 'NPCr2cm-cluster']
# Control ROIs: NPCl is contralateral to the stimulation site, NF1/NTO are
# numerosity-tuned but far from it.
CONTROL_MASKS = ['NPCl', 'NF1', 'NTO']


# --------------------------------------------------------------------------- neural


SELECTIONS = ['cvr2pos', 'top100', 'all']


def clean_voxels(d, mask, selection='cvr2pos'):
    """Voxels inside `mask` with a non-degenerate m1 fit, under one selection rule.

    `cvr2pos` keeps every voxel the model predicts out of sample; `top100` keeps each
    subject's 100 best-cvR2 voxels, which fixes the voxel count across subjects (a few
    subjects have <20 voxels above zero); `all` drops the cvR2 gate entirely.
    """
    keep = d[f'in_{mask}'].copy()
    keep &= d['pref_n_m1'].between(1, 300) & d['sd_m1'].between(0.01, 20)
    # A fit that collapsed the gain to zero in either arm carries no gain change.
    keep &= (d['amplitude_m1_ips'].abs() > 1e-6) & (d['amplitude_m1_vertex'].abs() > 1e-6)
    sel = d[keep]

    if selection == 'cvr2pos':
        return sel[sel['cvr2_m1'] > 0]
    if selection == 'top100':
        return (sel.sort_values('cvr2_m1', ascending=False)
                   .groupby('subject', group_keys=False).head(100))
    if selection == 'all':
        return sel
    raise ValueError(selection)


def neural_table(voxel_tsv, out_tsv):
    d = pd.read_csv(voxel_tsv, sep='\t')
    d['d_amp'] = d['amplitude_m1_ips'] - d['amplitude_m1_vertex']
    # Relative gain change, symmetric in the two arms so it cannot blow up.
    d['d_amp_rel'] = d['d_amp'] / (0.5 * (d['amplitude_m1_ips'] + d['amplitude_m1_vertex']))
    d['log_amp_ratio'] = np.log(d['amplitude_m1_ips']) - np.log(d['amplitude_m1_vertex'])
    # m1's other per-session parameter. It has no role in the TMS hypothesis, so it
    # serves as a within-model control for generic session-quality differences.
    d['d_baseline'] = d['baseline_m1_ips'] - d['baseline_m1_vertex']

    rows = []
    for mask, selection in product(MASKS + CONTROL_MASKS, SELECTIONS):
        sel = clean_voxels(d, mask, selection)
        for sub, g in sel.groupby('subject'):
            r = {'subject': sub, 'mask': mask, 'selection': selection,
                 'n_voxels': len(g)}
            if len(g) < 20:
                logging.warning(f'sub-{sub} {mask}/{selection}: only {len(g)} voxels')

            r['d_amp_mean'] = g['d_amp'].mean()
            r['d_amp_median'] = g['d_amp'].median()
            r['d_amp_rel_mean'] = g['d_amp_rel'].mean()
            r['d_amp_rel_median'] = g['d_amp_rel'].median()
            r['log_amp_ratio_median'] = g['log_amp_ratio'].median()
            r['d_baseline_median'] = g['d_baseline'].median()
            r['d_baseline_mean'] = g['d_baseline'].mean()
            r['amp_vertex_median'] = g['amplitude_m1_vertex'].median()
            r['pref_n_median'] = g['pref_n_m1'].median()

            # --- split by preferred numerosity -------------------------------
            # "low" = tuned inside the payoff window where the behavioural noise
            # function is credible (7-14 CHF); "high" = above the presented range.
            low = g[g['pref_n_m1'] <= 14]
            high = g[g['pref_n_m1'] > 14]
            r['n_low'], r['n_high'] = len(low), len(high)
            r['d_amp_low'] = low['d_amp'].median() if len(low) >= 10 else np.nan
            r['d_amp_high'] = high['d_amp'].median() if len(high) >= 10 else np.nan
            r['d_amp_rel_low'] = low['d_amp_rel'].median() if len(low) >= 10 else np.nan
            r['d_amp_rel_high'] = high['d_amp_rel'].median() if len(high) >= 10 else np.nan
            r['d_amp_lowminushigh'] = r['d_amp_low'] - r['d_amp_high']
            r['d_amp_rel_lowminushigh'] = r['d_amp_rel_low'] - r['d_amp_rel_high']

            # --- tuning-weighted profile over the payoff grid ----------------
            # w_v(n) = the voxel's Gaussian tuning evaluated at log(n): how much
            # voxel v contributes to the population response to a payoff of n.
            mu, sd = g['mu_m1'].values, g['sd_m1'].values
            for n in PAYOFFS:
                w = np.exp(-0.5 * ((np.log(n) - mu) / sd) ** 2)
                if w.sum() < 1e-8:
                    r[f'd_amp_w{int(n)}'] = np.nan
                    r[f'd_amp_rel_w{int(n)}'] = np.nan
                    continue
                r[f'd_amp_w{int(n)}'] = float(np.sum(w * g['d_amp'].values) / w.sum())
                r[f'd_amp_rel_w{int(n)}'] = float(
                    np.sum(w * g['d_amp_rel'].values) / w.sum())
            r['d_amp_wslope'] = r['d_amp_w7'] - r['d_amp_w28']
            r['d_amp_rel_wslope'] = r['d_amp_rel_w7'] - r['d_amp_rel_w28']
            rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}  {out.shape}')
    return out


# ------------------------------------------------------------------------ decoding


def decoding_tables(bids_folder, subjects, masks=MASKS):
    """Trial-wise decoded posteriors -> trial / run / session summaries.

    Uses the paper's decoding configuration: `n_voxels=1` (cross-validated voxel
    selection), `denoise=True`, `smoothed=False`, `natural_space=True`, i.e.
    `derivatives/decoded_pdfs.volume.cv_voxel_selection.denoise.natural_space/`,
    whose posteriors were decoded with m1 parameters (`decode_select_voxels_cv.py`
    line 157: `get_prf_parameters(model_label=1, ...)`).
    """
    from tms_risk.utils.data import get_all_behavior, get_pdf

    beh = get_all_behavior(bids_folder=bids_folder, drop_no_responses=False)

    trials = []
    for sub, session, mask in product(subjects, [1, 2, 3], masks):
        pdf = get_pdf(sub, session, False, True, False, str(bids_folder), mask, 1, True)
        if pdf.shape[0] == 0:
            continue
        grid = pdf.columns.values.astype(float)
        p = pdf.values
        E = np.trapz(p * grid[np.newaxis, :], grid, axis=1)
        # Posterior spread: mean |s - E| under the posterior (`get_decoding_info`'s
        # `sd` column), i.e. the decoder's own uncertainty.
        spread = np.trapz(np.abs(E[:, np.newaxis] - grid[np.newaxis, :]) * p, grid, axis=1)
        t = pd.DataFrame({'E': E, 'post_spread': spread},
                         index=pdf.index.get_level_values('trial_nr'))
        t['subject'], t['session'], t['mask'] = int(sub), int(session), mask
        trials.append(t.reset_index())

    trials = pd.concat(trials)
    b = beh.reset_index()[['subject', 'session', 'stimulation_condition', 'run',
                           'trial_nr', 'n1', 'n2', 'n_risky', 'n_safe',
                           'risky_first', 'chose_risky', 'rt']]
    trials = trials.merge(b, on=['subject', 'session', 'trial_nr'], how='inner')

    trials['abs_err'] = (trials['E'] - trials['n1']).abs()
    trials['log_abs_err'] = (np.log(trials['E']) - np.log(trials['n1'])).abs()
    trials['n1_low'] = trials['n1'] <= trials.groupby('subject')['n1'].transform('median')

    # --- run level --------------------------------------------------------
    def _acc(g):
        if g['n1'].nunique() < 3 or len(g) < 8:
            return np.nan
        return np.corrcoef(g['E'], g['n1'])[0, 1]

    runs = (trials.groupby(['subject', 'session', 'stimulation_condition', 'mask', 'run'])
            .apply(lambda g: pd.Series({
                'r_En1': _acc(g),
                'abs_err': g['abs_err'].mean(),
                'log_abs_err': g['log_abs_err'].mean(),
                'post_spread': g['post_spread'].mean(),
                'n_trials': len(g)}))
            .reset_index())

    # --- session level ----------------------------------------------------
    ses = (runs.groupby(['subject', 'session', 'stimulation_condition', 'mask'])
           [['r_En1', 'abs_err', 'log_abs_err', 'post_spread']].mean().reset_index())

    # decoding accuracy separately for low / high presented numerosity
    for lab, m in [('low', True), ('high', False)]:
        sub_ = trials[trials['n1_low'] == m]
        s = (sub_.groupby(['subject', 'session', 'mask'])
             .apply(lambda g: pd.Series({
                 f'log_abs_err_{lab}': g['log_abs_err'].mean(),
                 f'post_spread_{lab}': g['post_spread'].mean()}))
             .reset_index())
        ses = ses.merge(s, on=['subject', 'session', 'mask'], how='left')

    return trials, runs, ses


# ------------------------------------------------------------------------ behaviour


def _logit_fit(y, x, ridge=1e-2):
    """Slope and indifference point of a logistic psychometric on log(risky/safe).

    Penalised MLE: the ridge only bites when a session is quasi-separated, which the
    adaptive staircase makes fairly common (log-ratio SD is ~0.2 within a session).
    """
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 20 or y.std() == 0:
        return np.nan, np.nan

    def nll(b):
        z = np.clip(b[0] + b[1] * x, -30, 30)
        return -np.sum(y * z - np.logaddexp(0, z)) + ridge * b[1] ** 2

    res = minimize(nll, np.array([0.0, 1.0]), method='BFGS')
    b0, b1 = res.x
    if not np.isfinite(b1) or abs(b1) < 1e-6:
        return np.nan, np.nan
    return b1, -b0 / b1


def behavior_table(bids_folder, out_tsv):
    from tms_risk.utils.data import get_all_behavior

    df = get_all_behavior(bids_folder=bids_folder).reset_index()
    df = df[df['session'].isin([2, 3])].copy()
    df['lr'] = df['log(risky/safe)']

    rows = []
    for (sub, ses), g in df.groupby(['subject', 'session']):
        r = {'subject': sub, 'session': ses,
             'stimulation_condition': g['stimulation_condition'].iloc[0],
             'n_trials': len(g)}
        r['p_risky'] = g['chose_risky'].mean()
        for lab, m in [('rfirst', g['risky_first']), ('rsecond', ~g['risky_first'])]:
            r[f'p_risky_{lab}'] = g.loc[m, 'chose_risky'].mean()

        # magnitude split on the clean axis (n_safe is orthogonal to log ratio)
        med = g['n_safe'].median()
        r['p_risky_lowsafe'] = g.loc[g['n_safe'] <= med, 'chose_risky'].mean()
        r['p_risky_highsafe'] = g.loc[g['n_safe'] > med, 'chose_risky'].mean()

        slope, indiff = _logit_fit(g['chose_risky'], g['lr'])
        r['consistency'] = slope
        r['indifference'] = indiff
        for lab, m in [('rfirst', g['risky_first']), ('rsecond', ~g['risky_first'])]:
            s, i = _logit_fit(g.loc[m, 'chose_risky'], g.loc[m, 'lr'])
            r[f'consistency_{lab}'], r[f'indifference_{lab}'] = s, i
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}  {out.shape}')
    return out


# ----------------------------------------------------------------------------- cli


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['neural', 'decoding', 'behavior', 'all'])
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--voxel_tsv', default=str(DATA / 'prf_voxel_table.tsv'))
    p.add_argument('--out_prefix', default=str(DATA / 'bb'))
    args = p.parse_args()

    pre = Path(args.out_prefix)
    if args.stage in ('neural', 'all'):
        neural_table(args.voxel_tsv, f'{pre}_neural.tsv')

    if args.stage in ('decoding', 'all'):
        from tms_risk.modeling.scripts.extract_prf_voxel_table import SUBJECTS
        trials, runs, ses = decoding_tables(args.bids_folder, SUBJECTS)
        trials.to_csv(f'{pre}_decoding_trials.tsv', sep='\t', index=False)
        runs.to_csv(f'{pre}_decoding_runs.tsv', sep='\t', index=False)
        ses.to_csv(f'{pre}_decoding.tsv', sep='\t', index=False)
        print(f'wrote decoding tables: {trials.shape} / {runs.shape} / {ses.shape}')

    if args.stage in ('behavior', 'all'):
        behavior_table(args.bids_folder, f'{pre}_behavior.tsv')


if __name__ == '__main__':
    main()
