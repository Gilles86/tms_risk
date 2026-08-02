"""Is the cTBS amplitude attenuation concentrated in low-preferred-numerosity,
genuinely-tuned (non-monotonic) voxels?

The paper's one robust neural effect is a drop in nPRF *amplitude* after
parietal cTBS (1.30→1.04 PSC, p=0.027 one-sided), with μ and σ unchanged. The
theoretical payoff is that this attenuation hits the voxels tuned to *small*
numerosities (preferred-numerosity IQR [6,10] vs presented [13,30]), raising
neurocognitive noise for small payoffs. So the effect should be strongest in
voxels whose Gaussian nPRF actually peaks within the presented range (i.e. the
response is non-monotonic over the stimulus interval) at a *low* preferred
numerosity — not in degenerate huge-σ / out-of-range fits that behave
monotonically over the range.

We use the canonical **m1** PRF (μ, σ shared across sessions; amplitude
per-session), so the IPS-vs-Vertex contrast is on amplitude alone with tuning
held fixed. 'ips' = parietal-stimulation session, 'vertex' = sham.

Monotonicity over the presented interval [n_lo, n_hi]: a Gaussian peaks at μ,
so it is non-monotonic over the interval iff n_lo < preferred_n < n_hi, and
monotonic otherwise. 'Low preferred' = peak in the lower part of the range.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from tms_risk.utils.data import Subject, get_all_behavior
from tms_risk.modeling.scripts.plot_spherical_expected_uncertainty import apply_style
from tms_risk.behavior.utils import stimulation_palette  # IPS green, Vertex red


SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31,
            34, 35, 36, 37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
ROI = 'NPC12r'
# Presented numerosity interval that defines monotonicity over the stimulus.
N_LO, N_HI = 7, 112          # full presented payoff range
LOW_HI = 14                  # upper edge of the 'low preferred numerosity' band
# stimulation_palette is (green, red); IPS (stimulated) = red, Vertex = green.
COLOR = {'ips': stimulation_palette[1], 'vertex': stimulation_palette[0]}


def collect(bids_folder='/data/ds-tmsrisk', model_label=1):
    beh = get_all_behavior(bids_folder=bids_folder)
    cond_map = (beh.reset_index()[['subject', 'session', 'stimulation_condition']]
                    .drop_duplicates())
    frames = []
    for sid in SUBJECTS:
        sub = Subject(sid, bids_folder=bids_folder)
        cond = cond_map[cond_map.subject == sid]
        ses_of = {c: int(cond[cond.stimulation_condition == c]['session'].values[0])
                  for c in ('ips', 'vertex')
                  if len(cond[cond.stimulation_condition == c])}
        if {'ips', 'vertex'} - ses_of.keys():
            continue
        per = {}
        ok = True
        for c, ses in ses_of.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    per[c] = sub.get_prf_parameters(model_label=model_label,
                                                    session=ses, roi=ROI)
            except FileNotFoundError:
                ok = False
        if not ok:
            continue
        ips, vtx = per['ips'], per['vertex']
        signal = (ips['cvr2'] > 0) & (vtx['cvr2'] > 0)
        if signal.sum() == 0:
            continue
        df = pd.DataFrame({
            'subject':    sid,
            'mu':         ips.loc[signal, 'mu'].values,        # shared in m1
            'sd':         ips.loc[signal, 'sd'].values,
            'amp_ips':    ips.loc[signal, 'amplitude'].values,
            'amp_vertex': vtx.loc[signal, 'amplitude'].values,
        })
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out['pref_n'] = np.exp(out['mu'])
    out['amp_diff'] = out['amp_ips'] - out['amp_vertex']   # predicted < 0
    return out


def paired_test(df, label):
    """Per-subject mean amplitude IPS vs Vertex; one-sided paired t (IPS<Vertex)."""
    g = df.groupby('subject')[['amp_ips', 'amp_vertex']].mean().dropna()
    if len(g) < 3:
        print(f'  {label:38s} n={len(g)}  (too few)')
        return
    t, p_two = stats.ttest_rel(g['amp_ips'], g['amp_vertex'])
    p_one = p_two / 2 if t < 0 else 1 - p_two / 2
    d = (g['amp_ips'] - g['amp_vertex']).mean()
    nvox = df.groupby('subject').size().mean()
    print(f'  {label:38s} n={len(g):2d}  Δamp(IPS-V)={d:+.3f}  '
          f't={t:+.2f}  p_1s={p_one:.3f}  (~{nvox:.0f} vox/subj)')
    return g


def main(model_label=1, bids_folder='/data/ds-tmsrisk', from_tsv=False):
    apply_style()
    tsv = Path(f'notes/data/m{model_label}_amplitude_by_pref_n.tsv')
    if from_tsv:
        df = pd.read_csv(tsv, sep='\t')
    else:
        df = collect(bids_folder=bids_folder, model_label=model_label)
        tsv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tsv, sep='\t', index=False)
        print(f'wrote {tsv}  ({len(df)} voxels, {df.subject.nunique()} subjects)')

    nonmono = (df.pref_n > N_LO) & (df.pref_n < N_HI)     # peak in presented range
    low = nonmono & (df.pref_n <= LOW_HI)                 # low preferred numerosity
    high = nonmono & (df.pref_n > LOW_HI)
    mono = ~nonmono                                        # peak outside range

    print('\nPaired amplitude tests (one-sided, predict IPS < Vertex):')
    paired_test(df, 'Full ROI (all signal voxels)')
    paired_test(df[mono], 'Monotonic (peak outside range)')
    paired_test(df[nonmono], 'Non-monotonic (peak in range)')
    g_low = paired_test(df[low], f'  -> low pref n (<= {LOW_HI})')
    paired_test(df[high], f'  -> high pref n (> {LOW_HI})')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                   constrained_layout=True,
                                   gridspec_kw={'width_ratios': [1.5, 1]})

    # Panel A: amplitude shift vs preferred numerosity (per-subject binned means)
    edges = np.array([3, 7, 11, 17, 27, 45, 75, 130])
    centers = np.sqrt(edges[:-1] * edges[1:])
    df = df.assign(nbin=pd.cut(df.pref_n, edges, labels=centers))
    per_sub = (df.dropna(subset=['nbin'])
                 .groupby(['subject', 'nbin'], observed=True)['amp_diff'].mean()
                 .reset_index())
    grp = per_sub.groupby('nbin', observed=True)['amp_diff'].agg(['mean', 'sem', 'count'])
    axA.axhline(0, color='0.6', lw=0.8, zorder=1)
    axA.axvspan(N_LO, LOW_HI, color='0.9', zorder=0)
    axA.errorbar(grp.index.astype(float), grp['mean'], yerr=grp['sem'],
                 marker='o', color='#3B5BA5', capsize=3, lw=1.5, zorder=3)
    axA.set_xscale('log')
    axA.set_xlabel('Preferred numerosity (peak of nPRF)')
    axA.set_ylabel('Amplitude shift  IPS − Vertex')
    axA.set_title('Amplitude attenuation vs preferred numerosity\n'
                  '(shaded = low-pref band; negative = attenuated by parietal cTBS)',
                  fontsize=9)
    axA.set_xticks([5, 10, 20, 40, 80])
    axA.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    sns.despine(ax=axA, offset=3)

    # Panel B: paired amplitude (IPS vs Vertex) in the low-pref non-monotonic subset
    if g_low is not None:
        for _, r in g_low.iterrows():
            axB.plot([0, 1], [r['amp_ips'], r['amp_vertex']],
                     color='0.6', lw=0.5, alpha=0.5, zorder=2)
        for x, key in [(0, 'amp_ips'), (1, 'amp_vertex')]:
            c = COLOR['ips' if key == 'amp_ips' else 'vertex']
            axB.scatter([x] * len(g_low), g_low[key], s=16, color=c,
                        alpha=0.6, edgecolor='none', zorder=3)
            axB.errorbar([x], [g_low[key].mean()], yerr=[g_low[key].sem()],
                         marker='D', markersize=9, mew=1.5, color='black',
                         mfc=c, capsize=4, lw=1.5, zorder=5)
        t, p2 = stats.ttest_rel(g_low['amp_ips'], g_low['amp_vertex'])
        p1 = p2 / 2 if t < 0 else 1 - p2 / 2
        axB.set_xticks([0, 1]); axB.set_xticklabels(['IPS', 'Vertex'])
        axB.set_xlim(-0.5, 1.5)
        axB.set_ylabel('nPRF amplitude (PSC)')
        axB.set_title(f'Low-pref non-monotonic voxels (n≤{LOW_HI})\n'
                      f'n={len(g_low)}, t={t:+.2f}, p(1-sided)={p1:.3f}',
                      fontsize=9)
        sns.despine(ax=axB, offset=3, trim=True)

    fig.suptitle('cTBS amplitude effect localised to low-preferred-numerosity tuned voxels '
                 f'({ROI}, m{model_label})', fontsize=11, y=1.05)
    out = Path(f'notes/figures/m{model_label}_amplitude_by_pref_n')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f'wrote {out}.{{pdf,png}}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_label', type=int, default=1)
    p.add_argument('--from_tsv', action='store_true')
    args = p.parse_args()
    main(model_label=args.model_label, from_tsv=args.from_tsv)
