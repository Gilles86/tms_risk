"""Supplementary: which model, on two criteria that disagree.

Kept out of Figure 4 deliberately. Figure 4 answers "what did cTBS do"; this
answers "which model", and interleaving the two made both harder to read.

a  Expected log predictive density relative to the primary model, with the
   standard error of each PAIRED difference. Note what this panel cannot do:
   the prior-width-only model is a few nats behind, well inside its own error,
   so ELPD does not establish that the noise term is needed.
b  The check that does. The quantity is the cTBS effect on the psychometric
   slope for risky-second, low-stake trials -- published Figure 3B's headline.
   For each model, choices were simulated from its posterior and refitted with
   the same maximum-likelihood probit applied to the real choices, so no
   linearisation or change of estimator enters the comparison.

Same models in the same order in both panels, so the disagreement is legible.

    python -m tms_risk.behavior.scripts.plot_model_comparison_supp \
        --model_label log-power-n2psd
"""
import argparse
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
BAD = '#C44E52'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def main(data_dir, out_stem, label):
    dd = Path(data_dir)
    loo = pd.concat([pd.read_csv(f, **READ)
                     for f in glob.glob(str(dd / 'loo_anchor/loo.*.tsv'))],
                    ignore_index=True).set_index('label')
    # An ELPD computed on a posterior that never mixed is not a number anyone
    # should rank. 68 of 227 anchor traces fail the gate (r_hat <= 1.01,
    # ESS >= 400) and 41 of those carry a LOO, so the unfiltered ladder ranks
    # fits whose posteriors are not characterised. Mark them here rather than
    # dropping them silently: a reader who knows the model should see why it
    # is absent from the comparison.
    _cf = dd / 'all_anchor_check.tsv'
    conv = pd.read_csv(_cf, **READ).set_index('trace')['ok'].to_dict() if _cf.exists() else {}
    form, place = label.split('-')[1], label.split('-')[2]
    null = 'null' if place in ('null', 'perc', 'mem', 'percmem') else 'nullind'
    picks = [(label, 'Noise (2nd option) + prior width'),
             (f'log-{form}-n1n2{place[2:]}', '+ noise on 1st option'),
             (f'log-weber-{place}', 'Weber noise'),
             ('log-weber-psd', 'Weber noise'),
             ('log-power-psd', 'Prior width only'),
             (f'log-{form}-{place[:2]}', 'Noise only'),
             (f'log-{form}-{null}', 'No cTBS effect')]
    have, seen, names = set(loo.index), set(), set()
    picks = [(p_, n) for p_, n in picks
             if p_ in have and p_ not in seen and n not in names
             and not seen.add(p_) and not names.add(n)]

    def dse(other):
        fa, fb = (dd / f'loo_anchor/looi.{label}.npy',
                  dd / f'loo_anchor/looi.{other}.npy')
        if not (fa.exists() and fb.exists()):
            return np.nan
        a, b = np.load(fa), np.load(fb)
        if a.shape != b.shape:
            return np.nan
        d_ = b - a
        return float(np.sqrt(len(d_)) * d_.std(ddof=1))

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.5), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1]))
    y = np.arange(len(picks))[::-1]
    base = loo.elpd_loo.get(label, np.nan)

    ax = axes[0]
    for yy, (lab, nm) in zip(y, picks):
        d = loo.elpd_loo.get(lab, np.nan) - base
        e = 0.0 if lab == label else dse(lab)
        within = np.isfinite(e) and e > 0 and abs(d) < e
        col = '0.45' if (lab == label or within) else BAD
        ax.barh(yy, d, height=.5, color=col, alpha=.85, lw=0, zorder=2)
        if np.isfinite(e) and e > 0:
            ax.plot([d - e, d + e], [yy, yy], color='0.25', lw=1.0, zorder=3)
        bad_conv = conv.get(lab, True) in (False, 'False')
        if bad_conv:
            col = '0.72'
        txt = 'reference' if lab == label else f'{d:+.0f}'
        xend = min(d, 0) if not (np.isfinite(e) and e > 0) else min(d - e, 0)
        ax.text(xend - 4, yy, txt, ha='right', va='center', fontsize=6.5,
                color=col, fontweight='bold')
        note = ('  (did not converge)' if bad_conv
                else '  (within noise)' if within else '')
        ax.text(4, yy, nm + note, fontsize=6.3, va='center',
                color='0.6' if bad_conv else '0.25')
    ax.axvline(0, color='0.4', lw=.8)
    ax.set_yticks([])
    ax.set_ylim(-.95, len(picks) - .4)
    ax.set_xlabel('ELPD relative to the primary model (nats)')
    ax.set_title('Model comparison', fontsize=8)
    ax.set_xlim(min(-140, ax.get_xlim()[0]), abs(ax.get_xlim()[0]) * .62)
    # glyph, not the word for the glyph
    ax.plot([.80, .87], [.055, .055], transform=ax.transAxes, color='0.25',
            lw=1.0, clip_on=False)
    ax.text(.885, .055, '±1 SE of the paired difference', fontsize=5.8,
            color='0.5', transform=ax.transAxes, va='center')

    ax = axes[1]
    obs_v = None
    for yy, (lab, nm) in zip(y, picks):
        f = dd / f'probit_ppc_ml/probit_ppc_ml.{lab}.tsv'
        if not f.exists():
            ax.text(.02, yy, 'not run', fontsize=5.8, color='0.65',
                    va='center', transform=ax.get_yaxis_transform())
            continue
        q = pd.read_csv(f, **READ)
        q = q[(q.parameter == 'slope') & (q.order == 'Risky second')
              & (q.stake == 'low')]
        if not len(q):
            continue
        q = q.iloc[0]
        obs_v = q.observed
        fail = q.ppp > .95 or q.ppp < .05
        col = BAD if fail else '0.35'
        ax.plot([q.lo, q.hi], [yy, yy], color=col, lw=5, alpha=.30,
                solid_capstyle='butt')
        ax.plot([q.model] * 2, [yy - .22, yy + .22], color=col, lw=1.4)
        ax.text(q.hi + .02, yy, f'p = {q.ppp:.2f}' + ('  fails' if fail else ''),
                fontsize=6, va='center', color=col)
    if obs_v is not None:
        ax.axvline(obs_v, color='0.12', lw=1.2)
        ax.text(obs_v - .02, len(picks) - .55, 'Observed', fontsize=6.3,
                color='0.12', ha='right', va='bottom')
    ax.set_yticks([])
    ax.set_ylim(-.95, len(picks) - .4)
    ax.set_xlabel('cTBS effect on the psychometric slope')
    ax.set_title('Predictive check (risky second, low stake)', fontsize=8)
    # glyph, not the word for the glyph
    ax.add_patch(plt.Rectangle((.62, .04), .07, .028, transform=ax.transAxes,
                               facecolor='0.6', alpha=.55, lw=0, clip_on=False))
    ax.text(.705, .054, '95% predictive interval', fontsize=5.8, color='0.5',
            transform=ax.transAxes, va='center')

    for ax, s_ in zip(axes, 'ab'):
        ax.text(-.04, 1.06, s_, transform=ax.transAxes, fontsize=8.5,
                fontweight='bold', family='Arial', va='bottom')
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n2psd')
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/supp_model_comparison'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label)
