"""RETIRED 2026-08-03 -- do not use for the paper. See plot_fig3_probit.py.

This was one of two candidates for Figure 3. The probit won, on the grounds that a
probit already assumes nothing beyond "there is a psychophysical curve", so it *is*
the model-free argument, and this panel is a second and much harder-to-read route to
the same conclusion. Kept only because the signature argument below is still the
cleanest refutation of the pure-flattening account if it is ever needed in text.

---

Figure 3: the cTBS effect, argued without the cognitive model.

The preprint explained the effect as a flattening of the psychometric function. That
explanation makes a hard prediction, and it can be checked with no model at all:
flattening pulls every choice proportion toward 0.5, so wherever the baseline
proportion is ABOVE 0.5 it must push the proportion DOWN. Observed Delta P is positive
in those bins. Whatever cTBS did, it was not a pure loss of consistency.

    a, b   Delta P(chose risky) against the payoff ratio, per presentation order,
           with the two pure signatures overlaid: a consistency-only change (pure
           flattening) and a preference-only change (pure shift). Bins whose baseline
           sits above 0.5 are the diagnostic ones and are marked.
    c      Delta P against the risky payoff, both orders. The effect is carried by
           small payoffs, which is where the stimulated populations are tuned.

    python -m tms_risk.behavior.scripts.plot_fig3_modelfree

Reads notes/data/localnoise_{signatures,delta_by_ratio,delta_by_nrisky}.tsv. No model.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
# Colour is semantic across the paper: green = vertex, red = IPS. A DIFFERENCE
# between them is a third quantity, not one of the conditions, so it gets its own
# near-black ink rather than borrowing the IPS red.
DIFF = '#1a1a1a'
FLAT, SHIFT = '#3B5BA5', '#b8860b'      # consistency-only, preference-only
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')
PANEL = dict(fontsize=11, fontweight='bold', va='bottom', ha='right')


def main(data_dir, out_stem):
    data = Path(data_dir)
    rat = pd.read_csv(data / 'localnoise_delta_by_ratio.tsv', sep='\t')
    sig = pd.read_csv(data / 'localnoise_signatures.tsv', sep='\t')
    nrk = pd.read_csv(data / 'localnoise_delta_by_nrisky.tsv', sep='\t')
    bins = sorted(rat.bin.unique(), key=lambda b: int(b.rstrip('%')))
    xs = np.arange(len(bins))

    fig = plt.figure(figsize=(7.25, 2.6))
    gs = fig.add_gridspec(1, 3, wspace=.46, left=.075, right=.985, top=.82, bottom=.20)

    axes = []
    for col, order in enumerate(ORDERS):
        ax = fig.add_subplot(gs[0, col]); axes.append(ax)
        r = rat[rat.order == order].set_index('bin').reindex(bins).reset_index()
        ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)

        # the two pure signatures, rescaled onto the binned x-axis
        s = sig[sig.order == order]
        for curve, colr, lab in [('Consistency only', FLAT, 'Pure flattening'),
                                 ('Preference only', SHIFT, 'Pure shift')]:
            c = s[s.curve == curve].sort_values('x')
            if not len(c):
                continue
            xi = np.interp(r.vertex.values, c.p.values + c.delta.values * 0, c.x.values) \
                if False else None
            # map the signature onto the observed baseline proportion of each bin
            ax.plot(xs, np.interp(r.vertex.values, c.p.values, c.delta.values),
                    color=colr, lw=1.4, zorder=2, label=lab)

        # bins whose baseline is above 0.5 -- where flattening MUST be negative
        above = r.vertex.values > .5
        ax.errorbar(xs, r.delta, yerr=r['sem'], fmt='o', color=DIFF, ms=4.4, lw=0,
                    elinewidth=1.1, capsize=0, zorder=4)
        if above.any():
            ax.scatter(xs[above], r.delta.values[above], s=95, facecolor='none',
                       edgecolor='.25', lw=.8, zorder=5)
        ax.set_xticks(xs); ax.set_xticklabels(bins, fontsize=7)
        ax.set_xlabel('Risky/safe payoff ratio (percentile bin)')
        ax.set_title(order, fontsize=8.5, color='.2', pad=4)
        if col == 0:
            ax.set_ylabel('Δ P(chose risky)\nIPS − vertex')

    lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    axes[1].set_yticklabels([])
    axes[0].text(.04, .95, 'Pure flattening', transform=axes[0].transAxes,
                 fontsize=6.8, color=FLAT, va='top')
    axes[0].text(.04, .84, 'Pure shift', transform=axes[0].transAxes,
                 fontsize=6.8, color=SHIFT, va='top')
    axes[0].text(.04, .73, 'Observed', transform=axes[0].transAxes,
                 fontsize=6.8, color=DIFF, va='top')
    axes[1].text(.97, .04, 'Ringed: baseline > 0.5,\nwhere flattening must be negative',
                 transform=axes[1].transAxes, fontsize=6.3, color='.3', va='bottom',
                 ha='right', linespacing=1.25)

    # --- c: the effect against the risky payoff
    ax = fig.add_subplot(gs[0, 2])
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    order_bins = list(dict.fromkeys(nrk.n_risky_bin))
    xb = np.arange(len(order_bins))
    for o, colr, mk in [('Risky second', DIFF, 'o'), ('Risky first', '.62', 's')]:
        g = nrk[nrk.order == o].set_index('n_risky_bin').reindex(order_bins).reset_index()
        ax.errorbar(xb + (.07 if o == 'Risky second' else -.07), g.delta, yerr=g['sem'],
                    fmt=mk, color=colr, ms=4.2, lw=0, elinewidth=1.1, capsize=0,
                    zorder=3 if o == 'Risky second' else 2)
    ax.set_xticks(xb); ax.set_xticklabels(order_bins, fontsize=7)
    ax.set_xlabel('Risky payoff (CHF)')
    ax.set_ylabel('Δ P(chose risky)\nIPS − vertex')
    ax.set_title('By payoff size', fontsize=8.5, color='.2', pad=4)
    ax.text(.97, .95, 'Risky second', transform=ax.transAxes, fontsize=6.8,
            color=DIFF, va='top', ha='right')
    ax.text(.97, .84, 'Risky first', transform=ax.transAxes, fontsize=6.8,
            color='.62', va='top', ha='right')

    for a, letter in zip([axes[0], axes[1], ax], 'abc'):
        a.text(-.20, 1.06, letter, transform=a.transAxes, **PANEL)
    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf')
    print('\nbins whose baseline P(chose risky) exceeds 0.5 '
          '-- a pure flattening must give a NEGATIVE effect there:')
    for order in ORDERS:
        r = rat[rat.order == order]
        a = r[r.vertex > .5]
        print(f'  {order:14s} {len(a)}/{len(r)} such bins; observed Δ = '
              + ', '.join(f'{v:+.3f}' for v in a.delta) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/fig3_modelfree')
    a = parser.parse_args()
    main(a.data_dir, a.out)
