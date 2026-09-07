"""Why the cTBS effect appears only when the risky option is presented second.

No parameter in the Flexible PMC codes for presentation order, yet the behavioural
effect is order-specific. It emerges, and this figure walks the four steps. Every
panel is drawn from a tracked TSV -- no trace, no bauer, no GPU.

  a  How noisy are perceived payoffs? On log-log axes the perceptual noise function
     has slope ~0.5, i.e. nu ~ sqrt(n), NOT the slope of 1 that Weber's law (scalar
     invariance) predicts. Memory noise is flat. cTBS lifts the perceptual curve.
                                                   [noisecurve_reparam]
  b  A Bayesian observer with more likelihood noise shrinks harder toward its prior.
     The fitted prior sits near 5 CHF, below every payoff, so percepts are pulled
     DOWN -- and after cTBS the safe option is pulled down more than the risky one.
                                                   [pmc_percepts]
  c  That asymmetric pull, not the extra randomness, is what moves choices: the
     bias-only channel reproduces the full effect, the noise-only channel is flat.
     And it is roughly 5x larger when the risky option comes second.
                                                   [pmc_channels_by_ratio]
  d  The behavioural signature the model has to reproduce.        [ppc_fig3a]

    python -m tms_risk.behavior.scripts.plot_order_asymmetry --label flexible1nf
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'      # sham green, stimulated red (see CLAUDE.md)
SAFE, RISKY = '#4d4d4d', '#b2182b'
# Presentation order needs its own hues: green and red already mean vertex and IPS
# everywhere in this paper, so reusing them here would read as a stimulation contrast.
FIRST, SECOND = '#3B5BA5', '#7b3294'    # blue = risky first, purple = risky second

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
    'lines.linewidth': 1.3, 'lines.markersize': 4,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')

PANEL = dict(fontsize=11, fontweight='bold', va='bottom', ha='right')
LOGTICKS = [7, 14, 28, 56, 112]


def panel_a(ax, data, label):
    """Noise vs magnitude on log-log axes: sub-Weber scaling, lifted by cTBS."""
    c = pd.read_csv(data / f'noisecurve_reparam.{label}.tsv', sep='\t')
    slopes = {}
    for term, ls in [('perceptual', '-'), ('memory', '--')]:
        for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
            s = c[(c.term == term) & (c.stimulation == stim)].sort_values('payoff')
            if not len(s):
                continue
            if term == 'perceptual':
                ax.fill_between(s.payoff, s.lo, s.hi, color=colr, alpha=.16, lw=0,
                                zorder=1)
            ax.plot(s.payoff, s.nu, color=colr, ls=ls, lw=1.3 if term == 'perceptual'
                    else 1.0, zorder=2)
            if stim == 'vertex':
                x, y = np.log(s.payoff.values), np.log(s.nu.values)
                slopes[term] = np.polyfit(x, y, 1)[0]

    # Weber reference: slope 1 through the perceptual curve at its low end.
    p = c[(c.term == 'perceptual') & (c.stimulation == 'vertex')].sort_values('payoff')
    x0, y0 = p.payoff.iloc[0], p.nu.iloc[0]
    xs = np.array([x0, p.payoff.iloc[-1]])
    ax.plot(xs, y0 * xs / x0, color='.6', lw=.7, ls=':', zorder=0)
    ax.text(xs[-1] * .92, y0 * xs[-1] / x0 * .93, "Weber\n(slope 1)", fontsize=6.5,
            color='.5', ha='right', va='top', linespacing=.95)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(LOGTICKS)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Representational noise ν (CHF)')
    ax.text(.03, .97, 'Perceptual', transform=ax.transAxes, fontsize=7.5,
            color='.25', va='top')
    ax.text(.03, .12, 'Memory (dashed)', transform=ax.transAxes, fontsize=7,
            color='.45', va='top')
    ax.annotate(f'Slope {slopes["perceptual"]:.2f}, not 1:\nnoise grows with the\nsquare root of payoff',
                xy=(28, float(p.nu.iloc[np.abs(p.payoff.values - 28).argmin()])),
                xytext=(8.4, 5.4), fontsize=7, color='.3', ha='left', va='center',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=.25',
                                color='.45', lw=.6))
    return slopes


def panel_b(ax, data, label):
    """Prior attraction: percepts pulled down, the safe option most."""
    p = pd.read_csv(data / f'pmc_percepts.{label}.tsv', sep='\t')
    ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
    for opt, colr in [('safe', SAFE), ('risky', RISKY)]:
        s = p[p.option == opt].sort_values('n_safe')
        ax.fill_between(s.n_safe, s.lo, s.hi, color=colr, alpha=.20, lw=0, zorder=1)
        ax.plot(s.n_safe, s.delta, color=colr, marker='o', ms=3.6, zorder=2)
    ax.set_xlabel('Safe payoff (CHF)')
    ax.set_ylabel('Δ perceived value\nIPS − vertex (CHF)')
    ax.set_xticks([7, 10, 14, 20, 28])
    s = p[p.option == 'safe'].sort_values('n_safe')
    r = p[p.option == 'risky'].sort_values('n_safe')
    ax.text(29, s.delta.iloc[-1], 'Safe', color=SAFE, fontsize=7.5, va='center')
    ax.text(29, r.delta.iloc[-1], 'Risky', color=RISKY, fontsize=7.5, va='center')
    ax.set_xlim(6, 34)
    ax.annotate('The safe option loses more,\nso risky looks relatively better',
                xy=(19.5, float(s.delta.iloc[3])), xytext=(6.9, s.delta.min() * .97),
                fontsize=7, color='.3', ha='left', va='center',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-.25',
                                color='.45', lw=.6))
    return float(s.delta.mean()), float(r.delta.mean())


def panel_c(ax_first, ax_second, data, label):
    """Which channel carries the effect -- and why it is order-specific."""
    ch = pd.read_csv(data / f'pmc_channels_by_ratio.{label}.tsv', sep='\t')
    show = [('full', '0.15', '-', 'Full model'),
            ('bias_only', '#1b7837', '-', 'Bias only'),
            ('noise_only', '#762a83', '-', 'Noise only')]
    order_bins = sorted(ch.bin.unique(), key=lambda b: int(b.rstrip('%')))
    xs = np.arange(len(order_bins))
    peaks = {}
    for ax, name in [(ax_first, 'Risky first'), (ax_second, 'Risky second')]:
        ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
        for chan, colr, ls, _ in show:
            s = ch[(ch.order == name) & (ch.channel == chan)]
            s = s.set_index('bin').reindex(order_bins).reset_index()
            if chan == 'full':
                ax.fill_between(xs, s.lo, s.hi, color=colr, alpha=.18, lw=0, zorder=1)
            ax.plot(xs, s.delta, color=colr, ls=ls, lw=1.3, zorder=2)
            if chan == 'bias_only':
                peaks[name] = float(s.delta.iloc[0])
        ax.set_xticks(xs[::2])
        ax.set_xticklabels([order_bins[i] for i in range(0, len(order_bins), 2)],
                           fontsize=7)
        ax.set_xlabel('Risky/safe ratio (bin)')
        ax.set_ylim(-.03, .115)
        ax.text(.04, .96, name, transform=ax.transAxes, fontsize=8, color='.2',
                va='top')
    ax_first.set_ylabel('Δ P(chose risky)\nIPS − vertex')
    ax_second.set_yticklabels([])
    for chan, colr, _, nm in show:
        s = ch[(ch.order == 'Risky second') & (ch.channel == chan)]
        s = s.set_index('bin').reindex(order_bins)
        ax_second.text(len(xs) - .85, float(s.delta.iloc[-1]), nm, color=colr,
                       fontsize=7, va='center', ha='left')
    ax_second.set_xlim(-.4, len(xs) + 1.6)
    ax_first.set_xlim(-.4, len(xs) + 1.6)
    return peaks


def panel_d(ax_first, ax_second, data, label):
    """The behavioural signature."""
    p = pd.read_csv(data / f'ppc_fig3a.{label}.tsv', sep='\t')
    for ax, name in [(ax_first, 'Risky first'), (ax_second, 'Risky second')]:
        for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
            m = p[(p.order == name) & (p.stim == stim)].sort_values('frac')
            ax.fill_between(m.frac, m.lo, m.hi, color=colr, alpha=.20, lw=0, zorder=1)
            ax.plot(m.frac, m['mean'], color=colr, lw=1.2, zorder=2)
            ax.errorbar(m.frac, m.observed, yerr=m.observed_sem, fmt='o', color=colr,
                        ms=3.6, lw=0, elinewidth=.9, capsize=0, zorder=4)
        ax.axhline(.5, color='.85', lw=.6, ls='--', zorder=0)
        ax.set_ylim(.12, .92)
        ax.set_yticks([.2, .4, .6, .8])
        ax.set_xticks([1.5, 2.0, 2.5, 3.0])
        ax.set_xlabel('Risky/safe payoff ratio')
        ax.text(.04, .96, name, transform=ax.transAxes, fontsize=8, color='.2',
                va='top')
    ax_first.set_ylabel('P(chose risky)')
    ax_second.set_yticklabels([])
    ax_first.text(3.25, .23, 'Vertex', color=VERTEX, fontsize=7.5, ha='right')
    ax_first.text(3.25, .155, 'IPS', color=IPS, fontsize=7.5, ha='right')


def main(data_dir, label, out_stem):
    data = Path(data_dir)
    fig = plt.figure(figsize=(7.25, 5.6))
    gs = fig.add_gridspec(2, 4, hspace=.55, wspace=.55,
                          left=.085, right=.955, top=.90, bottom=.09)

    a = fig.add_subplot(gs[0, :2])
    slopes = panel_a(a, data, label)
    b = fig.add_subplot(gs[0, 2:])
    dsafe, drisky = panel_b(b, data, label)
    c1, c2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    peaks = panel_c(c1, c2, data, label)
    d1, d2 = fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])
    panel_d(d1, d2, data, label)

    for ax, letter in [(a, 'a'), (b, 'b'), (c1, 'c'), (d1, 'd')]:
        ax.text(-.19, 1.09, letter, transform=ax.transAxes, **PANEL)

    fig.suptitle('cTBS adds noise, so the Bayesian observer shrinks percepts harder toward '
                 'its low prior, and the safe option loses most',
                 fontsize=9, y=.972, color='.15')
    sns.despine(fig=fig, offset=3)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf')
    print(f'  log-log slope: ' + ', '.join(f'{k} {v:.2f}' for k, v in slopes.items())
          + '   (Weber = 1.0)')
    print(f'  mean Δ perceived value: safe {dsafe:+.3f} CHF, risky {drisky:+.3f} CHF')
    print('  bias-only ΔP(risky) at the lowest ratio bin: '
          + ', '.join(f'{k} {v:+.3f}' for k, v in peaks.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='flexible1nf')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    main(args.data_dir, args.label,
         args.out or f'/Users/gdehol/git/tms_risk/notes/figures/order_asymmetry.{args.label}')
