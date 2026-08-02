"""Why the cTBS effect is specific to trials where the risky option came SECOND.

The model has no order parameter. The asymmetry falls out of three facts, one per
panel, and the last panel is the behaviour they predict:

  a  The first-presented option's noise is built from BOTH noise components, the
     second's from the shared perceptual component alone. So the same cTBS
     perturbation raises the first option's noise ~1.4x more.  [pmcpars_curves]
  b  cTBS devalues percepts by pushing them toward the low prior. How much an option
     loses therefore depends on how noisy it already is -- so the safe option loses
     substantially more when it comes first, while the risky option, already noisy,
     barely notices its position.                        [pmc_percepts_by_order]
  c  Choices track the GAP between the two options' devaluations. That gap is ~3x
     larger when the safe option comes first.            [pmc_percepts_by_order]
  d  Which is exactly when the behavioural effect appears. [pmc_channels_by_ratio]

    python -m tms_risk.behavior.scripts.plot_why_risky_second --label flexible2nf
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
SAFE, RISKY = '#4d4d4d', '#b2182b'
FIRST, SECOND = '#3B5BA5', '#7b3294'    # blue = presented first, purple = second

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


def panel_a(ax, data, label):
    """cTBS lands harder on the first-presented option.

    The first option's noise is built from BOTH noise components, the second's from
    the shared perceptual component alone, so the same cTBS perturbation produces a
    larger increase in position 1. Plotting the increase itself (rather than the two
    noise functions, which nearly coincide) is what makes that visible.
    """
    c = pd.read_csv(data / f'pmcpars_curves.{label}.tsv', sep='\t')
    got = {}
    for term, pos, colr in [('n1_evidence_sd', 'first', FIRST),
                            ('n2_evidence_sd', 'second', SECOND)]:
        i = c[(c.term == term) & (c.stimulation == 'ips')].set_index('payoff').nu
        v = c[(c.term == term) & (c.stimulation == 'vertex')].set_index('payoff').nu
        if not len(i):
            raise FileNotFoundError(f'pmcpars_curves.{label}.tsv has no {term} curve')
        d = (i - v)
        got[pos] = d
        ax.plot(d.index.values, d.values, color=colr)
    ax.set_xscale('log'); ax.set_xticks([7, 14, 28, 56, 112])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Δ representational noise\nIPS − vertex (CHF)')
    ax.set_ylim(0, max(got['first'].max(), got['second'].max()) * 1.35)
    ax.axvspan(7, 28, color=SAFE, alpha=.07, lw=0, zorder=0)
    ax.text(14, ax.get_ylim()[1] * .04, 'Safe payoffs', color=SAFE, fontsize=6.8,
            ha='center')
    ax.text(8, float(got['first'].iloc[0]) * 1.06, 'Presented first', color=FIRST,
            fontsize=7.5, ha='left', va='bottom')
    ax.text(8, float(got['second'].iloc[0]) * .94, 'Presented second', color=SECOND,
            fontsize=7.5, ha='left', va='top')
    k = (got['first'].index >= 7) & (got['first'].index <= 28)
    ratio = got['first'].values[k].mean() / got['second'].values[k].mean()
    ax.annotate(f'Over the safe range, cTBS\nadds {ratio:.1f}× more noise to\n'
                'whichever option came first',
                xy=(18, float(got['first'].values[k].mean())),
                xytext=(30, ax.get_ylim()[1] * .82), fontsize=7, color='.3',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-.25',
                                color='.45', lw=.6))
    return ratio


def panel_b(ax, data, label):
    """Devaluation by cTBS, per option, split by whether it was presented first."""
    d = pd.read_csv(data / f'pmc_percepts_by_order.{label}.tsv', sep='\t')
    g = d.groupby(['option', 'position']).delta.mean().unstack()
    xs = np.arange(2)
    w = .34
    ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
    for k, (pos, colr) in enumerate([('first', FIRST), ('second', SECOND)]):
        vals = [g.loc['safe', pos], g.loc['risky', pos]]
        ax.bar(xs + (k - .5) * w, vals, width=w, color=colr, alpha=.85,
               edgecolor='none', zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(['Safe option', 'Risky option'])
    ax.set_ylabel('Δ perceived value\nIPS − vertex (CHF)')
    ax.invert_yaxis()
    ax.text(-.5 * w, g.loc['safe', 'first'] * 1.06, 'First', color=FIRST,
            fontsize=7, ha='center', va='top')
    ax.text(.5 * w, g.loc['safe', 'second'] * 1.06, 'Second', color=SECOND,
            fontsize=7, ha='center', va='top')
    ax.annotate('Position matters\nfor the safe option',
                xy=(0, g.loc['safe'].mean()), xytext=(.55, g.loc['safe', 'first'] * .62),
                fontsize=7, color='.3', ha='left', va='center',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=.25',
                                color='.45', lw=.6))
    ax.annotate('but not for the risky one',
                xy=(1, g.loc['risky'].mean()), xytext=(.55, g.loc['risky'].mean() * .42),
                fontsize=7, color='.3', ha='left', va='center',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-.2',
                                color='.45', lw=.6))
    return g


def panel_c(ax, g):
    """The gap that choices actually track."""
    gaps = {'Risky first': g.loc['safe', 'second'] - g.loc['risky', 'first'],
            'Risky second': g.loc['safe', 'first'] - g.loc['risky', 'second']}
    xs = np.arange(2)
    ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
    ax.bar(xs, [gaps['Risky first'], gaps['Risky second']], width=.55,
           color=['.68', IPS], edgecolor='none', zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(['Risky\nfirst', 'Risky\nsecond'])
    ax.set_ylabel('Extra value lost by the safe\noption, vs risky (CHF)')
    ax.invert_yaxis()
    for x, k in zip(xs, ['Risky first', 'Risky second']):
        ax.text(x, gaps[k] * 1.04, f'{gaps[k]:.3f}', ha='center', va='top',
                fontsize=7.5, color='.2')
    ratio = gaps['Risky second'] / gaps['Risky first']
    ax.annotate(f'{ratio:.1f}× larger', xy=(1, gaps['Risky second'] * .55),
                xytext=(.18, gaps['Risky second'] * .34), fontsize=7.5, color='.25',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=.25',
                                color='.45', lw=.6))
    return gaps, ratio


def panel_d(ax, data, label):
    """The behaviour those gaps predict."""
    ch = pd.read_csv(data / f'pmc_channels_by_ratio.{label}.tsv', sep='\t')
    ch = ch[ch.channel == 'full']
    order_bins = sorted(ch.bin.unique(), key=lambda b: int(b.rstrip('%')))
    xs = np.arange(len(order_bins))
    ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
    peaks = {}
    for name, colr in [('Risky first', '.55'), ('Risky second', IPS)]:
        s = ch[ch.order == name].set_index('bin').reindex(order_bins).reset_index()
        ax.fill_between(xs, s.lo, s.hi, color=colr, alpha=.18, lw=0, zorder=1)
        ax.plot(xs, s.delta, color=colr, zorder=2)
        peaks[name] = float(s.delta.iloc[0])
        ax.text(xs[-1] + .12, float(s.delta.iloc[-1]), name.replace(' ', '\n'),
                color=colr, fontsize=7, va='center', ha='left', linespacing=.95)
    ax.set_xticks(xs[::2])
    ax.set_xticklabels([order_bins[i] for i in range(0, len(order_bins), 2)])
    ax.set_xlabel('Risky/safe ratio (bin)')
    ax.set_ylabel('Δ P(chose risky)\nIPS − vertex')
    ax.set_xlim(-.4, len(xs) + 1.4)
    return peaks


def main(data_dir, label, out_stem):
    data = Path(data_dir)
    fig = plt.figure(figsize=(7.25, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=.62, wspace=.42,
                          left=.10, right=.95, top=.88, bottom=.11)
    a = fig.add_subplot(gs[0, 0]); nratio = panel_a(a, data, label)
    b = fig.add_subplot(gs[0, 1]); g = panel_b(b, data, label)
    c = fig.add_subplot(gs[1, 0]); gaps, ratio = panel_c(c, g)
    d = fig.add_subplot(gs[1, 1]); peaks = panel_d(d, data, label)

    for ax, letter in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        ax.text(-.20, 1.08, letter, transform=ax.transAxes, **PANEL)
    fig.suptitle('The first-presented option is the noisiest, so cTBS costs it most — '
                 'and that matters only when it is the safe one',
                 fontsize=9, y=.965, color='.15')
    sns.despine(fig=fig, offset=3)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf')
    print(f'  cTBS noise increase, first vs second position: {nratio:.2f}x')
    print(f'  Δ perceived value (CHF):\n{g.round(3).to_string()}')
    print(f'  safe-minus-risky gap: ' + ', '.join(f'{k} {v:.3f}' for k, v in gaps.items())
          + f'   ratio {ratio:.2f}x')
    print('  ΔP(risky) at the lowest ratio bin: '
          + ', '.join(f'{k} {v:+.3f}' for k, v in peaks.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    main(args.data_dir, args.label,
         args.out or f'/Users/gdehol/git/tms_risk/notes/figures/why_risky_second.{args.label}')
