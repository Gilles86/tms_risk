"""Figure 3: what cTBS did to the psychometric function, from the probit fits alone.

Laid out like the preprint's Figure 3: the psychometric functions stacked by
presentation order on the left, the parameter shifts on the right.

No cognitive model is involved. The probit has exactly two parameters, and they
correspond to the two competing explanations of the effect:

    slope  choice consistency. A pure loss of consistency -- the "flattening" account
           -- moves this and nothing else.
    RNP    the risk-neutral probability, where the psychometric crosses 0.5. A pure
           change in preference moves this and nothing else.

Both move, and both only when the risky option came second, so neither pure account
covers it. That is the observation the cognitive model has to explain.

The probit is fitted on log(risky/safe), which is why the x-axis is logarithmic --
and why a constant absolute noise increase reads as magnitude-specific on this scale.

    python -m tms_risk.behavior.scripts.plot_fig3_probit

Reads notes/data/localnoise_{signatures,group_posterior}.tsv for the model and
ppc_fig3a.<label>.tsv for the observed proportions and their between-subject SEMs.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as ss

VERTEX, IPS = '#2ca02c', '#d62728'
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
XT = [1.5, 2, 2.5, 3]


# NOTE on what the model line is.
# Evaluating Phi at the MEAN posterior parameters gives a curve that is systematically
# too steep: the observed statistic is a mean over subjects of sigmoids with different
# indifference points and slopes, and by Jensen that average is flatter than the
# sigmoid at the average parameters. Measured against these data, the group-parameter
# curve has RMSE 0.069 while the properly aggregated prediction has RMSE 0.019.
# So the model line here is `probit_<stim>` from analyze_localized_noise, which is the
# prediction of the binned statistic, with its own credible interval.


def paired_delta(g, parameter, order):
    """IPS - vertex, paired within (chain, draw) so the difference keeps its posterior."""
    s = g[(g.parameter == parameter) & (g.order == order)]
    w = s.pivot_table(index=['chain', 'draw'], columns='stimulation_condition',
                      values='value')
    return (w['ips'] - w['vertex']).values


def pfmt(d):
    """One-sided posterior tail probability, in the paper's pBayesian convention."""
    p = min(float((d > 0).mean()), float((d < 0).mean()))
    return 'p < 0.001' if p < .001 else f'p = {p:.3f}'


def main(data_dir, label, out_stem):
    data = Path(data_dir)
    sig = pd.read_csv(data / 'localnoise_signatures.tsv', sep='\t')
    post = pd.read_csv(data / 'localnoise_group_posterior.tsv', sep='\t')
    rat = pd.read_csv(data / 'localnoise_delta_by_ratio.tsv', sep='\t')
    obs = pd.read_csv(data / f'ppc_fig3a.{label}.tsv', sep='\t')

    # Presentation order is the ROW variable in EVERY column: "Risky first" is always
    # the top row, "Risky second" always the bottom. So the cTBS effect reads as a
    # vertical displacement in the right-hand columns, in the same direction that
    # separates the two psychometric functions on the left. Putting order on the
    # x-axis of the right panels (as in the earlier draft) encoded the same variable
    # two different ways and the panels did not line up.
    fig = plt.figure(figsize=(7.25, 4.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=.28, wspace=.62,
                          left=.085, right=.98, top=.93, bottom=.11)

    # --- a, b: psychometric functions, observed proportions over the probit fit
    lefts = []
    for row, order in enumerate(ORDERS):
        ax = fig.add_subplot(gs[row, 0]); lefts.append(ax)
        ax.axhline(.5, color='.8', lw=.7, ls='--', zorder=0)
        o = obs[obs.order == order]
        r = rat[rat.order == order].set_index('bin')
        for stim, colr, mk in [('vertex', VERTEX, 'o'), ('ips', IPS, 's')]:
            g = o[o.stim == stim].sort_values('frac')
            fit = r.loc[g.bin, f'probit_{stim}'].values
            flo = r.loc[g.bin, f'probit_{stim}_lo'].values
            fhi = r.loc[g.bin, f'probit_{stim}_hi'].values
            ax.fill_between(g.frac, flo, fhi, color=colr, alpha=.22, lw=0, zorder=1)
            ax.plot(g.frac, fit, color=colr, lw=1.4, zorder=2)
            ax.plot(g.frac, g.observed, mk, color=colr, ms=4.4, lw=0,
                    mfc='white' if stim == 'ips' else colr, mew=1.1, zorder=4)
        ax.set_xscale('log')
        ax.set_xticks(XT)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlim(1.42, 3.45)
        ax.set_ylim(.12, .95); ax.set_yticks([.25, .5, .75])
        ax.set_ylabel('P(chose risky)')
        ax.text(.035, .95, order, transform=ax.transAxes, fontsize=8, color='.2',
                va='top')
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Risky/safe payoff ratio')
    lefts[0].plot([], [], color=VERTEX, marker='o', ms=4.4, lw=1.4, label='Vertex')
    lefts[0].plot([], [], color=IPS, marker='s', ms=4.4, mfc='white', mew=1.1,
                  lw=1.4, label='IPS')
    lefts[0].legend(loc='lower right', fontsize=7, handlelength=1.6, borderpad=.3,
                    labelspacing=.25)

    # --- c, d: the cTBS change in each probit parameter
    stats, rights = {}, []
    specs = [('rnp', 'Δ risk-neutral probability\nIPS − vertex'),
             ('slope', 'Δ psychometric slope\nIPS − vertex')]
    for row, (par, ylab) in enumerate(specs):
        ax = fig.add_subplot(gs[row, 1]); rights.append(ax)
        ax.axhline(0, color='.8', lw=.7, ls='--', zorder=0)
        for i, order in enumerate(ORDERS):
            d = paired_delta(post, par, order)
            lo, hi = np.quantile(d, [.025, .975])
            stats[(par, order)] = (d.mean(), lo, hi, pfmt(d))
            colr = IPS if order == 'Risky second' else '.55'
            parts = ax.violinplot([d], positions=[i], widths=.7, showextrema=False)
            for b in parts['bodies']:
                b.set_facecolor(colr); b.set_alpha(.22); b.set_edgecolor('none')
            ax.plot([i, i], [lo, hi], color=colr, lw=1.7, solid_capstyle='round',
                    zorder=3)
            ax.plot([i], [d.mean()], 'o', color=colr, ms=5.5, mec='white', mew=.8,
                    zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Risky\nfirst', 'Risky\nsecond'] if row else ['', ''])
        ax.set_xlim(-.62, 1.62)
        ax.set_ylabel(ylab)
        # p-values under each distribution, in the paper's pBayesian convention
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.set_ylim(ax.get_ylim()[0] - .16 * span, ax.get_ylim()[1])
        ylo = ax.get_ylim()[0]
        for i, order in enumerate(ORDERS):
            _, _, _, p = stats[(par, order)]
            bold = 'bold' if p.startswith('p <') or float(p.split('=')[-1]) < .05 else 'normal'
            ax.text(i, ylo + .03 * span, p, ha='center', va='bottom', fontsize=6.8,
                    color='.15' if bold == 'bold' else '.45', fontweight=bold)

    for a, letter in zip([lefts[0], lefts[1], rights[0], rights[1]], 'abcd'):
        a.text(-.15, 1.03, letter, transform=a.transAxes, **PANEL)
    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf\n')
    print(f'{"parameter":<8}{"order":<15}{"Δ (IPS−vertex)":>16}{"95% CrI":>24}{"":>4}')
    for (par, order), (m, lo, hi, p) in stats.items():
        print(f'{par:<8}{order:<15}{m:>+16.4f}   [{lo:+.4f}, {hi:+.4f}]   {p}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/fig3_probit')
    a = parser.parse_args()
    main(a.data_dir, a.label, a.out)
