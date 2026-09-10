"""Figure 3, split by stake: the same probit argument at low and high payoffs.

Same grammar as plot_fig3_probit.py -- block A the psychometric functions, block B
the cTBS effect on the two probit parameters -- but every row is now one
(presentation order x stake) cell rather than one presentation order. Stake is
(n_safe + n_risky) / 2 split at each participant's own median; the four probits are
fitted independently by analyze_probit_by_stake.py.

Why look: the cognitive-model account says the noise cTBS adds should bite hardest
where the payoffs are small, because the nPRF population at the stimulation site
prefers numerosities well below the presented range. That predicts a low-stake
effect and little at high stakes.

One deliberate difference from the main figure. There the four p-values sit far from
0.05, so shading the credible ones dark and the null ones grey is a safe guide. Here
the eight land at 0.006-0.48 with three of them within a whisker of 0.05, and
thresholded ink would turn a 0.047/0.059 pair into a black/grey cliff the data does
not support. So every cell gets the same ink and the reader compares the interval
against the marked zero, which is the honest reading of a forest plot.

    python -m tms_risk.behavior.scripts.plot_fig3_probit_stake

Reads notes/data/probit_stake_{group_posterior,by_ratio}.tsv -- nothing else, no
trace, no bauer.
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
DIFF = '#2b2b2b'
ORDERS = ['Risky first', 'Risky second']
STAKES = ['Low stake', 'High stake']
CELLS = [(o, s) for o in ORDERS for s in STAKES]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8.5, 'axes.labelsize': 9.5, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
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

# macOS ships Helvetica as a .ttc whose faces matplotlib cannot index, so every weight
# resolves to Regular and `fontweight='bold'` is silently a no-op. Arial Bold is a
# separate file and is metric-compatible, so route only the bold text through it.
BOLD = dict(fontname='Arial', fontweight='bold')

XT = [1.5, 2, 2.5, 3]
RISK_NEUTRAL = 1 / .55
#: chance level on the RNP axis, the risk-neutral probability itself
RISK_NEUTRAL_P = .55

#: Panel B plots the two conditions as LEVELS with their credible intervals,
#: not the difference as a density. Figure 3b,c shows exactly these two
#: parameters in exactly that grammar, and this figure is the same analysis
#: split by stake -- a reader moving between them should not have to relearn
#: how to read a panel. The difference is still reported, as the annotated
#: delta and p, which is what the density was for.
SPECS = [
    dict(par='slope', ylabel='Psychometric slope', fmt='{:+.2f}',
         title='Choice consistency',
         anchors=('Less consistent', 'More consistent')),
    dict(par='rnp', ylabel='Risk-neutral\nprobability', fmt='{:+.3f}',
         title='Risk attitude',
         anchors=('Risk-averse', 'Risk-seeking')),
]


def paired_delta(g, parameter, order, stake):
    """IPS - vertex, paired within draw so the difference keeps its posterior."""
    s = g[(g.parameter == parameter) & (g.order == order) & (g.stake == stake)]
    w = s.pivot_table(index='draw', columns='stimulation_condition', values='value')
    return (w['ips'] - w['vertex']).values


def pfmt(d):
    p = min(float((d > 0).mean()), float((d < 0).mean()))
    return 'p < 0.001' if p < .001 else f'p = {p:.3f}'


def main(data_dir, out_stem, tag):
    data = Path(data_dir)
    suffix = f'.{tag}' if tag else ''
    post = pd.read_csv(data / f'probit_stake_group_posterior{suffix}.tsv', sep='\t')
    rat = pd.read_csv(data / f'probit_stake_by_ratio{suffix}.tsv', sep='\t')
    # two versions of this figure differ only in the random-effects structure of the
    # four fits, which is invisible in the panels -- so the figure states it
    re = post['random_effects'].iloc[0] if 'random_effects' in post else 'full'
    RE_NOTE = {
        'full': 'Random effects: intercept, slope and both cTBS shifts vary by subject',
        'intercept': 'Random intercept only, as in the published model — slope '
                     'contrasts assume independent trials',
    }[re]

    # Height is set by the margins the type needs, not by the row count: block title
    # on top, tick labels + axis label + anchors underneath. Everything left over is
    # divided among the four rows, which leaves each psychometric panel about 2:1 --
    # the shape a stimulus-axis curve wants anyway.
    H = 4.9
    fig = plt.figure(figsize=(7.25, H))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.02, 2.], wspace=.22,
                             left=.16, right=.985, top=1 - .30 / H, bottom=.46 / H)
    # nested so the gap between the two presentation orders is visibly larger than
    # the gap between the two stake bands inside each -- the grouping is the point
    gsA = outer[0].subgridspec(2, 1, hspace=.13)
    gsB = outer[1].subgridspec(2, 1, hspace=.13)
    axA, axB = {}, {}
    for gi, order in enumerate(ORDERS):
        sa = gsA[gi].subgridspec(2, 1, hspace=.16)
        sb = gsB[gi].subgridspec(2, 2, hspace=.16, wspace=.16)
        for si, stake in enumerate(STAKES):
            axA[(order, stake)] = fig.add_subplot(sa[si])
            for ci, spec in enumerate(SPECS):
                axB[(order, stake, spec['par'])] = fig.add_subplot(sb[si, ci])

    last = CELLS[-1]

    # --- A: psychometric functions per cell
    for cell in CELLS:
        order, stake = cell
        ax = axA[cell]
        ax.axhline(.5, color='.75', lw=.7, ls='--', zorder=0)
        ax.axvline(RISK_NEUTRAL, color='.75', lw=.7, ls='--', zorder=0)
        r = rat[(rat.order == order) & (rat.stake == stake)].sort_values('frac')
        for stim, colr, mk in [('vertex', VERTEX, 'o'), ('ips', IPS, 's')]:
            ax.fill_between(r.frac, r[f'probit_{stim}_lo'], r[f'probit_{stim}_hi'],
                            color=colr, alpha=.22, lw=0, zorder=1)
            ax.plot(r.frac, r[f'probit_{stim}'], color=colr, lw=1.4, zorder=2)
            ax.plot(r.frac, r[f'observed_{stim}'], mk, color=colr, ms=3.7, lw=0,
                    zorder=4)
        ax.set_xscale('log')
        ax.set_xticks(XT)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlim(1.5, 3.25)
        ax.set_ylim(.18, .92); ax.set_yticks([.25, .5, .75])
        if cell == CELLS[0]:
            ax.text(RISK_NEUTRAL * 1.04, .91, 'Risk-neutral', fontsize=7.5,
                    color='.45', ha='left', va='top', style='italic')
        if cell == last:
            ax.set_xlabel('Risky/safe payoff ratio')
            sns.despine(ax=ax, offset=3)
        else:
            ax.set_xticklabels([]); ax.tick_params(axis='x', length=0)
            sns.despine(ax=ax, offset=3, bottom=True)

    a0 = axA[CELLS[0]]
    a0.plot([], [], color=VERTEX, marker='o', ms=3.7, lw=1.4, label='Vertex')
    a0.plot([], [], color=IPS, marker='s', ms=3.7, lw=1.4, label='IPS')
    a0.legend(loc='lower right', fontsize=8, handlelength=1.5, borderpad=.2,
              labelspacing=.2, borderaxespad=.2)

    # --- B: the two conditions as levels, one y-range per parameter across all
    # four cells so the panels compare. Same grammar as Figure 3b,c.
    stats = {}
    for spec in SPECS:
        par = spec['par']
        deltas = {c: paired_delta(post, par, *c) for c in CELLS}
        lv = {}
        for cell in CELLS:
            for stim in ('vertex', 'ips'):
                v = post[(post.parameter == par) & (post.order == cell[0])
                         & (post.stake == cell[1])
                         & (post.stimulation_condition == stim)].value.values
                lv[cell + (stim,)] = (v.mean(), *np.quantile(v, [.025, .975]))
        allv = np.concatenate([[q[1], q[2]] for q in lv.values()])
        pad = .10 * np.ptp(allv)
        ylo, yhi = allv.min() - pad, allv.max() + pad * 2.4

        for cell in CELLS:
            ax = axB[(cell[0], cell[1], par)]
            d = deltas[cell]
            lo, hi = np.quantile(d, [.025, .975])
            pstr = pfmt(d)
            stats[(par,) + cell] = (d.mean(), lo, hi, pstr)
            credible = (lo > 0) or (hi < 0)

            if par == 'rnp':
                ax.axhline(RISK_NEUTRAL_P, color='.75', lw=.7, ls='--', zorder=1)
            m_v, m_i = lv[cell + ('vertex',)][0], lv[cell + ('ips',)][0]
            # the connector carries the contrast: dark when the interval on the
            # difference clears zero, grey when it does not, so the reader sees
            # the verdict on the same mark that shows the change
            ax.plot([0, 1], [m_v, m_i], '-', lw=1.4,
                    color='.15' if credible else '.62', zorder=2)
            for x, stim, col, mk in ((0, 'vertex', VERTEX, 'o'),
                                     (1, 'ips', IPS, 's')):
                mu, q_lo, q_hi = lv[cell + (stim,)]
                ax.plot([x, x], [q_lo, q_hi], color=col, lw=1.6,
                        solid_capstyle='butt', zorder=3)
                ax.plot([x], [mu], mk, ms=4.6, color=col, zorder=4)
            ax.text(.5, .985, f"Δ {spec['fmt'].format(d.mean())}   {pstr}",
                    transform=ax.transAxes, ha='center', va='top', fontsize=7.5,
                    color='.15' if credible else '.5',
                    **(BOLD if credible else {}))
            ax.set_xlim(-.55, 1.55)
            ax.set_ylim(ylo, yhi)
            ax.set_xticks([0, 1])
            if cell == last:
                sns.despine(ax=ax, offset={'bottom': 3, 'left': 3})
                ax.set_xticklabels(['Vertex', 'IPS'])
                ax.set_xlabel('')
            else:
                sns.despine(ax=ax, bottom=True, offset={'left': 3})
                ax.set_xticklabels([])
                ax.tick_params(axis='x', length=0)
            # every row, not just the top one: the four rows are separate
            # axes with their own tick labels, and a single label at the top
            # makes the reader carry it three panels down
            ax.set_ylabel(spec['ylabel'], fontsize=7.5,
                          linespacing=1.15)
            ax.tick_params(labelsize=7.5)

    # --- row labels once at the far left, order on top of stake
    for cell in CELLS:
        pos = axA[cell].get_position()
        fig.text(.004, (pos.y0 + pos.y1) / 2, f'{cell[0]}\n{cell[1]}', fontsize=9,
                 ha='left', va='center', linespacing=1.3, color='.15')

    # One title per COLUMN, each centred over the column it names, rather than
    # one title spanning both parameters -- the two columns are different
    # quantities and a shared heading made the reader look for the y-label to
    # find out which was which.
    y = axA[CELLS[0]].get_position().y1 + .028
    blocks = [('A', a0, a0, 'Proportion of risky choices')]
    for i, spec in enumerate(SPECS):
        ax0 = axB[CELLS[0] + (spec['par'],)]
        blocks.append(('B' if i == 0 else '', ax0, ax0, spec['title']))
    for letter, first, lastax, title in blocks:
        x0, x1 = first.get_position().x0, lastax.get_position().x1
        if letter:
            fig.text(x0 - .048, y, letter, fontsize=11.5, va='baseline',
                     ha='left', **BOLD)
        fig.text((x0 + x1) / 2, y, title, fontsize=9.5, ha='center',
                 va='baseline', **BOLD)

    # below the axes area entirely; the tight bbox grows to include it
    fig.text(.004, -.028, RE_NOTE, fontsize=7.5, color='.45', style='italic',
             ha='left', va='bottom')

    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf\n')
    print(f'{"parameter":<8}{"order":<15}{"stake":<12}{"Δ":>10}{"95% CrI":>22}')
    for k, (m, lo, hi, p) in stats.items():
        print(f'{k[0]:<8}{k[1]:<15}{k[2]:<12}{m:>+10.4f}   '
              f'[{lo:+.4f}, {hi:+.4f}]   {p}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/fig3_probit_stake')
    parser.add_argument('--tag', default='',
                        help='which fit to read, e.g. --tag ri for random-intercept-only')
    a = parser.parse_args()
    main(a.data_dir, a.out, a.tag)
