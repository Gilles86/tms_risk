"""Figure 4: which model the data prefer, and what its noise function looks like.

    a   Every fitted model on one ELPD axis, as a difference from the best with its
        dSE. The models that carry an explicit PRESENTATION-POSITION parameter sit far
        below the one that has none, which is the quantitative answer to the objection
        that the order effect was fitted rather than emergent.
    b   The winning model's perceptual noise function on log-log axes, both stimulation
        conditions, against a slope-1 (Weber) reference. The fitted slope is ~0.5:
        noise grows with the square root of payoff, not in proportion to it.
    c   The cTBS increase as a percentage of baseline, with its credible interval. This
        is the scale on which the psychophysical analyses operate, and the scale on
        which the effect is magnitude-specific.

    python -m tms_risk.behavior.scripts.plot_fig4_model

Reads notes/data/table1_flexible8.tsv (or --table), noisecurve_reparam.<label>.tsv and
pmcpars_relative.<label>.tsv.
"""
import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
FLEX, WEBER = '#3B5BA5', '#9c9c9c'
XT = [7, 14, 28, 56, 112]

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

SHORT = [
    (r'.*perceptual noise only\)$',              'Perceptual noise'),
    (r'.*perceptual and memory noise\)$',        'Perceptual + memory'),
    (r'.*memory noise only\)$',                  'Memory noise'),
    (r'.*first- and second-option noise\)$',     'First + second option'),
    (r'.*first-presented option only\)$',        'First option'),
    (r'.*second-presented option only\)$',       'Second option'),
    (r'.*null model$',                           'No cTBS effect'),
]
# Models in which the cTBS effect is CONFINED TO ONE PRESENTATION POSITION. Note the
# *full* models are not in this set: 'First + second option' and 'Perceptual + memory'
# are exact reparameterisations of one another (c1 = memory + perceptual, c2 =
# perceptual), so they share a likelihood and differ only in prior coordinates -- the
# gap between them is a prior effect, not a difference of mechanism. It is only the
# restricted models that impose genuinely different claims about where cTBS acts.
POSITIONAL = {'First option', 'Second option'}


def shorten(name):
    for pat, s in SHORT:
        if re.fullmatch(pat, name):
            return s
    return name


def main(data_dir, table, label, out_stem):
    data = Path(data_dir)
    t = pd.read_csv(table, sep='\t', index_col=0).sort_values('elpd_diff')
    t['base'] = t.name.map(shorten)          # the cTBS locus, family-agnostic
    t['weber'] = t.index.str.startswith('weber')
    t['family'] = t.index.str.extract(r'(?:flexible|weber)([12])')[0].values
    # With both noise families present the locus alone is ambiguous, and the two nulls
    # within a family differ only in prior coordinates, so both need spelling out.
    t['short'] = [('Weber: ' if r.weber else 'Flexible: ') + r.base
                  for _, r in t.iterrows()]
    dup = t.short.duplicated(keep=False)
    t.loc[dup, 'short'] = [f'{r.short} ({r.family})' for _, r in t[dup].iterrows()]

    # Row 1 is the model comparison at full width -- the model names need the room and
    # it is the panel the argument turns on. Row 2 gives the three curves real width.
    n_models = len(t)
    h_top, h_bot = .17 * n_models, 2.0
    fig = plt.figure(figsize=(7.25, h_top + h_bot))
    # Two gridspecs, not one: the model-comparison row needs a wide left margin for
    # its labels, while the curve panels below should use the full page width.
    split = h_bot / (h_top + h_bot)
    gs_top = fig.add_gridspec(1, 1, left=.28, right=.985,
                              top=.985, bottom=split + .085)
    gs = fig.add_gridspec(1, 3, wspace=.46, left=.095, right=.985,
                          top=split - .085, bottom=.115)

    # --- a: ELPD, as a cost relative to the best model
    ax = fig.add_subplot(gs_top[0, 0])
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        colr = WEBER if r.weber else FLEX
        mark = 'D' if r.base in POSITIONAL else 'o'
        ax.errorbar(r.elpd_diff, yi, xerr=r.dse, fmt=mark, color=colr, ms=4.4,
                    lw=0, elinewidth=1.1, capsize=0, zorder=3)
    ax.axvline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_yticks(y)
    lbl = [f'{r.short}' + ('' if r.weber else '') for _, r in t.iterrows()]
    ax.set_yticklabels(lbl, fontsize=7)
    for tick, (_, r) in zip(ax.get_yticklabels(), t.iterrows()):
        if r.base in POSITIONAL:
            tick.set_color('.15')
            tick.set_fontweight('bold')
    ax.set_xlabel('ELPD cost vs the best model (nats)')
    ax.set_ylim(-.8, len(t) - .2)
    ax.invert_xaxis()
    best = t.index[0]
    ax.annotate('Curves below', xy=(t.elpd_diff.iloc[0], y[0]),
                xytext=(t.elpd_diff.iloc[2], y[0] + .45), fontsize=6.5, color='.35',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-.2',
                                color='.5', lw=.6))
    ax.text(.98, .05, 'Diamonds: cTBS effect confined to\none presentation position',
            transform=ax.transAxes, fontsize=6.3, color='.3', ha='right', va='bottom',
            linespacing=1.25)

    # --- b: the noise functions, log-log
    # Both components are shown. NOTE the memory curve is softplus(eta_memory), the
    # parameter as bauer defines it -- it is NOT nu_1 - nu_2. The model composes
    # nu_1 = softplus(eta_mem + eta_perc), so a positive memory curve does not imply
    # the first-presented option is noisier; that requires eta_mem > 0, which fails
    # below ~12 CHF. The curve is drawn to show that memory noise, unlike perceptual
    # noise, does not scale with magnitude.
    ax = fig.add_subplot(gs[0, 0])
    c = pd.read_csv(data / f'pmcpars_curves.{label}.tsv', sep='\t')
    TERM, MEM = 'perceptual_noise_sd', 'memory_noise_sd'
    p = c[(c.term == TERM) & (c.stimulation == 'vertex')].sort_values('payoff')
    for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
        s_ = c[(c.term == TERM) & (c.stimulation == stim)].sort_values('payoff')
        ax.fill_between(s_.payoff, s_.lo, s_.hi, color=colr, alpha=.16, lw=0, zorder=1)
        ax.plot(s_.payoff, s_.nu, color=colr, zorder=2)
    # The memory CONTRIBUTION is nu_1 - nu_2, not softplus(eta_memory). Plotting the
    # latter is misleading: it is positive by construction, so it suggests the
    # first-presented option is always noisier, while the model composes
    # nu_1 = softplus(eta_mem + eta_perc) and the contribution is negative wherever
    # eta_mem < 0. A log axis cannot show a negative value, so nu_1 is drawn alongside
    # nu_2 and the memory contribution is the gap between them -- including where it
    # reverses at small payoffs.
    n1 = c[(c.term == 'n1_evidence_sd') & (c.stimulation == 'vertex')].sort_values('payoff')
    n2 = c[(c.term == 'n2_evidence_sd') & (c.stimulation == 'vertex')].sort_values('payoff')
    ax.plot(n1.payoff, n1.nu, color='.2', ls=(0, (3.5, 2)), lw=1.3, zorder=3)
    gap = n1.set_index('payoff').nu - n2.set_index('payoff').nu
    cross = gap.index[np.argmin(np.abs(gap.values))]

    x0, y0 = p.payoff.iloc[0], p.nu.iloc[0]
    xs = np.array([x0, p.payoff.iloc[-1]])
    ax.plot(xs, y0 * xs / x0, color='.6', lw=.7, ls=':', zorder=0)
    slope = np.polyfit(np.log(p.payoff.values), np.log(p.nu.values), 1)[0]
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(XT); ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Representational noise ν (CHF)')
    ax.set_ylim(min(0.8, float(n1.nu.min()) * .88), None)
    ax.text(.96, .13, 'IPS', transform=ax.transAxes, fontsize=7.2, color=IPS, ha='right')
    ax.text(.96, .04, 'Vertex', transform=ax.transAxes, fontsize=7.2, color=VERTEX,
            ha='right')
    ax.text(.03, .97, f'Slope {slope:.2f}', transform=ax.transAxes,
            fontsize=6.8, color='.25', va='top')
    ax.text(.03, .88, 'Dashed: first option', transform=ax.transAxes,
            fontsize=6.4, color='.35', va='top')
    ax.text(.97, .93, 'Weber, slope 1', transform=ax.transAxes, fontsize=6.4,
            color='.5', ha='right', va='top')

    # --- c: the memory contribution, on a linear axis where its sign is legible
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax.plot(gap.index.values, gap.values, color='.25', lw=1.4, zorder=2)
    ax.set_xscale('log'); ax.set_xticks([7, 28, 112])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('ν₁ − ν₂ (CHF)')
    ax.set_title('Memory contribution', fontsize=7, color='.3', pad=3)
    ax.text(.96, .06, 'First option less noisy', transform=ax.transAxes,
            fontsize=6.2, color='.4', va='bottom', ha='right')

    # --- d: the increase as a percentage, with its credible interval
    ax = fig.add_subplot(gs[0, 2])
    rel = pd.read_csv(data / f'pmcpars_relative.{label}.tsv', sep='\t')
    rel = rel[rel.term == 'perceptual_noise_sd'].sort_values('payoff')
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax.fill_between(rel.payoff, rel.lo, rel.hi, color=IPS, alpha=.18, lw=0, zorder=1)
    ax.plot(rel.payoff, rel.pct, color=IPS, zorder=2)
    ax.set_xscale('log'); ax.set_xticks(XT)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Δ noise, IPS − vertex (%)')
    lo7 = rel.iloc[(rel.payoff - 7).abs().argmin()]
    hi7 = rel.iloc[(rel.payoff - 112).abs().argmin()]

    for a, letter in zip(fig.axes, 'abcd'):
        a.text(0.0, 1.10, letter, transform=a.transAxes, fontsize=11,
               fontweight='bold', va='bottom', ha='left')
    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf   ({len(t)} models)')
    print(f'  log-log slope {slope:.3f} (Weber = 1)')
    print(f'  memory contribution nu1 - nu2: {gap.iloc[0]:+.3f} CHF at '
          f'{gap.index[0]:.0f}, {gap.iloc[-1]:+.3f} at {gap.index[-1]:.0f}; '
          f'crosses zero near {cross:.1f} CHF')
    print(f'  relative effect: {lo7.pct:+.1f}% at 7 CHF [{lo7.lo:+.1f}, {lo7.hi:+.1f}], '
          f'{hi7.pct:+.1f}% at 112 [{hi7.lo:+.1f}, {hi7.hi:+.1f}]')
    pos = t[t.base.isin(POSITIONAL)]
    print('  positional models cost: '
          + ', '.join(f'{r.short} +{r.elpd_diff:.0f}' for _, r in pos.iterrows()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--table', default='/Users/gdehol/git/tms_risk/notes/data/table1_flexible8.tsv')
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/fig4_model')
    a = parser.parse_args()
    main(a.data_dir, a.table, a.label, a.out)
