"""Figure 4: does the flexible noise function earn its parameters, and what does it say?

    a   Posterior predictive check, Weber PMC beside Flexible PMC, split by
        presentation order and stake. This is the qualitative half of the model
        comparison: the order x stake x stimulation interaction is visible in the data,
        the Weber model flattens it, the flexible one reproduces it. Deliberately the
        same panel as Fig 4A of the v8 draft.
    b   The winning model's perceptual noise function on log-log axes, both stimulation
        conditions, against a slope-1 (Weber) reference. The fitted slope is ~0.5:
        noise grows with the square root of payoff, not in proportion to it. The first
        option's total noise nu_1 is drawn alongside, so the memory contribution is the
        (small) gap between the two.
    c   The cTBS increase as a percentage of baseline, with its credible interval. This
        is the scale on which the psychophysical analyses operate, and the scale on
        which the effect is magnitude-specific.
    d   Every fitted model on one ELPD axis, as a difference from the best with its
        dSE. The models that carry an explicit PRESENTATION-POSITION parameter sit far
        below the one that has none, which is the quantitative answer to the objection
        that the order effect was fitted rather than emergent.

    python -m tms_risk.behavior.scripts.plot_fig4_model

Reads notes/data/table1_all16.tsv (or --table), ppc_by_stake.<label>.tsv for both
the flexible and the Weber fit, pmcpars_curves.<label>.tsv and
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
# Colour is semantic across the paper: green = vertex, red = IPS. A DIFFERENCE
# between them is a third quantity, not one of the conditions, so it gets its own
# near-black ink rather than borrowing the IPS red.
DIFF = '#1a1a1a'
FLEX, WEBER = '#3B5BA5', '#9c9c9c'
XT = [7, 14, 28, 56, 112]
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'mathtext.fontset': 'stixsans',      # Helvetica has no Greek; nu falls back to a v
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


def ppc_panel(axes, data, label, weber_label):
    """Panel a: observed vs predicted P(risky) per stake tercile, two models.

    Columns are models, rows are presentation order, colour is stimulation. The point
    of the panel is a THREE-way pattern, so nothing here may be collapsed: the cTBS
    gap exists only in the risky-second row and only at the low stakes, which is
    exactly the cell the Weber model cannot reach.
    """
    cols = [(weber_label, 'Weber PMC'), (label, 'Flexible PMC')]
    frames = {i: pd.read_csv(data / f'ppc_by_stake.{lbl}.tsv', sep='\t')
              for i, (lbl, _) in enumerate(cols)}

    allv = pd.concat(frames.values())
    ylo = min(allv.lo.min(), (allv.observed - allv.observed_sem).min()) - .012
    yhi = max(allv.hi.max(), (allv.observed + allv.observed_sem).max()) + .012
    stakes = np.sort(allv.stake.unique())
    x = np.arange(len(stakes))

    for col, (_, title) in enumerate(cols):
        d = frames[col]
        for row, order in enumerate(ORDERS):
            ax = axes[row, col]
            for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
                s = d[(d.order == order) & (d.stim == stim)].sort_values('stake')
                ax.fill_between(x, s.lo, s.hi, color=colr, alpha=.20, lw=0, zorder=1)
                ax.plot(x, s['mean'], color=colr, lw=1.2, zorder=2)
                # nudge the two conditions apart so overlapping SEMs stay readable
                dx = .07 if stim == 'ips' else -.07
                ax.errorbar(x + dx, s.observed, yerr=s.observed_sem, fmt='o',
                            color=colr, ms=3.4, lw=0, elinewidth=.9, capsize=0,
                            zorder=4)
            ax.set_ylim(ylo, yhi)
            ax.set_xlim(-.42, len(stakes) - .58)
            ax.set_xticks(x)
            ax.set_yticks([.5, .55, .6, .65])
            if row == 0:
                ax.set_title(title, fontsize=7.6, color='.15', pad=4)
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels([f'{v:.0f}' for v in stakes])
                ax.set_xlabel('Stake (CHF)')
            if col == 0:
                ax.set_ylabel('P(chose risky)')
                ax.text(.04, .94, order, transform=ax.transAxes, fontsize=7,
                        color='.25', va='top', style='italic')
            else:
                ax.set_yticklabels([])
    axes[0, 1].text(.96, .92, 'IPS', transform=axes[0, 1].transAxes, fontsize=7,
                    color=IPS, ha='right', va='top')
    axes[0, 1].text(.96, .78, 'Vertex', transform=axes[0, 1].transAxes, fontsize=7,
                    color=VERTEX, ha='right', va='top')
    axes[1, 0].text(.5, .045, 'No cTBS gap', transform=axes[1, 0].transAxes,
                    fontsize=6.3, color='.4', ha='center')
    axes[1, 1].text(.5, .045, 'cTBS gap at low stakes',
                    transform=axes[1, 1].transAxes, fontsize=6.3, color='.4',
                    ha='center')


def main(data_dir, table, label, weber_label, out_stem):
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

    # Two bands. The top one carries the qualitative check and the mechanism; the
    # bottom one is the model comparison, which needs the full width for its labels
    # and a row of height for every model.
    n_models = len(t)
    h_top, h_bot = 3.05, .175 * n_models
    fig = plt.figure(figsize=(7.25, h_top + h_bot))
    split = h_bot / (h_top + h_bot)
    top, bot = split + .925 * (1 - split), split + .125 * (1 - split)
    gs_ppc = fig.add_gridspec(2, 2, left=.085, right=.415, top=top, bottom=bot,
                              hspace=.16, wspace=.10)
    gs_cur = fig.add_gridspec(1, 2, left=.545, right=.985, top=top, bottom=bot,
                              wspace=.44)
    gs_bot = fig.add_gridspec(1, 1, left=.28, right=.985,
                              top=split - .06, bottom=.075)

    # --- a: posterior predictive check, Weber against Flexible
    ppc_axes = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            ppc_axes[r, c] = fig.add_subplot(gs_ppc[r, c])
    ppc_panel(ppc_axes, data, label, weber_label)

    # --- b: the noise functions, log-log
    ax_b = fig.add_subplot(gs_cur[0, 0])
    c = pd.read_csv(data / f'pmcpars_curves.{label}.tsv', sep='\t')
    TERM = 'perceptual_noise_sd'
    p = c[(c.term == TERM) & (c.stimulation == 'vertex')].sort_values('payoff')
    for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
        s_ = c[(c.term == TERM) & (c.stimulation == stim)].sort_values('payoff')
        ax_b.fill_between(s_.payoff, s_.lo, s_.hi, color=colr, alpha=.16, lw=0,
                          zorder=1)
        ax_b.plot(s_.payoff, s_.nu, color=colr, zorder=2)
    # The memory CONTRIBUTION is nu_1 - nu_2, not softplus(eta_memory). Plotting the
    # latter is misleading: it is positive by construction, so it suggests the
    # first-presented option is always noisier, while the model composes
    # nu_1 = softplus(eta_mem + eta_perc) and the contribution is negative wherever
    # eta_mem < 0. A log axis cannot show a negative value, so nu_1 is drawn alongside
    # nu_2 and the memory contribution is simply the gap between them. It is a small
    # quantity -- a tenth of a franc against a noise level of one to six -- and is
    # drawn at that scale rather than blown up into a panel of its own.
    n1 = c[(c.term == 'n1_evidence_sd')
           & (c.stimulation == 'vertex')].sort_values('payoff')
    n2 = c[(c.term == 'n2_evidence_sd')
           & (c.stimulation == 'vertex')].sort_values('payoff')
    ax_b.plot(n1.payoff, n1.nu, color='.25', ls=(0, (3.5, 2)), lw=1.1, zorder=3)
    gap = n1.set_index('payoff').nu - n2.set_index('payoff').nu

    x0, y0 = p.payoff.iloc[0], p.nu.iloc[0]
    xs = np.array([x0, p.payoff.iloc[-1]])
    ax_b.plot(xs, y0 * xs / x0, color='.6', lw=.7, ls=':', zorder=0)
    slope = np.polyfit(np.log(p.payoff.values), np.log(p.nu.values), 1)[0]
    ax_b.set_xscale('log'); ax_b.set_yscale('log')
    ax_b.set_xticks(XT); ax_b.set_yticks([1, 2, 4, 8])
    ax_b.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_b.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_b.set_xlabel('Payoff (CHF)')
    ax_b.set_ylabel(r'Representational noise $\nu$ (CHF)')
    ax_b.set_ylim(min(0.8, float(n1.nu.min()) * .88), None)
    ax_b.text(.96, .13, 'IPS', transform=ax_b.transAxes, fontsize=7.2, color=IPS,
              ha='right')
    ax_b.text(.96, .04, 'Vertex', transform=ax_b.transAxes, fontsize=7.2, color=VERTEX,
              ha='right')
    ax_b.text(.03, .97, f'Slope {slope:.2f}', transform=ax_b.transAxes,
              fontsize=6.8, color='.25', va='top')
    ax_b.text(.03, .90, r'Dashed: first option ($\nu_1$)', transform=ax_b.transAxes,
              fontsize=6.2, color='.35', va='top')
    # Label the reference line ALONG it. A slope-1 line on log-log is only drawn at
    # 45 degrees when the decades are equally long on both axes, which they are not
    # here, so the angle has to come from the rendered positions.
    fig.canvas.draw()
    (px0, py0), (px1, py1) = ax_b.transData.transform(
        np.column_stack([xs, y0 * xs / x0]))
    ax_b.text(18., y0 * 18. / x0, '  Weber, slope 1', fontsize=6.2, color='.5',
              ha='left', va='bottom', rotation=np.degrees(np.arctan2(py1 - py0,
                                                                     px1 - px0)),
              rotation_mode='anchor')

    # --- c: the increase as a percentage, with its credible interval
    ax_c = fig.add_subplot(gs_cur[0, 1])
    rel = pd.read_csv(data / f'pmcpars_relative.{label}.tsv', sep='\t')
    rel = rel[rel.term == 'perceptual_noise_sd'].sort_values('payoff')
    ax_c.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax_c.fill_between(rel.payoff, rel.lo, rel.hi, color=DIFF, alpha=.16, lw=0, zorder=1)
    ax_c.plot(rel.payoff, rel.pct, color=DIFF, zorder=2)
    ax_c.set_xscale('log'); ax_c.set_xticks(XT)
    ax_c.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_c.set_xlabel('Payoff (CHF)')
    ax_c.set_ylabel(r'$\Delta$ noise, IPS − vertex (%)')
    lo7 = rel.iloc[(rel.payoff - 7).abs().argmin()]
    hi7 = rel.iloc[(rel.payoff - 112).abs().argmin()]

    # --- d: ELPD, as a cost relative to the best model
    ax_d = fig.add_subplot(gs_bot[0, 0])
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        colr = WEBER if r.weber else FLEX
        mark = 'D' if r.base in POSITIONAL else 'o'
        ax_d.errorbar(r.elpd_diff, yi, xerr=r.dse, fmt=mark, color=colr, ms=4.4,
                      lw=0, elinewidth=1.1, capsize=0, zorder=3)
    ax_d.axvline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([r.short for _, r in t.iterrows()], fontsize=7)
    for tick, (_, r) in zip(ax_d.get_yticklabels(), t.iterrows()):
        if r.base in POSITIONAL:
            tick.set_color('.15')
            tick.set_fontweight('bold')
    ax_d.set_xlabel('ELPD cost vs the best model (nats)')
    ax_d.set_ylim(-.8, len(t) - .2)
    ax_d.invert_xaxis()
    ax_d.annotate('Shown in a–c', xy=(t.elpd_diff.iloc[0], y[0]),
                  xytext=(t.elpd_diff.iloc[2], y[0] + .45), fontsize=6.5, color='.35',
                  ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-.2',
                                  color='.5', lw=.6))
    ax_d.text(.98, .05, 'Diamonds: cTBS effect confined to\none presentation position',
              transform=ax_d.transAxes, fontsize=6.3, color='.3', ha='right',
              va='bottom', linespacing=1.25)

    sns.despine(fig=fig, offset=4)
    # Panel letters in FIGURE coordinates: the four panels sit in three different
    # gridspecs with different margins, so axes-relative offsets would not line up.
    for xf, yf, letter in [(.012, top + .055 * (1 - split), 'a'),
                           (.462, top + .055 * (1 - split), 'b'),
                           (.735, top + .055 * (1 - split), 'c'),
                           (.012, split - .035, 'd')]:
        fig.text(xf, yf, letter, fontsize=11, fontweight='bold', va='bottom',
                 ha='left')
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf   ({len(t)} models)')
    print(f'  log-log slope {slope:.3f} (Weber = 1)')
    gv, gx = gap.values, gap.index.values
    print(f'  memory contribution nu1 - nu2: {gap.iloc[0]:+.3f} CHF at '
          f'{gap.index[0]:.0f}, peak {gv.max():+.3f} at {gx[gv.argmax()]:.0f}, '
          f'{gap.iloc[-1]:+.3f} at {gap.index[-1]:.0f}')
    print(f'  relative effect: {lo7.pct:+.1f}% at 7 CHF [{lo7.lo:+.1f}, {lo7.hi:+.1f}], '
          f'{hi7.pct:+.1f}% at 112 [{hi7.lo:+.1f}, {hi7.hi:+.1f}]')
    for lbl in [weber_label, label]:
        d = pd.read_csv(data / f'ppc_by_stake.{lbl}.tsv', sep='\t')
        w = d.pivot_table(index=['order', 'stake'], columns='stim',
                          values=['mean', 'observed'])
        pred = w[('mean', 'ips')] - w[('mean', 'vertex')]
        obs = w[('observed', 'ips')] - w[('observed', 'vertex')]
        print(f'  {lbl}: cTBS gap (IPS - vertex), predicted vs observed')
        for k in pred.index:
            print(f'    {k[0]:<13} stake {k[1]:5.1f}  '
                  f'pred {pred[k]:+.3f}  obs {obs[k]:+.3f}')
    pos = t[t.base.isin(POSITIONAL)]
    print('  positional models cost: '
          + ', '.join(f'{r.short} +{r.elpd_diff:.0f}' for _, r in pos.iterrows()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--table',
                        default='/Users/gdehol/git/tms_risk/notes/data/table1_all16.tsv')
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--weber_label', default='weber2nf')
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/fig4_model')
    a = parser.parse_args()
    main(a.data_dir, a.table, a.label, a.weber_label, a.out)
