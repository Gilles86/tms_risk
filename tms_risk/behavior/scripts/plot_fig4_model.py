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


# macOS ships Helvetica as a .ttc from which matplotlib registers ONLY the regular
# face, so fontweight='bold' on a Helvetica text silently renders at regular weight --
# it does not warn, and the panel letters have been quietly un-bold. Arial is
# metrically identical to Helvetica and does register a real bold, so anything that
# has to be heavier asks for Arial by name.
BOLD = dict(family='Arial', fontweight='bold')


def shorten(name):
    for pat, s in SHORT:
        if re.fullmatch(pat, name):
            return s
    return name


def ppc_panel(axes, data, label, weber_label):
    """Panel a: observed vs predicted P(risky) per stake tercile, two models.

    Four axes in one row: the two presentation orders under the Weber fit, then the
    same two under the flexible fit. The point of the panel is a THREE-way pattern, so
    nothing here may be collapsed: the cTBS gap exists only in the risky-second row and
    only at the low stakes, which is exactly the cell the Weber model cannot reach.

    Returns the count of arrowed misses per model, so the caller can report them.
    """
    models = [(weber_label, 'Weber PMC'), (label, 'Flexible PMC')]
    frames = {i: pd.read_csv(data / f'ppc_by_stake.{lbl}.tsv', sep='\t')
              for i, (lbl, _) in enumerate(models)}

    allv = pd.concat(frames.values())
    ylo = min(allv.lo.min(), allv.observed.min()) - .012
    yhi = max(allv.hi.max(), allv.observed.max()) + .012
    stakes = np.sort(allv.stake.unique())
    x = np.arange(len(stakes))

    misses = {}
    for m, (lbl, _) in enumerate(models):
        d = frames[m]
        for o, order in enumerate(ORDERS):
            ax = axes[2 * m + o]
            for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
                s = d[(d.order == order) & (d.stim == stim)].sort_values('stake')
                ax.fill_between(x, s.lo, s.hi, color=colr, alpha=.20, lw=0, zorder=1)
                ax.plot(x, s['mean'], color=colr, lw=1.2, zorder=2)
                # No error bar on the observed points. The uncertainty that matters
                # for a predictive check is the model's, and that is the shaded band;
                # a between-subject SEM on top of it invites reading the two as
                # comparable when they answer different questions.
                dx = .06 if stim == 'ips' else -.06
                ax.plot(x + dx, s.observed, 'o', color=colr, ms=3.8, lw=0, zorder=4)
                # Arrow every observed point the model's own 95% predictive interval
                # fails to cover. Found from the numbers rather than placed by hand, so
                # a refit cannot leave an arrow pointing at a cell that now fits.
                for _, r in s.iterrows():
                    if r.lo <= r.observed <= r.hi:
                        continue
                    up = r.observed > r.hi
                    xi = float(x[np.argmin(np.abs(stakes - r.stake))]) + dx
                    ax.annotate('', xy=(xi, r.observed), xycoords='data',
                                xytext=(-15, 13 if up else -13),
                                textcoords='offset points', zorder=6,
                                arrowprops=dict(arrowstyle='-|>', color='.1', lw=.9,
                                                shrinkA=0, shrinkB=3.5,
                                                mutation_scale=6))
                    misses[lbl] = misses.get(lbl, 0) + 1
            ax.set_ylim(ylo, yhi)
            ax.set_xlim(-.42, len(stakes) - .58)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{v:.0f}' for v in stakes])
            ax.set_yticks([.5, .55, .6, .65])
            ax.set_title(order, fontsize=7.2, color='.3', pad=4, style='italic')
            if 2 * m + o == 0:
                ax.set_ylabel('P(chose risky)')
            if o == 1:                       # only the second panel of each pair
                ax.set_yticklabels([])
    axes[0].text(.97, .96, 'IPS', transform=axes[0].transAxes, fontsize=7,
                 color=IPS, ha='right', va='top')
    axes[0].text(.97, .82, 'Vertex', transform=axes[0].transAxes, fontsize=7,
                 color=VERTEX, ha='right', va='top')
    axes[1].text(.96, .04, 'Arrows: observed outside\nthe model\'s 95% interval',
                 transform=axes[1].transAxes, fontsize=6, color='.35', va='bottom',
                 ha='right', linespacing=1.25)
    axes[3].text(.96, .04, 'cTBS gap reproduced', transform=axes[3].transAxes,
                 fontsize=6, color='.35', va='bottom', ha='right')
    return misses


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

    # Three bands, one per argument: the qualitative check, the mechanism, the model
    # comparison. Laid out in INCHES and converted, because the row heights are set by
    # what each row contains -- the ELPD panel needs a line of height per model, the
    # curves want to be wide rather than tall -- and figure fractions would have to be
    # re-derived by hand every time the model count changes.
    n_models = len(t)
    H_A, H_BC, H_D = 1.30, 1.45, .145 * n_models
    # PAD_TOP carries three stacked lines above row a (panel title, the two model
    # headings, the per-axes order titles); each gap carries the row above's tick
    # labels and axis label plus the next row's title.
    PAD_TOP, GAP_A, GAP_BC, PAD_BOT = .60, .80, .72, .38
    H = PAD_TOP + H_A + GAP_A + H_BC + GAP_BC + H_D + PAD_BOT
    fig = plt.figure(figsize=(7.25, H))

    def band(top_in, height_in):
        return dict(top=1 - top_in / H, bottom=1 - (top_in + height_in) / H)

    row_a = band(PAD_TOP, H_A)
    row_bc = band(PAD_TOP + H_A + GAP_A, H_BC)
    row_d = band(PAD_TOP + H_A + GAP_A + H_BC + GAP_BC, H_D)

    # Panel a is two PAIRS, not four equal columns: the gap between the pairs is what
    # tells the reader the comparison is Weber-vs-flexible and not four unrelated cells.
    PAIRS = [(.075, .495), (.575, .995)]
    gs_a = [fig.add_gridspec(1, 2, left=l, right=r, wspace=.12, **row_a)
            for l, r in PAIRS]
    gs_bc = fig.add_gridspec(1, 2, left=.095, right=.975, wspace=.30, **row_bc)
    gs_d = fig.add_gridspec(1, 1, left=.28, right=.985, **row_d)

    # --- a: posterior predictive check, Weber against Flexible
    ppc_axes = [fig.add_subplot(gs_a[m][0, o]) for m in (0, 1) for o in (0, 1)]
    misses = ppc_panel(ppc_axes, data, label, weber_label)
    for (l, r), nm in zip(PAIRS, ['Weber PMC', 'Flexible PMC']):
        fig.text((l + r) / 2, row_a['top'] + .19 / H, nm, ha='center', va='bottom',
                 fontsize=9.5, color='.1')
        # The shared x-label has to clear the tick labels of BOTH panels of the pair --
        # centred under a pair, it lands exactly between the '42' of one and the '13'
        # of the next, so it needs the vertical room rather than the horizontal.
        fig.text((l + r) / 2, row_a['bottom'] - .42 / H, 'Stake (CHF)', ha='center',
                 va='bottom', fontsize=8.5)

    # --- b: the noise functions, log-log
    ax_b = fig.add_subplot(gs_bc[0, 0])
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
    ax_b.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_b.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_b.set_xticks(XT)
    ax_b.set_xlabel('Payoff (CHF)')
    ax_b.set_ylabel(r'Representational noise $\nu$ (CHF)')
    # The y-range comes from the NOISE CURVES only. The Weber reference reaches 21 CHF
    # at the right edge, so letting it autoscale the axis squeezes the three curves the
    # panel is actually about into its lower half. The reference is a guide to the eye
    # for a slope, and a guide to the eye may run off the top.
    drawn = pd.concat([c[(c.term == TERM) & (c.stimulation != 'ips - vertex')], n1])
    ylo, yhi = float(drawn.lo.min()) * .93, float(drawn.hi.max()) * 1.07
    ax_b.set_ylim(ylo, yhi)
    ax_b.set_yticks([v for v in [1, 1.5, 2, 3, 4, 6, 8] if ylo < v < yhi])
    ax_b.minorticks_off()
    ax_b.text(.97, .14, 'IPS', transform=ax_b.transAxes, fontsize=7.2, color=IPS,
              ha='right')
    ax_b.text(.97, .04, 'Vertex', transform=ax_b.transAxes, fontsize=7.2, color=VERTEX,
              ha='right')
    ax_b.text(.03, .96, f'Slope {slope:.2f}', transform=ax_b.transAxes,
              fontsize=6.8, color='.25', va='top')
    ax_b.text(.03, .84, r'Dashed: first option ($\nu_1$)', transform=ax_b.transAxes,
              fontsize=6.2, color='.35', va='top')
    # Label the reference line ALONG it, at the height where it is clear of the noise
    # bands. A slope-1 line on log-log is only drawn at 45 degrees when the decades are
    # equally long on both axes, which they are not here, so the angle has to come from
    # the rendered positions rather than from the slope.
    fig.canvas.draw()
    (px0, py0), (px1, py1) = ax_b.transData.transform(
        np.column_stack([xs, y0 * xs / x0]))
    y_lab = np.exp(np.log(ylo) + .62 * (np.log(yhi) - np.log(ylo)))
    ax_b.text(x0 * y_lab / y0, y_lab, 'Weber, slope 1', fontsize=6.2, color='.5',
              ha='center', va='bottom',
              rotation=np.degrees(np.arctan2(py1 - py0, px1 - px0)),
              rotation_mode='anchor')

    # --- c: the increase as a percentage, with its credible interval
    ax_c = fig.add_subplot(gs_bc[0, 1])
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
    ax_d = fig.add_subplot(gs_d[0, 0])
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        colr = WEBER if r.weber else FLEX
        ax_d.errorbar(r.elpd_diff, yi, xerr=r.dse, fmt='o', color=colr, ms=4.4,
                      lw=0, elinewidth=1.1, capsize=0, zorder=3)
    ax_d.axvline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([r.short for _, r in t.iterrows()], fontsize=7)
    ax_d.set_xlabel('ELPD cost vs the best model (nats)')
    ax_d.set_ylim(-.8, len(t) - .2)
    ax_d.invert_xaxis()
    ax_d.annotate('Shown in a–c', xy=(t.elpd_diff.iloc[0], y[0]),
                  xytext=(t.elpd_diff.iloc[2], y[0] + .45), fontsize=6.5, color='.35',
                  ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-.2',
                                  color='.5', lw=.6))

    sns.despine(fig=fig, offset=4)
    # Panel letters in FIGURE coordinates: the four panels sit in three different
    # gridspecs with different margins, so axes-relative offsets would not line up.
    # Letter hard left, title centred over the panel it names. Centring is read from
    # the rendered axes rather than written down, because the three rows sit in three
    # gridspecs with different margins and hand-set centres drift the moment one of
    # those margins changes.
    for ax_l, ax_r, yf, letter, title in [
            (ppc_axes[0], ppc_axes[3], row_a['top'] + .36 / H, 'a',
             'Posterior predictive checks'),
            (ax_b, ax_b, row_bc['top'] + .09 / H, 'b',
             'Noise as a function of magnitude'),
            (ax_c, ax_c, row_bc['top'] + .09 / H, 'c', 'Effect of cTBS on noise'),
            (ax_d, ax_d, row_d['top'] + .07 / H, 'd', 'Model comparison')]:
        fig.text(.008 if letter != 'c' else .507, yf, letter, fontsize=11,
                 va='bottom', ha='left', **BOLD)
        fig.text((ax_l.get_position().x0 + ax_r.get_position().x1) / 2, yf + .006,
                 title, fontsize=8.5, color='.1', va='bottom', ha='center', **BOLD)
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
