"""One PDF that shows the whole anchor grid: fit, noise shape, and where cTBS acts.

The grid has three dimensions -- inference SPACE, noise FORM, and cTBS PLACEMENT
-- and only two fit on a page, so the report picks the split by what each
dimension is for. Placement is the scientific question (does cTBS act on the
perceptual channel, the memory channel, the first option, the second?), so it
gets its own page. Form is a robustness dimension (does the answer survive a
different noise shape?), so it runs across the columns of every page, and the
reader checks stability by reading a row. Space is a modelling commitment rather
than a comparison, so each space gets its own PDF.

    p1      ELPD ladder + convergence gate for every model in this space
    p2      Contact sheet: the cTBS effect on the noise function, whole grid
    p3      Contact sheet: the cTBS effect on choices, whole grid, model vs data
    p4..    One page per placement; columns are noise forms, rows are
            sigma(x) / delta-sigma(x) / choice PPC / effect PPC

Reads only TSVs -- no trace, no bauer, no cluster:
    notes/data/anchor_curves.tsv                (extract_anchor_curves.py)
    notes/data/ppc_anchor/ppc_anchor.*.tsv      (extract_anchor_ppc.py)
    notes/data/loo_anchor/loo.*.tsv             (extract_anchor_loo.py)
    notes/data/anchor_ess.tsv                   (diagnose_anchor_ess.py)

    python -m tms_risk.behavior.scripts.make_anchor_report --space log
"""
import argparse
from glob import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

IPS, VERTEX = '#d62728', '#2ca02c'
N1, N2 = '0.55', '0.15'                 # presentation position never gets a hue
FIRST, SECOND = '0.62', '0.15'          # ... and neither does presentation order

FORMS = ['weber', 'affine', 'power', 'genweber', 'spl3', 'spl5']
FORM_LABEL = {'weber': 'Weber', 'affine': 'Affine', 'power': 'Power',
              'genweber': 'Gen. Weber', 'spl3': 'Spline-3', 'spl5': 'Spline-5'}
FORM_SUB = {'weber': 'σ constant', 'affine': 'σ ~ log x', 'power': 'log σ ~ log x',
            'genweber': 'σ = k + c/x', 'spl3': '3 anchors', 'spl5': '5 anchors'}

SHARED = ['null', 'perc', 'mem', 'percmem']
INDEP = ['nullind', 'n1', 'n2', 'n1n2']
PLACEMENTS = SHARED + INDEP
PLACE_LABEL = {
    'null': 'Null · no cTBS effect', 'perc': 'cTBS on perceptual noise',
    'mem': 'cTBS on memory noise', 'percmem': 'cTBS on perceptual + memory',
    'nullind': 'Null · no cTBS effect', 'n1': 'cTBS on first-presented option',
    'n2': 'cTBS on second-presented option', 'n3': '', 'n1n2': 'cTBS on both options'}
FAMILY = {**{p: 'Shared perceptual + memory channels (σ_n1 = σ_perc + σ_mem)'
             for p in SHARED},
          **{p: 'Independent first/second channels (σ_n1, σ_n2 free)'
             for p in INDEP}}
NULL_OF = {**{p: 'null' for p in SHARED}, **{p: 'nullind' for p in INDEP}}

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 6.5, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'lines.linewidth': 1.0, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 130, 'savefig.dpi': 300,
})
PAGE = (11.69, 8.27)                      # A4 landscape

#: 'null' and 'nullind' are PLACEMENT NAMES. Pandas reads the bare string
#: 'null' as NaN by default, which silently drops both null models out of every
#: panel -- exactly the models the rest of the grid is compared against.
READ = dict(sep='\t', keep_default_na=False, na_values=[''])


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def logx(ax, ticks=(7, 14, 28, 56, 112), labels=True):
    ax.set_xscale('log')
    ax.set_xticks(list(ticks))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    if labels:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    else:
        ax.set_xticklabels([])


def blank(ax, msg='no fit'):
    ax.text(.5, .5, msg, transform=ax.transAxes, ha='center', va='center',
            fontsize=6.5, color='0.65')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def key_block(fig, rect=(.575, .862, .40, .118), orders=True):
    """The page key. n1/n2 are the thing readers get wrong -- they are
    PRESENTATION POSITIONS, not the risky and safe option -- so the key says so
    in words rather than leaving it to a dashed line."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fc='0.965', ec='0.87', lw=.6, zorder=0))
    tx, lx = .105, (.025, .09)

    def row(y, text, color='0.15', ls='-', lw=1.2, sample=True, size=6.3):
        if sample:
            ax.plot(lx, [y, y], color=color, ls=ls, lw=lw,
                    solid_capstyle='butt')
        ax.text(tx if sample else .025, y, text, va='center', fontsize=size,
                color='0.15')

    short = orders          # the order key needs the right-hand half
    row(.88, 'σ$_{n1}$  ·  presented FIRST' + ('' if short else
        ' — held in memory until the choice'), color=N2, ls='-')
    row(.68, 'σ$_{n2}$  ·  presented SECOND' + ('' if short else
        ' — on screen at the choice'), color=N1, ls=(0, (2.4, 1.3)))
    ax.text(.025, .49,
            'n1 / n2 are PRESENTATION POSITIONS, not risky vs safe: either '
            'option can come first.', fontsize=5.8, color='0.45', va='center')
    ax.text(.025, .33,
            'Shared family:  σ$_{n1}$ = σ$_{perc}$ + σ$_{mem}$,   '
            'σ$_{n2}$ = σ$_{perc}$.   Independent family: both free.',
            fontsize=5.8, color='0.45', va='center')
    ax.plot(lx, [.14, .14], color=IPS, lw=1.6)
    ax.text(tx, .14, 'IPS (stimulated)', color=IPS, fontsize=6.3, va='center')
    ax.plot((.40, .465), [.14, .14], color=VERTEX, lw=1.6)
    ax.text(.48, .14, 'Vertex (sham)', color=VERTEX, fontsize=6.3, va='center')
    ax.text(.70, .14, 'Bands: 95% CrI', fontsize=6, color='0.45', va='center')
    if orders:
        # only where marker style is what encodes order; on the placement pages
        # order is the ROW, so the key would name an encoding that is not used
        ax.plot([.42], [.88], 'o', ms=3.4, color=SECOND, mfc=SECOND)
        ax.text(.455, .88, 'Risky second', fontsize=5.8, color='0.3', va='center')
        ax.plot([.42], [.68], 'o', ms=3.4, color=FIRST, mfc='white', mew=.8)
        ax.text(.455, .68, 'Risky first', fontsize=5.8, color='0.3', va='center')


def header(fig, title, subtitle, stamp):
    fig.text(.035, .965, title, fontsize=11, fontweight='bold', family='Arial',
             va='top')
    fig.text(.035, .928, subtitle, fontsize=8, color='0.30', va='top',
             linespacing=1.5)
    fig.text(.985, .008, stamp, fontsize=5.6, color='0.6', va='bottom', ha='right')


# --------------------------------------------------------------------------
# page 1 -- the ladder
# --------------------------------------------------------------------------
def page_ladder(pdf, loo, ess, space, stamp):
    fig = plt.figure(figsize=PAGE)
    header(fig, f'Anchor grid · {"log" if space == "log" else "natural"}-space observer',
           'Model comparison and convergence. ΔELPD is relative to the best\n'
           'model in this space; the gate is r̂ ≤ 1.01 and ESS ≥ 400 on\n'
           'group-level parameters.', stamp)
    key_block(fig, rect=(.60, .885, .37, .088), orders=False)
    # left margin has to hold a 20-character model label at 5.4 pt
    gs = fig.add_gridspec(1, 2, left=.115, right=.965, top=.82, bottom=.10,
                          width_ratios=[1.35, 1], wspace=.30)

    ax = fig.add_subplot(gs[0])
    if len(loo):
        # A ladder is a ranking, so sort by it: best model at the top. Grouping
        # by placement instead would bury the ordering the page exists to show;
        # placement stays readable because the label carries it and the tick is
        # coloured by family.
        d = loo.copy()
        d['d_elpd'] = d.elpd_loo - d.elpd_loo.max()
        d = d.sort_values('elpd_loo', ascending=False)
        y = np.arange(len(d))[::-1]
        # truncate mako: its top end is near-white and the Spline-5 markers and
        # key entry were washing out against the page
        cols = dict(zip(FORMS, sns.color_palette('mako', as_cmap=True)(
            np.linspace(.12, .72, len(FORMS)))))
        ok = d.label.isin(ess.loc[ess.ok_group, 'label']) if len(ess) else True
        for yi, (_, r) in zip(y, d.iterrows()):
            c = cols[r.form]
            good = bool(ok.loc[r.name]) if hasattr(ok, 'loc') else True
            ax.errorbar(r.d_elpd, yi, xerr=r.se, fmt='o', ms=3.6, color=c,
                        mfc=c if good else 'white', mew=.9, elinewidth=.7,
                        capsize=0, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(d.label, fontsize=5.4)
        for t, place in zip(ax.get_yticklabels(), d.placement):
            t.set_color('0.15' if place in SHARED else '#3B5BA5')
        ax.axvline(0, color='0.75', lw=.6, ls='--', zorder=0)
        ax.set_xlabel('ΔELPD (LOO) relative to the best model')
        ax.set_ylim(-1, len(d))
        # the null models sit far left, so the lower-right of the panel is empty
        for i, f in enumerate(FORMS):
            ax.text(.70, .30 - i * .028, FORM_LABEL[f], color=cols[f],
                    transform=ax.transAxes, fontsize=6, va='top')
        ax.text(.70, .30 - len(FORMS) * .028, 'Open marker: failed the gate',
                transform=ax.transAxes, fontsize=6, color='0.45', va='top')
        ax.text(.70, .30 - (len(FORMS) + 1) * .028, 'Label: shared perc/mem',
                color='0.15', transform=ax.transAxes, fontsize=6, va='top')
        ax.text(.70, .30 - (len(FORMS) + 2) * .028, '          independent n1/n2',
                color='#3B5BA5', transform=ax.transAxes, fontsize=6, va='top')
    else:
        blank(ax, 'no LOO yet')

    ax = fig.add_subplot(gs[1])
    if len(ess):
        M = ess.pivot_table(index='form', columns='placement',
                            values='min_ess_group')
        M = M.reindex(index=[f for f in FORMS if f in M.index],
                      columns=[p for p in PLACEMENTS if p in M.columns])
        im = ax.imshow(np.log10(M.values), cmap='rocket_r', vmin=0.7, vmax=3.7,
                       aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=5.6,
                        color='0.15' if v < 300 else 'white')
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(M.columns, rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels([FORM_LABEL[f] for f in M.index], fontsize=6)
        ax.set_title('Minimum group-level ESS (bulk)', fontsize=7.5, pad=4)
        cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
        cb.set_ticks(np.log10([10, 100, 400, 1000, 4000]))
        cb.set_ticklabels(['10', '100', '400', '1000', '4000'])
        cb.ax.tick_params(labelsize=5.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.text(0, -.075,
                'ESS below ~50 means chains that are stuck, not chains that are '
                'slow:\nthose cells are a geometry problem, not a draws problem.',
                transform=ax.transAxes, fontsize=6, color='0.35', va='top')
    else:
        blank(ax, 'no diagnostics yet')

    sns.despine(fig=fig, offset=3)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# pages 2-3 -- contact sheets over the whole grid
# --------------------------------------------------------------------------
def page_contact_noise(pdf, curves, space, stamp):
    fig = plt.figure(figsize=PAGE)
    header(fig, 'The cTBS effect on the noise function, whole grid',
           'Δσ = IPS − vertex, differenced within draw; median and 95% CrI. '
           'Solid: first-presented option. Dashed: second. Rows are noise '
           'forms, columns are where cTBS was allowed to act.', stamp)
    key_block(fig, orders=False)
    d = curves[(curves.space == space) & (curves.condition == 'delta')]
    if not len(d):
        blank(fig.add_subplot(111), 'no curves yet')
        pdf.savefig(fig); plt.close(fig); return
    m = 1.1 * max(abs(d.lo.min()), abs(d.hi.max()))
    forms = [f for f in FORMS if f in set(d.form)]
    gs = fig.add_gridspec(len(forms), 8, left=.065, right=.98, top=.85,
                          bottom=.075, hspace=.28, wspace=.22)
    for i, form in enumerate(forms):
        for j, place in enumerate(PLACEMENTS):
            ax = fig.add_subplot(gs[i, j])
            s = d[(d.form == form) & (d.placement == place)]
            if not len(s):
                blank(ax)
            else:
                ax.axhline(0, color='0.75', lw=.5, ls='--', zorder=0)
                for chan, ls, col in [('n1', '-', N2), ('n2', (0, (2.4, 1.3)), N1)]:
                    c = s[s.channel == chan].sort_values('x')
                    ax.fill_between(c.x, c.lo, c.hi, color=col, alpha=.20, lw=0)
                    ax.plot(c.x, c['mid'], color=col, ls=ls, lw=1.0)
                ax.set_ylim(-m, m)
                logx(ax, labels=(i == len(forms) - 1))
                ax.set_yticks([-.05, 0, .05] if m > .06 else [-.02, 0, .02])
                if j:
                    ax.set_yticklabels([])
            if i == 0:
                ax.set_title(place, fontsize=7, pad=3,
                             color='0.15' if place in SHARED else '#3B5BA5')
            if j == 0:
                ax.set_ylabel(FORM_LABEL[form], fontsize=7)
    fig.text(.065, .832, 'Shared perc/mem family', fontsize=6.5, color='0.15')
    fig.text(.40, .832, 'Independent n1/n2 family', fontsize=6.5, color='#3B5BA5')
    fig.text(.5, .022, 'Payoff (CHF)', ha='center', fontsize=7)
    sns.despine(fig=fig, offset=2)
    pdf.savefig(fig)
    plt.close(fig)


def page_contact_ppc(pdf, delta, space, meta, stamp):
    fig = plt.figure(figsize=PAGE)
    header(fig, 'The cTBS effect on choices, whole grid',
           'ΔP(chose risky) = IPS − vertex against stake, formed within subject. '
           'Points are the data (identical in every panel, ± SEM); bands are the '
           'model. Dark: risky presented second. Light: risky first.', stamp)
    key_block(fig)
    if not len(delta):
        blank(fig.add_subplot(111), 'no PPC yet')
        pdf.savefig(fig); plt.close(fig); return
    forms = [f for f in FORMS if f in set(meta.form)]
    gs = fig.add_gridspec(len(forms), 8, left=.065, right=.98, top=.80,
                          bottom=.075, hspace=.28, wspace=.22)
    for i, form in enumerate(forms):
        for j, place in enumerate(PLACEMENTS):
            ax = fig.add_subplot(gs[i, j])
            lbl = f'{space}-{form}-{place}'
            s = delta[delta.label == lbl]
            if not len(s):
                blank(ax)
            else:
                ax.axhline(0, color='0.75', lw=.5, ls='--', zorder=0)
                for order, col, mfc in [('Risky first', FIRST, 'white'),
                                        ('Risky second', SECOND, SECOND)]:
                    o = s[s.order == order].sort_values('stake_chf')
                    ax.fill_between(o.stake_chf, o.lo, o.hi, color=col,
                                    alpha=.20, lw=0)
                    ax.plot(o.stake_chf, o.model, color=col, lw=1.0)
                    ax.errorbar(o.stake_chf, o.observed, yerr=o.observed_sem,
                                fmt='o', ms=2.8, color=col, mfc=mfc, mew=.7,
                                lw=0, elinewidth=.7, capsize=0, zorder=4)
                ax.set_ylim(-.13, .17)
                ax.set_yticks([-.1, 0, .1])
                ax.set_xscale('log')
                ax.set_xticks(sorted(s.stake_chf.unique()))
                ax.minorticks_off()
                if i == len(forms) - 1:
                    ax.set_xticklabels([f'{v:.0f}'
                                        for v in sorted(s.stake_chf.unique())])
                else:
                    ax.set_xticklabels([])
                if j:
                    ax.set_yticklabels([])
            if i == 0:
                ax.set_title(place, fontsize=7, pad=3,
                             color='0.15' if place in SHARED else '#3B5BA5')
            if j == 0:
                ax.set_ylabel(FORM_LABEL[form], fontsize=7)
    fig.text(.5, .022, 'Stake (CHF, within-subject terciles)', ha='center',
             fontsize=7)
    sns.despine(fig=fig, offset=2)
    pdf.savefig(fig)
    plt.close(fig)


def page_priors(pdf, priors, payoffs, space, stamp):
    """Does the observer's fitted prior look like the payoffs it was shown?

    Nothing in the model forces it to. The PMC's mechanism is shrinkage toward
    this prior, so a prior that sits far from the presented distribution is a
    substantive claim about the participant, not a nuisance parameter.
    """
    fig = plt.figure(figsize=PAGE)
    header(fig, 'Fitted magnitude priors vs the payoffs actually presented',
           'Grey histogram: every option shown to every participant.\n'
           'Curves: each model\'s group-level prior, N(μ, σ) over log payoff.\n'
           'Nothing in the model ties the two together.', stamp)
    pr = priors[priors.space == space]
    if not len(pr) or not len(payoffs):
        blank(fig.add_subplot(111), 'no priors yet')
        pdf.savefig(fig); plt.close(fig); return
    gs = fig.add_gridspec(2, 2, left=.07, right=.97, top=.80, bottom=.09,
                          hspace=.42, wspace=.22, height_ratios=[1.5, 1])
    cols = dict(zip(FORMS, sns.color_palette('mako', as_cmap=True)(
        np.linspace(.12, .72, len(FORMS)))))

    for j, which in enumerate(['safe', 'risky']):
        ax = fig.add_subplot(gs[0, j])
        pay = payoffs[payoffs.which == which]
        v = np.repeat(np.log(pay.payoff.values), pay.n.values)
        # The safe option takes only five discrete values, so a density
        # histogram there is five spikes 6x taller than any prior curve and the
        # comparison of SHAPES -- which is the point -- becomes unreadable.
        # Scale both to peak 1 instead; the y axis is then explicitly relative.
        h, edges = np.histogram(v, bins=40, density=True)
        ax.bar(edges[:-1], h / h.max(), width=np.diff(edges), align='edge',
               color='0.85', edgecolor='white', lw=.3, zorder=1)
        grid = np.linspace(v.min() - 1.2, v.max() + 1.2, 400)
        for _, r in pr[pr.which == which].iterrows():
            dens = np.exp(-.5 * ((grid - r.mu) / r.sd) ** 2)
            ax.plot(grid, dens, color=cols.get(r.form, '0.4'), lw=.8, alpha=.75,
                    zorder=3)
        ax.set_ylim(0, 1.28)
        ax.axvline(v.mean(), color='0.45', lw=.8, ls='--', zorder=2)
        ax.text(v.mean() + .05, 1.26, 'Mean of presented', fontsize=6,
                color='0.45', va='top')
        ticks = [7, 14, 28, 56, 112]
        ax.set_xticks(np.log(ticks))
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel('Payoff (CHF, log axis)')
        ax.set_ylabel('Relative frequency /\ndensity (peak = 1)')
        ax.set_title(f'{which.capitalize()} option', fontsize=8, color='0.15')
        if j == 0:
            for i, f in enumerate(FORMS):
                if f in set(pr.form):
                    ax.text(.015, .97 - i * .05, FORM_LABEL[f], color=cols[f],
                            transform=ax.transAxes, fontsize=6.5, va='top')

        # how far off, in the model's own units, model by model
        ax = fig.add_subplot(gs[1, j])
        d = pr[pr.which == which].copy()
        d['pos'] = d.placement.map({p: i for i, p in enumerate(PLACEMENTS)})
        d = d.sort_values(['pos', 'form'])
        x = np.arange(len(d))
        ax.axhline(v.mean(), color='0.45', lw=.8, ls='--', zorder=0)
        ax.axhline(v.std(), color='#C97B2E', lw=.8, ls=':', zorder=0)
        ax.errorbar(x, d.mu, yerr=[d.mu - d.mu_lo, d.mu_hi - d.mu], fmt='o',
                    ms=2.6, color='0.2', elinewidth=.6, capsize=0, lw=0)
        ax.errorbar(x, d.sd, yerr=[d.sd - d.sd_lo, d.sd_hi - d.sd], fmt='s',
                    ms=2.6, color='#C97B2E', elinewidth=.6, capsize=0, lw=0)
        # one tick per PLACEMENT block, not per model: 48 rotated model names
        # at 4.6 pt is ink, not information
        blocks = d.groupby('placement', sort=False).apply(
            lambda g: x[d.placement.values == g.name].mean())
        for b in d.placement.drop_duplicates().index[1:]:
            ax.axvline(list(d.index).index(b) - .5, color='0.9', lw=.6, zorder=0)
        ax.set_xticks(blocks.values)
        ax.set_xticklabels(blocks.index, rotation=45, ha='right', fontsize=6)
        ax.set_xlim(-1, len(d))
        ax.set_ylabel('Prior μ and σ\n(log CHF)')
        ax.set_ylim(0, 3.9)
        ax.text(.005, .30, 'μ  (dashed: mean of presented)', color='0.2',
                transform=ax.transAxes, fontsize=6, va='top')
        ax.text(.005, .18, 'σ  (dotted: SD of presented)', color='#C97B2E',
                transform=ax.transAxes, fontsize=6, va='top')

    sns.despine(fig=fig, offset=3)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# one page per placement
# --------------------------------------------------------------------------
def page_placement(pdf, place, curves, rung, delta, stake_ppc, loo, ess,
                   space, stamp):
    fig = plt.figure(figsize=PAGE)
    header(fig, f'{PLACE_LABEL[place]}  ·  {place}', FAMILY[place], stamp)
    key_block(fig, orders=False)
    forms = [f for f in FORMS if len(curves[(curves.space == space)
                                            & (curves.form == f)])]
    # top leaves room for the three-line column headers UNDER the key block
    gs = fig.add_gridspec(4, len(forms), left=.075, right=.985, top=.755,
                          bottom=.065, hspace=.44, wspace=.24)

    c_all = curves[(curves.space == space) & (curves.placement == place)]
    # explicitly the two stimulation conditions: 'delta' and 'delta_pct' are
    # also rows here, and the percentage one runs to ~80
    ymax = 1.06 * curves[(curves.space == space)
                         & curves.condition.isin(['ips', 'vertex'])
                         & curves.channel.isin(['n1', 'n2'])].hi.max()
    dall = curves[(curves.space == space) & (curves.condition == 'delta')]
    dm = 1.1 * max(abs(dall.lo.min()), abs(dall.hi.max()))
    unit = 'log CHF' if space == 'log' else 'CHF'

    nullbase = loo.loc[loo.placement == NULL_OF[place]].set_index('form')['elpd_loo'] \
        if len(loo) else pd.Series(dtype=float)

    for j, form in enumerate(forms):
        lbl = f'{space}-{form}-{place}'
        c = c_all[c_all.form == form]

        # -- column header: what the model is, and whether to trust it -----
        ax = fig.add_subplot(gs[0, j])
        e = ess[ess.label == lbl]
        l_ = loo[loo.label == lbl]
        bits = [f'{FORM_LABEL[form]}  ·  {FORM_SUB[form]}']
        if len(l_):
            r = l_.iloc[0]
            gain = (r.elpd_loo - nullbase.get(form, np.nan))
            bits.append(f'ELPD {r.elpd_loo:,.0f} ± {r.se:.0f}'
                        + ('' if np.isnan(gain) else f'   vs null {gain:+.1f}'))
        if len(e):
            r = e.iloc[0]
            bits.append(f'r̂ {r.max_rhat_group:.3f} · ESS {r.min_ess_group:.0f} · '
                        f'{r.divergences:.0f} div')
        ax.set_title('\n'.join(bits), fontsize=6.4, pad=3, linespacing=1.6,
                     color='0.15' if (len(e) and e.iloc[0].ok_group) else '#8c2d04')

        # -- row 1: the noise function -------------------------------------
        if not len(c):
            blank(ax)
        else:
            for chan, ls in [('n1', '-'), ('n2', (0, (2.4, 1.3)))]:
                for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
                    s = c[(c.channel == chan) & (c.condition == cond)].sort_values('x')
                    ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.15, lw=0)
                    ax.plot(s.x, s['mid'], color=col, ls=ls, lw=1.1)
            ax.set_ylim(0, ymax)
            logx(ax, labels=False)
            if j == 0:
                ax.set_ylabel(f'σ ({unit})')
                # direct labels rather than a legend, in the panel's own corner
                ax.text(.04, .97, 'IPS', color=IPS, transform=ax.transAxes,
                        va='top', fontsize=6.5)
                ax.text(.04, .84, 'Vertex', color=VERTEX, transform=ax.transAxes,
                        va='top', fontsize=6.5)

        # -- row 2: the cTBS effect on it ----------------------------------
        ax = fig.add_subplot(gs[1, j])
        if not len(c):
            blank(ax)
        else:
            ax.axhline(0, color='0.75', lw=.5, ls='--', zorder=0)
            for chan, ls, col in [('n1', '-', N2), ('n2', (0, (2.4, 1.3)), N1)]:
                s = c[(c.channel == chan) & (c.condition == 'delta')].sort_values('x')
                ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.20, lw=0)
                ax.plot(s.x, s['mid'], color=col, ls=ls, lw=1.1)
            ax.set_ylim(-dm, dm)
            logx(ax)
            ax.set_xlabel('Payoff (CHF)')
            if j == 0:
                ax.set_ylabel('Δσ, IPS − vertex')

        # -- rows 3-4: the three-way interaction, one row per ORDER --------
        # cTBS is the colour, stake is x, and order is the row. The claim the
        # model has to reproduce is a red/green gap that opens with stake in the
        # bottom row and not in the top one -- readable by looking down a column.
        for k, order in enumerate(['Risky first', 'Risky second']):
            ax = fig.add_subplot(gs[2 + k, j])
            s = stake_ppc[(stake_ppc.label == lbl) & (stake_ppc.order == order)]
            if not len(s):
                blank(ax)
                continue
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                o = s[s.stim == stim].sort_values('stake_chf')
                ax.fill_between(o.stake_chf, o.lo, o.hi, color=col, alpha=.20, lw=0)
                ax.plot(o.stake_chf, o.model, color=col, lw=1.0)
                ax.errorbar(o.stake_chf, o.observed, yerr=o.observed_sem, fmt='o',
                            ms=3.0, color=col, lw=0, elinewidth=.8, capsize=0,
                            zorder=4)
            ax.axhline(.5, color='0.85', lw=.5, ls='--', zorder=0)
            ax.set_ylim(.40, .74)
            ax.set_yticks([.45, .55, .65])
            ax.set_xscale('log')
            ax.set_xticks(sorted(s.stake_chf.unique()))
            ax.set_xticklabels([f'{v:.0f}' for v in sorted(s.stake_chf.unique())])
            ax.minorticks_off()
            if k == 1:
                ax.set_xlabel('Stake (CHF)')
            if j == 0:
                ax.set_ylabel(f'P(risky)\n{order.lower()}')

    fig.text(.5, .018,
             'Rows: the fitted noise function · the cTBS effect on it · choices '
             'when the risky option came first · when it came second.   The '
             'result to look for is a red-above-green gap present at EVERY '
             'stake in the bottom row and absent in the top one: the effect is '
             'order-specific, and roughly flat in stake.',
             ha='center', fontsize=6, color='0.4')
    sns.despine(fig=fig, offset=2)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
def load(data_dir, space):
    dd = Path(data_dir)
    curves = pd.read_csv(dd / 'anchor_curves.tsv', **READ)

    def cat(pattern, cols):
        """Concatenate a glob, or an EMPTY frame that still has the columns --
        every panel below filters on `label`, and a bare empty DataFrame would
        raise instead of drawing the 'no fit yet' placeholder."""
        fs = sorted(glob(str(dd / pattern)))
        if fs:
            return pd.concat([pd.read_csv(f, **READ) for f in fs],
                             ignore_index=True)
        return pd.DataFrame(columns=cols)

    rung = cat('ppc_anchor/ppc_anchor.rung.*.tsv',
               ['label', 'order', 'rung', 'stim', 'model', 'lo', 'hi',
                'observed', 'observed_sem', 'frac', 'stake_chf'])
    stake_ppc = cat('ppc_anchor/ppc_anchor.stake.*.tsv',
                    ['label', 'order', 'stake_bin', 'stim', 'model', 'lo', 'hi',
                     'observed', 'observed_sem', 'frac', 'stake_chf'])
    delta = cat('ppc_anchor/ppc_anchor.delta_stake.*.tsv',
                ['label', 'order', 'stake_bin', 'model', 'lo', 'hi', 'p_gt0',
                 'observed', 'observed_sem', 'frac', 'stake_chf'])
    loo = cat('loo_anchor/loo.*.tsv',
              ['label', 'space', 'form', 'placement', 'memory_model', 'n_par',
               'elpd_loo', 'se', 'p_loo', 'frac_k_gt_07', 'max_k'])
    priors = (pd.read_csv(dd / 'anchor_priors.tsv', **READ)
              if (dd / 'anchor_priors.tsv').exists()
              else pd.DataFrame(columns=['label', 'space', 'form', 'placement',
                                         'which', 'mu', 'sd']))
    payoffs = (pd.read_csv(dd / 'anchor_payoffs.tsv', **READ)
               if (dd / 'anchor_payoffs.tsv').exists()
               else pd.DataFrame(columns=['which', 'subject', 'payoff', 'n']))
    ess = (pd.read_csv(dd / 'anchor_ess.tsv', **READ)
           if (dd / 'anchor_ess.tsv').exists()
           else pd.DataFrame(columns=['label', 'space', 'form', 'placement',
                                      'divergences', 'max_rhat_group',
                                      'min_ess_group', 'ok_group']))
    for df in (loo, ess):
        if len(df):
            df.drop(df.index[df.space != space], inplace=True)
    return curves, rung, delta, stake_ppc, loo, ess, priors, payoffs


def main(data_dir, out_pdf, space):
    curves, rung, delta, stake_ppc, loo, ess, priors, payoffs = load(
        data_dir, space)
    n = curves[curves.space == space].label.nunique()
    stamp = (f'{n} models · anchors from the trace stamps · '
             f'PRIOR_SPEC v1-2026-08-28')
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        page_ladder(pdf, loo, ess, space, stamp)
        page_contact_noise(pdf, curves, space, stamp)
        page_contact_ppc(pdf, delta, space, curves[curves.space == space], stamp)
        page_priors(pdf, priors, payoffs, space, stamp)
        for place in PLACEMENTS:
            if len(curves[(curves.space == space) & (curves.placement == place)]):
                page_placement(pdf, place, curves, rung, delta, stake_ppc, loo,
                               ess, space, stamp)
    print(f'wrote {out_pdf}')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--space', default='log', choices=['log', 'chf'])
    ap.add_argument('--out_pdf', default=None)
    a = ap.parse_args()
    out = a.out_pdf or str(REPO / 'notes/figures'
                           / f'anchor_results_{"log" if a.space == "log" else "natural"}.pdf')
    main(a.data_dir, out, a.space)
