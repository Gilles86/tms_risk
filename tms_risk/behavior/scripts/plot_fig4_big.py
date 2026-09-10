"""Figure 4, full page: what cTBS did to the noise, and everything that follows.

The old Figure 5 does not survive the anchor refit -- its per-option perceived
value panels show the two halves of a ratio separately, which is not what drives
choice, and its bottom row spends a whole axis on a two-point difference. So the
mechanism moves here, and Figure 4 carries the whole argument in four rows:

Row 1  the two fitted noise terms, and the cTBS effect on each. This is the
       measurement: free parameters ARE the noise SD at named payoffs.
Row 2  the observer that noise sits inside -- where its priors are -- and the
       two things cTBS does to the decision variable. The choice index is
       (perceived log ratio + log p) / decision SD, and cTBS moves both parts;
       they compete, which is why the consequence in row 3 is smaller than
       either.
Row 3  the consequence for choice, and the model comparison.

`--ppc psychometric` swaps row 3 for the full psychometric function in three
stake terciles x two orders. That version verifies the SHAPE -- a bias statistic
can be matched by a model whose slope is wrong -- but it is the worse main-text
panel and belongs in the supplement. The cTBS effect is the same size either way
(model IPS-vertex separation 0.0124 vs 0.0121), but splitting six ways widens
the band from 0.044 to 0.079 and the sigmoid forces a 0-1 axis, so the contrast
falls from 3.6% to 1.4% of the panel height: 2.6x less legible against 1.8x more
noise.

Columns 3 and 4 are risky-first and risky-second in row 2, and columns 1 and 2
in row 3, so a presentation order can be followed across the page.

The probit panels this figure briefly carried (slope and risk-neutral
probability, model against data) are behind --with_probit and are OFF by
default. The model's choice function is NOT a probit in log(frac): nu depends
on payoff and the risky payoff is frac * n_safe, so w_R and diff_sd both move
along the ladder and the index is not affine in log frac. The closed form
linearises at each cell's mean payoffs. Measured at the fitted parameters, that
costs up to 3.7 percentage points of choice probability and a 14% error in the
slope -- concentrated at risky-second, high payoff, which is exactly the
order x payoff contrast the paper claims. P(chose risky) needs no linearisation
on either side, so panels h and i carry the consequence instead.

    python -m tms_risk.behavior.scripts.plot_fig4_big --model_label log-power-n1n2
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
#: modes that stack SEVERAL views of the SAME choices inside h and i. Each
#: entry lists the rows, top to bottom.
#:
#: `slope` is the psychometric SLOPE per stake bin -- the direct signature of a
#: noise change, because noise is what flattens a psychometric function. The
#: P(risky) rows show the CONSEQUENCE for choice; only the slope row shows that
#: what changed is discriminability rather than preference.
SPLIT_ROWS = {
    'safe_ratio':       ('safe', 'ratio'),
    'safe_stake':       ('safe', 'stake'),
    'slope_stake':      ('slope', 'stake'),
    'slope_stake_safe': ('slope', 'stake', 'safe'),
}
SPLIT_PPC = tuple(SPLIT_ROWS)


def glyph_key(ax, entries, x=.04, y=.96, dy=.085, seg=.075, fs=6.0):
    """Inline legend drawn as REAL GLYPHS, never as the word for a glyph.

    "Pale: payoffs shown", "Whiskers: 95% CrI", "Dots: observed" all make the
    reader hold a word->ink mapping in their head while looking somewhere else,
    and the word is always a worse description of the mark than the mark is.
    So draw a miniature of the actual mark -- same colour, style, alpha, cap --
    beside its label. Only quantities with no visual form (what a p means, an n)
    stay as words.

    entries: (label, colour, kind, opts); kind in
      'line' | 'bar' | 'band' | 'whisker' | 'marker'.
    """
    for i, (lab, col, kind, o) in enumerate(entries):
        yy = y - i * dy
        tf = ax.transAxes
        if kind == 'band':
            ax.add_patch(plt.Rectangle((x, yy - .020), seg, .040, transform=tf,
                                       facecolor=col, alpha=o.get('alpha', .20),
                                       lw=0, clip_on=False, zorder=5))
        elif kind == 'whisker':
            ax.plot([x, x + seg], [yy] * 2, transform=tf, color=col,
                    lw=o.get('lw', 1.1), clip_on=False, zorder=5)
            for xe in (x, x + seg):
                ax.plot([xe] * 2, [yy - .020, yy + .020], transform=tf,
                        color=col, lw=o.get('lw', 1.1), clip_on=False, zorder=5)
        elif kind == 'marker':
            ax.plot(x + seg / 2, yy, o.get('marker', 'o'), transform=tf,
                    ms=o.get('ms', 3.6), color=col, clip_on=False, zorder=5)
        else:
            ax.plot([x, x + seg], [yy] * 2, transform=tf, color=col,
                    ls=o.get('ls', '-'),
                    lw=o.get('lw', 2.8 if kind == 'bar' else 1.4),
                    alpha=o.get('alpha', 1), solid_capstyle='butt',
                    clip_on=False, zorder=5)
        ax.text(x + seg + .030, yy, lab, transform=tf, color=o.get('tc', '0.35'),
                fontsize=fs, va='center', zorder=5)
IPS, VERTEX = '#d62728', '#2ca02c'
MEM, PERC = '0.25', '0.25'
FIRST_C, SECOND_C = '0.62', '0.15'
RISKY, SAFE = '#8172B2', '0.25'
P_RISKY = 0.55
#: Presentation order is DISPLAYED by the option that came FIRST, matching the
#: manuscript. The data key stays `Risky second` -- it is what `risky_first`
#: maps to throughout the repo and what every TSV contains -- and only the
#: drawn text changes. They are the same trials: if the risky option came
#: second, the safe one came first.
ORDER_LABEL = {'Risky first': 'Risky first', 'Risky second': 'Safe first'}
ORDERS = ['Risky first', 'Risky second']
STAKE = {0: 'Low', 1: 'High'}

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
XT = [7, 14, 28, 56, 112]


def logx(ax, ticks=XT):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def band(ax, s, col, ls='-', lw=1.3, alpha=.18):
    s = s.sort_values('x')
    ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=alpha, lw=0)
    ax.plot(s.x, s['mid'], color=col, ls=ls, lw=lw)


def sig_strip(ax, s, col=None):
    """Black bar spanning where the 95% CrI on the difference excludes zero.

    A rug of ticks reads as data; a single solid bar reads as an annotation,
    which is what it is. Contiguous runs are drawn as separate segments so a
    gap in credibility stays visible.
    """
    if 'p_gt0' not in s.columns or not len(s):
        return
    s = s.sort_values('x')
    # a channel with no cTBS regressor has delta identically 0; p_gt0 is then
    # degenerate and would mark the whole range as 'credible'
    if float(np.abs(s['mid']).max()) < 1e-9:
        return
    # ONE-SIDED. cTBS is a disruption protocol, so the hypothesis is
    # directional: noise goes up, and a credible DECREASE would have no
    # interpretation. p_gt0 is already the posterior mass above zero, so
    # the directional criterion is simply P(delta > 0) > .95.
    m = (s.p_gt0 > .95).values
    if not m.any():
        return
    x = s.x.values
    y = ax.get_ylim()[1] * .93
    edges = np.flatnonzero(np.diff(m.astype(int)) != 0) + 1
    for run in np.split(np.arange(len(m)), edges):
        if m[run[0]]:
            ax.plot(x[[run[0], run[-1]]], [y, y], color='0.1', lw=2.2,
                    solid_capstyle='butt')


def load_probit(dd, label, observed_tsv):
    """Model and observed probit parameters, per subject, on matched axes."""
    f = dd / 'probit_derived' / f'probit_subject.{label}.tsv'
    if not f.exists():
        return None, None
    m = pd.read_csv(f, **READ)
    m = (m[m.parameter.isin(['slope', 'rnp'])]
         .pivot_table(index=['subject', 'order', 'stake2',
                             'stimulation_condition'],
                      columns='parameter', values='median').reset_index())
    # extract_anchor_probit reports p_R * frac*, the EV ratio at indifference;
    # the observed fit reports 1/frac*, the risk-neutral probability itself.
    # Same number from opposite sides: RNP = p_R / (p_R frac*).
    m['rnp'] = P_RISKY / m['rnp']
    o = pd.read_csv(observed_tsv or dd / 'probit_observed_subject.tsv', **READ)
    return m, o


def probit_panel(ax, mod, obs, par, order, show_y):
    xs = np.array([0., 1.])
    dx = .055
    for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
        off = -dx if stim == 'vertex' else dx
        for src, kind in [(mod, 'line'), (obs, 'point')]:
            g = src[(src.order == order) & (src.stimulation_condition == stim)]
            g = g.groupby('stake2')[par].agg(['mean', 'sem']).reindex([0, 1])
            if kind == 'line':
                ax.plot(xs + off, g['mean'].values, color=col, lw=1.3, zorder=2)
                ax.fill_between(xs + off, g['mean'] - g['sem'],
                                g['mean'] + g['sem'], color=col, alpha=.20,
                                lw=0, zorder=1)
            else:
                ax.errorbar(xs + off, g['mean'].values, yerr=g['sem'].values,
                            fmt='o', ms=3.6, color=col, lw=0, elinewidth=.9,
                            capsize=0, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([STAKE[0], STAKE[1]])
    ax.set_xlim(-.42, 1.42)
    if not show_y:
        ax.set_yticklabels([])


def _two_panel_span(fig, ax_left, ax_right):
    """Bounding box covering two side-by-side axes, for a single wide panel."""
    fig.canvas.draw()
    a, b = ax_left.get_position(), ax_right.get_position()
    from matplotlib.transforms import Bbox
    return Bbox.from_extents(a.x0 + .012, a.y0, b.x1, a.y1)


def main(data_dir, out_stem, label, observed_tsv, with_probit=False,
         ppc_kind='safe', bids_folder='/data/ds-tmsrisk'):
    dd = Path(data_dir)
    c = pd.read_csv(dd / 'anchor_curves.tsv', **READ)
    c = c[c.label == label]
    pri = pd.read_csv(dd / 'anchor_priors.tsv', **READ)
    pri = pri[pri.label == label]
    # Per-session priors, when the model has them. `anchor_priors.tsv` stores mu
    # in LOG units; this table stores it in CHF -- do not log it twice.
    _pcf = dd / 'anchor_priors_by_condition.tsv'
    pcond = pd.read_csv(_pcf, **READ) if _pcf.exists() else None
    if pcond is not None:
        pcond = pcond[pcond.label.isin([label, label.split('.')[0]])]

    prior_varies = bool(pcond is not None and len(pcond)
                        and (pcond.varies == 1).any())

    def _logmu(which, cond):
        """Group prior mean in LOG CHF, per condition when the model has one."""
        if pcond is not None:
            q_ = pcond[(pcond.which == which) & (pcond.kind == 'mu')
                       & (pcond.condition == cond)]
            if len(q_):
                return float(np.log(q_['mid'].iloc[0]))
        r_ = pri[pri.which == which]
        return float(r_['mu'].iloc[0]) if len(r_) else np.nan
    pay = pd.read_csv(dd / 'anchor_payoffs.tsv', **READ)
    dff = dd / f'decision_function/decision_function.{label}.tsv'
    dfun = pd.read_csv(dff, **READ) if dff.exists() else None
    fname = 'stakerung' if ppc_kind == 'psychometric2' else 'stake3rung'
    psyf = dd / f'ppc_anchor/ppc_anchor.{fname}.{label}.tsv'
    psy = pd.read_csv(psyf, **READ) if psyf.exists() else None
    safef = dd / f'ppc_anchor/ppc_anchor.safe.{label}.tsv'
    absf = dd / f'ppc_anchor/ppc_anchor.stake.{label}.tsv'
    # 'stake' asks for the same curve against the trial's STAKE -- the mean of
    # the two payoffs, binned into within-participant terciles -- rather than
    # against the safe payoff. Same quantity, a different slice of the design:
    # the safe payoff moves with the ladder, the stake with how much is at
    # issue on the trial.
    if ppc_kind == 'stake' and absf.exists():
        pp, XKEY, XLAB = pd.read_csv(absf, **READ), 'stake_bin', 'Stake (CHF)'
        ppc_kind = 'safe'
    elif safef.exists():
        pp, XKEY, XLAB = pd.read_csv(safef, **READ), 'n_safe', 'Safe payoff (CHF)'
    elif absf.exists():
        pp, XKEY, XLAB = pd.read_csv(absf, **READ), 'stake_chf', 'Stake (CHF)'
    else:
        pp, XKEY, XLAB = None, 'n_safe', 'Safe payoff (CHF)'
    if ppc_kind.startswith('psychometric') and psy is None:
        ppc_kind = 'safe'
    # 'ratio' plots the psychometric function itself -- P(risky) against the
    # payoff ratio, exactly Figure 3a's x-axis -- rather than collapsing the
    # ladder into one number per safe payoff. The collapsed version hides where
    # a misfit lives along the curve, and the misfit here is real: every model
    # in the family is about 1.7x worse on risky-SECOND trials than on
    # risky-first (RMSE .032-.035 vs .019-.023).
    rungf = dd / f'ppc_anchor/ppc_anchor.rung.{label}.tsv'
    rung = pd.read_csv(rungf, **READ) if rungf.exists() else None
    if ppc_kind in ('ratio', 'safe_ratio', 'safe_stake') and rung is None:
        ppc_kind = 'safe'
    dltf = dd / f'ppc_anchor/ppc_anchor.delta_stake.{label}.tsv'
    dlt = pd.read_csv(dltf, **READ) if dltf.exists() else None
    if ppc_kind == 'delta' and dlt is None:
        ppc_kind = 'safe'
    # 'slope'  pools participants inside a cell (attenuated, see the extractor)
    # 'slope2' fits the slope per participant and averages (not attenuated)
    _sk = 'slope2' if ppc_kind == 'slope2' else 'slope'
    stf = dd / f'ppc_anchor/ppc_stats.{label}.tsv'
    sta = pd.read_csv(stf, **READ) if stf.exists() else None
    if ppc_kind == 'stats' and sta is None:
        ppc_kind = 'safe'
    slpf = dd / f'ppc_anchor/ppc_anchor.{_sk}.{label}.tsv'
    slp = pd.read_csv(slpf, **READ) if slpf.exists() else None
    if (ppc_kind.startswith('slope') or ppc_kind in SPLIT_PPC) and slp is None:
        ppc_kind = 'safe'
    loo = pd.concat([pd.read_csv(f, **READ)
                     for f in glob.glob(str(dd / 'loo_anchor/loo.*.tsv'))],
                    ignore_index=True)
    mprob, oprob = load_probit(dd, label, observed_tsv) if with_probit else (None, None)

    shared = set(c.channel) >= {'perc', 'mem'}
    CH = ([('mem', 'Memory term'), ('perc', 'Perceptual term')] if shared else
          [('n1', 'First-presented option'),
           ('n2', 'Second-presented option')])

    nrow = 3
    # the psychometric rows carry a sigmoid, which needs less height than the
    # noise curves to be read; squeezing them keeps the page from running long
    hr = [1, 1, 1]
    tight = True
    # 'safe_ratio' shows the same choices twice -- against the safe payoff and
    # against the ratio -- so it gets a real fourth ROW rather than a
    # subgridspec inside the third. Nesting collapsed the sub-axes and left
    # the parameter panel stretched over dead space.
    if ppc_kind in SPLIT_PPC:
        _n = len(SPLIT_ROWS[ppc_kind])
        nrow, hr = 2 + _n, [1, 1] + [.52] * _n
    fig = plt.figure(figsize=(7.25, (2.15 if tight else 2.42) * sum(hr)),
                     constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=.02 if tight else .045, h_pad=.03 if tight else .05,
        wspace=.02 if tight else .07, hspace=.06 if tight else .12)
    # left column carries a y-label AND tick labels in every row, the
    # right two carry less, so equal grid columns render unequal axes
    gs = fig.add_gridspec(nrow, 12, height_ratios=hr,
                          width_ratios=[.98] * 4 + [.96] * 4 + [.99] * 4)

    def _unaff(chan):
        d_ = c[(c.channel == chan) & (c.condition == 'delta')]
        return bool(len(d_)) and float(np.abs(d_['mid']).max()) < 1e-9

    # ALWAYS three panels in the top row, so every row shares the same column
    # boundaries. Four 3-column panels above three 4-column ones line up
    # nowhere, constrained_layout cannot form gutters, and the row-1 x-labels
    # end up on top of the row-2 titles. The two difference curves share units
    # anyway, so one axis for both is also the better plot.
    ROW1 = ['a', 'b', 'c']
    AX = {}
    for i, k in enumerate(ROW1):
        # a and b are the SAME quantity (nu, log units) for the two channels,
        # so they share a scale. Without it the reader cannot see that the
        # second-presented option is about half as noisy at 7 CHF, which is
        # what makes the effect there large in relative terms. c is a
        # difference and keeps its own scale.
        AX[k] = fig.add_subplot(gs[0, i * 4:(i + 1) * 4],
                                sharey=AX['a'] if k == 'b' else None)
    # Panel d carries two questions -- WHERE the priors sit on the payoff axis,
    # and WHETHER cTBS moved them -- and cramming both into one axis (pale
    # strip, mu +/- sigma bar, CrI whiskers, split IPS/vertex bars, numbers,
    # deltas) made it unreadable. Split it: the payoff axis shows the posterior
    # MEAN of mu and sigma only, and a short strip underneath carries the
    # uncertainty on the cTBS difference.
    AX['e'] = fig.add_subplot(gs[1, 0:4])     # where the priors sit
    # the uncertainty strip is an INSET of that axis, not a nested gridspec:
    # nesting inside constrained_layout breaks the column alignment of the
    # whole row and puts every title on top of the row above
    # ...and it is created ONLY when the model has per-session priors. For a
    # model that holds them fixed the strip was an empty band with a sentence
    # in it, which is dead space plus a note saying what the absence of red and
    # green bars already says.
    # Created ONLY when the model has per-session priors -- and here, after AX
    # exists. It used to be built at the top of main(), which raised
    # UnboundLocalError for exactly the models it was written for; no such model
    # had been plotted until log-power-percpsd.
    # the cTBS effect on the priors is now printed on panel d's own rows
    AX['e2'] = None
    AX['f'] = fig.add_subplot(gs[1, 4:8])     # mechanism, risky first
    AX['g'] = fig.add_subplot(gs[1, 8:12])    # mechanism, risky second
    AX['p'] = fig.add_subplot(gs[2:nrow, 0:4])  # group-level parameters
    # Model comparison and the predictive checks are NOT in this figure. They
    # answer "which model", not "what did cTBS do", and interleaving the two
    # questions made the figure hard to read. They are the supplementary figure
    # built by `plot_model_comparison_supp.py`.
    NCOL = 2 if ppc_kind == 'psychometric2' else 3
    if False:
        # the panels share both axes: the point is the comparison between
        # them, and six independently scaled panels would destroy it
        P = np.empty((2, NCOL), dtype=object)
        for ri in range(2):
            for ci in range(NCOL):   # not `c` -- that is the curves DataFrame
                P[ri, ci] = fig.add_subplot(
                    gs[2 + ri, ci * 3:(ci + 1) * 3],
                    sharey=P[0, 0] if (ri or ci) else None,
                    sharex=P[0, 0] if (ri or ci) else None)
        AX['h'] = P[0, 0]
        AX['l'] = fig.add_subplot(gs[2:4, NCOL * 3:12])
    P = None
    if ppc_kind in SPLIT_PPC:
        # Two views of the same choices, stacked. Against SAFE PAYOFF the
        # ladder is collapsed, so the cTBS separation is a clean vertical gap
        # at each of the five design levels; against the RATIO it is the
        # psychometric function itself, which shows WHERE along the curve a
        # misfit sits. Neither view alone answers both questions.
        # rows in a column share the x-axis when they plot against the same
        # quantity, so only the bottom row carries ticks and a label
        _same_x = len(set(SPLIT_ROWS[ppc_kind])) and all(
            v in ('slope', 'stake') for v in SPLIT_ROWS[ppc_kind])
        for r_ in range(len(SPLIT_ROWS[ppc_kind])):
            kh = 'h' if r_ == 0 else f'h{r_ + 1}'
            ki = 'i' if r_ == 0 else f'i{r_ + 1}'
            AX[kh] = fig.add_subplot(
                gs[2 + r_, 4:8], sharex=AX['h'] if (_same_x and r_) else None)
            AX[ki] = fig.add_subplot(
                gs[2 + r_, 8:12], sharey=AX[kh],
                sharex=AX['i'] if (_same_x and r_) else None)
    else:
        AX['h'] = fig.add_subplot(gs[2, 4:8])
        # h and i plot the same quantity for the two presentation orders. They
        # MUST share a scale: with independent limits the largest IPS-vertex
        # gap in the figure was drawn 15% smaller than on h's scale, while i's
        # blanked tick labels invited the reader to assume the scales matched.
        AX['i'] = fig.add_subplot(gs[2, 8:12], sharey=AX['h'])


    # -- a, b: the two noise terms ---------------------------------------
    def unaffected(chan):
        d_ = c[(c.channel == chan) & (c.condition == 'delta')]
        return len(d_) and float(np.abs(d_['mid']).max()) < 1e-9

    for k, (chan, nm) in zip('ab', CH):
        ax = AX[k]
        if unaffected(chan):
            # one curve, not two identical ones: this channel carries no cTBS
            # regressor in this model, so IPS and vertex are the same by
            # construction and two overlaid lines would imply a null result
            band(ax, c[(c.channel == chan) & (c.condition == 'vertex')], '0.35')
            ax.text(.04, .95, 'Shared across conditions\n(no cTBS term)',
                    transform=ax.transAxes, va='top', fontsize=5.8, color='0.45',
                    linespacing=1.4)
        else:
            for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
                band(ax, c[(c.channel == chan) & (c.condition == cond)], col)
        logx(ax)
        if not tight:
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.set_title(nm, fontsize=7.5)
        ax.set_xlabel('Payoff (CHF)')
        ax.set_ylabel('Representational noise ν (log units)'
                      if k == 'a' else '')
    key = AX['b'] if unaffected(CH[0][0]) else AX['a']
    # rising bands leave the UPPER left empty; at (.04, .12) these sat inside
    # both bands at 7 CHF
    key.text(.04, .95, 'IPS', color=IPS, transform=key.transAxes, fontsize=7,
             va='top')
    key.text(.04, .86, 'Vertex', color=VERTEX, transform=key.transAxes,
             fontsize=7, va='top')

    # -- c, d: the cTBS effect on each term ------------------------------
    dsel = c[c.condition == 'delta']
    dm = 1.15 * max(abs(dsel.lo.min()), abs(dsel.hi.max())) if len(dsel) else .05
    ax = AX['c']
    ax.axhline(0, color='0.45', lw=1.3, zorder=0)
    strips = []
    for (chan, base), col, ls in zip(CH, (FIRST_C, SECOND_C),
                                     ((0, (3, 1.6)), '-')):
        sd_ = dsel[dsel.channel == chan]
        if not len(sd_) or float(np.abs(sd_['mid']).max()) < 1e-9:
            continue
        # Draw the non-credible stretch at reduced weight. At full weight the
        # curve's sign reversal above ~56 CHF reads as a claim; P(dnu > 0) there
        # is 0.16, so the model has no view on it and the figure should not
        # imply one.
        band(ax, sd_, col, ls=ls, lw=1.0)
        q_ = sd_.sort_values('x')
        cred = (q_.p_gt0 > .95).values if 'p_gt0' in q_ else np.zeros(len(q_), bool)
        if cred.any():
            lo_i, hi_i = np.flatnonzero(cred)[[0, -1]]
            ax.plot(q_.x.values[lo_i:hi_i + 1], q_['mid'].values[lo_i:hi_i + 1],
                    color=col, ls=ls, lw=2.0, zorder=6, solid_capstyle='round')
        strips.append((sd_, col))
        q = sd_.sort_values('x')
        ax.annotate(base.split('-')[0], (q.x.iloc[-1], q['mid'].iloc[-1]),
                    xytext=(4, 0), textcoords='offset points', color=col,
                    fontsize=6.6, va='center', clip_on=False,
                    annotation_clip=False)
    ax.set_ylim(-dm, dm)
    # only now is the top of the axis known; sig_strip anchors to it
    for sd_, col in strips:
        sig_strip(ax, sd_)
    logx(ax)
    ax.set_xlim(6.4, 118)
    ax.set_title('cTBS effect on noise', fontsize=7.5)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Δν, IPS − Vertex')
    # the interval type stays here: the BAR's meaning is the statistical claim,
    # so naming the interval is part of naming the mark, not caption bookkeeping
    glyph_key(ax, [('P(Δν > 0) > 0.95', 'k', 'bar', dict(lw=3.0))],
        x=.04, y=.06, fs=5.8)

    # -- e: where the priors sit -----------------------------------------
    ax = AX['e']
    for i, (which, col) in enumerate([('risky', RISKY), ('safe', SAFE)]):
        r = pri[pri.which == which]
        if not len(r):
            continue
        r = r.iloc[0]
        v = pay[pay.which == which]
        obs_p = np.repeat(v.payoff.values, v.n.values)
        y = 1 - i
        # the payoffs actually shown, as context for where the prior sits
        ax.plot([obs_p.min(), obs_p.max()], [y + .20] * 2, color=col, lw=4,
                alpha=.22, solid_capstyle='butt')
        ax.plot(np.exp(np.mean(np.log(obs_p))), y + .20, 'v', ms=3.5, color=col,
                clip_on=False)
        if i == 0:                       # name the strip once, on the strip
            ax.text(obs_p.max() * .95, y + .38, 'Payoffs shown', ha='right',
                    va='center', fontsize=5.4, color='0.45')
        # posterior MEAN of mu and sigma only. The uncertainty lives in the
        # strip below, so this axis answers one question: where does the prior
        # sit relative to the payoffs people saw?
        cw = pcond[(pcond.label.isin([label, label.split('.')[0]]))
                   & (pcond.which == which)] if pcond is not None else None
        varies = (cw is not None and len(cw)
                  and bool(cw[cw.varies == 1].shape[0]))
        if varies:
            for cond, ccol, dy in (('ips', IPS, +.10), ('vertex', VERTEX, -.10)):
                mu_c = _logmu(which, cond)
                q_ = cw[(cw.kind == 'sd') & (cw.condition == cond)]
                sd_c = float(q_['mid'].iloc[0]) if len(q_) else float(r.sd)
                ax.plot([np.exp(mu_c - sd_c), np.exp(mu_c + sd_c)],
                        [y + dy] * 2, color=ccol, lw=2.6, alpha=.55,
                        solid_capstyle='butt')
                ax.plot(np.exp(mu_c), y + dy, 'o', ms=3.6, color=ccol)
        else:
            ax.plot([np.exp(r.mu - r.sd), np.exp(r.mu + r.sd)], [y] * 2,
                    color=col, lw=3, alpha=.45, solid_capstyle='butt')
            ax.plot(np.exp(r.mu), y, 'o', ms=5, color=col)
        ax.text(.03, y - .34,
                f'{np.exp(r.mu):.0f} CHF  (±σ: {np.exp(r.mu - r.sd):.0f}–'
                f'{np.exp(r.mu + r.sd):.0f})',
                transform=ax.get_yaxis_transform(), fontsize=5.6, color=col,
                ha='left', va='center')
    # the cTBS effect on each prior, printed on the row it belongs to. It used
    # to be a separate inset forest below, which needed its own axis, its own
    # x-label and its own tick row to carry two numbers -- far more furniture
    # than the content justified, and it crowded the panel.
    if pcond is not None:
        for i, (which, col) in enumerate([('risky', RISKY), ('safe', SAFE)]):
            dq = pcond[(pcond.which == which) & (pcond.kind == 'mu')
                       & (pcond.condition == 'delta')]
            if len(dq) and float(dq['mid'].iloc[0]) != 0:
                pv = float(dq['p_gt0'].iloc[0])
                ax.text(.99, (1 - i) + .20,
                        f'Δμ {100 * (np.exp(float(dq["mid"].iloc[0])) - 1):+.0f}%'
                        f'  p {min(pv, 1 - pv):.2f}',
                        transform=ax.get_yaxis_transform(), fontsize=5.6,
                        color=col, ha='right', va='center')
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['Risky', 'Safe'], fontsize=7)
    ax.set_ylim(-.75, 1.55)
    logx(ax)
    ax.set_title('Where the priors sit', fontsize=7.5)
    ax.set_xlabel('Payoff (CHF)')
    # no corner key: panel b already names IPS and vertex in the same palette,
    # and the only other mark is the pale strip, labelled on itself above

    # -- d, lower strip: did cTBS move them? -------------------------------
    # One forest of the IPS - vertex differences the model allows, with a zero
    # line. Separating this from the payoff axis above is the whole point: the
    # question "where is the prior" and the question "did it move" have
    # different units and different uncertainties.
    ax2 = AX['e2']
    rows_ = []
    if ax2 is not None and pcond is not None:
        for which, col in (('risky', RISKY), ('safe', SAFE)):
            cw = pcond[pcond.which == which]
            for kind, sym in (('mu', 'μ'), ('sd', 'σ')):
                dq = cw[(cw.kind == kind) & (cw.condition == 'delta')]
                if len(dq) and float(dq['mid'].iloc[0]) != 0:
                    rows_.append((f'{which.capitalize()} {sym}', col,
                                  float(dq['lo'].iloc[0]),
                                  float(dq['mid'].iloc[0]),
                                  float(dq['hi'].iloc[0]),
                                  float(dq['p_gt0'].iloc[0])))
    if rows_:
        ax2.axvline(0, color='0.45', lw=1.0, zorder=0)
        for k, (lab_, col, lo_, mid_, hi_, p_) in enumerate(rows_):
            yy = len(rows_) - 1 - k
            ax2.plot([lo_, hi_], [yy] * 2, color=col, lw=1.4,
                     solid_capstyle='butt')
            ax2.plot(mid_, yy, 'o', ms=4, color=col)
            # same convention as panel g: the posterior mass on the opposite
            # sign, so the two panels' p's are comparable
            ax2.text(1.02, yy, f'p {min(p_, 1 - p_):.2f}',
                     transform=ax2.get_yaxis_transform(), va='center',
                     fontsize=5.4, color='0.45')
        ax2.set_yticks(range(len(rows_)))
        ax2.set_yticklabels([r_[0] for r_ in rows_[::-1]], fontsize=6)
        ax2.set_ylim(-.7, len(rows_) - .3)
        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel('cTBS effect on prior (IPS − Vertex, log)',
                       fontsize=5.6, labelpad=2)
        ax2.tick_params(labelsize=5.6, length=2, pad=1)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(True)
        ax2.patch.set_alpha(0)

    # -- f, g: the mechanism, per option and in the decision variable ------
    # Four traces. The two SOLID ones are what actually enters the choice: the
    # perceived risky/safe ratio, and the decision SD it is divided by. The two
    # DASHED ones decompose the ratio into what happened to each option, which
    # is where the order-specificity comes from: cTBS makes the SECOND-presented
    # option noisier, a noisier option is pulled toward its own prior, and which
    # prior that is depends on the order. The safe option's trace crossing zero
    # near 12 CHF -- its prior mean -- is the clearest single sign of that.
    OPT_R, OPT_S = '#8172B2', '0.55'
    # NOT red: red means IPS in a, b, g, h and i, and '#C44E52' is
    # indistinguishable from '#d62728' at print size. Near-black for the
    # perceived ratio (the derived quantity), the paper's model-contrast blue
    # for decision noise.
    RATIO_C, NOISE_C = '0.15', '#3B5BA5'
    # These quantities are NONLINEAR in the parameters (perceived value depends
    # on w = sd^2/(sd^2 + nu^2)), so neither the value at the average
    # parameters nor the value at a participant's median parameters is the
    # average value. Two separate corrections, both measured in
    # `notes/analyses/aggregation_check.md`:
    #   * compute per participant, then average -- group-level parameters give
    #     a hypothetical average participant and understate the mean effect by
    #     up to 2.9x in models with a per-subject prior contrast;
    #   * compute per posterior DRAW, then average -- plugging in a
    #     participant's median parameters scores r = 0.977 against the model's
    #     own simulated PPC and flips the sign of the risky-first effect, while
    #     integrating over draws scores r = 0.991 and reproduces it to 0.004.
    # So prefer the draw-integrated table when it exists, and fall back to the
    # plug-in reconstruction only for models it has not been run for.
    mech = None
    mf = dd / f'anchor_mechanism.{label}.tsv'
    if not mf.exists():
        mf = dd / f"anchor_mechanism.{label.split('.')[0]}.tsv"
    if mf.exists():
        mech = pd.read_csv(mf, **READ)
        print(f'  mechanism: draw-integrated, from {mf.name}')
    psf, csf = (dd / 'anchor_priors_subject.tsv',
                dd / 'anchor_curves_subject.tsv')
    if psf.exists() and csf.exists():
        P_ = pd.read_csv(psf, **READ)
        C_ = pd.read_csv(csf, **READ)
        # `.pathfinder` etc. are filename suffixes, not part of the label the
        # trace stamps, so the subject-level tables can be keyed either way
        base = label.split('.')[0]
        P_ = P_[P_.label.isin([label, base])]
        C_ = C_[C_.label.isin([label, base])]
    else:
        P_ = C_ = pd.DataFrame()
    if mech is None and len(P_) and len(C_):
        print('  mechanism: plug-in from per-subject medians (fallback)')
        from tms_risk.behavior.fit_model import get_data
        tr = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
        tr['order'] = tr['risky_first'].map({True: 'Risky first',
                                             False: 'Risky second'})
        rows = []
        for (order_, ns), g_ in tr.groupby(['order', 'n_safe']):
            rf = order_ == 'Risky first'
            chR, chS = ('n1', 'n2') if rf else ('n2', 'n1')
            per = []
            for sub, gs in g_.groupby('subject'):
                Cs = C_[C_.subject == sub]

                def _nu(ch, cond, Cs=Cs):
                    # The shared family stores `perc`/`mem`, not `n1`/`n2`:
                    # sigma_n1 = perc + mem (the first option is also held in
                    # memory), sigma_n2 = perc. Map the request accordingly.
                    def _one(name):
                        q_ = Cs[Cs.channel == name]
                        raw = [t.split('sd')[-1] for t in q_.parameter]
                        ys = q_[f'sigma_{cond}'].values
                        # Weber has a single anchor and its parameter carries no
                        # numeric suffix (`..._sd`), so there is nothing to
                        # interpolate: the channel is constant in payoff.
                        if len(ys) == 1 or any(t == '' for t in raw):
                            const = float(ys[0])
                            return lambda v: np.full_like(np.asarray(v, float),
                                                          const)
                        xs = np.array([float(t) for t in raw])
                        o = np.argsort(xs)
                        return lambda v: np.exp(np.interp(np.log(v),
                                                          np.log(xs[o]),
                                                          np.log(ys[o])))
                    have = set(Cs.channel.unique())
                    if ch in have:
                        return _one(ch)
                    if {'perc', 'mem'} <= have:
                        fp, fm = _one('perc'), _one('mem')
                        return ((lambda v: fp(v) + fm(v)) if ch == 'n1' else fp)
                    raise KeyError(f'no channel {ch} for this model')

                def _pv(w_, k_, cond, sub=sub):
                    r_ = P_[(P_.subject == sub) & (P_.which == w_)
                            & (P_.kind == k_) & (P_.condition == cond)]
                    return float(r_.value.iloc[0])

                post, dsd = {}, {}
                for cond in ('vertex', 'ips'):
                    vR = _nu(chR, cond)(gs.n_risky.values)
                    vS = _nu(chS, cond)(gs.n_safe.values)
                    sR, sS = _pv('risky', 'sd', cond), _pv('safe', 'sd', cond)
                    mR = np.log(_pv('risky', 'mu', cond))
                    mS = np.log(_pv('safe', 'mu', cond))
                    wR = sR ** 2 / (sR ** 2 + vR ** 2)
                    wS = sS ** 2 / (sS ** 2 + vS ** 2)
                    post[(cond, 'r')] = wR * np.log(gs.n_risky.values) + (1 - wR) * mR
                    post[(cond, 's')] = wS * np.log(gs.n_safe.values) + (1 - wS) * mS
                    dsd[cond] = float(np.mean(np.sqrt(vR ** 2 + vS ** 2)))
                dR = float(np.mean(post[('ips', 'r')] - post[('vertex', 'r')]))
                dS = float(np.mean(post[('ips', 's')] - post[('vertex', 's')]))
                per.append((100 * np.expm1(dR), 100 * np.expm1(dS),
                            100 * np.expm1(dR - dS),
                            100 * (dsd['ips'] / dsd['vertex'] - 1)))
            per = np.array(per)
            row = dict(order=order_, n_safe=ns, n_sub=len(per))
            for j, k_ in enumerate(['risky', 'safe', 'ratio', 'noise']):
                row[k_] = per[:, j].mean()
                row[k_ + '_sem'] = per[:, j].std(ddof=1) / np.sqrt(len(per))
                row[k_ + '_sd'] = per[:, j].std(ddof=1)
            rows.append(row)
        mech = pd.DataFrame(rows)

    TRACES = [(OPT_R, 'risky', 1.1, 3.2, (0, (2.5, 1.5)), 'Risky option'),
              (OPT_S, 'safe', 1.1, 3.2, (0, (2.5, 1.5)), 'Safe option'),
              (RATIO_C, 'ratio', 1.9, 4.4, '-', 'Perceived ratio'),
              (NOISE_C, 'noise', 1.9, 4.4, '-', 'Decision noise')]
    if mech is not None:
        keys_ = ['risky', 'safe', 'ratio', 'noise']
        # Every column that actually gets drawn: the band edges where a
        # channel has one (noise, ratio), and the plain value everywhere,
        # including the two DASHED option lines, which have no band. Taking
        # the extent of only the banded columns (the previous form) let the
        # dashed lines run past the axis limits used to size the legend's
        # reserved strip below, which is exactly why the legend collided
        # with them.
        cols = [k_ for k_ in keys_ if k_ in mech]
        cols += [k_ + s for k_ in keys_ for s in ('_hi', '_lo') if k_ + s in mech]
        allv = pd.concat([mech[c_] for c_ in cols])
        hi_, lo_ = float(allv.max()), float(allv.min())
        rng_ = hi_ - lo_
        # a slim reserved strip for the legend, and just enough headroom to
        # clear the topmost band -- the effects are a few percent, so every
        # wasted unit of axis costs visibility
        YL = (lo_ - .02 * rng_ - .10 * rng_, hi_ + .015 * rng_)
    for k, order in zip('fg', ORDERS):
        ax = AX[k]
        ax.axhline(0, color='0.45', lw=1.3, zorder=0)
        if mech is not None:
            q = mech[mech.order == order].sort_values('n_safe')
            x = np.arange(len(q))
            for col, key, lw, ms, ls, _ in TRACES:
                # a channel with no cTBS term is exactly zero, and the ratio is
                # then identical to the other channel: drawing both is drawing
                # the same line twice
                if float(np.abs(q[key]).max()) < 1e-9:
                    continue
                # the 95% CrI on the group mean, not an s.e.m. across
                # participants: this is a model-derived quantity and the band
                # has to be its posterior (CLAUDE.md, "Conventions worth
                # knowing"). `_sem` is still written to the TSV for reference.
                if key + '_lo' in q:
                    ax.fill_between(x, q[key + '_lo'], q[key + '_hi'],
                                    color=col, alpha=.16, lw=0, zorder=1)
                # markers only on traces whose interval actually clears zero
                # somewhere. The two dashed decomposition traces are
                # intermediate quantities and their bands cover zero at every
                # safe payoff; drawn with dots they read as measured points and
                # the eye follows the line instead of the band.
                clears = (key + '_lo' not in q or
                          bool(((q[key + '_lo'] > 0) | (q[key + '_hi'] < 0)).any()))
                if clears:
                    ax.plot(x, q[key], 'o', ms=ms, color=col, zorder=3)
                ax.plot(x, q[key], ls=ls, lw=lw if clears else lw * .85,
                        color=col, alpha=1.0 if clears else .75, zorder=2)
            ax.set_ylim(*YL)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(['7', '10', '14', '20', '28'])
        ax.set_xlabel('Safe payoff (CHF)')
        ax.set_title(ORDER_LABEL[order], fontsize=7.5)
        if k == 'f':
            ax.set_ylabel('cTBS effect (%)')
        else:
            ax.tick_params(labelleft=False)
            if mech is not None:
                from matplotlib.lines import Line2D
                handles = [Line2D([], [], color=col, ls=ls,
                                  lw=1.7 if ls == '-' else 1.2, marker='o',
                                  ms=3.2, label=lab)
                           for col, key, _, _, ls, lab in TRACES
                           if float(np.abs(q[key]).max()) >= 1e-9]
                ax.legend(handles=handles, loc='lower center', ncol=2,
                          fontsize=5.8, frameon=False, handlelength=1.5,
                          handletextpad=.5, columnspacing=1.1,
                          labelspacing=.35, borderaxespad=.15)

    # -- h(,i): the consequence for choice -------------------------------
    if ppc_kind == 'stats':
        # Targeted posterior predictive checks. Every other PPC in this figure
        # plots cells of 12-20 trials per participant, where the point's own
        # standard error is as wide as the model's band and the eye reads
        # sampling noise as misfit. These statistics aggregate over the whole
        # design instead, so each one is a single number with little noise
        # left, and each asks a specific question the paper actually makes a
        # claim about -- rather than asking whether the bulk choice curve is
        # right, which every model in the grid gets right.
        NAMES = {
            'dp_second_mean':   'cTBS effect, risky second',
            'dp_second_high':   '… at the largest stakes',
            'order_contrast':   'Effect is bigger when risky is second',
            'stake_slope_second': 'Effect grows with stake (risky second)',
            'three_way':        'Stake dependence differs by order',
            'slope_contrast':   'cTBS flattens the psychometric curve',
            'slope_second_ctbs': '… on risky-second trials',
        }
        q = [r for r in sta.to_dict('records') if r['statistic'] in NAMES]
        q = sorted(q, key=lambda r: list(NAMES).index(r['statistic']))
        AX['i'].set_visible(False)
        ax = AX['h']
        ax.set_position(_two_panel_span(fig, AX['h'], AX['i']))
        # names go INSIDE the panel, so the axes can start flush against the
        # panel to its left instead of reserving a gutter for tick labels
        y = np.arange(len(q))[::-1]
        xl = min(r['lo'] for r in q) - .012
        xr = max(r['hi'] for r in q) + .075
        for yy, r in zip(y, q):
            ok = bool(r['covered'])
            col = '0.25' if ok else IPS
            ax.plot([r['lo'], r['hi']], [yy, yy], color='0.62', lw=4.0,
                    alpha=.40, solid_capstyle='butt', zorder=1)
            ax.plot(r['model_median'], yy, '|', ms=10, color='0.30', mew=1.5,
                    zorder=2)
            ax.plot(r['observed'], yy, 'o', ms=5.4, color=col, zorder=4)
            ax.text(xl + .004, yy + .30, NAMES[r['statistic']], fontsize=6.6,
                    va='bottom', ha='left', color='0.2')
            ax.text(xr - .004, yy, f"p = {r['ppp']:.3f}", ha='right',
                    va='center', fontsize=6.6,
                    color='0.45' if ok else IPS)
        ax.axvline(0, color='0.45', lw=.9, zorder=0)
        ax.set_yticks([])
        ax.set_xlim(xl, xr)
        ax.set_ylim(-1.5, len(q) - .25)
        ax.set_xlabel('Effect on P(chose risky)')
        ax.set_title('Does the model produce what was measured?', fontsize=8)
        glyph_key(ax, [('Observed', '.25', 'marker', dict(ms=5.4)),
                       ('Model, 95% predictive', '0.62', 'bar',
                        dict(lw=4.0, alpha=.40))],
                  x=.02, y=.10, dy=.075, seg=.055)
    elif ppc_kind in ('slope', 'slope2'):
        # The psychometric SLOPE, split by stake -- the quantity Figure 3
        # reports, so the model and the data are finally the same thing on the
        # same axis. Pooled over stake the slope contrast comes out with the
        # wrong sign, because the fitted cTBS effect ROTATES the noise function
        # (up at low payoffs, down at high) and the high-stake trials dominate.
        # Both sides are the same statistic: a linear-probability slope on
        # log(risky/safe), computed on each posterior draw's simulated choices
        # for the band and on the real choices for the points.
        for k, order in zip(('h', 'i'), ORDERS):
            ax = AX[k]
            o_ = slp[slp.order == order]
            xs = np.arange(o_.stake_bin.nunique())
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                q = o_[o_.stim == stim].sort_values('stake_chf')
                ax.fill_between(xs, q.lo, q.hi, color=col, alpha=.18, lw=0,
                                zorder=1)
                ax.plot(xs, q.slope, color=col, lw=1.4, zorder=3)
                ax.plot(xs, q.observed, 'o', ms=4.4, color=col, zorder=5)
            ax.set_xticks(xs)
            ax.set_xticklabels([f'{v:.0f}' for v in
                                o_.groupby('stake_bin').stake_chf.mean()])
            ax.set_xlim(-.35, len(xs) - .65)
            ax.set_xlabel('Stake (CHF)')
            ax.set_title(ORDER_LABEL[order], fontsize=7.5)
            if k == 'h':
                ax.set_ylabel('Psychometric slope\n(ΔP per log ratio'
                              + (', per participant)' if ppc_kind == 'slope2'
                                 else ', pooled)'))
                ax.text(.05, .12, 'IPS', transform=ax.transAxes, color=IPS,
                        fontsize=7)
                ax.text(.05, .03, 'Vertex', transform=ax.transAxes,
                        color=VERTEX, fontsize=7)
            else:
                ax.tick_params(labelleft=False)
                glyph_key(ax, [('Observed', '.25', 'marker', dict(ms=4.4)),
                               ('95% predictive', '.5', 'band',
                                dict(alpha=.18))],
                          x=.05, y=.14, dy=.095)
    elif ppc_kind == 'delta':
        # Two overlapping psychometric curves are an accurate picture of a
        # model that predicts a ~1-percentage-point separation, and a useless
        # one. The difference, with its own predictive band, is the same
        # information at a resolution where it can be read -- and it shows
        # plainly where the observed effect exceeds what the model predicts.
        for k, order in zip(('h', 'i'), ORDERS):
            ax, o_ = AX[k], dlt[dlt.order == order].sort_values('stake_chf')
            x = np.arange(len(o_))
            ax.axhline(0, color='0.45', lw=.9, zorder=1)
            ax.fill_between(x, 100 * o_.lo, 100 * o_.hi, color='0.55',
                            alpha=.18, lw=0, zorder=2)
            ax.plot(x, 100 * o_.model, color='0.2', lw=1.4, zorder=3)
            ax.errorbar(x, 100 * o_.observed, yerr=100 * o_.observed_sem,
                        fmt='o', ms=4.2, color=IPS, ecolor=IPS, elinewidth=1.0,
                        capsize=2, zorder=5)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{v:.0f}' for v in o_.stake_chf])
            ax.set_xlim(-.45, len(o_) - .55)
            ax.set_ylim(-9, 14)
            ax.set_xlabel('Stake (CHF)')
            ax.set_title(ORDER_LABEL[order], fontsize=7.5)
            if k == 'h':
                ax.set_ylabel('cTBS effect on P(risky)\n(IPS − Vertex, %%points)'
                              .replace('%%', '%'))
            else:
                ax.tick_params(labelleft=False)
                glyph_key(ax, [('Observed ±1 s.e.m.', IPS, 'whisker',
                                dict(lw=1.0)),
                               ('Model', '.2', 'line', dict(lw=1.4)),
                               ('95% predictive', '.55', 'band',
                                dict(alpha=.18))],
                          x=.05, y=.95, dy=.095)
    elif ppc_kind in SPLIT_PPC:
        xs = np.sort(rung.frac.unique()) if rung is not None else np.array([1.])
        stk = (pd.read_csv(dd / f'ppc_anchor/ppc_anchor.stake.{label}.tsv',
                           **READ) if absf.exists() else None)

        def _draw(ax, view, order, first):
            """One row of h/i. `view` names the x-axis and the quantity."""
            if view == 'slope':
                o_ = slp[slp.order == order]
                xk, ycols = 'stake_bin', ('slope', 'lo', 'hi', 'observed')
            elif view == 'stake':
                o_, xk = stk[stk.order == order], 'stake_bin'
                ycols = ('model', 'lo', 'hi', 'observed')
            elif view == 'safe':
                o_, xk = pp[pp.order == order], 'n_safe'
                ycols = ('model', 'lo', 'hi', 'observed')
            else:                                            # 'ratio'
                o_, xk = rung[rung.order == order], 'frac'
                ycols = ('model', 'lo', 'hi', 'observed')
            m_, lo_, hi_, ob_ = ycols
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                q = o_[o_.stim == stim].sort_values(xk)
                ax.fill_between(q[xk], q[lo_], q[hi_], color=col, alpha=.20,
                                lw=0, zorder=1)
                ax.plot(q[xk], q[m_], color=col, lw=1.3, zorder=2)
                ax.plot(q[xk], q[ob_], 'o', ms=3.6, color=col, zorder=4)
            if view == 'slope':
                # a slope of zero is a flat psychometric function: no
                # discrimination at all. It is the meaningful reference here,
                # not 0.5.
                ax.axhline(0, color='0.88', lw=.6, ls='--', zorder=0)
            else:
                ax.axhline(.5, color='0.88', lw=.6, ls='--', zorder=0)
            if view in ('slope', 'stake'):
                lab_ = o_.groupby('stake_bin')['stake_chf'].mean()
                xv = sorted(o_.stake_bin.unique())
                ax.set_xticks(xv)
                ax.set_xticklabels([f'{lab_[v]:.0f}' for v in xv])
                ax.set_xlim(-.35, max(xv) + .35)
                ax.set_xlabel('Stake (CHF)')
                if view == 'slope':
                    ax.set_ylim(0, .82)
                    ax.set_yticks([0, .25, .5, .75])
                else:
                    ax.set_ylim(.40, .74)
                    ax.set_yticks([.45, .55, .65])
            elif view == 'safe':
                ax.set_xscale('log')
                xv = sorted(o_.n_safe.unique())
                ax.set_xticks(xv)
                ax.set_xticklabels([f'{v:.0f}' for v in xv])
                ax.minorticks_off()
                ax.set_ylim(.40, .74)
                ax.set_yticks([.45, .55, .65])
                ax.set_xlabel('Safe payoff (CHF)')
            else:
                ax.axvline(1 / P_RISKY, color='0.75', lw=.7, ls=':', zorder=0)
                ax.set_xscale('log')
                tk = [t for t in (1.5, 2, 2.5, 3) if xs.min() <= t <= xs.max()]
                ax.set_xticks(tk)
                ax.set_xticklabels([f'{t:g}' for t in tk])
                ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
                ax.set_xlim(xs.min() * .93, xs.max() * 1.07)
                ax.set_ylim(.15, .95)
                ax.set_yticks([.25, .5, .75])
                ax.set_xlabel('Risky / safe payoff')
            if first:
                ax.set_ylabel('Psychometric slope' if view == 'slope'
                              else 'P(chose risky)')
            else:
                ax.tick_params(labelleft=False)

        views = SPLIT_ROWS[ppc_kind]
        for c_, (k, order) in enumerate(zip(('h', 'i'), ORDERS)):
            for r_, view in enumerate(views):
                ax = AX[k if r_ == 0 else f'{k}{r_ + 1}']
                _draw(ax, view, order, first=(k == 'h'))
                if _same_x and r_ < len(views) - 1:
                    ax.tick_params(labelbottom=False)
                    ax.set_xlabel('')
                if r_ == 0:
                    # h/i are the only panels showing DATA rather than
                    # parameters, so they say so rather than leaving the reader
                    # to infer it from the presence of markers
                    ax.set_title(f'Posterior predictive\n{ORDER_LABEL[order].lower()}',
                                 fontsize=7.5, linespacing=1.35)
                    if k == 'h':
                        # descending slope curves leave the LOWER left empty;
                        # a rising/flat P(risky) leaves the UPPER left
                        yy = .07 if view == 'slope' else .78
                        glyph_key(ax, [('IPS', IPS, 'line', dict(lw=1.6)),
                                       ('Vertex', VERTEX, 'line', dict(lw=1.6))],
                                  x=.04, y=yy, dy=.15, seg=.10, fs=6.5)
                if r_ == len(views) - 1 and k == 'i':
                    glyph_key(ax, [('Observed', '0.35', 'marker', dict(ms=3.6)),
                                   ('95% predictive', '0.55', 'band',
                                    dict(alpha=.25))],
                              x=.55, y=.13, dy=.11, seg=.09, fs=5.6)
    elif ppc_kind == 'ratio':
        xs = np.sort(rung.frac.unique())
        for k, order in zip(('h', 'i'), ORDERS):
            ax = AX[k]
            o_ = rung[rung.order == order]
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                q = o_[o_.stim == stim].sort_values('frac')
                ax.fill_between(q.frac, q.lo, q.hi, color=col, alpha=.20, lw=0,
                                zorder=1)
                ax.plot(q.frac, q.model, color=col, lw=1.3, zorder=2)
                ax.plot(q.frac, q.observed, 'o', ms=3.6, color=col, zorder=4)
            ax.axvline(1 / P_RISKY, color='0.75', lw=.7, ls=':', zorder=0)
            ax.axhline(.5, color='0.88', lw=.7, ls='--', zorder=0)
            ax.set_xscale('log')
            # the ladder only spans ~1.4-3.2x, so matplotlib's log locator
            # gives '2' and '3 x 10^0'. Label the rungs themselves.
            tk = [t for t in (1.5, 2, 2.5, 3) if xs.min() <= t <= xs.max()]
            ax.set_xticks(tk)
            ax.set_xticklabels([f'{t:g}' for t in tk])
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.set_xlim(xs.min() * .93, xs.max() * 1.07)
            ax.set_ylim(.15, .95)
            ax.set_xlabel('Risky / safe payoff')
            ax.set_title(ORDER_LABEL[order], fontsize=7.5)
            # a RISING psychometric leaves the top-left and bottom-right empty
            if k == 'h':
                ax.set_ylabel('P(chose risky)')
                ax.text(.05, .95, 'IPS', transform=ax.transAxes, color=IPS,
                        fontsize=7, va='top')
                ax.text(.05, .86, 'Vertex', transform=ax.transAxes,
                        color=VERTEX, fontsize=7, va='top')
                ax.text(1 / P_RISKY, .40, 'Risk neutral', fontsize=6,
                        color='0.45', ha='center', va='bottom',
                        rotation=90, rotation_mode='anchor')
            else:
                ax.tick_params(labelleft=False)
                glyph_key(ax, [('Observed', '.25', 'marker', dict(ms=3.6)),
                               ('95% predictive', '.45', 'band',
                                dict(alpha=.20))],
                          x=.52, y=.16, dy=.10)
    elif ppc_kind.startswith('psychometric'):
        slab = psy.groupby('stake_grp')['stake_chf'].mean().round(0).astype(int)
        xs = np.sort(psy.frac.unique())
        for r, order in enumerate(ORDERS):
            for cc, grp in enumerate(sorted(psy.stake_grp.unique())):
                ax = P[r, cc]
                o_ = psy[(psy.order == order) & (psy.stake_grp == grp)]
                for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                    q = o_[o_.stim == stim].sort_values('frac')
                    ax.fill_between(q.frac, q.lo, q.hi, color=col, alpha=.20, lw=0)
                    ax.plot(q.frac, q.model, color=col, lw=1.2)
                    ax.plot(q.frac, q.observed, 'o', ms=3.4, color=col, zorder=4)
                ax.axvline(1 / P_RISKY, color='0.8', lw=.6, ls=':', zorder=0)
                ax.axhline(.5, color='0.88', lw=.6, ls='--', zorder=0)
                ax.set_xscale('log')
                ax.set_xlim(xs.min() * .96, xs.max() * 1.04)
                ax.set_ylim(.12, 1.0)
                ax.set_xticks([1.5, 2, 3])
                ax.set_xticklabels(['1.5', '2', '3'])
                ax.set_yticks([.25, .5, .75, 1])
                ax.minorticks_off()
                if r == 0:
                    ax.set_title(f'Stake ≈ {slab[grp]} CHF', fontsize=7.5)
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel('Risky / safe payoff')
                if cc == 0:
                    ax.set_ylabel(f'P(chose risky)\n{order.lower()}')
                else:
                    ax.tick_params(labelleft=False)
        P[0, 0].text(.05, .95, 'IPS', color=IPS, transform=P[0, 0].transAxes,
                     va='top', fontsize=7)
        P[0, 0].text(.05, .81, 'Vertex', color=VERTEX,
                     transform=P[0, 0].transAxes, va='top', fontsize=7)
        P[1, NCOL - 1].text(.97, .04, '',
                     transform=P[1, NCOL - 1].transAxes, ha='right', va='bottom',
                     fontsize=5.6, color='0.45', linespacing=1.4)
        glyph_key(P[1, NCOL - 1],
            [('Observed', '0.35', 'marker', dict(ms=3.4)),
             ('95% predictive', '0.55', 'band', dict(alpha=.25))],
            x=.62, y=.14, dy=.10, seg=.09, fs=5.6)
    else:
        # Collapsed over the ratio ladder, which is what makes the contrast
        # legible: the same 0.012 model separation occupies 3.6% of this panel
        # against 1.4% of a full-range psychometric panel, and the band is
        # 0.044 rather than 0.079 because no cell is split six ways.
        for k, order in zip('hi', ORDERS):
            ax = AX[k]
            o = pp[pp.order == order] if pp is not None else None
            if o is None or not len(o):
                ax.text(.5, .5, 'no PPC', transform=ax.transAxes, ha='center',
                        color='0.6')
                ax.set_xticks([]); ax.set_yticks([])
            else:
                for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                    q = o[o.stim == stim].sort_values(XKEY)
                    ax.fill_between(q[XKEY], q.lo, q.hi, color=col, alpha=.20,
                                    lw=0)
                    ax.plot(q[XKEY], q.model, color=col, lw=1.3)
                    ax.plot(q[XKEY], q.observed, 'o', ms=3.8, color=col,
                            zorder=4)
                ax.axhline(.5, color='0.88', lw=.6, ls='--', zorder=0)
                ax.set_ylim(.40, .74)
                ax.set_yticks([.45, .55, .65])
                if XKEY == 'stake_bin':
                    # bins are 0/1/2; label them by the mean stake they hold
                    lab_ = o.groupby(XKEY)['stake_chf'].mean()
                    ax.set_xticks(sorted(o[XKEY].unique()))
                    ax.set_xticklabels([f'{lab_[v]:.0f}' for v in
                                        sorted(o[XKEY].unique())])
                    ax.set_xlim(-.3, o[XKEY].max() + .3)
                else:
                    ax.set_xscale('log')
                    ax.set_xticks(sorted(o[XKEY].unique()))
                    ax.set_xticklabels([f'{v:.0f}' for v in
                                        sorted(o[XKEY].unique())])
                    ax.minorticks_off()
            ax.set_xlabel(XLAB)
            ax.set_title(ORDER_LABEL[order], fontsize=7.5)
            if k == 'h':
                ax.set_ylabel('P(chose risky)')
                ax.text(.04, .95, 'IPS', color=IPS, transform=ax.transAxes,
                        va='top', fontsize=7)
                ax.text(.04, .82, 'Vertex', color=VERTEX, transform=ax.transAxes,
                        va='top', fontsize=7)
            else:
                ax.tick_params(labelleft=False)
                glyph_key(ax, [('Observed', '0.35', 'marker', dict(ms=3.6)),
                         ('95% predictive', '0.55', 'band', dict(alpha=.25))],
                    x=.60, y=.15, dy=.10, seg=.09, fs=5.6)
                ax.text(.97, .04, '',
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=5.6, color='0.45', linespacing=1.4)

    for row, par, ylab, keys in []:      # probit panels retired, see docstring
        axs = []
        for kk, order in zip(keys, ORDERS):
            ax = AX[kk]
            axs.append(ax)
            if mprob is None:
                ax.text(.5, .5, 'no probit', transform=ax.transAxes,
                        ha='center', color='0.6')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            probit_panel(ax, mprob, oprob, par, order, show_y=(kk in 'jm'))
            ax.set_xlabel('Stake')
            if kk in 'jm':
                ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(ORDER_LABEL[order], fontsize=7.5)
        if mprob is not None:
            lo = min(a.get_ylim()[0] for a in axs)
            hi = max(a.get_ylim()[1] for a in axs)
            for a in axs:
                a.set_ylim(lo, hi)
                if par == 'rnp':
                    a.axhline(P_RISKY, color='0.75', lw=.6, ls=':', zorder=0)
    if False:
        AX['n'].text(1.02, P_RISKY, ' Risk\n neutral', transform=AX['n'].get_yaxis_transform(),
                     fontsize=5.6, color='0.55', va='center', ha='left',
                     linespacing=1.3)

    # -- p: every group-level parameter on one axis -----------------------
    # The free parameters ARE interpretable quantities -- the noise SD at the
    # two anchor payoffs, and the prior mean and spread -- so they can be shown
    # directly rather than through a link function.
    ax = AX['p']
    rows = []
    for chan, nm in CH:
        for xa in sorted(c.x.unique())[::len(c.x.unique()) - 1]:
            for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
                q = c[(c.channel == chan) & (c.condition == cond)
                      & (np.isclose(c.x, xa))]
                if not len(q):
                    continue
                q = q.iloc[0]
                short = ('first-presented' if nm.startswith('First')
                         else 'second-presented' if nm.startswith('Second')
                         else nm.split()[0].lower())
                lab = f'ν {short}, {xa:.0f} CHF'
                rows.append((lab, q['mid'], q.lo, q.hi,
                             '0.35' if unaffected(chan) else col))
            if unaffected(chan):          # one entry, not two identical ones
                rows = rows[:-1]
    n_noise = len(rows)
    # No priors here: panel e already shows where they sit and how wide they
    # are. Repeating them turns this panel into a table of everything instead
    # of the thing it is for -- the noise parameters, which ARE the noise SD at
    # the two anchor payoffs and can be read straight off the axis.
    yy = np.arange(len(rows))[::-1]
    for y_, (lab, m_, lo_, hi_, col) in zip(yy, rows):
        ax.plot([lo_, hi_], [y_, y_], color=col, lw=1.2, solid_capstyle='round')
        ax.plot(m_, y_, 'o', ms=4, color=col)
    # Bayesian p-value on the IPS - vertex difference, printed once per pair of
    # rows. `p_gt0` is the posterior probability that the difference exceeds
    # zero, so the 95% CrI excludes zero when it is above .975 or below .025.
    xr_ = ax.get_xlim()[1]
    for i, (lab, m_, lo_, hi_, col) in enumerate(rows):
        if i + 1 >= len(rows) or rows[i + 1][0] != lab:
            continue                       # not the first of a condition pair
        anch = lab.split('@')[-1].replace('CHF', '').strip()
        chan = 'n1' if '1st' in lab else 'n2'
        dsel_ = c[(c.channel == chan) & (c.condition == 'delta')]
        if not len(dsel_) or 'p_gt0' not in dsel_:
            continue
        j = int(np.argmin(np.abs(dsel_.x.values - float(anch))))
        pg = float(dsel_.p_gt0.values[j])
        sig = pg > .95          # one-sided; see the note on panel c
        # a dedicated strip INSIDE the axis, not the axis edge: flush right the
        # labels sat in the gutter against the next panel's y-axis
        xend = max(hi_, rows[i + 1][3])
        ax.text(xend + .012, (yy[i] + yy[i + 1]) / 2,
                # the SAME statistic panel c's key names, in the same
                # direction: P(IPS - vertex > 0). Printing 1 - that as "p" made
                # 0.84 read as "no difference" when it means the point estimate
                # is REVERSED (P = 0.16).
                f'P = {pg:.3f}' if sig else f'P = {pg:.2f}',
                clip_on=False, ha='left', va='center', fontsize=6.6,
                color='0.15' if sig else '0.5',
                fontweight='bold' if sig else 'normal')
    # separate the two presentation positions, not the noise/prior blocks --
    # the priors left this panel
    n_first = sum(1 for r_ in rows if '1st' in r_[0] or 'Mem' in r_[0])
    ax.axhline(len(rows) - n_first - .5, color='0.85', lw=.7, zorder=0)

    # The prior MEANS live on a payoff scale, two orders of magnitude away from
    # the SDs, so they get their own axis at the top rather than squashing the
    # rest of the panel. Same rows, same panel, commensurate reading.
    mu_rows = []

    _l, _r = ax.get_xlim()
    ax.set_xlim(_l, _r + .13 * (_r - _l))   # just enough for the P labels
    # each stimulated parameter appears twice, once per condition. Colour
    # already separates them, so the label belongs to the PAIR -- put the tick
    # at its midpoint, not on whichever row happens to come first.
    ticks, labs, i_ = [], [], 0
    while i_ < len(rows):
        if i_ + 1 < len(rows) and rows[i_ + 1][0] == rows[i_][0]:
            ticks.append((yy[i_] + yy[i_ + 1]) / 2); i_ += 2
        else:
            ticks.append(yy[i_]); i_ += 1
        labs.append(rows[i_ - 1][0].replace(' @ ', ', '))
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=6.6)
    # extra room at the bottom purely for the footnote, so it never lands on
    # the lowest pair of bars
    ax.set_ylim(-1.75, len(rows) - .3)
    ax.set_xlabel('ν (log units)')
    ax.set_title('Noise parameters', fontsize=7.5)
    # inside the axes, in the strip reserved above -- an axes-fraction offset
    # BELOW the spine collides with the x-label, which constrained_layout
    # cannot see and therefore cannot move out of the way. x in axes fraction,
    # y in data coordinates.


    # Panel letters sit a fixed distance (in POINTS) left of each axes' own
    # top-left corner, not a fixed axes-FRACTION -- a fraction lands at a
    # different absolute distance on every panel because axes widths differ
    # (they even differ for the same panel between --ppc safe's 3 rows and
    # --ppc safe_stake's 4, since constrained_layout re-solves the margins).
    # That is what let panel a's letter collide with its own rotated y-label
    # in one layout while clearing it in the other. Measuring how far left
    # each axis' rendered y-tick labels + y-label actually reach and adding a
    # fixed points-buffer is layout-agnostic by construction.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _letter_dx_pts(ax, buffer=4.0, lo=-62.0, hi=-6.0):
        x0 = ax.get_window_extent(renderer=renderer).x0
        left_px = x0
        for tick in ax.get_yticklabels():
            if not tick.get_visible() or not tick.get_text():
                continue
            bb = tick.get_window_extent(renderer=renderer)
            if bb.width > 0:
                left_px = min(left_px, bb.x0)
        ylab = ax.yaxis.label
        if ylab.get_visible() and ylab.get_text():
            bb = ylab.get_window_extent(renderer=renderer)
            if bb.width > 0:
                left_px = min(left_px, bb.x0)
        pts = (x0 - left_px) * 72 / fig.dpi + buffer
        return float(np.clip(-pts, lo, hi))

    keys = list(ROW1) + ['e', 'f', 'g', 'p', 'h', 'i']
    for letter, k in zip('abcdefghi', keys):
        if k in AX and AX[k].get_visible():
            ax = AX[k]
            ann = ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                              xytext=(_letter_dx_pts(ax), 6),
                              textcoords='offset points', fontsize=9,
                              fontweight='bold', family='Arial', va='bottom',
                              annotation_clip=False)
            ann.set_in_layout(False)
    # NOT trim=True: it re-derives ticks and drops the categorical row
    # labels in d and g. c's and g's overlong x-spines are fixed at source.
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2.mapjitter.klw')
    ap.add_argument('--observed_tsv', default=None)
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--out_stem', default=None)
    ap.add_argument('--ppc', default='safe',
                    choices=['safe', 'safe_ratio', 'safe_stake', 'slope_stake',
                             'slope_stake_safe', 'stake',
                             'ratio', 'delta', 'slope', 'slope2',
                             'stats', 'psychometric', 'psychometric2'],
                    help="'safe' collapses over the ratio ladder (main text); "
                         "'ratio' shows the psychometric function itself, on "
                         "Figure 3a's x-axis, one panel per order; 'delta' "
                         "shows the cTBS DIFFERENCE with its predictive band, "
                         "which is the only honest way to show an effect the "
                         "model puts at ~1 percentage point; "
                         "'psychometric' shows the full curve in three stake "
                         'terciles (supplement)')
    ap.add_argument('--with_probit', action='store_true',
                    help='add the probit slope / RNP panels. The model is not '
                         'exactly a probit in log(frac) -- see module docstring')
    a = ap.parse_args()
    stem = a.out_stem or str(REPO / f'notes/figures/fig5_{a.model_label}')
    if a.ppc.startswith('psychometric') and a.out_stem is None:
        stem += '_' + a.ppc
    main(a.data_dir, stem, a.model_label, a.observed_tsv, a.with_probit, a.ppc,
         a.bids_folder)
