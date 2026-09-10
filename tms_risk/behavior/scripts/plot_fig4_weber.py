"""Figure 4: representational noise grows with magnitude. Weber's law does not hold.

Placed between the psychophysics (Fig 3) and the cTBS result (Fig 5), because
it answers a different question -- what kind of observer is this? -- and it
answers it without reference to stimulation. Motivating the change of model
class here, on pre-stimulation data, is what keeps it from reading as post hoc.

Top row, the data. a, b  Psychometric functions for low- and high-stake trials,
   in the BASELINE session before anyone was stimulated and using the full
   cohort including participants never stimulated. Weber's law says these two
   curves should be the same. c  The same thing as a slope, with the size of
   the violation.
Bottom row, the model. d  Fitted noise against payoff for the two presentation
   positions; Weber is a flat line here. e  The same curve under every noise
   form we fitted -- they agree, which is why two parameters suffice. f  Panel
   c's quantity predicted: Weber is flat by construction, the power law
   reproduces the decline.

    python -m tms_risk.behavior.scripts.plot_fig4_weber --model_label log-power-n2psd
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
IPS, VERTEX = '#d62728', '#2ca02c'
#: canonical, and matching CLAUDE.md rather than drifting from it
FIRST, SECOND = '0.62', '0.15'
#: ORANGE, not red. Red means cTBS-to-IPS in every other figure in this paper,
#: and '#C44E52' is indistinguishable from '#d62728' at print size -- a reader
#: arriving from Figure 3 or 5 reads this reference line as a stimulation
#: condition. Weber-versus-flexible is a MODEL contrast, and the project
#: reserves blue/orange for those.
WEBER = '#E1812C'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
NPAR = {'weber': 1, 'affine': 2, 'power': 2, 'genweber': 2,
        'spl3': 3, 'spl5': 5, 'spl7': 7, 'spl9': 9}


def key(ax, entries, x=.04, y=.96, dy=.085, lw=1.4, seg=.075, fs=8):
    """Draw a legend as short line segments in the panel, not as words.

    Writing "solid: X, dashed: Y" makes the reader hold a mapping in their head
    while looking somewhere else. A short stroke in the actual style, next to
    the label, is read once and done.
    """
    for i, (label, col, ls) in enumerate(entries):
        yy = y - i * dy
        ax.plot([x, x + seg], [yy, yy], transform=ax.transAxes, color=col,
                ls=ls, lw=lw, solid_capstyle='butt', clip_on=False)
        ax.text(x + seg + .028, yy, label, transform=ax.transAxes, color=col,
                fontsize=fs, va='center')


def logx(ax, ticks=(7, 14, 28, 56, 112)):
    ax.set_xscale('log')
    ax.set_xticks(list(ticks))
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


#: Which decomposition the model panels (d, e) show.
#:
#: 'position' -- the first- and second-presented option, what the participant
#:     saw. Descriptive, but the fit is poorly identified: on the baseline the
#:     independent family reaches only r_hat 1.05 / ESS 115.
#: 'stage' -- perceptual and memory noise, sigma_n2 = perc and
#:     sigma_n1 = perc + mem. Samples cleanly (ESS 8388) and states the result
#:     as a claim about PROCESSING rather than serial position: the
#:     payoff-dependence is in the perceptual stage, and the memory term runs
#:     the other way. It is also the parameterisation Figure 5 uses, so the two
#:     figures share coordinates.
CHANNELS = {
    'position': [('n1', 'First-presented'), ('n2', 'Second-presented')],
    'stage':    [('perc', 'Perceptual'), ('mem', 'Memory')],
}
#: labels of the baseline fit to read, per decomposition
BASE_LABELS = {'position': ('log-power-nullind', 'log-weber-nullind', 'nullind'),
               'stage':    ('log-power-null', 'log-weber-null', 'null')}


def main(data_dir, out_stem, label, weber_label, bids_folder,
         panels='abde', channels='stage', curves_tsv=None, forms=None):
    dd = Path(data_dir)
    CHANS = CHANNELS[channels]
    # Panels c and f are OFF by default. c ("size of the violation") only
    # restates the -22% / -37% already printed in the insets of a and b, and f
    # ("predicted consistency") derives its slopes analytically rather than by
    # simulate-and-refit, so its absolute values are not on the same footing as
    # the measured slopes it sits beside. Four panels at this type size read
    # far better than six. `--panels abcdef` puts them back.
    keys = [k for k in 'abcdef' if k in panels]
    ncol = 2 if len(keys) <= 4 else 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig = plt.figure(figsize=(3.6 * ncol, 2.55 * nrow), constrained_layout=True)
    gs = fig.add_gridspec(nrow, ncol)
    AX = {k: fig.add_subplot(gs[i // ncol, i % ncol])
          for i, k in enumerate(keys)}
    # every panel block below is guarded, so a missing key is simply skipped
    _skip = plt.figure().add_subplot(111)      # scratch axis for skipped panels
    for k in 'abcdef':
        AX.setdefault(k, _skip)

    # -- a, b: the curves themselves --------------------------------------
    # blue/orange: stake level is neither a stimulation condition nor a
    # model contrast, but red here would still read as IPS to anyone
    # coming from Figures 2, 3 or 5
    LOW, HIGH = '#3B5BA5', '#E1812C'
    # Hierarchical Bayesian probit and its posterior predictive -- never the
    # maximum-likelihood fit. An s.e.m. bar on an observed point answers how
    # precisely we measured it; the band here answers whether the model could
    # have produced it (simulated choices at the real trials, aggregated the
    # same way as the data). See CLAUDE.md, "Conventions worth knowing".
    hp = pd.read_csv(dd / 'weber_baseline_hier.ses1.ppc.tsv', **READ)
    hs = pd.read_csv(dd / 'weber_baseline_hier.ses1.slopes.tsv', **READ)
    pse = pd.read_csv(dd / 'weber_baseline_curves.ses1.pse.tsv', **READ)
    _p = {(r.order, r.stake): r.pse_logfrac for _, r in pse.iterrows()}
    hp['x_centred'] = [np.log(f) - _p[(o, st)]
                       for f, o, st in zip(hp.frac, hp.order, hp.stake)]
    for k, order in zip('ab', ['Risky first', 'Risky second']):
        ax = AX[k]
        for stake, col in [('low', LOW), ('high', HIGH)]:
            o = hp[(hp.order == order) & (hp.stake == stake)].sort_values('x_centred')
            ax.fill_between(o.x_centred, o.lo, o.hi, color=col, alpha=.22,
                            lw=0, zorder=1)
            ax.plot(o.x_centred, o.model, color=col, lw=1.5, zorder=2)
            ax.plot(o.x_centred, o.observed, 'o', ms=3.8, color=col, zorder=4)
        ax.axhline(.5, color='0.88', lw=.6, ls='--', zorder=0)
        ax.axvline(0, color='0.88', lw=.6, ls='--', zorder=0)
        ax.set_xlim(-.85, .85)
        ax.set_xticks([-.7, 0, .7])
        ax.set_ylim(.02, 1.0)
        ax.set_yticks([.25, .5, .75, 1])
        ax.set_xlabel('Log ratio, relative to\nthat cell\'s indifference point')
        ax.set_title(order, fontsize=9.5)
        # The slope belongs beside the curves it summarises, not two panels
        # away: the reader should be able to check that the number matches what
        # they just saw.
        ins = ax.inset_axes([.60, .07, .37, .30])
        q = hs[hs.order == order].set_index('stake').reindex(['low', 'high'])
        q = q.rename(columns={'slope_lo': 'lo', 'slope_hi': 'hi'})
        for i, (stake, col) in enumerate([('low', LOW), ('high', HIGH)]):
            r_ = q.loc[stake]
            ins.plot([i, i], [r_.lo, r_.hi], color=col, lw=1.1)
            ins.plot(i, r_.slope, 'o', ms=4, color=col)
        ins.plot([-.3, 1.3], [q.slope.iloc[0]] * 2, color=WEBER, lw=1.0,
                 ls=(0, (2.5, 1.8)), zorder=0)
        # the contrast is computed PER DRAW upstream, so quote it directly
        pct = 100 * (q.slope.iloc[1] / q.slope.iloc[0] - 1)
        ins.text(.5, .04, f'{pct:+.0f}%',
                 transform=ins.transAxes, ha='center', fontsize=7.5,
                 color='0.2', fontweight='bold')
        ins.set_xticks([0, 1])
        ins.set_xticklabels(['Low', 'High'], fontsize=7)
        ins.set_xlim(-.45, 1.45)
        ins.tick_params(labelsize=7, length=2)
        ins.set_ylabel('Probit slope', fontsize=7.2, labelpad=1)
        ins.set_title('Consistency', fontsize=7.2, pad=2)
        for sp in ('top', 'right'):
            ins.spines[sp].set_visible(False)

        if k == 'a':
            ax.set_ylabel('P(chose risky)')
            key(ax, [('Low stake', LOW, '-'), ('High stake', HIGH, '-')],
                x=.05, y=.93, dy=.095)
            # the rest of this belongs in the caption; three lines of prose
            # inside a panel of rising curves has nowhere to sit

        else:
            ax.set_yticklabels([])

    # -- c: the same thing as a number ------------------------------------
    _noise_ylim = []
    ax = AX['c']
    xs = {'low': 0, 'high': 1}
    for order, col in [('Risky first', FIRST), ('Risky second', SECOND)]:
        q = (hs[hs.order == order].set_index('stake').reindex(['low', 'high'])
             .rename(columns={'slope_lo': 'lo', 'slope_hi': 'hi'}))
        x = np.array([xs[s] for s in q.index])
        ax.plot(x, q.slope, 'o-', color=col, ms=5, lw=1.4, zorder=3)
        ax.errorbar(x, q.slope, yerr=[q.slope - q.lo, q.hi - q.slope],
                    fmt='none', ecolor=col, elinewidth=1.1, capsize=0, zorder=3)
        # Weber's prediction: whatever the low-stake slope is, unchanged
        ax.plot([0, 1], [q.slope.iloc[0]] * 2, color=WEBER, lw=1.1, ls=(0, (3, 2)),
                zorder=2)
        ax.annotate(order, (1, q.slope.iloc[1]), xytext=(6, 0),
                    textcoords='offset points', color=col, fontsize=8,
                    va='center')
        ax.text(1.06, q.slope.iloc[1] - .13,
                f'{100 * (q.slope.iloc[1] / q.slope.iloc[0] - 1):+.0f}%',
                color=col, fontsize=7.5, va='top')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low', 'High'])
    ax.set_xlim(-.25, 1.75)
    ax.set_xlabel('Stake')
    ax.set_ylabel('Choice consistency\n(probit slope)')
    ax.set_title('Size of the violation', fontsize=9.5)
    key(ax, [("Weber's law", WEBER, (0, (3, 2)))], x=.05, y=.95)

    # -- d: the same thing from the fitted model --------------------------
    ax = AX['d']
    base = dd / 'anchor_curves_baseline.tsv'
    if base.exists():
        c_all = pd.read_csv(base, **READ)
        _g = pd.read_csv(dd / 'weber_baseline_slopes_bygroup.ses1.tsv', **READ)
        n_sub = int(_g.loc[_g.group == 'All', 'n_sub'].iloc[0])
        label, weber_label, _place = BASE_LABELS[channels]
        place_note = f'n = {n_sub} participants'
    else:                                   # baseline fits not extracted yet
        c_all = pd.read_csv(dd / 'anchor_curves.tsv', **READ)
        place_note = 'TMS cohort, vertex sessions'
    if curves_tsv:                 # e.g. the shared-family baseline extraction
        extra = pd.read_csv(curves_tsv, **READ)
        c_all = pd.concat([c_all[~c_all.label.isin(extra.label.unique())], extra])
    # the shared-family baseline traces carry the `.mapjitter.klw` stamp
    if label not in set(c_all.label) and f'{label}.mapjitter.klw' in set(c_all.label):
        label, weber_label = f'{label}.mapjitter.klw', f'{weber_label}.mapjitter.klw'
    c = c_all[c_all.label == label]
    lab = dict(CHANS)
    for chan, col in [(CHANS[0][0], FIRST), (CHANS[1][0], SECOND)]:
        s = c[(c.channel == chan) & (c.condition == 'vertex')].sort_values('x')
        if not len(s):
            continue
        ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.18, lw=0)
        ax.plot(s.x, s['mid'], color=col, lw=1.4)
        r = s['mid'].iloc[-1] / s['mid'].iloc[0]
        # both endpoints get the label ABOVE-right: the two curves end far
        # apart (0.36 vs 0.05), so they cannot collide, and -19 pushed the
        # falling channel's label off the bottom of the axes
        dy = 14
        ax.annotate(f'{lab[chan]}\n{r:.2f}× over the range',
                    (s.x.iloc[-1], s['mid'].iloc[-1]),
                    xytext=(4, dy), textcoords='offset points', color=col,
                    fontsize=7.8, va='center', linespacing=1.4)
        # The ACTUAL Weber fit, not a horizontal line through the power law's
        # value at 7 CHF. Those are different quantities, and drawing the
        # second one with the first one's dash and label put two different
        # things behind one mark -- panel d draws the fitted Weber model, so
        # the two panels disagreed about what the dashed line meant and sat at
        # visibly different heights.
        w = c_all[(c_all.label == weber_label) & (c_all.channel == chan)
                  & (c_all.condition == 'vertex')].sort_values('x')
        if len(w):
            ax.plot([w.x.iloc[0], w.x.iloc[-1]], [w['mid'].iloc[0]] * 2,
                    color=WEBER, lw=1.0, ls=(0, (3, 2)), zorder=1)
    logx(ax)
    ax.set_xlim(6.4, 420)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Representational noise ν (log units)')
    ax.set_title('Fitted noise', fontsize=9.5)
    key(ax, [("Weber's law (constant ν)", WEBER, (0, (3, 2)))], x=.05, y=.95)
    _noise_ylim.append(ax.get_ylim())

    # -- e: every noise form we fitted, on one axis -------------------------
    # The Occam panel. If the flexible forms found structure the power law
    # misses, their curves would separate. They do not, so the extra parameters
    # buy nothing -- which is the same conclusion the ELPD column reaches, but
    # visible rather than asserted.
    ax = AX['e']
    place = label.split('-')[2]
    # smooth forms first: the point of the bundle is that the shape does not
    # depend on the functional form, and a piecewise-linear curve invites
    # the reader to look at its kinks instead of its shape. `--forms` can
    # override.
    forms = forms or ['power', 'cspl3', 'cspl5', 'cspl7', 'affine']
    avail = set(c_all.label.unique())
    forms = [f_ for f_ in forms if f'log-{f_}-{place}' in avail]
    # Both channels, every form. The forms are drawn as one bundle rather than
    # six labelled lines: that they are indistinguishable IS the result, and
    # naming each would invite the reader to look for a difference that is not
    # there. Weber is the one that separates, so it is the one that is marked.
    # DASH MEANS WEBER, here as in panel c -- and only Weber. It used to
    # encode the channel in this panel and the Weber reference in the one
    # beside it, so the same mark meant two things within one figure. The two
    # channels are already separated by colour, by a direct label, and by
    # sitting in non-overlapping bands of the y-axis; they do not need a third
    # encoding, and certainly not one that is spoken for.
    for chan, ls, nm, ycol in [(CHANS[0][0], '-', CHANS[0][1], FIRST),
                               (CHANS[1][0], '-', CHANS[1][1], SECOND)]:
        for f_ in forms:
            q = c_all[(c_all.label == f'log-{f_}-{place}')
                      & (c_all.channel == chan)
                      & (c_all.condition == 'vertex')].sort_values('x')
            if len(q):
                ax.plot(q.x, q['mid'], color=ycol, lw=1.0, ls=ls, alpha=.75)
        qw = c_all[(c_all.label == f'log-weber-{place}') & (c_all.channel == chan)
                   & (c_all.condition == 'vertex')].sort_values('x')
        if len(qw):
            ax.plot(qw.x, qw['mid'], color=WEBER, lw=1.5, ls=(0, (3, 2)),
                    zorder=3)
        q2 = c_all[(c_all.label == f'log-power-{place}') & (c_all.channel == chan)
                   & (c_all.condition == 'vertex')].sort_values('x')
        if len(q2):
            ax.annotate(nm, (q2.x.iloc[-1], q2['mid'].iloc[-1]),
                        xytext=(5, 10 if chan == CHANS[1][0] else -10),
                        textcoords='offset points', color=ycol, fontsize=7.8,
                        va='center')
    logx(ax)
    ax.set_xlim(6.4, 620)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Representational noise ν (log units)')
    ax.set_title(f'{len(forms)} flexible forms, and Weber', fontsize=9.5)
    _noise_ylim.append(ax.get_ylim())
    key(ax, [("Weber's law (constant ν)", WEBER, (0, (3, 2))),
             (f'{len(forms)} flexible forms', '0.35', '-')], x=.05, y=.96,
        dy=.075, seg=.06)

    if len(_noise_ylim) == 2:
        ylo = min(v[0] for v in _noise_ylim)
        yhi = max(v[1] for v in _noise_ylim)
        for k_ in ('c', 'e'):
            if k_ in AX:
                AX[k_].set_ylim(ylo, yhi)

    # -- f: panel c's quantity, predicted -----------------------------------
    # A residual on choice PROPORTIONS cannot separate these models: both fit
    # the level, and Weber's failure is in the SLOPE. So predict the slope.
    # slope = w_R / diff_sd, evaluated on the trials participants actually saw
    # and averaged within each stake half -- the same arithmetic for both
    # models, so the comparison is of shape, which is the claim.
    ax = AX['f']
    from tms_risk.behavior.fit_model import get_data
    from scipy.stats import norm                                   # noqa: F401
    trials = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    trials['order'] = trials['risky_first'].map({True: 'Risky first',
                                                 False: 'Risky second'})
    trials['avg'] = (trials.n_safe + trials.n_risky) / 2
    trials['hi'] = (trials.groupby('subject')['avg']
                    .transform(lambda v: (v > v.median()).astype(int)))
    pb = dd / 'anchor_priors_baseline.tsv'
    pri_all = pd.read_csv(pb if pb.exists() else dd / 'anchor_priors.tsv', **READ)

    def predicted_slope(lb):
        cc = c_all[(c_all.label == lb) & (c_all.condition == 'vertex')]
        pp_ = pri_all[pri_all.label == lb]
        if not len(cc) or not len(pp_):
            return None
        nu = {}
        for ch in ('n1', 'n2'):
            g = cc[cc.channel == ch].sort_values('x')
            nu[ch] = (np.log(g.x.values), np.log(g['mid'].values))
        f = lambda ch, v: np.exp(np.interp(np.log(v), *nu[ch]))
        P = {w_: pp_[pp_.which == w_].iloc[0] for w_ in ('risky', 'safe')}
        out = {}
        for (order, hi_), g in trials.groupby(['order', 'hi']):
            rf = order == 'Risky first'
            vR = f('n1' if rf else 'n2', g.n_risky.values)
            vS = f('n2' if rf else 'n1', g.n_safe.values)
            sR, sS = P['risky'].sd, P['safe'].sd
            wR = sR ** 2 / (sR ** 2 + vR ** 2)
            out[(order, int(hi_))] = float(np.mean(
                wR / np.sqrt(vR ** 2 + vS ** 2)))
        return out

    obs = hs.set_index(['order', 'stake'])
    for lb, col, nm in [(weber_label, WEBER, 'Weber'), (label, '0.2', 'Power')]:
        pr = predicted_slope(lb)
        if pr is None:
            continue
        for order, ls in [('Risky second', '-'), ('Risky first', (0, (2.5, 1.5)))]:
            v = [pr[(order, 0)], pr[(order, 1)]]
            ax.plot([0, 1], v, ls=ls, color=col, lw=1.4,
                    marker='o' if ls == '-' else 's', ms=4)
        ax.annotate(nm, (1, pr[('Risky second', 1)]), xytext=(5, 0),
                    textcoords='offset points', color=col, fontsize=8,
                    va='center')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low', 'High'])
    ax.set_xlim(-.25, 1.65)
    ax.set_xlabel('Stake')
    ax.set_ylabel('Predicted consistency\n(w$_R$ / decision SD)')
    ax.set_title('Predicted', fontsize=9.5)
    # open up the bottom so the note does not sit on the flat Weber line
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_ - .30 * (hi_ - lo_), hi_)
    key(ax, [('Risky second', '0.35', '-'),
             ('Risky first', '0.35', (0, (2.5, 1.5)))], x=.05, y=.16, dy=.095)

    fig.suptitle(f'Baseline session, before any stimulation · {place_note}',
                 fontsize=8.5, color='0.35', y=1.005)
    for letter, k in zip('abcdefgh', keys):
        AX[k].text(-.20, 1.07, letter, transform=AX[k].transAxes, fontsize=9,
                   fontweight='bold', family='Arial', va='bottom')
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--channels', default='stage', choices=list(CHANNELS),
                    help="'stage' (perceptual/memory, the default and what "
                         "Figure 5 uses) or 'position' (n1/n2, which does not "
                         "sample: baseline r_hat 1.05 / ESS 115)")
    ap.add_argument('--forms', nargs='+', default=None,
                    help='flexible noise forms to bundle in panel d (default: '
                         'power + the smooth cspl family)')
    ap.add_argument('--curves_tsv', default=None,
                    help='extra anchor_curves TSV to merge in, e.g. the '
                         'shared-family baseline extraction')
    # KLW labels. The old defaults -- log-power-n2psd and log-weber-n1n2 --
    # were raw-choice-rule fits, which have been archived, so this figure would
    # silently have found no curves. n2psd was also a prior-shift model, and
    # those are out of the paper.
    ap.add_argument('--model_label', default='log-power-n1n2.mapjitter.klw')
    ap.add_argument('--weber_label', default='log-weber-n1n2.mapjitter.klw')
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/fig4_weber'))
    ap.add_argument('--panels', default='abde',
                    help="which panels to draw. Default 'abde'; 'abcdef' adds "
                         "the redundant violation-size panel and the analytic "
                         "predicted-consistency panel")
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label, a.weber_label, a.bids_folder,
         a.panels, a.channels, a.curves_tsv, a.forms)
