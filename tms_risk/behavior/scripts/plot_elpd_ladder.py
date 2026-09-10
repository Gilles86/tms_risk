"""Model comparison for the cTBS models: WHERE the perturbation acts.

Every rung holds the noise function at the power law and differs only in what
cTBS is allowed to move -- perceptual noise, memory noise, the magnitude
priors, or combinations of those. That isolates the mechanism question from the
functional-form question, which is a different question on a different cohort
and has its own figure (`plot_noise_flexibility.py`, fitted on the 73
pre-stimulation baseline participants Figure 4 shows). Putting both in one
figure invites reading a flexibility result from one cohort as if it applied to
the other, which is exactly the mistake that separation prevents.

Differences are PAIRED against the reported model, with the standard error of
the difference; the marginal SE of each ELPD is several times larger and would
make every comparison look inconclusive. Models that fail the convergence gate
are drawn in outline and labelled: an ELPD from a posterior that never mixed is
not a number to rank.

Panel b answers the obvious follow-up. A reader who accepts that the effect is
perceptual will still ask whether the SIZE of it is an artefact of the
two-parameter noise function used to measure it. It is not: refitting the same
placement with every noise form from Weber to a seven-anchor spline moves the
estimated increase by a few percentage points and never changes its sign or
its payoff dependence. What the flexible forms do change is the PRECISION --
they widen the interval by about half for no change in the estimate, which is
the parsimony argument stated in the currency that matters.

    python -m tms_risk.behavior.scripts.plot_elpd_ladder
"""
import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])

#: WHICH DATA each panel is fitted on. These are different cohorts and the
#: distinction is easy to lose: the cTBS ladder can only be fitted where there
#: is stimulation (35 participants, sessions 2-3), while Figure 4's noise
#: functions come from the pre-stimulation baseline (73 participants, session
#: 1). Naming them on the panels themselves is the cheapest guard against
#: reading a flexibility result from one cohort as if it applied to the other.
COHORT = {'tms': 'cTBS cohort · n = 35',
          'baseline': 'Baseline session, before any stimulation · n = 73'}
REPO = Path(__file__).resolve().parents[3]
REF = 'log-power-percpmu'
GOOD, BAD, DEAD = '0.15', '0.48', '0.78'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 0,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})

PANELS = [
    # a  WHERE the cTBS effect acts, noise form held at the power law.
    # Ordered by what the perturbation is allowed to move, richest first, so
    # the reader walks DOWN to the null and the two rungs that matter --
    # dropping the prior shift, then dropping the noise change -- sit adjacent
    # to the model that keeps them.
    ('Where the cTBS effect acts', [
        ('log-power-percmempmu', 'Perceptual + memory noise, prior means'),
        ('log-power-percpmu',    'Perceptual noise + prior means'),
        ('log-power-percmem',    'Perceptual + memory noise'),
        ('log-power-perc',       'Perceptual noise only'),
        ('log-power-spmu',       'Prior means only, no noise change'),
        ('log-power-mem',        'Memory noise only'),
        ('log-power-null',       'No cTBS effect'),
    ]),
]

#: The spline family, plotted separately by `--splines`. These ask whether two
#: anchors are enough, which is a question about RESOLUTION rather than shape:
#: spl3 and spl5 place three and five anchors over the same payoff range, so a
#: flat ladder here says the two-anchor power law already has the resolution
#: the data support.
SPLINES = [
    ('Is two anchors enough?', [
        ('log-power-n1n2', 'Power law, 2 anchors'),
        ('log-spl3-n1n2',  'Spline, 3 anchors'),
        ('log-spl5-n1n2',  'Spline, 5 anchors'),
    ]),
    ('… and without a cTBS effect', [
        ('log-power-nullind', 'Power law, 2 anchors'),
        ('log-spl3-nullind',  'Spline, 3 anchors'),
        ('log-spl5-nullind',  'Spline, 5 anchors'),
    ]),
]




#: Preference order over the sampler/init variants of one model. A bare label
#: like `log-power-perc` names a MODEL; on disk there may be several fits of it
#: differing only in init or in a convergence-prior override, and the ladder
#: must pick one deterministically. Converged variants win, then this order.
# Sampler/init variants only. Fits under a DIFFERENT PRIOR (tn*, ti*, sps*)
# are deliberately absent: a ladder must compare models on one prior spec, and
# swapping in the better-mixing tau_noise fit for a single rung would make its
# ELPD incommensurable with every other bar. The mixing of the rung that needs
# it is reported as numbers on the row instead.
VARIANTS = ['.mapjitter.klw', '.pathfinder.klw', '.klw']


def resolve(base, ld, chk):
    """Bare model label -> the KLW trace to plot, or None.

    Raises rather than silently falling back to a NON-KLW file: the raw-rule
    fits normalise a shrunken numerator by an unshrunken denominator, so their
    ELPD is not on the same footing as a KLW fit and a ladder mixing the two is
    meaningless. That is exactly what happened once -- a KLW reference against
    raw-rule comparators -- and it looked completely plausible.
    """
    cands = [base + v for v in VARIANTS
             if (ld / f'looi.{base + v}.npy').exists()]
    if not cands:
        if (ld / f'looi.{base}.npy').exists():
            raise SystemExit(
                f'{base}: only a non-KLW fit is available. Refusing to mix '
                f'choice rules in one ladder.')
        return None
    # prefer a variant that passes; otherwise the best-mixing one, so a rung
    # is never represented by a worse fit than exists at the same prior
    scored = sorted(cands, key=lambda c: (
        not (c in chk.index and bool(chk.loc[c, 'ok'])),
        float(chk.loc[c, 'max_rhat']) if c in chk.index else 9.0))
    return scored[0]


def _pairs(rows, ld, chk, piw, ref, a):
    """(name, dELPD, SE, is_ref, ok, rhat, ess) for each rung that resolves."""
    seen = []
    for base, nm in rows:
        lab = resolve(base, ld, chk)
        if lab is None:
            print(f'  no KLW fit for {base}, skipped')
            continue
        b = piw(lab)
        if lab == ref:
            d = se = 0.0
        elif a is not None and b is not None and b.shape == a.shape:
            diff = b - a
            d = float(diff.sum())
            se = float(np.std(diff, ddof=1) * np.sqrt(len(diff)))
        else:
            print(f'  no comparable pointwise LOO for {lab}, skipped')
            continue
        # Diagnostics, not a verdict. r_hat 1.02 / ESS 384 and r_hat 1.12 /
        # ESS 42 are not the same object, and one label for both is misleading
        # in whichever direction the reader happens to guess.
        r_ = float(chk.loc[lab, 'max_rhat']) if lab in chk.index else np.nan
        e_ = float(chk.loc[lab, 'min_ess_bulk']) if lab in chk.index else np.nan
        ok = bool(chk.loc[lab, 'ok']) if lab in chk.index else True
        seen.append((nm, d, se, lab == ref, ok, r_, e_))
    return seen


def _ladder(ax, title, seen):
    y = np.arange(len(seen))[::-1]
    # names sit clear of the longest POSITIVE bar+whisker, so a model that
    # beats the reference does not draw its bar through its own label
    lo = min([r[1] - r[2] for r in seen] + [0])
    xn = max([r[1] + r[2] for r in seen] + [0]) + abs(lo) * .04
    for yy, (nm, d, se, is_ref, ok, r_, e_) in zip(y, seen):
        # grey out only what is genuinely unusable; a near-miss keeps its ink
        # and carries its numbers
        poor = (not ok) and (r_ > 1.05 or e_ < 100)
        col = GOOD if is_ref else (DEAD if poor else BAD)
        ax.barh(yy, d, height=.62, color=col, alpha=.35 if poor else .9,
                lw=.8 if poor else 0, edgecolor=col, zorder=2)
        if se > 0:
            ax.plot([d - se, d + se], [yy] * 2, color=DEAD if poor else '0.15',
                    lw=1.0, zorder=3, solid_capstyle='butt')
        txt = 'reported model' if is_ref else f'{d:+.0f}'
        ax.text(min(d - se, 0) - abs(lo) * .03, yy, txt, ha='right',
                va='center', fontsize=7.5, color=col,
                fontweight='bold' if is_ref else 'normal')
        ax.text(xn, yy, nm, fontsize=7.5, va='center',
                color=DEAD if poor else '0.2')
        if not ok:
            # flush right, so it can never run into a model name
            ax.text(.995, yy, f'r̂ {r_:.2f} · ESS {e_:.0f}',
                    fontsize=6.4, va='center', ha='right',
                    transform=ax.get_yaxis_transform(),
                    color=DEAD if poor else '0.45')
    ax.axvline(0, color='0.35', lw=.9, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-1.2, len(seen) - .35)
    ax.set_xlim(lo * 1.12, xn * 6.0)
    ax.set_title(title, fontsize=9, loc='left', x=.0)
    ax.set_xlabel('ΔELPD vs the reported model (nats)')


#: anchors per channel, for ordering the invariance panel. Mirrors
#: `plot_noise_flexibility.parse_form`, imported rather than duplicated.
def _invariance(ax, curves_tsv, suffix, ref_form, payoffs=(7, 112)):
    """The cTBS effect on perceptual noise, refitted under every noise form.

    Percentage change, IPS - vertex, at the two ends of the payoff range, with
    the 95% credible interval. One row per form, ordered by anchors per
    channel, the reported form picked out.
    """
    from tms_risk.behavior.scripts.plot_noise_flexibility import parse_form
    C = pd.read_csv(curves_tsv, **READ)
    C = C[(C.placement == suffix) & (C.condition == 'delta_pct')
          & (C.channel == 'perc')]
    if C.empty:
        raise SystemExit(f'no delta_pct perc rows for {suffix} in {curves_tsv}')
    rows = []
    for form, g in C.groupby('form'):
        k, basis = parse_form(form)
        g = g.sort_values('x')
        vals = {}
        for t in payoffs:
            r = g.iloc[(g.x - t).abs().argmin()]
            vals[t] = (float(r['mid']), float(r['lo']), float(r['hi']))
        rows.append(dict(form=form, k=k if k is not None else 99, vals=vals,
                         is_ref=(form == ref_form)))
    rows.sort(key=lambda r: (r['k'], r['form']))
    y = np.arange(len(rows))[::-1]
    # room on the LEFT for the row names, measured from the data rather than
    # guessed: a fixed offset put every label on top of its own interval
    allv = [v for r in rows for t in payoffs for v in r['vals'][t]]
    dlo, dhi = min(allv), max(allv)
    span = dhi - dlo
    xname = dlo - span * .04
    ax.set_xlim(dlo - span * .55, dhi + span * .04)
    # the two payoffs are the SAME quantity at two ends of the axis, so they
    # are two columns of one panel, offset, not two hues
    OFF = {payoffs[0]: .17, payoffs[1]: -.17}
    COL = {payoffs[0]: '0.15', payoffs[1]: '0.62'}
    ax.axvline(0, color='0.75', lw=.8, ls=(0, (3, 2)), zorder=1)
    for yy, r in zip(y, rows):
        for t in payoffs:
            m, lo, hi = r['vals'][t]
            c = COL[t]
            ax.plot([lo, hi], [yy + OFF[t]] * 2, color=c, lw=1.0,
                    alpha=.9 if r['is_ref'] else .5, zorder=2,
                    solid_capstyle='butt')
            ax.plot(m, yy + OFF[t], 'o', ms=4.4 if r['is_ref'] else 3.2,
                    color=c, mfc=c if r['is_ref'] else 'w', mew=1.0, zorder=3)
        nm = ('Power law' if r['is_ref'] else
              ('Weber' if r['form'] == 'weber' else
               ('Power + Weber' if '+' in r['form'] else
                f"{r['k']} anchors, "
                f"{'smooth' if r['form'].startswith('cspl') else 'linear'}")))
        ax.text(xname, yy, nm, fontsize=6.8, ha='right', va='center',
                color='0.15' if r['is_ref'] else '0.45',
                fontweight='bold' if r['is_ref'] else 'normal')
    ax.set_yticks([])
    ax.set_ylim(-1.15, len(rows) - .35)
    ax.set_xlabel('Change in perceptual noise, IPS − vertex (%)')
    ax.set_title('The estimate does not depend on the form', fontsize=9,
                 loc='left', x=.0)
    # the two payoffs are named by REDRAWING their marks
    for i, t in enumerate(payoffs):
        yy = .085 - i * .07
        ax.plot(.78, yy, 'o', transform=ax.transAxes, ms=3.6, color=COL[t],
                mfc=COL[t], mew=0, clip_on=False)
        ax.text(.81, yy, f'At {t} CHF', transform=ax.transAxes, fontsize=6.8,
                color=COL[t], va='center')
    return rows


#: canonical project palette. Red marks the active arm, green the sham one;
#: this mapping is fixed across every figure in the paper and is never
#: inverted, and no other variable in this figure is allowed a hue.
IPS_C, VER_C = '#d62728', '#2ca02c'


def _noise_grid(fig, curves_tsv, suffix, ref_form, ncol=6, elpd=None):
    """The perceptual noise function itself, IPS against vertex, per form.

    The forest panel compresses each fit to one number at one payoff, which is
    the right summary but asks to be taken on trust. This shows the object the
    number came from: under every noise form, the stimulated curve sits above
    the sham curve at low payoffs and the two converge by the top of the range.
    Small multiples rather than one crowded axes, because the comparison that
    matters is red-against-green WITHIN a form, not form against form.
    """
    from tms_risk.behavior.scripts.plot_noise_flexibility import parse_form
    C = pd.read_csv(curves_tsv, **READ)
    D = C[(C.placement == suffix) & (C.channel == 'perc')
          & (C.condition == 'delta')]
    # the memory channel, drawn as ONE curve. Under this placement the
    # stimulation regressor is on the perceptual channel only, so memory is
    # identical in the two arms by construction (its delta is exactly zero at
    # every payoff, checked) -- plotting it twice in red and green would imply
    # a comparison the model does not make. One neutral curve says instead
    # what is true: this is the part cTBS was not allowed to move.
    M = C[(C.placement == suffix) & (C.channel == 'mem')
          & (C.condition == 'ips')]
    C = C[(C.placement == suffix) & (C.channel == 'perc')
          & C.condition.isin(['ips', 'vertex'])]
    if C.empty:
        raise SystemExit(f'no perc ips/vertex rows for {suffix} in {curves_tsv}')
    forms = []
    for form in C.form.unique():
        k, basis = parse_form(form)
        forms.append((k if k is not None else 99, form, basis))
    forms.sort()
    # the reported form leads, so the eye starts at the curve the paper reports
    forms = ([f for f in forms if f[1] == ref_form]
             + [f for f in forms if f[1] != ref_form])
    # one spare cell for the key. Putting it inside the first panel meant it
    # sat on that panel's own curves and bands, which is the one place a key
    # must never be; a blank cell costs a little space and no ink.
    nrow = int(np.ceil((len(forms) + 1) / ncol))
    outer = fig.add_gridspec(nrow, ncol)
    ylo = float(min(C.lo.min(), M.lo.min() if len(M) else C.lo.min())) * .90
    yhi = float(C.hi.max()) * 1.10
    AX = []
    for i, (k, form, basis) in enumerate(forms):
        ax = fig.add_subplot(outer[i // ncol, i % ncol])
        AX.append(ax)
        mq = M[M.form == form].sort_values('x')
        if len(mq):
            ax.fill_between(mq.x, mq.lo, mq.hi, color='0.45', alpha=.14, lw=0,
                            zorder=1)
            ax.plot(mq.x, mq['mid'], '-', color='0.45', lw=1.0, zorder=2)
        for cond, col in (('vertex', VER_C), ('ips', IPS_C)):
            q = C[(C.form == form) & (C.condition == cond)].sort_values('x')
            ax.fill_between(q.x, q.lo, q.hi, color=col, alpha=.15, lw=0,
                            zorder=1)
            ax.plot(q.x, q['mid'], '-', color=col, lw=1.1, zorder=2)
        # Significance rug: the payoffs at which P(IPS > vertex) exceeds 0.95.
        # DIRECTIONAL, matching how the paper treats its other directional
        # predictions, and stated as such in the key -- the prediction under
        # test is that cTBS ADDS noise, not that it changes it either way. At
        # the reported model's 7 CHF anchor this is 0.963, so the two-sided
        # 95% interval marginally includes zero; the rug is drawn where the
        # directional claim holds and the bands show the full uncertainty.
        d = D[D.form == form].sort_values('x')
        if not d.empty:
            sig = pd.to_numeric(d.p_gt0, errors='coerce').fillna(0).values > .95
            xs, ytop = d.x.values, ylo + .90 * (yhi - ylo)
            run = None
            for j in range(len(xs) + 1):
                on = j < len(xs) and sig[j]
                if on and run is None:
                    run = j
                elif not on and run is not None:
                    ax.plot([xs[run], xs[j - 1]], [ytop] * 2, color='0.15',
                            lw=2.4, solid_capstyle='butt', zorder=5,
                            clip_on=False)
                    run = None
        ax.set_xscale('log')
        ax.set_ylim(ylo, yhi)
        ax.set_xticks([7, 28, 112])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        nm = ('Power law' if form == ref_form else
              'Weber' if form == 'weber' else
              'Power + Weber' if '+' in form else
              f"{k} anchors, {'smooth' if form.startswith('cspl') else 'linear'}")
        col = '0.15' if form == ref_form else '0.4'
        ax.set_title(nm, fontsize=6.6, color=col, pad=9,
                     fontweight='bold' if form == ref_form else 'normal')
        if elpd and form in elpd:
            d_, se_ = elpd[form]
            txt = ('reported' if form == ref_form
                   else f'ΔELPD {d_:+.0f} ({se_:.0f})')
            ax.text(.5, 1.005, txt, transform=ax.transAxes, fontsize=6.0,
                    color=col, ha='center', va='bottom')
        if i % ncol:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Noise ν (log units)', fontsize=7)
        if i // ncol == nrow - 1 or i + ncol >= len(forms):
            ax.set_xlabel('Payoff (CHF)', fontsize=7)
        ax.tick_params(labelsize=6.2)
        sns.despine(ax=ax, offset=3)
    # the marks are named by REDRAWING them, in the spare cell
    i = len(forms)
    axk = fig.add_subplot(outer[i // ncol, i % ncol])
    axk.axis('off')
    for j, (nm, col, lw, alpha) in enumerate(
            (('Perceptual, cTBS to IPS', IPS_C, 1.5, 1.0),
             ('Perceptual, vertex', VER_C, 1.5, 1.0),
             ('Memory, shared across arms', '0.45', 1.2, 1.0),
             ('95% credible interval', '0.45', 5.0, .28),
             ('P(IPS > vertex) > 0.95', '0.15', 2.6, 1.0))):
        yy = .86 - j * .135
        axk.plot([.02, .26], [yy, yy], transform=axk.transAxes, color=col,
                 lw=lw, alpha=alpha, solid_capstyle='butt', clip_on=False)
        axk.text(.32, yy, nm, transform=axk.transAxes, fontsize=6.2,
                 color=col if alpha == 1.0 else '0.4', va='center')
    return AX


def main(data_dir, out_stem, ref, panels=None, splines=False,
         curves_tsv=None):
    dd = Path(data_dir)
    ld = dd / 'loo_anchor'
    _chkf = dd / 'all_klw_check.tsv'
    chk = pd.read_csv(_chkf if _chkf.exists() else dd / 'all_anchor_check.tsv',
                      **READ).set_index('trace')
    piw = lambda l: (np.load(ld / f'looi.{l}.npy')
                     if (ld / f'looi.{l}.npy').exists() else None)
    ref = resolve(ref, ld, chk) or ref
    a = piw(ref)
    if a is None:
        raise SystemExit(f'no pointwise LOO for the reference {ref}')
    print(f'reference: {ref}')

    if splines:
        fig, AX = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True,
                               constrained_layout=True)
        for ax, (title, rows) in zip(AX, panels):
            _ladder(ax, title, _pairs(rows, ld, chk, piw, ref, a))
        sns.despine(fig=fig, left=True, offset={'bottom': 4})
    else:
        wide = curves_tsv is not None
        fig = plt.figure(figsize=(7.6 if wide else 4.7, 6.0 if wide else 2.7),
                         constrained_layout=True)
        if wide:
            # SUBFIGURES, not one grid: the two wide panels above need generous
            # space between them for their labels, and on a shared gridspec
            # that spacing propagates down and tears a gap through the middle
            # of the small-multiple rows. Separate subfigures lay out
            # independently.
            top, bot = fig.subfigures(2, 1, height_ratios=[1.45, 2.0])
            gs = top.add_gridspec(1, 2)
            ax = top.add_subplot(gs[0, 0])
        else:
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
        title, rows = panels[0]
        _ladder(ax, title, _pairs(rows, ld, chk, piw, ref, a))
        # one glyph key, drawn as the mark it explains
        ax.plot([.06, .13], [.035, .035], transform=ax.transAxes,
                color='0.15', lw=1.0)
        ax.text(.15, .035, '\u00b11 SE of the paired difference',
                transform=ax.transAxes, fontsize=6.8, va='center',
                color='0.45')
        sns.despine(ax=ax, left=True, offset={'bottom': 4})
        AX = [ax]
        if curves_tsv:
            suffix = ref.split('-')[-1].split('.')[0]
            ref_form = ref.split('-')[1]
            axb = top.add_subplot(gs[0, 1])
            _invariance(axb, curves_tsv, suffix, ref_form)
            sns.despine(ax=axb, left=True, offset={'bottom': 4})
            AX.append(axb)
            # paired dELPD per noise form, for the small-panel subtitles
            from tms_risk.behavior.scripts.plot_noise_flexibility import (
                parse_form)
            elpd = {}
            for f_ in ld.glob('looi.*.npy'):
                lab_ = f_.name[len('looi.'):-len('.npy')]
                m_ = re.match(rf'log-(.+)-{suffix}(\..*)?$', lab_)
                if not m_ or parse_form(m_.group(1))[1] is None:
                    continue
                b_ = np.load(f_)
                if b_.shape != a.shape:
                    continue
                if lab_ in chk.index and not bool(chk.loc[lab_, 'ok']):
                    continue
                dd_ = b_ - a
                elpd[m_.group(1)] = (float(dd_.sum()),
                                     float(np.std(dd_, ddof=1)
                                           * np.sqrt(len(dd_))))
            grid = _noise_grid(bot, curves_tsv, suffix, ref_form, ncol=6,
                               elpd=elpd)
            AX.append(grid[0])
    # the cohort is named on the figure: this ladder can only be fitted where
    # there is stimulation, and the flexibility figure is a different cohort
    fig.suptitle(COHORT['tms'], fontsize=8, color='0.4')
    if len(AX) > 1:
        fig.canvas.draw()
        for letter, ax in zip('abc', AX):
            bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
                fig.transFigure.inverted())
            fig.text(bb.x0 - .012, min(bb.y1 + .035, .995), letter, fontsize=9,
                     fontweight='bold', family='Arial', va='top', ha='left')
    fig.savefig(f'{out_stem}.pdf')
    fig.savefig(f'{out_stem}.png', dpi=200)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--reference', default=REF)
    ap.add_argument('--splines', action='store_true',
                    help='plot the spline-resolution comparison instead')
    ap.add_argument('--out_stem', default=None)
    ap.add_argument('--curves_tsv', default=None,
                    help='anchor_curves TSV for the cTBS cohort; adds panel b, '
                         'the same effect estimated under every noise form')
    a = ap.parse_args()
    stem = a.out_stem or str(REPO / ('notes/figures/supp_elpd_'
                                     + ('splines' if a.splines else 'ladder')))
    main(a.data_dir, stem, a.reference,
         SPLINES if a.splines else PANELS, a.splines, a.curves_tsv)
