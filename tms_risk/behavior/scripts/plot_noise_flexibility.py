"""How flexible does the noise function need to be, and what does it look like?

The companion to Figure 4. Figure 4 shows ONE noise function -- the power law
fitted to the pre-stimulation baseline -- and a reader is entitled to ask
whether that shape is a property of the data or of the two-parameter form
imposed on it. This figure answers that by fitting the same data with every
form from Weber (a single anchor, constant nu) up to a seven-anchor spline, in
both a piecewise-linear and a natural-cubic ("smooth") basis, and showing:

a  what each extra anchor BUYS -- expected log predictive density against the
   number of anchors per channel. A complexity curve, not a ranked ladder: the
   x-axis is ordered, and a bar chart would hide that.
b,c  what each extra anchor DOES -- the perceptual and memory noise functions
   the fits actually imply, all of them on one pair of axes.

Panel a alone says added flexibility does not pay. Panels b and c say why: the
flexible fits describe the same monotone functions and differ only in wiggle
the data cannot resolve. That is the argument, and it is visible at a glance.

**Cohort matters and is stated on the figure.** The flexibility question
belongs on the BASELINE data Figure 4 itself shows (73 participants, session 1,
before any stimulation). The same panels can be drawn on the 35-participant
cTBS cohort with `--cohort tms`, which is a different question -- how flexible
the noise function needs to be when a stimulation effect is also being fitted.
Mixing the two is the mistake this script's labelling exists to prevent.

    python -m tms_risk.behavior.scripts.plot_noise_flexibility \\
        --cohort baseline --curves_tsv notes/data/base_curves.tsv
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
REPO = Path(__file__).resolve().parents[3]

#: cohort -> (placement suffix, reference form, caption, pointwise-LOO dir)
#:
#: The LOO directory is PART OF THE COHORT, not a global setting. Both cohorts
#: have fits at the `null` placement -- the baseline ones because there is no
#: stimulation to model, the cTBS ones as the no-effect rung of the placement
#: ladder -- so a single flat directory silently lets a `log-power-null` from
#: one cohort become the reference for the other. That happened once: the
#: baseline figure was drawn with cTBS-cohort ELPDs under baseline noise
#: curves, and nothing raised, because the two datasets happen to have
#: comparable trial counts. Separate directories make it impossible.
COHORT = {
    'baseline': ('null', 'power',
                 'Baseline session, before any stimulation · n = 73',
                 'loo_baseline'),
    'tms':      ('percpmu', 'power', 'cTBS cohort · n = 35', 'loo_anchor'),
}

#: forms whose anchor count is not in their name
FIXED_ANCHORS = {'weber': 1, 'power': 2, 'affine': 2, 'genweber': 2}
#: a form giving the two channels DIFFERENT functions is not a point on an
#: "anchors per channel" axis at all, so it is drawn as a reference rule
MIXED = re.compile(r'\+')

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})


def parse_form(form):
    """`cspl5` -> (5, 'smooth'); `spl3` -> (3, 'linear'); `weber` -> (1, 'both').

    A one- or two-anchor form belongs to BOTH bases -- through two points a
    piecewise-linear and a natural-cubic interpolant are the same straight
    line -- so it anchors both series rather than floating in one of them.
    """
    if MIXED.search(form):
        return None, 'mixed'
    if form in FIXED_ANCHORS:
        return FIXED_ANCHORS[form], 'both'
    m = re.fullmatch(r'(c?)spl(\d+)', form)
    if not m:
        return None, None
    return int(m.group(2)), 'smooth' if m.group(1) else 'linear'


#: blue/orange, the repo's reserved model-contrast pair. Red and green stay
#: free for stimulation, which this figure never shows.
BASIS_COL = {'linear': '#3B5BA5', 'smooth': '#E1812C', 'both': '0.15'}
BASIS_NAME = {'linear': 'Piecewise linear', 'smooth': 'Smooth (natural cubic)'}
#: panels b/c: anchor count is ORDERED, so it gets a sequential ramp rather
#: than a categorical palette, truncated away from both extremes so no line is
#: near-white or near-black on the page
RAMP = 'mako'
#: the form whose credible band is drawn behind the family
REF_FORM = 'power'


def collect(curves_tsv, suffix):
    """Every fitted noise function in the file, tagged with anchors and basis."""
    C = pd.read_csv(curves_tsv, **READ)
    C = C[C.placement == suffix]
    if C.empty:
        raise SystemExit(f'no rows with placement={suffix!r} in {curves_tsv}')
    # baseline fits have no stimulation contrast, so ips and vertex are the
    # same draw; taking one avoids drawing every line twice
    keep = [c for c in ('baseline', 'ips', 'vertex') if c in set(C.condition)]
    C = C[C.condition == keep[0]]
    info = {f: parse_form(f) for f in C.form.unique()}
    C['n_anchor'] = C.form.map(lambda f: info[f][0])
    C['basis'] = C.form.map(lambda f: info[f][1])
    unknown = sorted({f for f, (k, b) in info.items() if b is None})
    if unknown:
        print(f'  unrecognised forms, not drawn: {", ".join(unknown)}')
    return C[C.basis.notna()]


def panel_complexity(ax, ld, chk, ref_label, suffix, resolve):
    """ELPD against anchors per channel: where flexibility stops paying."""
    a = np.load(ld / f'looi.{ref_label}.npy')
    rows = []
    for f in sorted(ld.glob('looi.*.npy')):
        lab = f.name[len('looi.'):-len('.npy')]
        m = re.match(rf'log-(.+)-{suffix}(\..*)?$', lab)
        if not m:
            continue
        k, basis = parse_form(m.group(1))
        b = np.load(f)
        if b.shape != a.shape:
            continue
        d = b - a
        rows.append(dict(label=lab, form=m.group(1), k=k, basis=basis,
                         d=float(d.sum()),
                         se=float(np.std(d, ddof=1) * np.sqrt(len(d))),
                         ok=bool(chk.loc[lab, 'ok']) if lab in chk.index else True))
    P = pd.DataFrame(rows)
    if P.empty:
        raise SystemExit(f'no pointwise LOO for any *-{suffix} fit in {ld}')
    P = P[P.ok]
    # one row per (anchors, basis): if several fits of the same rung exist,
    # the best-mixing one already won in `resolve`, so keep the first
    P = P.drop_duplicates(['k', 'basis'])
    for basis in ('linear', 'smooth'):
        q = P[P.basis.isin([basis, 'both'])].dropna(subset=['k']).sort_values('k')
        if len(q) < 2:
            continue
        c = BASIS_COL[basis]
        ax.plot(q.k, q.d, '-', color=c, lw=1.3, zorder=3)
        ax.fill_between(q.k, q.d - q.se, q.d + q.se, color=c, alpha=.16, lw=0,
                        zorder=2)
        ax.text(q.k.iloc[-1] + .15, q.d.iloc[-1], BASIS_NAME[basis], color=c,
                fontsize=6.8, va='center')
    kk = P.dropna(subset=['k'])
    ax.plot(kk.k, kk.d, 'o', ms=3.6, mfc='w', mew=1.1, color='0.35', zorder=4)
    r = P[P.label == ref_label]
    if len(r):
        ax.plot(r.k, r.d, 'o', ms=5.2, color='0.15', zorder=5)
    # a mixed form gives the channels different functions, so it is not a
    # point on this axis; it is drawn as the level it reaches
    for _, q in P[P.basis == 'mixed'].iterrows():
        ax.axhline(q.d, color='0.55', lw=.9, ls=(0, (4, 2.5)), zorder=1)
        # LEFT edge: the right edge is where both series put their own
        # endpoint labels, and this rule sits within a nat or two of zero,
        # which is exactly where a flat series ends up
        ax.text(kk.k.min(), q.d, 'Power + Weber', fontsize=6.4,
                color='0.5', va='top', ha='left')
    ax.axhline(0, color='0.35', lw=.9, zorder=1)
    # The Weber rung is tens of nats below everything else, and letting it set
    # the limits compresses the 2-to-7-anchor comparison -- the actual question
    # -- into a few pixels. So the axis is scaled to the rungs that are being
    # compared and Weber is drawn AT the clipped edge as a caret carrying its
    # own value: the reader sees both that it is far worse and by how much,
    # without losing the resolution where it matters.
    body = kk[kk.k >= 2]
    if len(body):
        lo = float((body.d - body.se).min())
        hi = float((body.d + body.se).max())
        pad = max((hi - lo) * .18, 2.0)
        ylo, yhi = lo - pad, hi + pad
        off = kk[(kk.d < ylo) | (kk.d > yhi)]
        for _, o in off.iterrows():
            below = o.d < ylo
            y = ylo if below else yhi
            ax.plot(o.k, y, marker='v' if below else '^', ms=5.0,
                    color='0.35', mfc='w', mew=1.1, zorder=6, clip_on=False)
            ax.annotate(f'{o.d:+.0f}', xy=(o.k, y), xytext=(7, 0),
                        textcoords='offset points', fontsize=6.6, color='0.35',
                        ha='left', va='center')
        ax.set_ylim(ylo, yhi)
    ax.set_xticks(sorted(kk.k.unique().astype(int)))
    ax.set_xlim(kk.k.min() - .4, kk.k.max() + 2.4)
    ax.set_xlabel('Anchors per noise channel')
    ax.set_ylabel('ΔELPD vs the power law (nats)')
    ax.set_title('What extra anchors buy', fontsize=9, loc='left', x=.0)
    return P


def panel_functions(axes, C):
    """Every fitted noise function, perceptual and memory."""
    ks = sorted(C.n_anchor.dropna().unique())
    ramp = sns.color_palette(RAMP, as_cmap=True)
    col = {k: ramp(v) for k, v in
           zip(ks, np.linspace(.20, .78, len(ks)) if len(ks) > 1 else [.4])}
    # a form that gives the two channels different functions has no anchor
    # count, so it cannot take a colour from the ramp; grey keeps it legible
    # without implying a position on the scale
    for ax, ch, nm in zip(axes, ('perc', 'mem'),
                          ('Perceptual noise', 'Memory noise')):
        # ONE credible band, for the reported form. Thirteen overlapping bands
        # would be unreadable, and the question these panels answer is not
        # "how wide is each fit" but "do the flexible fits say anything the
        # two-anchor fit does not" -- which is exactly whether they fall
        # inside its interval. So the band is drawn once, underneath, and
        # every other form is a line on top of it.
        ref = C[(C.channel == ch) & (C.form == REF_FORM)].sort_values('x')
        if not ref.empty:
            ax.fill_between(ref.x, ref.lo, ref.hi, color='0.55', alpha=.16,
                            lw=0, zorder=1)
        for (form, k, basis), q in C[C.channel == ch].groupby(
                ['form', 'n_anchor', 'basis'], dropna=False):
            q = q.sort_values('x')
            c = col.get(k, '0.6')
            # basis by DASH, anchor count by colour: two encodings on two
            # variables, so neither has to carry both
            ls = '-' if basis in ('smooth', 'both') else (0, (3.5, 1.8))
            ax.plot(q.x, q.mid, ls=ls, color=c, lw=1.15, zorder=3, alpha=.95)
        ax.set_xscale('log')
        ax.set_xticks([7, 14, 28, 56, 112])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel('Payoff (CHF)')
        ax.set_title(nm, fontsize=9, loc='left', x=.0)
    axes[0].set_ylabel('Representational noise ν (log units)')
    # the ramp is named by REDRAWING it: a strip of the actual line colours
    # with the end anchor counts written under it, and the two dash styles
    # shown as the dashes themselves
    # top-left of the perceptual panel: every curve rises from the bottom-left
    # to the top-right, so this corner is the one region no line reaches and
    # the key can sit at full size without crowding anything
    ax = axes[0]
    x0, w, y0 = .05, .038, .845
    for i, k in enumerate(ks):
        ax.add_patch(plt.Rectangle((x0 + i * w, y0), w, .042,
                                   transform=ax.transAxes, facecolor=col[k],
                                   lw=0, clip_on=False))
    ax.text(x0, y0 - .015, f'{int(ks[0])}', transform=ax.transAxes,
            fontsize=6.2, va='top', ha='center', color='0.35')
    ax.text(x0 + len(ks) * w, y0 - .015, f'{int(ks[-1])}',
            transform=ax.transAxes, fontsize=6.2, va='top', ha='center',
            color='0.35')
    ax.text(x0 + len(ks) * w / 2, y0 + .075, 'Anchors per channel',
            transform=ax.transAxes, fontsize=6.2, va='bottom', ha='center',
            color='0.35')
    entries = [('Smooth', '-', '0.35'), ('Piecewise linear', (0, (3.5, 1.8)),
                                         '0.35')]
    band = ('95% CrI, power law', None, '0.55')
    if (C.basis == 'mixed').any():
        entries.append(('Power + Weber', (0, (3.5, 1.8)), '0.6'))
    for i, (nm, ls, c) in enumerate(entries + [band]):
        yy = y0 - .085 - i * .072
        if ls is None:                       # the band, drawn as a band
            ax.add_patch(plt.Rectangle((x0, yy - .018), .075, .036,
                                       transform=ax.transAxes, facecolor=c,
                                       alpha=.16, lw=0, clip_on=False))
        else:
            ax.plot([x0, x0 + .075], [yy, yy], transform=ax.transAxes, ls=ls,
                    color=c, lw=1.15, clip_on=False)
        ax.text(x0 + .095, yy, nm, transform=ax.transAxes, fontsize=6.2,
                color=c, va='center')


def main(cohort, curves_tsv, data_dir, out_stem):
    from tms_risk.behavior.scripts.plot_elpd_ladder import resolve
    suffix, ref_form, caption, loo_dir = COHORT[cohort]
    dd = Path(data_dir)
    ld = dd / loo_dir
    _c = dd / 'all_klw_check.tsv'
    chk = pd.read_csv(_c if _c.exists() else dd / 'all_anchor_check.tsv',
                      **READ).set_index('trace')
    ref_label = resolve(f'log-{ref_form}-{suffix}', ld, chk) if ld.is_dir() \
        else None
    if ref_label is None:
        print(f'no pointwise LOO for log-{ref_form}-{suffix} under {ld} -- '
              f'panel a will be blank. Run extract_anchor_loo for this '
              f'cohort with --out_dir {ld}.')
    print(f'cohort {cohort}  reference {ref_label}')

    C = collect(curves_tsv, suffix)
    print(f'  {C.form.nunique()} forms: {", ".join(sorted(C.form.unique()))}')

    fig = plt.figure(figsize=(7.2, 2.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.08, 1.08])
    AX = [fig.add_subplot(gs[i]) for i in range(3)]
    try:
        if ref_label is None:
            raise SystemExit('no reference')
        P = panel_complexity(AX[0], ld, chk, ref_label, suffix, resolve)
        print(P[['form', 'k', 'basis', 'd', 'se']].to_string(index=False))
    except SystemExit as e:
        print(f'  panel a: {e}')
        AX[0].set_visible(False)
    panel_functions(AX[1:], C)
    fig.suptitle(caption, fontsize=8, color='0.4')
    for ax in AX:
        if ax.get_visible():
            sns.despine(ax=ax, offset=4)
    fig.canvas.draw()
    for letter, ax in zip('abc', [x for x in AX if x.get_visible()]):
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted())
        fig.text(bb.x0 - .012, min(bb.y1 + .035, .995), letter, fontsize=9,
                 fontweight='bold', family='Arial', va='top', ha='left')
    fig.savefig(f'{out_stem}.pdf')
    fig.savefig(f'{out_stem}.png', dpi=200)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='baseline', choices=list(COHORT))
    ap.add_argument('--curves_tsv', default=None,
                    help='default: base_curves.tsv for baseline, '
                         'flex_curves.tsv for tms')
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    dflt = {'baseline': 'base_curves.tsv', 'tms': 'flex_curves.tsv'}[a.cohort]
    main(a.cohort, a.curves_tsv or str(REPO / 'notes/data' / dflt), a.data_dir,
         a.out_stem or str(REPO / f'notes/figures/FIG_noise_flexibility_{a.cohort}'))
