"""The anchor-parameterised noise forms: what each can do, and where to anchor.

No fitted data here -- these are the functional forms and a property of the
design, both knowable before a single model is run. The point of the figure is
to make two decisions with eyes open:

  * which forms are genuinely different from each other, and where they differ;
  * where to place the anchors, which is not cosmetic -- anchors at the extremes
    of the payoff range sit where the data are thin, so their parameters trade
    off against each other.

    python -m tms_risk.behavior.scripts.plot_noise_forms
"""
import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.6,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

LO, HI = 7.0, 112.0
X = np.geomspace(LO, HI, 400)

C_FORM = {'weber': '#8A8A8A', 'affine': '#D8801F', 'power': '#5D8C3F',
          'genweber': '#C44E52', 'spl3': '#7B5EA7', 'spl5': '#3B5BA5'}
NICE = {'weber': 'Weber', 'affine': 'Affine', 'power': 'Power law',
        'genweber': 'Gen. Weber', 'spl3': 'Spline, 3 anchors',
        'spl5': 'Spline, 5 anchors'}


def logx(ax, label='Payoff (CHF)', ticks=(7, 14, 28, 56, 112)):
    ax.set_xscale('log')
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(label)


def letter(ax, s, dx=-0.22, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, family='Arial',
            fontweight='bold', va='bottom', ha='left')


def sigma(form, anchors, s_anchor, x):
    """Evaluate a form, using bauer's own interpolation so the figure cannot
    drift away from what the models actually do."""
    import bauer.models as bm
    m = bm.LogAnchorNoiseRiskModel.__new__(bm.LogAnchorNoiseRiskModel)
    m.noise_form = form
    m.n_anchors, m.anchor_link = bm.NOISE_FORMS[form]
    m._anchors = np.asarray(anchors, float)
    B = m.interp_matrix(x)
    th = np.log(np.asarray(s_anchor, float))
    return np.exp(B @ th) if m.anchor_link == 'log' else B @ np.exp(th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--bauer', default='libs/bauer')
    ap.add_argument('--out', default='notes/figures/noise_forms')
    args = ap.parse_args()
    sys.path.insert(0, args.bauer)

    from tms_risk.behavior.fit_model import get_data
    import bauer.models as bm
    df = get_data(args.bids_folder, model_label='lfx2-bs3-m2-dp-bm')

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.9), constrained_layout=True)

    # -- a: the six forms through the same two endpoint values ---------------
    ax = axes[0, 0]
    s_lo, s_hi = 0.13, 0.29
    for form in ['weber', 'genweber', 'affine', 'power']:
        k = bm.NOISE_FORMS[form][0]
        anc = np.array([LO]) if k == 1 else np.array([LO, HI])
        sa = [s_lo] if k == 1 else [s_lo, s_hi]
        ax.plot(X, sigma(form, anc, sa, X), color=C_FORM[form], lw=1.5)
    ax.plot([LO, HI], [s_lo, s_hi], 'o', color='.15', ms=5, zorder=6)
    for form, y, va in [('genweber', 0.255, 'bottom'), ('affine', 0.212, 'bottom'),
                        ('power', 0.176, 'top'), ('weber', 0.126, 'top')]:
        ax.text(30, y, NICE[form], color=C_FORM[form], fontsize=6.6,
                ha='center', va=va)
    logx(ax)
    ax.set_ylim(0.10, 0.33)
    ax.set_ylabel('Noise SD (log units)')
    ax.set_title('Same two anchors, four shapes', fontsize=7.6)
    ax.text(.97, .96, 'Dots: the free parameters', transform=ax.transAxes,
            fontsize=6.4, color='.45', ha='right', va='top')

    # -- b: affine vs power law, the whole difference ------------------------
    ax = axes[0, 1]
    u = (np.log(X) - np.log(LO)) / (np.log(HI) - np.log(LO))
    for r, sh in zip([1.5, 2.23, 4.0, 10.0], np.linspace(.30, 1.0, 4)):
        aff = s_lo * (1 - u) + s_lo * r * u
        pw = np.exp(np.log(s_lo) * (1 - u) + np.log(s_lo * r) * u)
        ax.plot(X, 100 * (aff / pw - 1), color='#8A5FB0', lw=1.4, alpha=sh)
        j = int(np.argmax(aff / pw))
        ax.text(X[j], 100 * (aff / pw - 1).max() + 1.6, f'r = {r:g}',
                color='#8A5FB0', alpha=sh, fontsize=6.4, ha='center',
                va='bottom')
    logx(ax)
    ax.set_ylabel('Affine above power law (%)')
    ax.set_title('They differ only between the anchors', fontsize=7.6)
    ax.set_ylim(-3, 100)
    ax.text(.97, .55, 'r = ratio of the two anchor values\n'
                      'Equal at r = 1 (both are Weber)',
            transform=ax.transAxes, fontsize=6.4, color='.35', ha='right',
            va='top', linespacing=1.3)

    # -- c: the two spline forms --------------------------------------------
    ax = axes[0, 2]
    for form, anc, sa in [('spl3', [7, 28, 112], [0.13, 0.24, 0.29]),
                          ('spl5', [7, 14, 28, 56, 112],
                           [0.13, 0.15, 0.24, 0.22, 0.29])]:
        ax.plot(X, sigma(form, anc, sa, X), color=C_FORM[form], lw=1.5)
        ax.plot(anc, sa, 'o', color=C_FORM[form], ms=4.5, zorder=6)
        ax.text(HI * 1.04, sa[-1] + (0.012 if form == 'spl3' else -0.012),
                NICE[form], color=C_FORM[form], fontsize=6.6, ha='left',
                va='bottom' if form == 'spl3' else 'top')
    logx(ax)
    ax.set_xlim(LO * .95, HI * 2.6)
    ax.set_ylim(0.10, 0.33)
    ax.set_ylabel('Noise SD (log units)')
    ax.set_title('More anchors, more freedom', fontsize=7.6)

    # -- d: where the payoffs actually are -----------------------------------
    ax = axes[1, 0]
    bins = np.geomspace(LO * .9, HI * 1.1, 26)
    for col, key, nm in [('.75', 'n_safe', 'Safe'), ('.45', 'n_risky', 'Risky')]:
        v = df[key].values
        ax.hist(v[np.isfinite(v)], bins=bins, color=col, alpha=.75, lw=0)
        ax.text(np.exp(np.log(v[np.isfinite(v)]).mean()), 0, f'  {nm}',
                color=col if col != '.75' else '.55', fontsize=6.6,
                ha='left', va='bottom')
    for a in (7, 28, 112):
        ax.axvline(a, color='#7B5EA7', lw=0.9, ls='--', zorder=5)
    logx(ax)
    ax.set_ylabel('Trials')
    ax.set_title('Anchors sit where the data are thin', fontsize=7.6)
    ax.text(.97, .95, 'Dashed: 7 / 28 / 112', color='#7B5EA7', fontsize=6.4,
            transform=ax.transAxes, ha='right', va='top')

    # -- e: anchor placement vs identifiability ------------------------------
    ax = axes[1, 1]
    pairs = [(7, 112), (10, 80), (14, 56), (7, 28), (28, 112)]
    rs = []
    for lo, hi in pairs:
        m = bm.LogAnchorNoiseRiskRegressionModel(
            df, noise_form='affine', anchors=(lo, hi),
            memory_model='shared_perceptual_noise', prior_estimate='full')
        rs.append(m.anchor_correlation()[0, 1])
    y = np.arange(len(pairs))[::-1]
    ax.barh(y, rs, height=.6, color='#8A5FB0', alpha=.85, lw=0)
    for yi, (p, r) in zip(y, zip(pairs, rs)):
        ax.text(r + .015, yi, f'{r:+.2f}', fontsize=6.4, va='center', color='.25')
        ax.text(-0.02, yi, f'{p[0]} & {p[1]}', fontsize=6.6, va='center',
                ha='right', color='.15')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 0.78)
    ax.set_xlabel('Correlation between the two anchor parameters')
    ax.set_title('The default (7 & 112) is the worst', fontsize=7.6)

    # -- f: what clamping outside the anchors costs --------------------------
    ax = axes[1, 2]
    for anc, col, nm in [((7, 112), '#C44E52', '7 & 112'),
                         ((28, 112), '#3B5BA5', '28 & 112')]:
        s = sigma('affine', anc, [0.13, 0.29], X)
        ax.plot(X, s, color=col, lw=1.5)
        ax.plot(anc, [0.13, 0.29], 'o', color=col, ms=4.5, zorder=6)
        ax.text(LO * 1.03, s[0] + (0.006 if anc[0] == 7 else -0.010), nm,
                color=col, fontsize=6.6, ha='left',
                va='bottom' if anc[0] == 7 else 'top')
    logx(ax)
    ax.set_ylim(0.10, 0.33)
    ax.set_ylabel('Noise SD (log units)')
    ax.set_title('Anchors must span the range', fontsize=7.6)
    ax.text(.97, .30, 'Below 28 the blue model\ncannot vary at all',
            transform=ax.transAxes, fontsize=6.4, color='.35', ha='right',
            va='top', linespacing=1.3)

    sns.despine(fig=fig, offset=3)
    for ax, s in zip(axes.ravel(), 'abcdef'):
        letter(ax, s)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.03)

    print('anchor-pair correlations (lower is better identified):')
    for p, r in zip(pairs, rs):
        print(f'  {p[0]:3d} & {p[1]:3d}   r = {r:+.3f}')
    m3 = bm.LogAnchorNoiseRiskRegressionModel(
        df, noise_form='spl3', anchors=(7, 28, 112),
        memory_model='shared_perceptual_noise', prior_estimate='full')
    C = m3.anchor_correlation()
    print(f'  spl3 7/28/112: 7-28 {C[0,1]:+.3f}  28-112 {C[1,2]:+.3f}  '
          f'7-112 {C[0,2]:+.3f}')
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
