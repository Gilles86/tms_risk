"""How perceptual and memory noise combine into the two options' noise.

Under the anchor parameterisation the first-presented option's noise is the
**sum of two positive functions**::

    sigma_n1(x) = sigma_perc(x) + sigma_mem(x)
    sigma_n2(x) = sigma_perc(x)

rather than `softplus(eta_perc + eta_mem)`. Three consequences, one per panel
in the bottom row:

* `sigma_n1 > sigma_n2` holds **by construction**, for every parameter value.
  Under the old composition it held only where `eta_mem > 0`, and the fitted
  memory coordinate dips below zero at the low end -- so the old model was free
  to say a *remembered* option was less noisy than one still on screen.
* The value-link forms are **closed under addition**: a sum of two affine
  functions is affine, of two generalized-Weber functions is generalized Weber,
  of two constants is constant. So n1 stays in the same family as its parts.
* The power law is **not** closed. exp(a) + exp(b) is not an exponential, so
  even when both channels are power laws the first option's noise is not one.
  That is a real modelling commitment, not a technicality.

    python -m tms_risk.behavior.scripts.plot_noise_composition
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

C_PERC, C_MEM = '#3B5BA5', '#C44E52'
C_N1, C_N2 = '#1A1A1A', '#8A8A8A'


def logx(ax, label='Payoff (CHF)', ticks=(7, 14, 28, 56, 112)):
    ax.set_xscale('log')
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(label)


def letter(ax, s, dx=-0.24, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, family='Arial',
            fontweight='bold', va='bottom', ha='left')


def sigma(form, anchors, s_anchor, x):
    """bauer's own interpolation, so the figure cannot drift from the models."""
    import bauer.models as bm
    m = bm.LogAnchorNoiseRiskModel.__new__(bm.LogAnchorNoiseRiskModel)
    m.noise_form = form
    m.n_anchors, m.anchor_link = bm.NOISE_FORMS[form]
    m._anchors = np.asarray(anchors, float)
    B = m.interp_matrix(x)
    th = np.log(np.asarray(s_anchor, float))
    return np.exp(B @ th) if m.anchor_link == 'log' else B @ np.exp(th)


def softplus(u):
    return np.log1p(np.exp(-np.abs(u))) + np.maximum(u, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bauer', default='libs/bauer')
    ap.add_argument('--out', default='notes/figures/noise_composition')
    args = ap.parse_args()
    sys.path.insert(0, args.bauer)

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.9), constrained_layout=True)

    # -- a: the composition, on an affine example ----------------------------
    ax = axes[0, 0]
    perc = sigma('affine', (LO, HI), [0.12, 0.26], X)
    mem = sigma('affine', (LO, HI), [0.20, 0.08], X)
    ax.fill_between(X, perc, perc + mem, color=C_MEM, alpha=.16, lw=0)
    ax.plot(X, perc, color=C_PERC, lw=1.5)
    ax.plot(X, mem, color=C_MEM, lw=1.5, ls='--')
    ax.plot(X, perc + mem, color=C_N1, lw=1.8)
    for y, c, nm, va in [(perc[-1], C_PERC, 'Perceptual = n2', 'top'),
                         (mem[-1], C_MEM, 'Memory', 'top'),
                         ((perc + mem)[-1], C_N1, 'n1 = perceptual + memory',
                          'bottom')]:
        ax.text(HI * 0.97, y, nm, color=c, fontsize=6.6, ha='right', va=va)
    logx(ax)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel('Noise SD (log units)')
    ax.set_title('The two channels add', fontsize=7.6)

    # -- b: closed under addition (affine) -----------------------------------
    ax = axes[0, 1]
    tot = perc + mem
    ref = sigma('affine', (LO, HI), [tot[0], tot[-1]], X)
    ax.plot(X, tot, color=C_N1, lw=2.4, alpha=.35)
    ax.plot(X, ref, color='#5D8C3F', lw=1.4, ls='--')
    ax.text(HI * 0.97, tot[-1] * 1.06, 'n1 (the sum)', color=C_N1,
            fontsize=6.6, ha='right', va='bottom')
    ax.text(HI * 0.97, tot[-1] * 0.90,
            'an affine function through\nits own two endpoints',
            color='#5D8C3F', fontsize=6.6, ha='right', va='top',
            linespacing=1.25)
    logx(ax)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel('Noise SD (log units)')
    ax.set_title('Affine is closed under addition', fontsize=7.6)
    ax.text(.03, .05, 'They coincide exactly', transform=ax.transAxes,
            fontsize=6.6, color='.35')

    # -- c: the power law is not ---------------------------------------------
    ax = axes[0, 2]
    p2 = sigma('power', (LO, HI), [0.12, 0.26], X)
    m2 = sigma('power', (LO, HI), [0.20, 0.08], X)
    t2 = p2 + m2
    r2 = sigma('power', (LO, HI), [t2[0], t2[-1]], X)
    ax.plot(X, t2, color=C_N1, lw=2.4, alpha=.35)
    ax.plot(X, r2, color='#D8801F', lw=1.4, ls='--')
    dev = 100 * np.abs(t2 - r2).max() / t2.mean()
    ax.text(HI * 0.97, t2[-1] * 1.06, 'n1 (the sum)', color=C_N1,
            fontsize=6.6, ha='right', va='bottom')
    ax.text(HI * 0.97, t2[-1] * 0.90,
            'the power law through\nits own two endpoints',
            color='#D8801F', fontsize=6.6, ha='right', va='top',
            linespacing=1.25)
    logx(ax)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel('Noise SD (log units)')
    ax.set_title('The power law is not', fontsize=7.6)
    ax.text(.03, .05, f'They differ by up to {dev:.1f}%',
            transform=ax.transAxes, fontsize=6.6, color='#D8801F')

    # -- d: the old composition could invert the order -----------------------
    ax = axes[1, 0]
    eta_p = -1.7
    em = np.linspace(-2.0, 1.5, 300)
    ax.plot(em, softplus(em + eta_p), color=C_N1, lw=1.6)
    ax.plot(em, softplus(eta_p) + softplus(em), color='#5D8C3F', lw=1.6)
    ax.axhline(softplus(eta_p), color=C_N2, lw=1.2, ls=':')
    bad = em[softplus(em + eta_p) < softplus(eta_p)]
    ax.axvspan(em.min(), bad.max(), color=C_MEM, alpha=.10, lw=0)
    ax.text(bad.max() - 0.05, 0.90, 'here the OLD model made the\n'
                                    'remembered option LESS noisy',
            color=C_MEM, fontsize=6.4, ha='right', va='top', linespacing=1.25)
    ax.text(1.45, softplus(1.5 + eta_p) + 0.02, 'OLD  softplus(perc + mem)',
            color=C_N1, fontsize=6.6, ha='right', va='bottom')
    ax.text(1.45, softplus(eta_p) + softplus(1.5) - 0.02,
            'NEW  perc + mem', color='#5D8C3F', fontsize=6.6,
            ha='right', va='top')
    ax.text(-1.95, softplus(eta_p) + 0.02, 'n2', color=C_N2, fontsize=6.6,
            ha='left', va='bottom')
    ax.set_xlabel('Memory coordinate (pre-transform)')
    ax.set_ylabel('n1 (log units)')
    ax.set_ylim(0, 1.9)
    ax.set_title('n1 > n2 is now guaranteed', fontsize=7.6)

    # -- e: example curves, three forms, same anchors ------------------------
    ax = axes[1, 1]
    for form, col in [('weber', '#8A8A8A'), ('genweber', '#C44E52'),
                      ('spl3', '#7B5EA7')]:
        anc = (LO,) if form == 'weber' else ((LO, HI) if form == 'genweber'
                                             else (LO, 20.0, HI))
        sp_ = [0.12] if form == 'weber' else ([0.12, 0.26] if form == 'genweber'
                                              else [0.12, 0.22, 0.26])
        sm = [0.20] if form == 'weber' else ([0.20, 0.08] if form == 'genweber'
                                             else [0.20, 0.11, 0.08])
        p_ = sigma(form, anc, sp_, X)
        m_ = sigma(form, anc, sm, X)
        ax.plot(X, p_, color=col, lw=1.2, ls='--')
        ax.plot(X, p_ + m_, color=col, lw=1.7)
        ax.text(HI * 0.97, (p_ + m_)[-1] + 0.012, form, color=col,
                fontsize=6.6, ha='right', va='bottom')
    logx(ax)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel('Noise SD (log units)')
    ax.set_title('Solid n1, dashed n2', fontsize=7.6)

    # -- f: the memory cost, as a ratio --------------------------------------
    ax = axes[1, 2]
    for form, col in [('affine', '#D8801F'), ('genweber', '#C44E52'),
                      ('power', '#5D8C3F')]:
        p_ = sigma(form, (LO, HI), [0.12, 0.26], X)
        m_ = sigma(form, (LO, HI), [0.20, 0.08], X)
        ax.plot(X, (p_ + m_) / p_, color=col, lw=1.5)
        ax.text(HI * 0.97, ((p_ + m_) / p_)[-1], f'  {form}', color=col,
                fontsize=6.6, ha='right', va='center')
    ax.axhline(1, color='.75', lw=0.8, ls=':')
    logx(ax)
    ax.set_ylabel('n1 / n2')
    ax.set_ylim(1, 3.0)
    ax.set_title('The memory cost, relative', fontsize=7.6)
    ax.text(.97, .95, 'Always > 1', transform=ax.transAxes, fontsize=6.6,
            color='.35', ha='right', va='top')

    sns.despine(fig=fig, offset=3)
    for ax, s in zip(axes.ravel(), 'abcdef'):
        letter(ax, s)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.03)
    print(f'power-law sum deviates from a power law by up to {dev:.1f}%')
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
