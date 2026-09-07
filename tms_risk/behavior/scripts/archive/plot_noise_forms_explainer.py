"""What generalized Weber is, and how it differs from the alternatives.

Didactic figure -- no data, no fits, just the functional forms, so the choice
between them can be made with eyes open.

The one idea: all four models say relative (log-scale) noise either is or is not
constant, and they differ only in HOW it departs from constant. Which of them can
even produce the shape our memory-noise curve shows is a question about the forms,
not about the fits.

    sigma_abs = absolute SD of the internal representation, in CHF
    sigma_rel = sigma_abs / payoff, i.e. SD on a log/ratio scale

    Weber              sigma_rel = k                    (1 parameter)
    Generalized Weber  sigma_rel = k + c/x              (2)  <- delta_x/(x+a) = k
    Power law          sigma_rel = c * x**(beta - 1)    (2)  Stevens / efficient coding
    Affine in log      sigma_rel = alpha + beta*log(x)  (2)  the convenient line

    python -m tms_risk.behavior.scripts.plot_noise_forms_explainer
"""
import argparse
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
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.4, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

C_WEB = '#7F7F7F'    # Weber
C_GW = '#C44E52'     # generalized Weber
C_POW = '#3B5BA5'    # power law
C_AFF = '#D8801F'    # affine in log

LO, HI = 7.0, 112.0
X = np.geomspace(LO, HI, 400)

# The fitted memory-noise curve, read off the primary model (lfx2-bs3-m2-dp-bm):
# 2 df = linear in log payoff, running 1.10 at 7 CHF to 0.70 at 112 CHF.
FIT_LO, FIT_HI = 1.10, 0.70


def affine_through(lo, hi):
    """sigma = alpha + beta*log x through the two endpoints."""
    beta = (hi - lo) / (np.log(HI) - np.log(LO))
    return lambda x: lo + beta * (np.log(x) - np.log(LO)), beta


def gw_through(lo, hi):
    """sigma = k + c/x through the two endpoints; returns also a = c/k in CHF."""
    c = (lo - hi) / (1 / LO - 1 / HI)
    k = lo - c / LO
    return (lambda x: k + c / x), k, c


def power_through(lo, hi):
    """sigma_rel = c * x**(beta-1) through the two endpoints."""
    p = np.log(hi / lo) / np.log(HI / LO)          # exponent on sigma_rel
    c = lo / LO ** p
    return (lambda x: c * x ** p), p


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 14, 28, 56, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='notes/figures/noise_forms_explainer')
    args = ap.parse_args()

    f_aff, beta_aff = affine_through(FIT_LO, FIT_HI)
    f_gw, k_gw, c_gw = gw_through(FIT_LO, FIT_HI)
    f_pow, p_pow = power_through(FIT_LO, FIT_HI)
    a_chf = c_gw / k_gw

    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.25), constrained_layout=True)

    # -- a: absolute noise. Weber is a line through the ORIGIN; generalized
    #       Weber is the same line lifted off it. That offset IS the model. ----
    ax = axes[0]
    xl = np.linspace(0, HI, 200)
    ax.plot(xl, k_gw * xl, color=C_WEB, lw=1.4)
    ax.plot(xl, k_gw * (xl + a_chf), color=C_GW, lw=1.6)
    ax.plot([0], [k_gw * a_chf], 'o', color=C_GW, ms=4, zorder=4)
    ax.annotate(f'Noise floor\na = {a_chf:.1f} CHF',
                xy=(0, k_gw * a_chf), xytext=(17, k_gw * a_chf + 21),
                fontsize=6.4, color=C_GW, ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', color=C_GW, lw=1.0,
                                mutation_scale=8, shrinkA=3, shrinkB=6,
                                relpos=(0.0, 0.3),
                                connectionstyle='angle3,angleA=0,angleB=70'))
    ax.text(HI * .99, k_gw * HI * .74, 'Weber', color=C_WEB, fontsize=6.5,
            ha='right', va='top')
    ax.text(52, k_gw * (52 + a_chf) + 9, 'Generalized Weber', color=C_GW,
            fontsize=6.5, ha='center', va='bottom', rotation=27,
            rotation_mode='anchor')
    ax.set_xlim(0, HI)
    ax.set_xticks([0, 28, 56, 84, 112])
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Absolute noise SD (CHF)')
    ax.set_ylim(0, 92)
    ax.set_title('Same slope, lifted off zero', fontsize=7.5)

    # -- b: the same two on the RELATIVE scale, which is what gets plotted ----
    ax = axes[1]
    ax.plot(X, np.full_like(X, k_gw), color=C_WEB, lw=1.4)
    ax.plot(X, f_gw(X), color=C_GW, lw=1.6)
    ax.axhline(k_gw, color=C_WEB, lw=.7, ls=':', zorder=0)
    ax.text(HI * .97, k_gw - .05, f'Asymptote k = {k_gw:.2f}', color=C_WEB,
            fontsize=6.4, ha='right', va='top')
    ax.text(9, 1.0, 'Generalized\nWeber', color=C_GW, fontsize=6.5,
            ha='left', va='top', linespacing=1.3)
    logx(ax)
    ax.set_ylabel('Relative noise SD (log units)')
    ax.set_ylim(0, 1.25)
    ax.set_title('A floor looks like a decline', fontsize=7.5)

    # -- c: what the two parameters do ---------------------------------------
    ax = axes[2]
    for a, sh in zip([0, 2, 5, 12], np.linspace(.25, .85, 4)):
        col = plt.get_cmap('rocket')(sh)
        ax.plot(X, k_gw * (1 + a / X), color=col, lw=1.3)
        ax.text(LO * 0.94, k_gw * (1 + a / LO), f'a = {a:g}',
                color=col, fontsize=6.2, ha='right', va='center')
    logx(ax)
    ax.set_ylabel('Relative noise SD (log units)')
    ax.set_ylim(0, 1.75)
    ax.set_xlim(LO * .62, HI * 1.06)
    ax.set_title(f'Bigger floor, steeper fall (k = {k_gw:.2f})', fontsize=7.5)

    # -- d: can each form reproduce the fitted memory curve? -----------------
    ax = axes[3]
    ax.plot(X, f_aff(X), color=C_AFF, lw=1.5)
    ax.plot(X, f_gw(X), color=C_GW, lw=1.5)
    ax.plot(X, f_pow(X), color=C_POW, lw=1.5, ls='--')
    ax.plot([LO, HI], [FIT_LO, FIT_HI], 'o', color='.15', ms=4.5, zorder=5)
    mid = 30.0
    ax.annotate('They differ\nmost here',
                xy=(mid, (f_aff(mid) + f_gw(mid)) / 2),
                xytext=(9.4, .43), fontsize=6.4, color='.25',
                ha='left', va='center', linespacing=1.3,
                arrowprops=dict(arrowstyle='-|>', color='.25', lw=1.0,
                                mutation_scale=8, shrinkA=3, shrinkB=7,
                                relpos=(1.0, 0.5),
                                connectionstyle='angle3,angleA=0,angleB=60'))
    ax.text(HI * .97, f_aff(HI) + .30, 'Affine in log', color=C_AFF,
            fontsize=6.4, ha='right', va='bottom')
    ax.text(HI * .97, f_gw(HI) - .07, 'Gen. Weber', color=C_GW,
            fontsize=6.4, ha='right', va='top')
    ax.text(HI * .97, f_pow(HI) + .10, 'Power law', color=C_POW,
            fontsize=6.4, ha='right', va='bottom')
    logx(ax)
    ax.set_ylabel('Relative noise SD (log units)')
    ax.set_ylim(0, 1.35)
    ax.set_title('All three hit the endpoints', fontsize=7.5)

    sns.despine(fig=fig, offset=3)
    for ax, letter in zip(axes, 'abcd'):
        ax.text(-0.30, 1.10, letter, transform=ax.transAxes, fontsize=8,
                family='Arial', fontweight='bold', va='bottom', ha='left')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.02)

    print(f'Through the fitted memory endpoints ({FIT_LO} at {LO:g} CHF, '
          f'{FIT_HI} at {HI:g} CHF):')
    print(f'  Generalized Weber   k = {k_gw:.3f}, c = {c_gw:.3f}  '
          f'-> noise floor a = c/k = {a_chf:.2f} CHF')
    print(f'  Affine in log       slope = {beta_aff:.3f} per log-CHF')
    print(f'  Power law           sigma_rel exponent = {p_pow:.3f} '
          f'(sigma_abs ~ x**{1 + p_pow:.3f})')
    print(f'  At 30 CHF: affine {f_aff(30):.3f} | gen-Weber {f_gw(30):.3f} '
          f'| power {f_pow(30):.3f}')
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
