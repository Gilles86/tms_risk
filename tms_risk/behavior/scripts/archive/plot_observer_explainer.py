"""What "the observer reasons in log space" actually means, in pictures.

Two things are easy to conflate, and they are independent:

  (1) THE SCALE OF INFERENCE — the observer represents log(payoff), holds a
      lognormal prior over payoffs, and combines the two by precision weighting
      on the log scale. Shrinkage is therefore multiplicative: a percept is
      pulled toward the prior by a RATIO, not by a number of CHF. This is what
      distinguishes LogFlexibleNoiseRiskModel from the natural-space model, and
      it is what the -4.63 CHF prior was a symptom of.

  (2) THE NOISE FUNCTION — how the SD of that log-scale representation varies
      with payoff. Weber says constant; generalized Weber says k + c/x; a
      spline says whatever fits.

Changing (2) does NOT change (1). Every lfx2 model, whatever its noise
function, does its Bayesian inference in log space. This figure shows what that
machinery does, using the fitted priors of the primary model.

    python -m tms_risk.behavior.scripts.plot_observer_explainer
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
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

C_PRIOR = '#7F7F7F'
C_LIK = '#3B5BA5'
C_POST = '#C44E52'
C_LOW = '#8FB3E0'      # low noise (perceptual only)
C_HIGH = '#1F4E96'     # high noise (perceptual + memory)

# Fitted priors of the primary log-space model, in payoff units:
# safe 9.19 CHF x/÷ 1.55, risky 15.73 x/÷ 1.43.
MU_P, SD_P = np.log(9.19), np.log(1.55)
LO, HI = 7.0, 112.0
X = np.geomspace(LO, HI, 300)

SIG_LOW = 0.15    # perceptual noise alone (second-presented option)
SIG_HIGH = 0.90   # perceptual + memory (first-presented, remembered)


def posterior_log(logx, sigma):
    """Precision-weighted combination on the LOG scale."""
    w = (1 / sigma ** 2) / (1 / sigma ** 2 + 1 / SD_P ** 2)
    return w * logx + (1 - w) * MU_P, w


def perceived(x, sigma):
    m, _ = posterior_log(np.log(x), sigma)
    return np.exp(m)


def logx_axis(ax, label='Payoff (CHF)'):
    ax.set_xscale('log')
    ax.set_xticks([7, 14, 28, 56, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='notes/figures/observer_explainer')
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.4), constrained_layout=True)

    # -- a: one trial, on the log axis ---------------------------------------
    ax = axes[0]
    grid = np.linspace(np.log(3), np.log(200), 500)
    true_x = 56.0
    for sigma, col, nm, off in [(SIG_HIGH, C_POST, 'Remembered\n(high noise)', 0)]:
        lik = np.exp(-.5 * ((grid - np.log(true_x)) / sigma) ** 2)
        pri = np.exp(-.5 * ((grid - MU_P) / SD_P) ** 2)
        m, _ = posterior_log(np.log(true_x), sigma)
        s = 1 / np.sqrt(1 / sigma ** 2 + 1 / SD_P ** 2)
        post = np.exp(-.5 * ((grid - m) / s) ** 2)
        ax.fill_between(np.exp(grid), 0, pri / pri.max(), color=C_PRIOR,
                        alpha=.25, lw=0)
        ax.plot(np.exp(grid), lik / lik.max(), color=C_LIK, lw=1.3)
        ax.plot(np.exp(grid), post / post.max(), color=C_POST, lw=1.5)
        ax.axvline(true_x, color='.2', lw=.7, ls='--', zorder=0)
        ax.annotate('', xy=(np.exp(m), 1.07), xytext=(true_x, 1.07),
                    arrowprops=dict(arrowstyle='-|>', color='.2', lw=1.1,
                                    mutation_scale=8))
    ax.text(np.exp(MU_P), .34, 'Prior', color='.35', fontsize=6.5, ha='center')
    ax.text(true_x * 1.05, 1.0, 'Noisy\nobservation', color=C_LIK, fontsize=6.4,
            ha='left', va='top', linespacing=1.25)
    ax.text(np.exp(m) * .93, .62, 'Percept', color=C_POST, fontsize=6.5,
            ha='right', va='center')
    ax.text(np.sqrt(true_x * np.exp(m)), 1.13, 'Shrinkage', fontsize=6.4,
            color='.2', ha='center', va='bottom')
    logx_axis(ax)
    ax.set_ylim(0, 1.32)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('Density')
    ax.set_title('Inference happens on the log axis', fontsize=7.5)

    # -- b: perceived vs objective -------------------------------------------
    ax = axes[1]
    ax.plot(X, X, color='.75', lw=.9, ls='--', zorder=0)
    ax.plot(X, perceived(X, SIG_LOW), color=C_LOW, lw=1.6)
    ax.plot(X, perceived(X, SIG_HIGH), color=C_HIGH, lw=1.6)
    ax.text(16, 22.5, 'Veridical', color='.55', fontsize=6.4,
            ha='center', va='bottom', rotation=37, rotation_mode='anchor')
    ax.text(HI * .96, perceived(HI, SIG_LOW) * .80, 'Low noise\n(σ = 0.15)',
            color=C_LOW, fontsize=6.4, ha='right', va='top', linespacing=1.25)
    ax.text(HI * .96, perceived(HI, SIG_HIGH) * .82, 'High noise\n(σ = 0.90)',
            color=C_HIGH, fontsize=6.4, ha='right', va='top', linespacing=1.25)
    ax.set_yscale('log')
    ax.set_yticks([7, 14, 28, 56, 112])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    logx_axis(ax)
    ax.set_ylabel('Perceived payoff (CHF)')
    ax.set_title('More noise, more compression', fontsize=7.5)

    # -- c: what a cTBS-sized noise increase does ----------------------------
    ax = axes[2]
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    for sig, col, nm in [(SIG_LOW, C_LOW, 'Second-presented\n(perceptual only)'),
                         (SIG_HIGH, C_HIGH, 'First-presented\n(+ memory)')]:
        d = 100 * (perceived(X, sig * 1.10) / perceived(X, sig) - 1)
        ax.plot(X, d, color=col, lw=1.6)
        ax.text(LO * 1.06, d[0] + .30, nm, color=col, fontsize=6.4,
                ha='left', va='bottom', linespacing=1.25)
    logx_axis(ax)
    ax.set_ylim(-8.4, 5.2)
    ax.set_ylabel('Δ perceived value (%)')
    ax.set_title('A 10% noise rise, by option role', fontsize=7.5)

    sns.despine(fig=fig, offset=3)
    axes[0].spines['left'].set_visible(False)
    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.22, 1.08, letter, transform=ax.transAxes, fontsize=8,
                family='Arial', fontweight='bold', va='bottom', ha='left')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.02)

    for sig, nm in [(SIG_LOW, 'perceptual only'), (SIG_HIGH, '+ memory')]:
        _, w = posterior_log(np.log(28), sig)
        print(f'sigma = {sig:.2f} ({nm:15s}): weight on the observation = {w:.3f}; '
              f'28 CHF perceived as {perceived(28.0, sig):.1f} CHF')
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
