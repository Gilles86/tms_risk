"""Explanatory simulation: diminishing sensitivity vs lognormal priors.

Four simulated panels (no fitting):
a  the two mechanisms draw the SAME line on log-log axes (slope = exponent);
b  simulated choices from both observers are indistinguishable;
c  where they part: raising encoding noise shifts preferences only for the
   lognormal-prior observer (whose exponent is noise-derived);
d  why a natural-space Gaussian prior + growing noise (the power-law model)
   can imitate multiplicative shrinkage inside the tested payoff range.

Writes notes/figures/sensitivity_vs_priors.pdf (exploratory, NOT paper).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'notes' / 'figures'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 9,
    'mathtext.fontset': 'stixsans',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.4, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

C_DS = '#C44E52'      # diminishing sensitivity (fixed exponent)
C_LN = '#3B5BA5'      # lognormal-prior observer (noise-derived exponent)

W = 0.6               # shared exponent
MU_L = np.log(20.0)   # lognormal prior median: 20 CHF
SIG_E = 0.35          # log-space encoding noise
P_RISKY = 0.55
rng = np.random.default_rng(7)


def logx(ax, ticks):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def panel_letter(ax, letter):
    ax.text(-0.20, 1.04, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='Arial', va='bottom', ha='right')


fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.6), constrained_layout=True)

# --- a: same line, two stories ---------------------------------------------
ax = axes[0, 0]
n = np.exp(np.linspace(np.log(2), np.log(130), 100))
v_ds = n ** W                              # utility curvature: v = n^alpha
v_ln = n ** W * np.exp((1 - W) * MU_L)     # shrinkage: x-hat = n^w e^((1-w)mu)
ax.plot(n, v_ds, color=C_DS, lw=1.4)
ax.plot(n, v_ln, color=C_LN, lw=1.4)
ax.plot([20], [20], 'o', ms=5, mfc='white', mec=C_LN, mew=1.2, zorder=4)
ax.plot(n, n, color='0.75', lw=0.7, ls='--', zorder=0)
logx(ax, [2, 5, 20, 50, 130])
ax.set_yscale('log')
ax.set_yticks([1, 3, 10, 30, 100])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Subjective value')
ax.set_title('Two stories, one line (log-log)', fontsize=9)
ax.text(90, 90 * 1.35, 'Veridical', color='0.55', fontsize=7.5,
        rotation=38, ha='center')
ax.text(34, 34 ** W * np.exp((1 - W) * MU_L) * 1.75,
        f'Lognormal prior:\nx̂ = n$^w$·e$^{{(1-w)μ}}$', color=C_LN, fontsize=7.5)
ax.text(38, 38 ** W * 0.42, f'Diminishing sensitivity:\nv = n$^α$,  α = w = {W}',
        color=C_DS, fontsize=7.5)
ax.annotate('Anchor: 20 CHF\nperceived veridically', xy=(20, 20),
            xytext=(3.1, 55), fontsize=7.5, ha='left', va='center',
            arrowprops=dict(arrowstyle='-|>', color='0.2', lw=1.0,
                            mutation_scale=8, shrinkA=2, shrinkB=6,
                            relpos=(0.6, 0.0)))
panel_letter(ax, 'a')

# --- b: simulated choices are indistinguishable ----------------------------
ratios = np.exp(np.linspace(np.log(0.6), np.log(4.5), 12))
SAFE = 20.0
THR = np.log(1 / P_RISKY)                  # log(p_safe / p_risky)


def p_choice_ln(ratio, sig_e, sig_p):
    w = sig_p ** 2 / (sig_p ** 2 + sig_e ** 2)
    # log DV: w*log(risky) - w*log(safe) vs threshold; anchor cancels
    return norm.cdf((w * np.log(ratio) - THR) / (np.sqrt(2) * sig_e))


def p_choice_ds(ratio, alpha, sig_c):
    return norm.cdf((alpha * np.log(ratio) - THR) / sig_c)


SIG_P = SIG_E * np.sqrt(W / (1 - W))       # gives w = W exactly
ax = axes[0, 1]
rr = np.exp(np.linspace(np.log(0.55), np.log(4.8), 200))
ax.plot(rr, p_choice_ln(rr, SIG_E, SIG_P), color=C_LN, lw=1.6)
ax.plot(rr, p_choice_ds(rr, W, np.sqrt(2) * SIG_E), color=C_DS, lw=1.6,
        ls=(0, (4, 3)))
for pfun, c, dx in [(lambda r: p_choice_ln(r, SIG_E, SIG_P), C_LN, 0.97),
                    (lambda r: p_choice_ds(r, W, np.sqrt(2) * SIG_E), C_DS, 1.03)]:
    sim = rng.binomial(300, pfun(ratios)) / 300
    ax.plot(ratios * dx, sim, 'o', ms=3.5, color=c, mec='white', mew=.4)
ax.axhline(.5, color='0.8', lw=.6, ls='--', zorder=0)
logx(ax, [0.6, 1, 1.82, 4.5])
ax.set_ylim(0, 1)
ax.set_yticks([0, .5, 1])
ax.set_xlabel('Risky / safe payoff ratio')
ax.set_ylabel('P(choose risky)')
ax.set_title('Simulated choices: identical', fontsize=9)
ax.text(0.03, 0.95, 'Solid: lognormal prior\nDashed: fixed exponent\n'
        'Dots: 300 simulated trials each', transform=ax.transAxes,
        fontsize=7.5, va='top')
panel_letter(ax, 'b')

# --- c: what a noise increase does -----------------------------------------
ax = axes[1, 0]
SIG_E2 = SIG_E * 1.5
for sig_e, lw_, alpha_ in [(SIG_E, 1.0, .45), (SIG_E2, 1.8, 1.0)]:
    ax.plot(rr, p_choice_ln(rr, sig_e, SIG_P), color=C_LN, lw=lw_, alpha=alpha_)
    ax.plot(rr, p_choice_ds(rr, W, np.sqrt(2) * sig_e), color=C_DS, lw=lw_,
            ls=(0, (4, 3)), alpha=alpha_)
# indifference ratios
r_ind1 = np.exp(THR / W)
w2 = SIG_P ** 2 / (SIG_P ** 2 + SIG_E2 ** 2)
r_ind2 = np.exp(THR / w2)
ax.vlines([r_ind1], 0, .5, color='0.6', lw=.7, ls=':')
ax.vlines([r_ind2], 0, .5, color=C_LN, lw=.7, ls=':')
ax.axhline(.5, color='0.8', lw=.6, ls='--', zorder=0)
logx(ax, [0.6, 1, 1.82, 4.5])
ax.set_ylim(0, 1)
ax.set_yticks([0, .5, 1])
ax.set_xlabel('Risky / safe payoff ratio')
ax.set_ylabel('P(choose risky)')
ax.set_title('Noise ×1.5 (thick): only the prior story\nshifts preference',
             fontsize=9)
ax.annotate('Indifference moves:\nrisk attitude changes', xy=(r_ind2, .5),
            xytext=(0.62, .80), fontsize=7.5, color=C_LN, ha='left',
            va='center',
            arrowprops=dict(arrowstyle='-|>', color=C_LN, lw=1.0,
                            mutation_scale=8, shrinkA=2, shrinkB=5,
                            relpos=(1.0, 0.5)))
ax.text(0.03, 0.97, 'Fixed exponent: curve only\nflattens around the same point',
        transform=ax.transAxes, fontsize=7.5, color=C_DS, va='top')
panel_letter(ax, 'c')

# --- d: Gaussian prior + growing noise imitates multiplicative shrinkage ---
ax = axes[1, 1]
MU_G, SIG_G = 23.5, 15.4          # fitted power2_full risky prior (natural CHF)
C_I, BETA = 0.53, 0.71            # fitted first-option noise: (e^perc+e^mem)~, exp
n_wide = np.exp(np.linspace(np.log(2), np.log(320), 200))
nu = 0.60 * n_wide ** BETA        # perc+mem combined scale (fitted approx)
w_g = SIG_G ** 2 / (SIG_G ** 2 + nu ** 2)
xhat_g = MU_G + w_g * (n_wide - MU_G)
# matched multiplicative observer: log-log LS fit inside the tested range
m = (n_wide >= 7) & (n_wide <= 112)
slope, inter = np.polyfit(np.log(n_wide[m]), np.log(np.maximum(xhat_g[m], .1)), 1)
xhat_m = np.exp(inter) * n_wide ** slope
ax.plot(n_wide, n_wide, color='0.8', lw=0.7, ls='--', zorder=0)
ax.plot(n_wide, xhat_g, color=C_LN, lw=1.5)
ax.plot(n_wide, xhat_m, color=C_DS, lw=1.5, ls=(0, (4, 3)))
ax.axvspan(7, 112, color='0.93', zorder=0)
logx(ax, [2, 7, 28, 112, 320])
ax.set_yscale('log')
ax.set_yticks([3, 10, 30, 100])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Perceived payoff (CHF)')
ax.set_title('Gaussian prior + growing noise ≈\npower law inside the tested range',
             fontsize=9)
ax.text(24, 8.3, 'Tested range 7–112', fontsize=7, color='0.45', ha='center')
ax.text(0.03, 0.95, f'Gaussian prior N({MU_G:.0f}, {SIG_G:.0f}) +\nσ(n) ~ n^{BETA}',
        transform=ax.transAxes, fontsize=7.5, color=C_LN, va='top')
ax.text(0.97, 0.16, f'Matched power law\nslope {slope:.2f}',
        transform=ax.transAxes, fontsize=7.5, color=C_DS, ha='right')
panel_letter(ax, 'd')

import seaborn as sns
sns.despine(fig=fig, offset=5, trim=True)
fig.savefig(OUT / 'sensitivity_vs_priors.pdf')
fig.savefig(OUT / 'sensitivity_vs_priors.png', dpi=150)
print('wrote', OUT / 'sensitivity_vs_priors.pdf')
