"""One page explaining the model family: why the observer works in log space, and
which noise function the data pick.

Written for a colleague who has not followed the model sweep. Three blocks:

  Row 1  WHY LOG SPACE.  Multiplying a payoff by the win probability p is a
         payoff-dependent operation on a linear axis (it moves 7 CHF by 3 CHF and
         112 CHF by 50, and rescales the belief's width) and a single rigid shift
         of log p on a log axis. The prior a linear-scale observer needs sits
         below every payoff ever shown. And the choice data themselves are far
         closer to a function of the ratio than of the CHF difference in expected
         value.

  Row 2  THE FIVE NOISE FUNCTIONS, as forms, with what each can and cannot do.

  Row 3  WHAT THE DATA PICK.  Fitted noise of the second-presented option under
         four different parameterizations; the ELPD ladder over converged fits;
         and the by-stake posterior predictive check that separates Weber from
         the magnitude-dependent forms.

Everything is read from TSVs already extracted from the traces (notes/data/cards,
notes/data/ladder_v12.tsv) plus the raw behaviour; no trace, no bauer, no GPU.

    python -m tms_risk.behavior.scripts.plot_model_overview
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7.6, 'axes.labelsize': 8.3, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.4, 'ytick.labelsize': 7.4, 'legend.fontsize': 7.1,
    'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

P_WIN = 0.55                       # the risky option's win probability, fixed by design
LO, HI = 7.0, 112.0                # payoff range of the paradigm
X = np.geomspace(LO, HI, 400)

# Family colours, used identically in rows 2 and 3.
C_WEB = '#8A8A8A'      # Weber
C_SPN = '#D8801F'      # spline, natural scale
C_SPL = '#3B5BA5'      # spline, log scale  (the primary model)
C_GW = '#C44E52'       # generalized Weber
C_POW = '#5D8C3F'      # power law
C_DATA = '#1A1A1A'
IPS, VERTEX = '#d62728', '#2ca02c'   # house palette: stimulated red, sham green

# Stake terciles get their own purple ramp: blue/orange/red/green are already
# spoken for by the model families, so a fourth hue keeps the semantics clean.
C_STAKE = ['#C3A8DC', '#8A5FB0', '#43226B']
STAKE_LABEL = ['7-17 CHF', '18-30 CHF', '31-70 CHF']

CARDS = Path('notes/data/cards')


def logx(ax, label='Payoff (CHF)', ticks=(7, 14, 28, 56, 112)):
    ax.set_xscale('log')
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(label)


def panel_letter(ax, letter, dx=-0.20, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9.2,
            family='Arial', fontweight='bold', va='bottom', ha='left')


def titled(ax, name, message=None, color='.12'):
    """Left-aligned title; the second line carries the panel's message."""
    txt = name if message is None else f'{name}\n{message}'
    ax.set_title(txt, fontsize=7.9, loc='left', color=color, linespacing=1.32,
                 pad=4)


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


# ----------------------------------------------------------------------------
# Row 1 -- why log space
# ----------------------------------------------------------------------------

def draw_multiply_linear(ax):
    """Multiplying by p on a linear payoff axis: a different move at every payoff."""
    sigma_rel = 0.22
    anchors = [7.0, 28.0, 112.0]
    grid = np.linspace(-20, 175, 900)
    base = 0.0
    for i, a in enumerate(anchors):
        y0 = i * 1.55
        for val, col, alpha in [(a, C_DATA, .16), (P_WIN * a, C_SPL, .30)]:
            d = gauss(grid, val, sigma_rel * val)
            ax.fill_between(grid, y0, y0 + 0.95 * d, color=col, alpha=alpha, lw=0)
            ax.plot(grid, y0 + 0.95 * d, color=col, lw=0.9)
        ax.annotate('', xy=(P_WIN * a, y0 + 1.03), xytext=(a, y0 + 1.03),
                    arrowprops=dict(arrowstyle='-|>', color='.25', lw=0.9,
                                    mutation_scale=6))
        ax.text(P_WIN * a, y0 + 1.07, f'\u2212{a - P_WIN * a:.0f} CHF',
                fontsize=6.8, color='.25', ha='left', va='bottom')
        base = y0
    ax.axvline(0, color='.75', lw=0.6, ls=':', zorder=0)
    ax.set_xlim(-20, 175)
    ax.set_ylim(-0.05, base + 1.45)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Payoff (CHF), linear axis')
    ax.set_ylabel('Belief')
    titled(ax, 'Linear scale',
           'Taking the expected value is a\ndifferent move at every payoff')
    ax.text(0.55, 0.31, 'Belief about the payoff', color=C_DATA, fontsize=6.9,
            ha='left', va='center', transform=ax.transAxes)
    ax.text(0.55, 0.19, 'Belief about its\nexpected value', color=C_SPL,
            fontsize=6.9, ha='left', va='center', linespacing=1.25,
            transform=ax.transAxes)


def draw_multiply_log(ax):
    """The same operation on a log axis: one rigid shift, shape untouched."""
    sigma = 0.30
    anchors = [7.0, 28.0, 112.0]
    grid = np.linspace(np.log(1.6), np.log(190), 700)
    shift = np.log(P_WIN)
    base = 0.0
    for i, a in enumerate(anchors):
        y0 = i * 1.55
        for val, col, alpha in [(np.log(a), C_DATA, .16),
                                (np.log(a) + shift, C_SPL, .30)]:
            d = gauss(grid, val, sigma)
            ax.fill_between(np.exp(grid), y0, y0 + 0.95 * d, color=col,
                            alpha=alpha, lw=0)
            ax.plot(np.exp(grid), y0 + 0.95 * d, color=col, lw=0.9)
        ax.annotate('', xy=(a * P_WIN, y0 + 1.03), xytext=(a, y0 + 1.03),
                    arrowprops=dict(arrowstyle='-|>', color='.25', lw=0.9,
                                    mutation_scale=6))
        ax.text(a * P_WIN, y0 + 1.07, f'\u2212{-shift:.2f} log units',
                fontsize=6.8, color='.25', ha='left', va='bottom')
        base = y0
    logx(ax, 'Payoff (CHF), log axis', ticks=(2, 7, 28, 112))
    ax.set_xlim(1.6, 190)
    ax.set_ylim(-0.05, base + 1.45)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    titled(ax, 'Log scale',
           'It is one rigid shift of log p,\nthe same at every payoff')
    ax.text(0.60, 0.20, 'Same width,\nsame shape,\nnever below zero', color='.3',
            fontsize=6.9, ha='left', va='center', linespacing=1.25,
            transform=ax.transAxes)


def draw_priors(ax, bids_folder):
    """The prior each observer needs, against the payoffs actually presented.

    Linear-scale numbers are the group posteriors of `flexible1_noisefix.head`
    (Gaussian, in CHF); log-scale ones are the primary model's lognormal priors,
    quoted as a median x/div spread.
    """
    from tms_risk.utils.data import get_all_behavior
    d = get_all_behavior(bids_folder=bids_folder)
    pay = np.concatenate([d['n_safe'].values, d['n_risky'].values])
    pay = pay[np.isfinite(pay)]

    bins = np.geomspace(LO * .9, HI * 1.15, 22)
    h, _ = np.histogram(pay, bins=bins, density=True)
    ax.bar(bins[:-1], h / h.max(), width=np.diff(bins), align='edge',
           color='.87', edgecolor='none', zorder=0)

    g = np.geomspace(1.2, 300, 3000)
    for mu, sd, nm in [(4.90, 1.06, 'Safe'), (9.95, 1.31, 'Risky')]:
        ax.plot(g, gauss(g, mu, sd), color=C_SPN, lw=1.3, zorder=3)
    for med, spread, nm in [(9.23, 1.556, 'Safe'), (15.75, 1.432, 'Risky')]:
        ax.plot(g, gauss(np.log(g), np.log(med), np.log(spread)),
                color=C_SPL, lw=1.3, zorder=2)

    ax.axvline(LO, color='.45', lw=0.7, ls='--', zorder=1)
    logx(ax, ticks=(7, 14, 28, 56, 112))
    ax.set_xlim(1.6, 260)
    ax.set_ylim(0, 1.95)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('Density (peak-scaled)')
    titled(ax, 'The prior each observer needs',
           'Linear scale: 3x below the mean payoff')
    ax.text(0.985, 0.99, 'Linear scale, Gaussian:\nsafe 4.9, risky 10.0 CHF',
            color=C_SPN, fontsize=6.8, ha='right', va='top',
            linespacing=1.25, transform=ax.transAxes)
    ax.text(0.985, 0.72, 'Log scale, lognormal:\nsafe 9.2, risky 15.8 CHF',
            color=C_SPL, fontsize=6.8, ha='right', va='top',
            linespacing=1.25, transform=ax.transAxes)
    ax.text(0.985, 0.17, 'Gray: payoffs shown (means 15.8, 36.1)\n'
                         'Dashed: 7 CHF, the smallest',
            color='.45', fontsize=6.3, ha='right', va='center',
            linespacing=1.25, transform=ax.transAxes)


def draw_perceived(ax, primary='lfx2-bs3-m2-dp-bm'):
    """Perceived vs objective payoff, for both observers.

    The linear-scale numbers are read straight off the fitted natural-space model
    (`decision_space.flexible1nf.tsv`, which tabulates the perceived value of each
    option). The log-scale ones are the precision-weighted combination the primary
    model implements, evaluated with its own fitted prior and noise curve:

        perceived = exp(w log x + (1 - w) log m),  w = s_p^2 / (s_p^2 + n2(x)^2)

    The point of the panel: the linear-scale observer's value scale SATURATES.
    Doubling 56 CHF to 112 CHF moves its percept by under 2 CHF, so what counts as
    a fair gamble has to change drastically with the stake. The log-scale observer
    compresses by a roughly constant ratio instead.
    """
    d = pd.read_csv('notes/data/decision_space.flexible1nf.tsv', sep='\t')
    d = d[d.order == 'Risky first'].copy()
    d['n_risky'] = (d.n_safe * d.ratio).round(1)
    lin = d.assign(pr=d.ev_risky_vertex / P_WIN).groupby('n_risky').pr.mean()

    g = _curve(primary, 'n2 (second)')
    x = g.payoff.values
    n2 = g['median'].values
    med, sp = 15.75, np.log(1.432)                 # primary model's risky prior
    w = sp ** 2 / (sp ** 2 + n2 ** 2)
    log_p = np.exp(w * np.log(x) + (1 - w) * np.log(med))

    ax.plot(x, x, color='.75', lw=0.9, ls='--', zorder=0)
    ax.plot(lin.index, lin.values, color=C_SPN, lw=1.5)
    ax.plot(x, log_p, color=C_SPL, lw=1.5)
    ax.plot([56, 112], [lin.loc[56.0], lin.loc[112.0]], 'o', ms=3.0,
            color=C_SPN, zorder=4)
    logx(ax, 'Objective payoff (CHF)')
    ax.set_yscale('log')
    ax.set_yticks([7, 14, 28, 56, 112])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(LO * .95, HI * 1.05)
    ax.set_ylim(5.5, 150)
    ax.set_ylabel('Perceived payoff (CHF)')
    titled(ax, 'What each observer perceives',
           'Same 5-df B-spline noise, different scale')
    ax.text(HI * 0.96, 150, 'Veridical', color='.55', fontsize=6.7,
            ha='right', va='top')
    ax.text(HI * 0.96, log_p[-1] * 0.84, 'Log scale', color=C_SPL, fontsize=6.7,
            ha='right', va='top')
    ax.annotate('Linear scale saturates:\n56 to 112 CHF moves the\npercept by only 1.9 CHF',
                xy=(86, lin.loc[112.0] * 1.06), xytext=(7.4, 128), fontsize=6.7,
                color=C_SPN, ha='left', va='top', linespacing=1.3,
                arrowprops=dict(arrowstyle='-|>', color=C_SPN, lw=0.9,
                                mutation_scale=7, shrinkA=2, shrinkB=4))


def draw_order_effect(ax, primary='lfx2-bs3-m2-dp-bm'):
    """The behavioural fingerprint of memory: the same gamble, two orders."""
    xb = pd.read_csv(CARDS / f'bins_ratio_bin.{primary}.tsv', sep='\t')
    xmap = dict(zip(xb.ratio_bin, xb.x))
    d = pd.read_csv(CARDS / f'ppc_by_ratio_order.{primary}.tsv', sep='\t')
    for order, col, mfc in [('Risky first', '.62', 'none'),
                            ('Risky second', '.15', '.15')]:
        g = d[d.order == order].sort_values('ratio_bin')
        x = g.ratio_bin.map(xmap).values
        ax.fill_between(x, g.lo, g.hi, color=col, alpha=.22, lw=0)
        ax.plot(x, g['median'], color=col, lw=1.1)
        ax.plot(x, g.observed, 'o', ms=3.4, color=col, mfc=mfc, mew=0.9)
        ax.text(x[-1] * 1.005, g.observed.iloc[-1], f'  {order}', color=col,
                fontsize=6.7, ha='left', va='center')
    ax.axhline(0.5, color='.85', lw=0.6, ls=':', zorder=0)
    ax.set_xlim(0.36, 1.60)
    ax.set_ylim(0.20, 0.88)
    ax.set_xlabel('log(risky / safe)')
    ax.set_ylabel('P(chose risky)')
    titled(ax, 'The order effect',
           'Risky-second is chosen more, at every ratio')
    ax.text(.97, .06, 'Log-scale spline model,\ncTBS on both channels (r-hat 1.00)',
            transform=ax.transAxes, fontsize=6.4, color='.45', ha='right',
            va='bottom', linespacing=1.25)


# The four converged families, for the by-stimulation comparison.
STIM_MODELS = [
    ('lfx2-bs3-m2-dp-bm', 'Spline, log scale', 'cTBS on both channels'),
    ('lfx2-bs3-m2-dp-bm-p2-fx', 'Affine, log scale', 'fixed cTBS slope'),
    ('lfx2-gw-m2-dp-bm-p2-fp', 'Generalized Weber', 'cTBS on both channels'),
    ('lfx2-bs3-w-dp-bm', 'Weber', 'cTBS on both channels'),
]


def draw_stim_curves(ax, label, title, sub, legend=False, ylim=(0.08, 0.44)):
    """Where cTBS lands, for one model: each option's noise by stimulation site."""
    for which, ls, nm in [('n1 (first)', '-', 'n1'), ('n2 (second)', '--', 'n2')]:
        for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
            g = _curve(label, which, stim=stim)
            if g is None:
                continue
            ax.fill_between(g.payoff, g.lo, g.hi, color=col, alpha=.12, lw=0)
            ax.plot(g.payoff, g['median'], color=col, lw=1.3, ls=ls)
        g = _curve(label, which, stim='vertex')
        if g is not None:
            ax.text(HI * 1.06, g['median'].iloc[-1] + (0.010 if nm == 'n1' else -0.012),
                    nm, color='.25', fontsize=6.9, ha='left', va='center')
    logx(ax)
    ax.set_xlim(LO * .95, HI * 1.6)
    ax.set_ylim(*ylim)
    ax.set_ylabel('Noise SD (log units)')
    titled(ax, title, sub)
    if legend:
        ax.text(.03, .97, 'IPS', color=IPS, fontsize=6.9, transform=ax.transAxes,
                ha='left', va='top')
        ax.text(.03, .87, 'Vertex', color=VERTEX, fontsize=6.9,
                transform=ax.transAxes, ha='left', va='top')
        ax.set_ylabel('Noise SD (log units)\nSolid n1, dashed n2; 95% CrI')


# Both rows of the forest come from the SAME affine-noise family, fitted twice:
# once with independent n1/n2 ('-p2-i'), once with the memory/perceptual
# composition and a fixed cTBS slope ('-p2-fx'). Only models that carry the
# contrast are shown; the generalized-Weber twin is left out because its memory
# coordinate at 112 CHF is unidentified (95% CrI -1.44 to +1.46).
FOREST = [
    ('lfx2-bs3-m2-dp-bm-p2-i', 'n2 noise @ 7 CHF', 'n2 @ 7 CHF', 0),
    ('lfx2-bs3-m2-dp-bm-p2-i', 'n2 noise @ 112 CHF', 'n2 @ 112 CHF', 0),
    ('lfx2-bs3-m2-dp-bm-p2-i', 'n1 noise @ 7 CHF', 'n1 @ 7 CHF', 0),
    ('lfx2-bs3-m2-dp-bm-p2-i', 'n1 noise @ 112 CHF', 'n1 @ 112 CHF', 0),
    ('lfx2-bs3-m2-dp-bm-p2-fx', 'perceptual noise @ 7 CHF', 'Perceptual @ 7 CHF', 1),
    ('lfx2-bs3-m2-dp-bm-p2-fx', 'perceptual noise @ 112 CHF', 'Perceptual @ 112 CHF', 1),
    ('lfx2-bs3-m2-dp-bm-p2-fx', 'memory noise @ 7 CHF', 'Memory @ 7 CHF', 1),
    ('lfx2-bs3-m2-dp-bm-p2-fx', 'memory noise @ 112 CHF', 'Memory @ 112 CHF', 1),
]


def draw_ctbs_forest(ax):
    """cTBS on each option and each channel, at the two ends of the payoff range."""
    rows = []
    for label, quantity, nice, grp in FOREST:
        d = pd.read_csv(CARDS / f'derived.{label}.tsv', sep='\t')
        r = d[(d.stim == 'IPS - vertex') & (d.quantity == quantity)]
        if not len(r):
            print(f'  ! missing contrast: {label} / {quantity}')
            continue
        rows.append((nice, grp, float(r['mean'].iloc[0]), float(r.lo.iloc[0]),
                     float(r.hi.iloc[0])))

    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r[1] != rows[i - 1][1]:
            y += 1.9
        ys.append(y)
        y += 1.0
    ys = [max(ys) - v for v in ys]

    ax.axvline(0, color='.35', lw=0.7, zorder=0)
    for yi, (nice, grp, m, lo, hi) in zip(ys, rows):
        col = C_DATA if grp == 0 else C_SPL
        excl = lo > 0 or hi < 0
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.4, solid_capstyle='butt')
        ax.plot([m], [yi], 'o', ms=4.0, color=col if excl else 'w', mec=col,
                mew=1.0, zorder=4)
        ax.text(-0.415, yi + 0.48, nice, fontsize=6.7, color=col, ha='left',
                va='center')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_ylim(-0.85, max(ys) + 1.95)
    ax.set_xlim(-0.42, 0.25)
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xlabel('cTBS effect, IPS - vertex (log units)')
    titled(ax, 'Where cTBS acts',
           'More noise at 7 CHF, less at 112: a crossover')
    ax.text(-0.415, max(ys) + 1.25, 'Affine noise, independent n1/n2  '
            '(r-hat 1.07)', color=C_DATA, fontsize=6.4, ha='left', va='center')
    ax.text(-0.415, ys[4] + 1.25, 'Affine noise, memory/perceptual, fixed '
            'cTBS slope  (r-hat 1.00)', color=C_SPL, fontsize=6.4, ha='left',
            va='center')
    ax.text(0.245, -0.62, 'Filled = 95% CrI excludes 0', color='.45',
            fontsize=6.4, ha='right', va='center')


def draw_convergence(ax, ladder_tsv):
    """Every fit that exists, and where the convergence gate cut it.

    Panel n shows converged models only -- a fit with r-hat 2.4 has no defensible
    ELPD -- so this panel says how many were dropped and how well they scored, to
    make the selection auditable instead of invisible.
    """
    d = pd.read_csv(ladder_tsv, sep='\t')
    d = d[d.elpd_loo > -4400]                      # keep the readable range
    shown = {lab for lab, *_ in LADDER}
    rng = np.random.default_rng(0)
    for i, (flag, col, face) in enumerate([(True, '.15', '.15'),
                                           (False, '.62', 'none')]):
        g = d[d.converged == flag]
        x = i + rng.uniform(-0.16, 0.16, len(g))
        ax.scatter(x, g.elpd_loo, s=13, facecolor=face, edgecolor=col, lw=0.7,
                   zorder=2 + i)
        sel = g.label.isin(shown).values
        ax.scatter(x[sel], g.elpd_loo.values[sel], s=30, facecolor='none',
                   edgecolor=C_SPL, lw=1.0, zorder=5)
        ax.text(i, -4308, f'n = {len(g)}', fontsize=6.7, color=col,
                ha='center', va='top')
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Passed', 'Excluded'])
    ax.set_ylim(-4320, -4130)
    ax.set_ylabel('ELPD')
    titled(ax, 'What the gate removed',
           'r-hat < 1.01 and ESS > 400')
    ax.text(0.97, 0.97, 'Blue rings:\nshown in panel n', color=C_SPL,
            fontsize=6.4, transform=ax.transAxes, ha='right', va='top',
            linespacing=1.25)


def psychometric_data(bids_folder, n_bins=6):
    """Binned P(chose risky) by stake tercile, on both candidate decision axes."""
    from tms_risk.utils.data import get_all_behavior
    d = get_all_behavior(bids_folder=bids_folder).reset_index()
    d = d.dropna(subset=['chose_risky']).copy()
    d['ev_diff'] = P_WIN * d['n_risky'] - d['n_safe']
    d['logratio'] = np.log(d['n_risky'] / d['n_safe'])
    # the whole repo (and every card TSV) defines stake this way; keep it
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    d['tercile'] = pd.qcut(d['stake'], 3, labels=['Low', 'Mid', 'High'])
    out = {}
    for var in ('ev_diff', 'logratio'):
        d['b'] = d.groupby('tercile', observed=True)[var].transform(
            lambda s: pd.qcut(s, n_bins, labels=False, duplicates='drop'))
        # subject means first, so the error bar is between-subject
        per_sub = (d.groupby(['tercile', 'b', 'subject'], observed=True)
                     .agg(x=(var, 'mean'), y=('chose_risky', 'mean')).reset_index())
        out[var] = (per_sub.groupby(['tercile', 'b'], observed=True)
                    .agg(x=('x', 'mean'), y=('y', 'mean'),
                         se=('y', lambda s: s.std(ddof=1) / np.sqrt(s.size)))
                    .reset_index())
    return d, out


def probit_slopes(d, var):
    """Probit slope of P(chose risky) on `var`, per stake tercile."""
    import statsmodels.api as sm
    sl = {}
    for t, g in d.groupby('tercile', observed=True):
        m = sm.GLM(g['chose_risky'].values,
                   sm.add_constant(g[var].values),
                   family=sm.families.Binomial(link=sm.families.links.Probit())
                   ).fit()
        sl[t] = (m.params[0], m.params[1])
    return sl


def draw_psychometric(ax, binned, slopes, var, xlabel, title, note, label_dx=1.0):
    """Binned choice data with the best-fitting probit per stake tercile."""
    from scipy.stats import norm
    for t, col, nm in zip(['Low', 'Mid', 'High'], C_STAKE, STAKE_LABEL):
        g = binned[binned.tercile == t].sort_values('x')
        ax.errorbar(g.x, g.y, yerr=g['se'], fmt='o', ms=3.0, color=col,
                    lw=0, elinewidth=0.8, capsize=0, zorder=3)
        a, b = slopes[t]
        xs = np.linspace(g.x.min(), g.x.max(), 100)
        ys = norm.cdf(a + b * xs)
        ax.plot(xs, ys, color=col, lw=1.4, zorder=2)
        ax.text(xs[-1] + label_dx, ys[-1], nm, color=col, fontsize=6.9,
                ha='left', va='center')
    ax.axhline(0.5, color='.85', lw=0.6, ls=':', zorder=0)
    ax.set_ylim(0.13, 0.92)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('P(chose risky)')
    ratio = slopes['Low'][1] / slopes['High'][1]
    titled(ax, title, f'Slope {ratio:.0f}x steeper at low than high stakes')
    ax.text(.97, .05, note, transform=ax.transAxes, fontsize=6.8, color='.35',
            ha='right', va='bottom', linespacing=1.25)


# ----------------------------------------------------------------------------
# Row 2 -- the five noise functions
# ----------------------------------------------------------------------------

FORMS = [
    ('Weber', C_WEB, r'$\sigma_{rel}=k$', '1 par., either scale',
     lambda: [np.full_like(X, k) for k in (0.13, 0.22, 0.38)]),
    ('Spline, linear scale', C_SPN, r'$\sigma_{abs}=\mathrm{spline}(x)$',
     '5 par., linear scale',
     lambda: [4.0 / X, 0.20 + 0.0 * X, 1.5 / X + 0.12]),
    ('Spline, log scale', C_SPL, r'$\sigma_{rel}=\mathrm{spline}(\log x)$',
     '5 par., log scale',
     lambda: [0.20 + 0.09 * np.sin(np.log(X / LO) / np.log(16) * np.pi * 1.5),
              0.11 * (X / LO) ** 0.32,
              0.40 * (X / LO) ** -0.34]),
    ('Generalized Weber', C_GW,
     r'$\sigma_{rel}=\mathrm{softplus}(k+c/x)$', '2 par.; c < 0 when fitted',
     lambda: [np.log1p(np.exp(-1.55 + c / X)) for c in (8.0, 0.0, -1.66)]),
    ('Power law', C_POW, r'$\sigma_{rel}=c\,x^{\beta-1}$', '2 par., either scale',
     lambda: [0.22 * (X / 28.) ** (b - 1) for b in (0.75, 1.0, 1.3)]),
]


def draw_form(ax, name, col, eq, note, curves):
    for i, y in enumerate(curves()):
        ax.plot(X, y, color=col, lw=1.25, alpha=[0.42, 0.70, 1.0][i])
    logx(ax, ticks=(7, 28, 112))
    ax.set_yscale('log')
    ax.set_ylim(0.03, 1.1)
    ax.set_yticks([0.05, 0.2, 0.8])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_title(f'{name}\n{eq}', fontsize=7.9, loc='left', color=col,
                 linespacing=1.45, pad=3)
    ax.text(0.03, 0.05, note, transform=ax.transAxes, fontsize=6.7,
            color='.45', ha='left', va='bottom')


# ----------------------------------------------------------------------------
# Row 3 -- what the data pick
# ----------------------------------------------------------------------------

# label offsets keep the right-hand labels from colliding, per panel
FITTED = [
    ('lfx2-bs3-w-dp-bm', 'Weber', C_WEB, -0.026, -0.022, -0.030),
    ('lfx2-gw-m2-dp-bm-p2-fp', 'Gen. Weber', C_GW, +0.016, +0.014, +0.020),
    ('lfx2-pl-m2-dp-m-p2-i', 'Power law', C_POW, -0.006, -0.006, None),
    ('lfx2-bs3-m2-dp-bm', 'Spline, log', C_SPL, +0.008, +0.006, +0.006),
]


def _curve(label, name, stim='vertex'):
    f = CARDS / f'curves.{label}.tsv'
    if not f.exists():
        return None
    d = pd.read_csv(f, sep='\t')
    g = d[(d.curve == name) & (d.stim == stim)].sort_values('payoff')
    return g if len(g) else None


def draw_fitted_curves(ax, which, title, message, ylim, band=True):
    """One option's fitted noise SD, under every converged noise form."""
    col_i = {'n1 (first)': 3, 'n2 (second)': 4}[which]
    for row in FITTED:
        label, nice, col, dy = row[0], row[1], row[2], row[col_i]
        if dy is None:
            continue
        g = _curve(label, which)
        if g is None:
            continue
        if band:
            ax.fill_between(g.payoff, g.lo, g.hi, color=col, alpha=.11, lw=0)
        ax.plot(g.payoff, g['median'], color=col, lw=1.4)
        ax.text(HI * 1.10, g['median'].iloc[-1] + dy, nice, color=col,
                fontsize=6.8, ha='left', va='center')
    logx(ax)
    ax.set_xlim(LO * .95, HI * 3.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel('Noise SD (log units)')
    titled(ax, title, message)


def draw_channels(ax):
    """The two coordinates the noise is built from -- NOT option noise.

    bauer composes n1 = softplus(memory + perceptual) and n2 = softplus(perceptual),
    so `memory` is one term of a sum, never the SD of anything observable. At 7 CHF
    the memory coordinate reads ~1.06 while n1 is 0.23.
    """
    for label, nice, col, _, _, dy in FITTED:
        if dy is None:
            continue
        for name, ls in [('memory', '-'), ('perceptual', '--')]:
            g = _curve(label, name)
            if g is None:
                continue
            ax.plot(g.payoff, g['median'], color=col, lw=1.3, ls=ls)
    logx(ax)
    ax.set_yscale('log')
    ax.set_xlim(LO * .95, HI * 1.9)
    ax.set_ylim(0.058, 1.75)
    ax.set_yticks([0.1, 0.3, 1.0])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylabel('Coordinate (log units)')
    ax.text(HI * 1.06, 0.80, 'Memory', color='.25', fontsize=6.9, ha='left',
            va='center')
    ax.text(HI * 1.06, 0.235, 'Perceptual\n(dashed)', color='.25', fontsize=6.9,
            ha='left', va='center', linespacing=1.2)
    titled(ax, 'The two channels behind them',
           'Memory falls where perceptual rises')
    ax.text(.97, .03, 'Terms of a sum, not option noise:\n'
                       r'$n_1=\mathrm{softplus}(\mathrm{perc}+\mathrm{mem})$',
            transform=ax.transAxes, fontsize=6.6, color='.45', ha='right',
            va='bottom', linespacing=1.35)


# The tag says exactly which quantities the cTBS regressor sits on. Models
# parameterised by CHANNEL carry memory and perceptual (n1 = softplus(mem+perc),
# n2 = softplus(perc)); models parameterised by OPTION carry n1 and n2 directly.
LADDER = [
    ('lfx2-bs3-m2-dp-bm', 'Spline, log', C_SPL, 'on memory + perceptual'),
    ('lfx2-gw-m2-dp-bm-p2-fp', 'Gen. Weber, log', C_GW, 'on memory + perceptual'),
    ('flexible2.4_noisefix.head', 'Spline, linear', C_SPN, 'on perceptual only'),
    ('flexible1_noisefix.head', 'Spline, linear', C_SPN, 'on n1 + n2'),
    ('lfx2-bs3-w-dp-bm', 'Weber, log', C_WEB, 'on memory + perceptual'),
    ('lfx2-pl-m2-dp-m-p2-i', 'Power law, log', C_POW, 'on n1 only'),
    ('lfx2-bs3-m2-dp-null', 'Spline, log', C_SPL, 'no cTBS'),
    ('lfx2-bs3-w-dp-null', 'Weber, log', C_WEB, 'no cTBS'),
    ('flexible1_noisefix_null.head', 'Spline, linear', C_SPN, 'no cTBS'),
]


def draw_ladder(ax, ladder_tsv):
    d = pd.read_csv(ladder_tsv, sep='\t').set_index('label')
    rows = []
    for lab, nice, col, tag in LADDER:
        if lab not in d.index:
            print(f'  ! missing from the ladder: {lab}')
            continue
        r = d.loc[lab]
        rows.append((nice, col, tag, r.elpd_loo, r.dse_best))
    best = max(r[3] for r in rows)
    # a gap between the models that carry a cTBS regressor and those that do not
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r[2] == 'no cTBS' and rows[i - 1][2] != 'no cTBS':
            y += 0.75
        ys.append(y)
        y += 1.0
    ys = [max(ys) - v for v in ys]

    for yi, (nice, col, tag, elpd, dse) in zip(ys, rows):
        dd = elpd - best
        ax.barh(yi, dd, height=.60, color=col, alpha=.85, lw=0)
        if dse > 0:
            ax.plot([dd - dse, dd + dse], [yi, yi], color='.15', lw=0.9)
        tail = tag if tag == 'no cTBS' else f'cTBS {tag}'
        ax.text(4, yi + 0.17, nice, fontsize=6.6, va='center', ha='left',
                color='.1')
        ax.text(4, yi - 0.28, tail, fontsize=6.1, va='center', ha='left',
                color='.55')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-142, 172)
    ax.set_ylim(min(ys) - 0.75, max(ys) + 0.75)
    ax.axvline(0, color='.3', lw=0.7)
    ax.set_xlabel('ELPD relative to the best model (nats)')
    ax.set_xticks([-120, -80, -40, 0])

    titled(ax, 'Predictive fit, converged fits only',
           'Bars are +/- the paired dSE')


PPC_MODELS = [('lfx2-bs3-w-dp-bm', 'Weber, log', C_WEB),
              ('flexible1_noisefix.head', '5-df spline, linear', C_SPN),
              ('lfx2-bs3-m2-dp-bm', '5-df spline, log', C_SPL)]


def draw_ppc(ax, order):
    xb = pd.read_csv(CARDS / 'bins_stake_bin.lfx2-bs3-m2-dp-bm.tsv', sep='\t')
    xmap = dict(zip(xb.stake_bin, xb.x))
    for i, (label, nice, col) in enumerate(PPC_MODELS):
        d = pd.read_csv(CARDS / f'ppc_by_stake.{label}.tsv', sep='\t')
        g = d[d.order == order].sort_values('stake_bin')
        x = g.stake_bin.map(xmap).values
        ax.fill_between(x, g.lo, g.hi, color=col, alpha=.22, lw=0, zorder=1 + i)
        ax.plot(x, g['median'], color=col, lw=1.0, zorder=3)
        ax.text(x[0] * 0.99, g['median'].iloc[0] + [-0.032, 0.020, 0.062][i],
                nice, color=col, fontsize=6.8, ha='left',
                va='top' if i == 0 else 'bottom')
        if i == 0:
            bad = g[(g.observed < g.lo) | (g.observed > g.hi)]
        if i == len(PPC_MODELS) - 1:
            ax.plot(x, g.observed, 'o', ms=3.6, color=C_DATA, zorder=5,
                    mec='w', mew=0.5)
    for _, r in bad.iterrows():
        ax.annotate('Weber\nmisses this', xy=(xmap[r.stake_bin], r.observed),
                    xytext=(xmap[r.stake_bin] * 0.60, r.observed + 0.028),
                    fontsize=6.7, color='.25', ha='center', va='bottom',
                    linespacing=1.2,
                    arrowprops=dict(arrowstyle='-|>', color='.25', lw=0.8,
                                    mutation_scale=6, shrinkA=1, shrinkB=3))
    logx(ax, 'Stake, (risky + safe) / 2 (CHF)', ticks=(11, 16, 23, 32, 47))
    ax.set_xlim(9.6, 56)
    ax.set_ylim(0.43, 0.75)
    ax.set_yticks([0.45, 0.55, 0.65])
    ax.set_ylabel('P(chose risky)')
    titled(ax, f'Predictive check: {order.lower()}',
           'The same 5-df spline, log vs linear scale')


def family_of(label):
    """Group a trace label into the model family the figure talks about."""
    if label.startswith('lfx2-gw'):
        return 'Gen. Weber, log', C_GW
    if label.startswith('lfx2-pl'):
        return 'Power law, log', C_POW
    if label.startswith('power'):
        return 'Power law, linear', C_POW
    if label.startswith('flexible') or label.startswith('logflex'):
        return ('Spline, linear' if label.startswith('flexible')
                else 'Spline, log'), C_SPN
    if label.startswith('lfx2'):
        if '-w-' in label:
            return 'Weber, log', C_WEB
        if '-p2' in label:
            return 'Affine, log', C_SPL
        return 'Spline, log', C_SPL
    return 'Other', '.5'


def draw_family_gate(ax, ladder_tsv):
    """Which model families converged, and which never did.

    The ELPD ladder can only show fits that passed; this says how often each
    family passed at all, which is a property of the family's geometry rather
    than of any one fit.
    """
    d = pd.read_csv(ladder_tsv, sep='\t')
    d = d[d.elpd_loo > -4400]
    fam = d.label.map(lambda l: family_of(l)[0])
    col = {family_of(l)[0]: family_of(l)[1] for l in d.label}
    t = (pd.DataFrame({'fam': fam, 'ok': d.converged.values})
         .groupby('fam').agg(n=('ok', 'size'), k=('ok', 'sum')))
    t['frac'] = t.k / t.n
    t = t.sort_values('frac')
    y = np.arange(len(t))
    for yi, (name, r) in zip(y, t.iterrows()):
        c = col[name]
        ax.barh(yi, r.n, height=.62, color=c, alpha=.20, lw=0)
        ax.barh(yi, r.k, height=.62, color=c, alpha=.90, lw=0)
        ax.text(r.n + 1.2, yi, f'{int(r.k)}/{int(r.n)}', fontsize=6.4,
                color='.35', va='center', ha='left')
        ax.text(0.8, yi + 0.52, name, fontsize=6.5, color='.15', va='center',
                ha='left')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_ylim(-0.75, len(t) - 0.15)
    ax.set_xlim(0, max(t.n) * 1.30)
    ax.set_xlabel('Fits attempted')
    titled(ax, 'Which families converge',
           'Solid: passed. Faint: attempted')


DELTA_PPC = [
    ('lfx2-bs3-m2-dp-null', 'No cTBS', '.62', True),
    ('lfx2-bs3-w-dp-bm', 'Weber', C_WEB, False),
    ('lfx2-bs3-m2-dp-bm', 'Spline, log', C_SPL, True),
]


def draw_ctbs_ppc(ax, order, legend=False):
    """The three-way cTBS x order x stake check, as the cTBS difference itself.

    Plotting IPS - vertex rather than the two levels is what makes the panel
    readable: the effect is ~0.06 on a probability that ranges over 0.5-0.65, so
    on a levels plot it is invisible. `ppc_delta_by_stake` holds the difference
    computed WITHIN each posterior draw, so the band is the correct joint
    credible interval, not a difference of two marginal ones.
    """
    ax.axhline(0, color='.8', lw=0.7, ls=':', zorder=0)
    obs = None
    for label, nice, col, band in DELTA_PPC:
        f = Path(f'notes/data/ppc_delta_by_stake.{label}.tsv')
        if not f.exists():
            print(f'  ! missing delta PPC: {label}')
            continue
        d = pd.read_csv(f, sep='\t')
        g = d[d.order == order].sort_values('stake')
        if band:
            ax.fill_between(g.stake, g.lo, g.hi, color=col, alpha=.20, lw=0)
            ax.plot(g.stake, g['median'], color=col, lw=1.2)
        else:
            ax.plot(g.stake, g['median'], color=col, lw=1.2, ls='--')
        obs = g
    ax.errorbar(obs.stake, obs.obs, yerr=obs.obs_sem, fmt='o', ms=4.0,
                color=C_DATA, lw=0, elinewidth=1.0, capsize=0, zorder=5,
                mec='w', mew=0.5)
    logx(ax, 'Stake, (risky + safe) / 2 (CHF)', ticks=(13, 23, 42))
    ax.set_xlim(10.6, 50)
    ax.set_ylim(-0.075, 0.105)
    ax.set_yticks([-0.05, 0, 0.05, 0.10])
    ax.set_ylabel('cTBS effect on P(chose risky)\n(IPS - vertex)')
    msg = ('No effect to explain' if order == 'Risky first'
           else 'A real effect; every model under-predicts it')
    titled(ax, f'cTBS x stake, {order.lower()}', msg)
    ax.text(.985, .97, 'IPS more risk-seeking', color=IPS, fontsize=6.5,
            transform=ax.transAxes, ha='right', va='top')
    ax.text(.985, .03, 'Vertex more risk-seeking', color=VERTEX, fontsize=6.5,
            transform=ax.transAxes, ha='right', va='bottom')
    if legend:
        for i, (_, nice, col, _) in enumerate(DELTA_PPC):
            ax.text(.03, .97 - .085 * i, nice, color=col, fontsize=6.5,
                    transform=ax.transAxes, ha='left', va='top')


# The three models that carry the full cTBS x order x stake predictive check.
# (label, human name). The r-hat line is read from the ladder rather than typed
# here -- the per-card meta only screened the group `*_mu` variables, which
# understated `-p2-i` as 1.07 when the full check puts it at 1.27 with ESS 11.
LEVEL_MODELS = [
    ('flexible1_noisefix_null.head', 'Linear spline, null'),
    ('flexible1_noisefix.head', 'Linear spline, n1+n2'),
    ('lfx2-bs3-m2-dp-bm', 'Log spline, mem+perc'),
    # The affine / power-law pair below is matched on everything but the link:
    # same design matrix ([1, log payoff]), same cTBS placement (n1 only), both
    # converged. Their sibling fits with cTBS on BOTH channels and a free random
    # slope all failed -- in both link families -- so those are not shown here.
    ('lfx2-bs3-m2-dp-m-p2-i', 'Log affine, n1'),
    ('lfx2-pl-m2-dp-m-p2-i', 'Log power law, n1'),
    ('lfx2-bs3-m2-dp-bm-p2-fx', 'Log affine, fixed slope'),
    ('lfx2-gw-m2-dp-bm-p2-fp', 'Gen. Weber, mem+perc'),
    # Same family, cTBS on the perceptual channel ALONE: converges cleanly and
    # ties the both-channels fit (-4161.9 vs -4162.1) with one fewer effect,
    # while the memory-only sibling is 46 nats worse. That is as clean a
    # localisation as this design supports.
    ('lfx2-gw-m2-dp-b-p2-fp', 'Gen. Weber, perceptual'),
    # The overall ELPD winner: log spline, cTBS on the perceptual channel alone.
    ('lfx2-bs2-m2-dp-b', 'Log spline, perceptual'),
]


DELTA_DIR = Path('notes/data/delta')


def _noise_files(label):
    """Prefer the delta/ extraction (it carries a matched difference), else cards."""
    c = DELTA_DIR / f'curves.{label}.tsv'
    d = DELTA_DIR / f'delta.{label}.tsv'
    if not c.exists():
        c = CARDS / f'curves.{label}.tsv'
    return (c if c.exists() else None), (d if d.exists() else None)


def is_log_space(label):
    """Which scale the observer in this fit reasons on."""
    return not label.startswith('flexible')


def to_absolute(x, sigma, log_space):
    """Noise SD in CHF.

    For a log-space observer sigma is the SD of log(payoff), so the SD in CHF is
    the lognormal one -- NOT x * sigma, which is only the small-sigma limit:

        SD(x) = x * exp(sigma^2 / 2) * sqrt(exp(sigma^2) - 1)

    at sigma = 0.9 the factor is 1.67, not 0.9. For a natural-space observer the
    curves arrive already divided by payoff (natdelta.py), so multiplying back is
    exact. Both maps are strictly increasing in sigma at fixed x, which is why the
    significance strips below are identical on the two rows: P(sigma_ips >
    sigma_vertex) is invariant under a monotone per-condition transform, and an
    equal-tailed 95% interval excluding zero is exactly that probability
    statement.
    """
    if log_space:
        return x * np.exp(sigma ** 2 / 2) * np.sqrt(np.exp(sigma ** 2) - 1)
    return sigma * x


def draw_noise_column(ax, label, nice, space='relative', ylabel=False,
                      legend=False, title=True):
    """One model's fitted noise functions, with a cTBS significance strip.

    Curves: n1 (solid) and n2 (dashed), by stimulation site. The strip marks the
    payoffs where the 95% credible interval of the WITHIN-draw difference
    sigma_ips(x) - sigma_vertex(x) excludes zero -- red where cTBS raised the
    noise, green where it lowered it. Marginal bands cannot answer that: IPS and
    vertex come from the same draws and are strongly correlated, so their bands
    overlap far more than the difference is uncertain.
    """
    cfile, dfile = _noise_files(label)
    if cfile is None:
        ax.text(.5, .5, 'noise curves\npending', transform=ax.transAxes,
                fontsize=6.8, color='.6', ha='center', va='center',
                linespacing=1.3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if title:
            titled(ax, nice, 'extraction running')
        return

    logsp = is_log_space(label)
    absolute = space == 'absolute'
    conv = ((lambda x, v: to_absolute(x, v, logsp)) if absolute
            else (lambda x, v: v))

    d = pd.read_csv(cfile, sep='\t')
    for which, ls in [('n1 (first)', '-'), ('n2 (second)', '--')]:
        for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
            g = d[(d.curve == which) & (d.stim == stim)].sort_values('payoff')
            if not len(g):
                continue
            x = g.payoff.values
            ax.fill_between(x, conv(x, g.lo.values), conv(x, g.hi.values),
                            color=col, alpha=.11, lw=0)
            ax.plot(x, conv(x, g['median'].values), color=col, lw=1.3, ls=ls)

    lo_y, hi_y = ((0.35, 42.0) if absolute else (0.055, 0.62))
    if dfile is not None:
        dd = pd.read_csv(dfile, sep='\t')
        for k, which in enumerate(['n1 (first)', 'n2 (second)']):
            g = dd[dd.curve == which].sort_values('payoff')
            if not len(g):
                continue
            y = lo_y * (1.16 + 0.30 * k) if absolute else lo_y * (1.14 + 0.16 * k)
            # p_gt0 rather than lo/hi: for an equal-tailed interval the two are
            # equivalent, and p_gt0 is the form that survives the CHF conversion.
            sig = (g.p_gt0 > 0.975) | (g.p_gt0 < 0.025)
            ax.plot(g.payoff, np.full(len(g), y), color='.90', lw=2.2,
                    solid_capstyle='butt', zorder=1)
            for up, col in [(True, IPS), (False, VERTEX)]:
                m = sig & ((g['median'] > 0) if up else (g['median'] < 0))
                if m.any():
                    xs = g.payoff.values.astype(float).copy()
                    xs[~m.values] = np.nan
                    ax.plot(xs, np.full(len(g), y), color=col, lw=2.2,
                            solid_capstyle='butt', zorder=2)
            ax.text(LO * 0.93, y, 'n1' if k == 0 else 'n2', fontsize=6.2,
                    color='.4', ha='right', va='center')

    logx(ax)
    ax.set_yscale('log')
    ax.set_xlim(LO * .84, HI * 1.05)
    ax.set_ylim(lo_y, hi_y)
    ax.set_yticks([1, 3, 10, 30] if absolute else [0.1, 0.2, 0.4])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if ylabel:
        ax.set_ylabel('Noise SD (CHF)\nSolid n1, dashed n2' if absolute
                      else 'Relative noise SD\nSolid n1, dashed n2')
    if title:
        titled(ax, nice,
               'Bars: 95% CrI of IPS - vertex excludes 0' if legend else None)


def diagnostics(label, ladder_tsv='notes/data/ladder_v12.tsv'):
    """One short line of convergence diagnostics, straight from the ladder."""
    d = pd.read_csv(ladder_tsv, sep='\t').set_index('label')
    if label in d.index:
        r = d.loc[label]
        rhat, ess, ok = float(r.max_rhat), int(r.min_ess), bool(r.converged)
    else:
        # the ladder is rebuilt from pointwise LOO files, so a fit newer than the
        # last rebuild is missing from it; the card carries the same diagnostics
        f = CARDS / f'meta.{label}.tsv'
        if not f.exists():
            return 'diagnostics unavailable'
        r = pd.read_csv(f, sep='\t').iloc[0]
        rhat, ess = float(r['rhat']), int(r['ess'])
        ok = rhat <= 1.01 and ess >= 400
    if ok:
        return f'r-hat {rhat:.2f}, ESS {ess}'
    return f'NOT CONVERGED: r-hat {rhat:.2f}'


def draw_levels(ax, label, nice, order, legend=False, xlabel=True):
    """One cell of the cTBS x order x stake predictive check, in the house palette.

    Levels rather than differences, because the reader has to see the size of the
    effect against the size of the quantity: the cTBS gap is ~0.06 on a
    probability that ranges over 0.45-0.68.
    """
    xb = pd.read_csv(CARDS / f'bins_stake3.{label}.tsv', sep='\t')
    xmap = dict(zip(xb.stake3, xb.x))
    d = pd.read_csv(CARDS / f'ppc_by_stake_stim_order.{label}.tsv', sep='\t')
    d = d[d.order == order]
    n_miss = 0
    for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
        g = d[d.stimulation_condition == stim].sort_values('stake3')
        x = g.stake3.map(xmap).values
        ax.fill_between(x, g.lo, g.hi, color=col, alpha=.18, lw=0)
        ax.plot(x, g['median'], color=col, lw=1.3)
        ax.plot(x, g.observed, 'o', ms=4.0, color=col, mec='w', mew=0.6, zorder=5)
        n_miss += int(((g.observed < g.lo) | (g.observed > g.hi)).sum())
        if legend:
            ax.text(x[-1] * 1.05, g.observed.iloc[-1],
                    'IPS' if stim == 'ips' else 'Vertex', color=col,
                    fontsize=6.7, ha='left', va='center')
    logx(ax, 'Stake, (risky + safe) / 2 (CHF)' if xlabel else '',
         ticks=(13, 23, 42))
    if not xlabel:
        ax.set_xticklabels([])
    ax.set_xlim(10.6, 62 if legend else 50)
    ax.set_ylim(0.42, 0.72)
    ax.set_yticks([0.45, 0.55, 0.65])
    ax.set_ylabel('P(chose risky)')
    msg = (f'{6 - n_miss}/6 cells covered' if n_miss
           else 'All 6 cells covered')
    diag = diagnostics(label)
    bad = diag.startswith('NOT')
    titled(ax, nice, diag, color=C_GW if bad else '.12')
    ax.text(.03, .05, msg, transform=ax.transAxes, fontsize=6.6,
            color=C_GW if n_miss else '.4', ha='left', va='bottom')


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--ladder', default='notes/data/ladder_v12.tsv')
    ap.add_argument('--out', default='notes/figures/model_overview')
    args = ap.parse_args()

    d, binned = psychometric_data(args.bids_folder)
    sl_ev = probit_slopes(d, 'ev_diff')
    sl_lr = probit_slopes(d, 'logratio')

    fig = plt.figure(figsize=(14.6, 26.0))
    outer = GridSpec(11, 1, figure=fig,
                     height_ratios=[1.00, 1.00, 1.00, 1.00, 0.94, 0.72, 1.42,
                                    0.94, 0.86, 0.90, 0.90],
                     hspace=0.86, left=0.058, right=0.988, top=0.978,
                     bottom=0.026)

    # -- row 1: what "the observer works in log space" actually means --------
    g1 = outer[0].subgridspec(1, 4, wspace=0.62)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(g1[0, i]) for i in range(4))
    draw_multiply_linear(ax_a)
    draw_multiply_log(ax_b)
    draw_perceived(ax_c)
    draw_priors(ax_d, args.bids_folder)

    # -- row 2: the choice data, on the two candidate decision axes ----------
    g2 = outer[1].subgridspec(1, 2, wspace=0.28)
    ax_e, ax_f = (fig.add_subplot(g2[0, i]) for i in range(2))
    draw_psychometric(ax_e, binned['ev_diff'], sl_ev, 'ev_diff',
                      'Expected-value difference (CHF)',
                      'A CHF decision axis',
                      'Points: data (mean +/- SEM over subjects)\nLines: best-fitting probit',
                      label_dx=1.2)
    draw_psychometric(ax_f, binned['logratio'], sl_lr, 'logratio',
                      'log(risky / safe)',
                      'A log-ratio decision axis',
                      'What is left over is what the\nnoise function has to explain',
                      label_dx=0.03)

    # -- row 3: the fitted noise, per option and per channel -----------------
    g3 = outer[2].subgridspec(1, 3, wspace=0.52)
    ax_g, ax_h, ax_i = (fig.add_subplot(g3[0, j]) for j in range(3))
    draw_fitted_curves(ax_g, 'n1 (first)', 'First-presented option (n1)',
                       'Remembered: perceptual + memory', (0.15, 0.34))
    draw_fitted_curves(ax_h, 'n2 (second)', 'Second-presented option (n2)',
                       'On screen: perceptual noise alone', (0.07, 0.40))
    draw_channels(ax_i)

    # -- row 4: the memory effect --------------------------------------------
    g4 = outer[3].subgridspec(1, 2, width_ratios=[0.9, 1.1], wspace=0.34)
    ax_j = fig.add_subplot(g4[0, 0])
    ax_l = fig.add_subplot(g4[0, 1])
    draw_order_effect(ax_j)
    draw_ctbs_forest(ax_l)

    # -- row 4b: the same question, one panel per converged family -----------
    g4b = outer[4].subgridspec(1, 4, wspace=0.58)
    stim_axes = []
    for i, (lab, nice, sub) in enumerate(STIM_MODELS):
        ax = fig.add_subplot(g4b[0, i])
        draw_stim_curves(ax, lab, nice, sub, legend=(i == 0))
        stim_axes.append(ax)
    ax_k = stim_axes[0]

    # -- row 5: the five noise functions -------------------------------------
    g5 = outer[5].subgridspec(1, 5, wspace=0.62)
    for i, (name, col, eq, note, curves) in enumerate(FORMS):
        ax = fig.add_subplot(g5[0, i])
        draw_form(ax, name, col, eq, note, curves)
        if i == 0:
            ax.set_ylabel('Relative noise SD\n(log units)')
            panel_letter(ax, 'm', dx=-0.46, dy=1.30)

    # -- row 6: predictive fit and predictive check --------------------------
    g6 = outer[6].subgridspec(1, 4, width_ratios=[1.60, 0.72, 0.86, 0.92],
                              wspace=0.46)
    ax_n = fig.add_subplot(g6[0, 0])
    ax_p = fig.add_subplot(g6[0, 1])
    ax_p2 = fig.add_subplot(g6[0, 2])
    ax_o = fig.add_subplot(g6[0, 3])
    draw_ladder(ax_n, args.ladder)
    draw_convergence(ax_p, args.ladder)
    draw_family_gate(ax_p2, args.ladder)
    draw_ppc(ax_o, 'Risky second')

    # -- rows 7-8: the three-way cTBS x order x stake predictive check -------
    # -- rows 7-8: the noise functions of exactly those models, twice --------
    avail0 = [m for m in LEVEL_MODELS
              if (CARDS / f'ppc_by_stake_stim_order.{m[0]}.tsv').exists()]
    headers = []
    noise_axes = []
    for row, space, head in [
            (7, 'absolute', 'Fitted noise functions \u2014 absolute (CHF)'),
            (8, 'relative', 'Fitted noise functions \u2014 relative '
                            '(Weber = a flat line)')]:
        gr = outer[row].subgridspec(1, len(avail0), wspace=0.58)
        axes_row = []
        for i, (lab, nice) in enumerate(avail0):
            ax = fig.add_subplot(gr[0, i])
            draw_noise_column(ax, lab, nice, space=space, ylabel=(i == 0),
                              legend=(i == 0 and space == 'absolute'),
                              title=(space == 'absolute'))
            axes_row.append(ax)
        noise_axes.append(axes_row)
        headers.append((axes_row, head))

    lvl = []
    for row, order, note in [(9, 'Risky first', 'the cTBS control'),
                             (10, 'Risky second', 'where the effect lives')]:
        avail = [m for m in LEVEL_MODELS
                 if (CARDS / f'ppc_by_stake_stim_order.{m[0]}.tsv').exists()]
        if len(avail) < len(LEVEL_MODELS):
            missing = [m[0] for m in LEVEL_MODELS if m not in avail]
            print(f'  ! no stake x stim x order card yet for: {missing}')
        gr = outer[row].subgridspec(1, len(avail), wspace=0.58)
        axes_row = []
        for i, (lab, nice) in enumerate(avail):
            ax = fig.add_subplot(gr[0, i])
            draw_levels(ax, lab, nice, order,
                        legend=(i == len(avail) - 1),
                        xlabel=(row == 10))
            axes_row.append(ax)
        lvl.append(axes_row)
        headers.append((axes_row,
                        f'Predictive check \u2014 {order.lower()} ({note})'))
    ax_q, ax_r = lvl[0][0], lvl[1][0]

    sns.despine(fig=fig, offset=3)
    for ax in (ax_a, ax_b, ax_d, ax_l, ax_n, ax_p2):
        ax.spines['left'].set_visible(False)

    for ax, letter, dx in [(ax_a, 'a', -0.17), (ax_b, 'b', -0.14),
                           (ax_c, 'c', -0.24), (ax_d, 'd', -0.14),
                           (ax_e, 'e', -0.10), (ax_f, 'f', -0.10),
                           (ax_g, 'g', -0.20), (ax_h, 'h', -0.20),
                           (ax_i, 'i', -0.20), (ax_j, 'j', -0.16),
                           (ax_k, 'k', -0.30), (ax_l, 'l', -0.06),
                           (ax_n, 'n', -0.04), (ax_p, 'o', -0.26),
                           (ax_p2, 'p', -0.10), (ax_o, 'q', -0.24),
                           (ax_q, 's', -0.24),
                           (ax_r, 't', -0.24),
                           (noise_axes[0][0], 'r', -0.24)]:
        panel_letter(ax, letter, dx=dx, dy=1.08)

    # one header per predictive-check row, instead of repeating the order in
    # all four panel titles
    for axes_row, text in headers:
        boxes = [ax.get_position() for ax in axes_row]
        fig.text(0.5 * (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)),
                 max(b.y1 for b in boxes) + 0.0125, text, ha='center',
                 va='bottom', fontsize=9.4, family='Arial', fontweight='bold',
                 color='.05')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.03)

    print('probit slopes (Low / Mid / High stake tercile)')
    for nm, sl in [('EV difference (per CHF)', sl_ev), ('log ratio', sl_lr)]:
        v = [sl[t][1] for t in ['Low', 'Mid', 'High']]
        print(f'  {nm:24s} {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}   '
              f'Low/High = {v[0] / v[2]:.1f}x')
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
