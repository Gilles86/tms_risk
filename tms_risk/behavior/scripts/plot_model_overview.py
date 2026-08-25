"""Model-comparison overview: what the choice models can and cannot do.

A  Choice PPC by stake for three models, plain-language labels.
B  The cTBS contrast (IPS - vertex) for the same three -- where they differ.
C  ELPD ladder over every converged candidate, annotated with the two
   contrasts that carry the argument.

Models are named by what they assume, not by their grid coordinate; the
coordinate is given in small type underneath for traceability.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
DATA, OUT = ROOT / 'notes' / 'data', ROOT / 'notes' / 'figures'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'mathtext.fontset': 'stixsans',
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
})

IPS, VERTEX = '#d62728', '#2ca02c'
PASS, FAIL = '#2c7a52', '#b0453b'
BOLD = dict(fontweight='bold', fontfamily='Arial')
ORDERS = ['Risky first', 'Risky second']

TRIO = [('lfx2-bs3-m2-dp-null', 'cTBS changes nothing',
         'Noise free to vary with payoff; no stimulation term',
         'lfx2-bs3-m2-dp-null'),
        ('lfx2-bs3-w-dp-bm', 'Relative noise constant (Weber)',
         'SD proportional to payoff; cTBS may rescale it',
         'lfx2-bs3-w-dp-bm'),
        ('lfx2-bs3-m2-dp-bm', 'Relative noise varies with payoff',
         'SD-to-payoff ratio free; cTBS on memory + perception',
         'lfx2-bs3-m2-dp-bm  (primary)')]

fig = plt.figure(figsize=(7.25, 8.6))

# ================================================================ panel A ==
byst = {k: pd.read_csv(DATA / f'ppc_by_stake.{k}.tsv', sep='\t')
        for k, _, _, _ in TRIO}
allv = pd.concat(byst.values())
stakes = np.sort(allv.stake.unique())
x = np.arange(len(stakes))
ylo, yhi = min(allv.lo.min(), allv.observed.min()) - .015, \
           max(allv.hi.max(), allv.observed.max()) + .015

SPANS = [(.075, .345), (.395, .665), (.715, .985)]
gs_a = [fig.add_gridspec(1, 2, left=l, right=r, wspace=.1, top=.900, bottom=.748)
        for l, r in SPANS]
for m, (key, name, sub, coord) in enumerate(TRIO):
    d, nmiss = byst[key], 0
    for o, order in enumerate(ORDERS):
        ax = fig.add_subplot(gs_a[m][0, o])
        for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
            s = d[(d.order == order) & (d.stim == stim)].sort_values('stake')
            ax.fill_between(x, s.lo, s.hi, color=colr, alpha=.20, lw=0)
            ax.plot(x, s['mean'], color=colr, lw=1.2)
            dx = .07 if stim == 'ips' else -.07
            ax.plot(x + dx, s.observed, 'o', color=colr, ms=3.6, lw=0, zorder=4)
            for _, r in s.iterrows():
                if r.lo <= r.observed <= r.hi:
                    continue
                nmiss += 1
                xi = float(x[np.argmin(np.abs(stakes - r.stake))]) + dx
                ax.annotate('', xy=(xi, r.observed), xycoords='data',
                            xytext=(-13, 11 if r.observed > r.hi else -11),
                            textcoords='offset points', zorder=6,
                            arrowprops=dict(arrowstyle='-|>', color='.1', lw=.9,
                                            shrinkA=0, shrinkB=3.5,
                                            mutation_scale=6))
        ax.set_ylim(ylo, yhi); ax.set_xlim(-.45, len(stakes) - .55)
        ax.set_xticks(x); ax.set_xticklabels([f'{v:.0f}' for v in stakes])
        ax.set_yticks([.5, .55, .6, .65])
        ax.set_title(order, fontsize=7, color='.35', pad=3, style='italic')
        if m == 0 and o == 0:
            ax.set_ylabel('P(chose risky)')
            ax.text(.05, .97, 'IPS (stimulated)', transform=ax.transAxes,
                    fontsize=6.8, color=IPS, va='top')
            ax.text(.05, .86, 'Vertex (control)', transform=ax.transAxes,
                    fontsize=6.8, color=VERTEX, va='top')
        else:
            ax.set_yticklabels([])
    l, r = SPANS[m]
    fig.text((l + r) / 2, .962, name, ha='center', fontsize=8, **BOLD)
    fig.text((l + r) / 2, .945, sub, ha='center', fontsize=6.4, color='.35')
    fig.text((l + r) / 2, .930, coord, ha='center', fontsize=5.8, color='.6',
             family='monospace')
    fig.text((l + r) / 2, .914,
             f'{nmiss} of 12 cells missed', ha='center', fontsize=7,
             color=(FAIL if nmiss else PASS))
    fig.text((l + r) / 2, .707, 'Stake (CHF)', ha='center', fontsize=8)

# ================================================================ panel B ==
dlt = {k: pd.read_csv(DATA / f'ppc_delta_by_stake.{k}.tsv', sep='\t')
       for k, _, _, _ in TRIO}
alld = pd.concat(dlt.values())
elo = min(alld.lo.min(), (alld.obs - alld.obs_sem).min()) - .012
ehi = max(alld.hi.max(), (alld.obs + alld.obs_sem).max()) + .012
gs_b = [fig.add_gridspec(1, 2, left=l, right=r, wspace=.1, top=.628, bottom=.475)
        for l, r in SPANS]
for m, (key, name, _, _c) in enumerate(TRIO):
    e = dlt[key]
    ncov = int(((e.obs >= e.lo) & (e.obs <= e.hi)).sum())
    for o, order in enumerate(ORDERS):
        ax = fig.add_subplot(gs_b[m][0, o])
        s = e[e.order == order].sort_values('stake')
        ax.fill_between(x, s.lo, s.hi, color='.82', lw=0)
        ax.plot(x, s['median'], color='.3', lw=1.5)
        ax.errorbar(x, s.obs, yerr=s.obs_sem, fmt='o', color='.05', ms=3.6,
                    elinewidth=.9, capsize=0, zorder=4)
        ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
        ax.set_ylim(elo, ehi); ax.set_xlim(-.45, len(stakes) - .55)
        ax.set_xticks(x); ax.set_xticklabels([f'{v:.0f}' for v in stakes])
        ax.set_title(order, fontsize=7, color='.35', pad=3, style='italic')
        ax.set_xlabel('Stake (CHF)')
        if m == 0 and o == 0:
            ax.set_ylabel('Effect of cTBS\nΔ P(chose risky)')
            ax.text(.05, .97, 'Dots = data', transform=ax.transAxes,
                    fontsize=6.8, color='.05', va='top')
            ax.text(.05, .86, 'Band = model', transform=ax.transAxes,
                    fontsize=6.8, color='.45', va='top')
        else:
            ax.set_yticklabels([])
    l, r = SPANS[m]
    fig.text((l + r) / 2, .651, f'{ncov} of 6 effect cells captured',
             ha='center', fontsize=7.5,
             color=(PASS if ncov == 6 else FAIL), **BOLD)

# ================================================================ panel C ==
ROWS = [  # (plain name, coordinate, elpd, converged, delta-ppc cells)
    ('Relative noise varies with payoff · cTBS on memory + perception',
     'lfx2-bs3-m2-dp-bm', -4149.0, True, 6),
    ('Relative noise varies with payoff · cTBS on perception only',
     'lfx2-bs2-m3-dp-b', -4153.2, True, 4),
    ('Relative noise varies with payoff · cTBS on both (quadratic memory)',
     'lfx2-bs2-m3-dp-bm', -4155.0, True, 6),
    ('Published model, natural space · cTBS on both',
     'flexible2', -4161.0, False, 6),
    ('Relative noise constant (Weber) · cTBS on both',
     'lfx2-bs3-w-dp-bm', -4195.1, True, 4),
    ('Published model, natural space · cTBS on memory only',
     'flexible2a', -4217.5, True, 6),
    ('cTBS changes nothing', 'lfx2-bs3-m2-dp-null', -4245.3, True, 4),
    ('cTBS changes nothing · relative noise constant (Weber)',
     'lfx2-bs3-w-dp-null', -4264.2, True, 4),
    ('cTBS changes nothing · natural space', 'flexible2_null', -4271.4, True, None),
]
best = max(r[2] for r in ROWS)
ys = np.arange(len(ROWS))[::-1]
YL = (-.85, len(ROWS) - .15)

axlab = fig.add_axes([.010, .055, .385, .315]); axlab.axis('off')
axlab.set_xlim(0, 1); axlab.set_ylim(*YL)
for y, (name, coord, e, conv, cov) in zip(ys, ROWS):
    axlab.text(1.0, y + .18, name, fontsize=6.6, ha='right', va='center',
               color=('.05' if y == ys[0] else '.25'),
               **(BOLD if y == ys[0] else {}))
    axlab.text(1.0, y - .30, coord, fontsize=5.5, ha='right', va='center',
               color='.62', family='monospace')

ax = fig.add_axes([.405, .055, .30, .315])
for y, (name, coord, e, conv, cov) in zip(ys, ROWS):
    d = e - best
    c = PASS if cov == 6 else (FAIL if cov is not None else '.45')
    if conv:
        ax.plot(d, y, 'o', color=c, ms=6, zorder=3)
    else:
        ax.plot(d, y, 'o', mfc='white', mec=c, mew=1.4, ms=6, zorder=3)
ax.axvline(0, color='.8', lw=.6, ls='--', zorder=0)
ax.set_xlim(-134, 14); ax.set_ylim(*YL)
ax.set_xticks([-120, -90, -60, -30, 0])
ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlabel('Predictive accuracy, ELPD − best (nats)\nhigher is better', fontsize=8)


# the two contrasts that carry the argument
def bracket(y0, y1, xpos, label, colr):
    ax.plot([xpos, xpos - 5, xpos - 5, xpos], [y0, y0, y1, y1], color=colr,
            lw=.9, clip_on=False)
    ax.text(xpos - 8, (y0 + y1) / 2, label, fontsize=6.4, ha='right',
            va='center', color=colr, **BOLD)
GAP = '#33619e'
for row, txt in [(4, '46 nats'), (6, '96 nats')]:
    d = ROWS[row][2] - best
    ax.annotate('', xy=(0, ys[row]), xytext=(d, ys[row]),
                arrowprops=dict(arrowstyle='<->', color=GAP, lw=.8,
                                shrinkA=3, shrinkB=3))
    ax.text(d / 2, ys[row] + .40, txt, fontsize=6.6, ha='center',
            va='center', color=GAP, **BOLD)

leg = fig.add_axes([.735, .055, .255, .315]); leg.axis('off')
leg.set_xlim(0, 1); leg.set_ylim(0, 1)
leg.text(0, 1.0, 'How to read this', fontsize=8, va='top',
         transform=leg.transAxes, **BOLD)
items = [
    (PASS, 'o', True, 'Captures all 6 cTBS-effect cells'),
    (FAIL, 'o', True, 'Misses ≥ 2 of them'),
    ('.45', 'o', False, 'Open = chains did not mix (r̂ > 1.01),\nso its verdict is not usable'),
]
for i, (c, mk, filled, txt) in enumerate(items):
    yy = .86 - i * .13
    if filled:
        leg.plot([.045], [yy], mk, color=c, ms=6, transform=leg.transAxes)
    else:
        leg.plot([.045], [yy], mk, mfc='white', mec=c, mew=1.4, ms=6,
                 transform=leg.transAxes)
    leg.text(.13, yy, txt, fontsize=6.6, va='center', color='.2',
             transform=leg.transAxes)
leg.text(0, .49, 'Blue arrows', fontsize=6.8, va='top', color='#33619e',
         transform=leg.transAxes, **BOLD)
leg.text(0, .43,
         '46 nats: the noise-to-payoff ratio must\nvary with payoff — noise merely\n'
         'proportional to payoff (Weber) is not\nenough.\n'
         '96 nats: cTBS must be allowed to\nchange the noise at all.',
         fontsize=6.5, va='top', color='.25', transform=leg.transAxes)
leg.text(0, .195,
         'Every model shown fits a Bayesian prior over\npayoffs; pinning it at the payoff statistics\n'
         'costs a further 583 ± 31 nats.',
         fontsize=6.2, va='top', color='.45', transform=leg.transAxes)
leg.text(0, .085,
         'Bands: 95% posterior-predictive intervals from 200\ndraws, computed per subject then averaged across\n'
         'the 35 subjects within draw. Data: ±1 SEM.',
         fontsize=6.2, va='top', color='.45', transform=leg.transAxes)

for xf, yf, t in [(.012, .980, 'A'), (.012, .680, 'B'), (.012, .392, 'C')]:
    fig.text(xf, yf, t, fontsize=12, va='bottom', ha='left', **BOLD)
fig.text(.040, .986, 'Does the model reproduce the choices?', fontsize=8.5,
         color='.25', style='italic', va='bottom')
fig.text(.040, .682, 'Does it reproduce the EFFECT of stimulation?',
         fontsize=8.5, color='.25', style='italic', va='bottom')
fig.text(.040, .394, 'Model comparison over every converged candidate',
         fontsize=8.5, color='.25', style='italic', va='bottom')

for a in fig.axes:
    if a not in (leg, axlab):
        sns.despine(ax=a, offset=3)
fig.savefig(OUT / 'model_overview.pdf')
fig.savefig(OUT / 'model_overview.png', dpi=150)
print('wrote', OUT / 'model_overview.pdf')
