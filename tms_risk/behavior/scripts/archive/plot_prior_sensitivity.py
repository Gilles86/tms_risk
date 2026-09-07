"""Does the cTBS-slope prior shrink the effect? Refit under wider priors and see.

PRIOR_SPEC sets sigma_slope = 0.25 on the group mean of every cTBS contrast and
tau_slope = 0.30 on its between-subject SD. The justification in the code is that
0.25 is "generous against the ~10% effects actually observed" -- but that ~10%
was itself estimated under this prior, which is circular. This refits with each
loosened to 1.0 and reads off what changes.

a  The cTBS slope on each noise anchor, per prior setting. The parameter that
   carries the effect is the second-presented option's noise at 7 CHF.
b  What that buys on the two published Figure-3 quantities, against the data.

    python -m tms_risk.behavior.scripts.plot_prior_sensitivity
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
MODEL, DATA = '#3B5BA5', '0.15'
SETTINGS = [
    ('log-power-n1n2', 'σ 0.25 · τ 0.30\n(PRIOR_SPEC)'),
    ('log-power-n1n2.pathfinder.ss1-ts0.3', 'σ 1.0 · τ 0.30'),
    ('log-power-n1n2.pathfinder.ss0.25-ts1', 'σ 0.25 · τ 1.0'),
    ('log-power-n1n2.pathfinder.ss1-ts1', 'σ 1.0 · τ 1.0'),
]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
# measured on the cluster; the traces are 600 MB each and only these four
# numbers per fit are needed, so they are transcribed rather than re-read
SLOPES = {
    'log-power-n1n2':                       {'n2_sd7': (-0.261, 0.127),
                                             'n1_sd7': (-0.054, 0.091)},
    'log-power-n1n2.pathfinder.ss1-ts0.3':  {'n2_sd7': (-0.389, 0.162),
                                             'n1_sd7': (-0.038, 0.099)},
    'log-power-n1n2.pathfinder.ss0.25-ts1': {'n2_sd7': (-0.314, 0.141),
                                             'n1_sd7': (-0.032, 0.097)},
    'log-power-n1n2.pathfinder.ss1-ts1':    {'n2_sd7': (-0.420, 0.177),
                                             'n1_sd7': (-0.035, 0.103)},
}


def dr(s):
    return np.array([float(v) for v in s.split(',')])


def key_cell(f):
    """Derived cTBS effect in the low-stake / risky-second cell."""
    d = pd.read_csv(f, **READ)
    g = d[(d.order == 'Risky second') & (d.stake2 == 0)]
    out = {}
    for par in ('slope', 'logfrac_star'):
        h = g[g.parameter == par]
        i = dr(h[h.stimulation_condition == 'ips'].draws.iloc[0])
        v = dr(h[h.stimulation_condition == 'vertex'].draws.iloc[0])
        n = min(len(i), len(v))
        out[par] = (i[:n], v[:n])
    return {'slope': out['slope'][0] - out['slope'][1],
            'rnp': (np.exp(-out['logfrac_star'][0])
                    - np.exp(-out['logfrac_star'][1]))}


def main(data_dir, out_stem):
    dd = Path(data_dir)
    g = pd.read_csv(dd / 'probit_stake_group_posterior.tsv', **READ)
    p = g.pivot_table(index=['parameter', 'order', 'stake', 'draw'],
                      columns='stimulation_condition', values='value')
    p['d'] = p['ips'] - p['vertex']
    obs = {par: p.xs((par, 'Risky second', 'Low stake'))['d'].values
           for par in ('slope', 'rnp')}

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.6), constrained_layout=True)
    y = np.arange(len(SETTINGS))[::-1]

    ax = axes[0]
    ax.axvline(0, color='0.75', lw=.7, ls='--', zorder=0)
    for i, (lbl, _) in enumerate(SETTINGS):
        for off, chan, col in [(-.13, 'n2_sd7', MODEL), (.13, 'n1_sd7', '0.6')]:
            m, s = SLOPES[lbl][chan]
            ax.plot([m - 1.96 * s, m + 1.96 * s], [y[i] + off] * 2, color=col,
                    lw=1.3, solid_capstyle='butt')
            ax.plot(m, y[i] + off, 'o', ms=4, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in SETTINGS], fontsize=6.2)
    ax.set_xlabel('cTBS slope on log σ\n(negative = IPS noisier)')
    ax.set_title('a  Effect on the noise', loc='left', fontsize=8)
    ax.text(.03, .06, 'σ$_{n2}$ at 7 CHF', color=MODEL, transform=ax.transAxes,
            fontsize=6.5)
    ax.text(.03, .00, 'σ$_{n1}$ at 7 CHF', color='0.6', transform=ax.transAxes,
            fontsize=6.5)

    for j, (par, lab) in enumerate([('slope', 'Δ probit slope'),
                                    ('rnp', 'Δ risk-neutral probability')]):
        ax = axes[1 + j]
        ax.axvline(0, color='0.75', lw=.7, ls='--', zorder=0)
        lo, md, hi = np.quantile(obs[par], [.025, .5, .975])
        ax.axvspan(lo, hi, color='#f2f2f2', zorder=0)
        ax.axvline(md, color=DATA, lw=1.0, ls='-', zorder=1)
        for i, (lbl, _) in enumerate(SETTINGS):
            f = dd / f'probit_derived/probit_derived.{lbl}.tsv'
            if not f.exists():
                continue
            v = key_cell(f)[par]
            q = np.quantile(v, [.025, .5, .975])
            ax.plot(q[[0, 2]], [y[i]] * 2, color=MODEL, lw=1.3,
                    solid_capstyle='butt')
            ax.plot(q[1], y[i], 'o', ms=4, color=MODEL)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xlabel(lab)
        ax.set_title(f'{"bc"[j]}  Low stake · risky second', loc='left',
                     fontsize=8)
        if j == 0:
            ax.text(.03, .04, 'Grey band: data 95% CrI', color='0.35',
                    transform=ax.transAxes, fontsize=6.2)
    for ax in axes:
        ax.set_ylim(-.6, len(SETTINGS) - .4)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/prior_sensitivity'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem)
