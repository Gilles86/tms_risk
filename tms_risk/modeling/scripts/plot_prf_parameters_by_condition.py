"""Diagnostic figure 2: best-fitting nPRF parameters in signal voxels, IPS vs vertex.

Not a paper figure -- all the relevant data is on the page, including the per-voxel
distributions a paper figure would collapse to a bar.

Parameters come from the MAIN fits (all of sessions 2+3, no folds held out), so these
are the best-fitting estimates rather than cross-validated ones. Which parameters are
free depends on the model: under m1 only `amplitude` differs between sessions, under m4
only `mu`/`sd`, under m5 `amplitude`/`baseline`, under m2 all four. A parameter that is
POOLED in a given model is identical across conditions by construction, and is drawn
greyed out so that a flat line is never mistaken for a null result.

    top row     per-voxel distribution per condition (pooled over subjects), so the
                shape and any bimodality is visible
    bottom row  per-subject paired change, IPS − vertex, with the paired t-test. This
                is the inferential panel; the top row is descriptive.

Units: `mu` is log preferred numerosity, so it is shown as exp(mu) in CHF-equivalent
numerosity; `sd` is a width in LOG units (a Weber-like coefficient), not a numerosity.

    python -m tms_risk.modeling.scripts.plot_prf_parameters_by_condition \
        --model_label 5 --roi NPCr2cm-cluster

Reads notes/data/prf_params_by_condition.tsv (see `extract_prf_params_by_condition.py`).
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

IPS, VERTEX = '#d62728', '#2ca02c'
PARAMS = [('pref_n', 'Preferred numerosity  exp(mu)', True),
          ('sd', 'Dispersion  (log units)', False),
          ('amplitude', 'Amplitude', False),
          ('baseline', 'Baseline', False)]
# which parameters are free per session, by model label
SESSION_VARYING = {0: [], 1: ['amplitude'], 2: ['amplitude', 'mu', 'sd', 'baseline'],
                   3: ['amplitude', 'sd'], 4: ['mu', 'sd'], 5: ['amplitude', 'baseline']}

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def main(data_dir, model_label, roi, out_stem, log_x):
    d = pd.read_csv(Path(data_dir) / 'prf_params_by_condition.tsv', sep='\t')
    d = d[(d.model == model_label) & (d.roi == roi)]
    if not len(d):
        raise SystemExit(f'no rows for model {model_label}, roi {roi}')
    varying = SESSION_VARYING.get(model_label, [])
    n_sub = d.subject.nunique()

    fig, axes = plt.subplots(2, len(PARAMS), figsize=(12.0, 5.4),
                             constrained_layout=True)

    for j, (par, label, as_exp) in enumerate(PARAMS):
        free = par.replace('pref_n', 'mu') in varying
        cols = [f'{par}_vertex', f'{par}_ips']
        if not all(c in d.columns for c in cols):
            for ax in (axes[0, j], axes[1, j]):
                ax.set_visible(False)
            continue

        # ---------------------------------------------------- top: distributions
        ax = axes[0, j]
        lo, hi = np.nanpercentile(d[cols].values, [0.5, 99.5])
        bins = (np.geomspace(max(lo, 1e-3), hi, 45) if (as_exp and log_x and lo > 0)
                else np.linspace(lo, hi, 45))
        for c, col, lab in [(cols[0], VERTEX, 'Vertex'), (cols[1], IPS, 'IPS')]:
            ax.hist(d[c].dropna(), bins=bins, density=True, histtype='step',
                    color=col if free else '0.6', lw=1.3, label=lab)
        if as_exp and log_x and lo > 0:
            ax.set_xscale('log')
        ax.set_xlabel(label)
        ax.set_ylabel('Density' if j == 0 else '')
        if j == 0:
            ax.legend(loc='upper right')
        if not free:
            ax.text(.5, .5, 'pooled across sessions\n(identical by construction)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=7.5,
                    color='.45')

        # ------------------------------------------------ bottom: paired change
        ax = axes[1, j]
        ps = d.groupby('subject')[cols].mean().dropna()
        diff = (ps[cols[1]] - ps[cols[0]]).values
        ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
        for k, (a_, b_) in enumerate(zip(ps[cols[0]], ps[cols[1]])):
            ax.plot([0, 1], [a_, b_], color='.75', lw=.6, zorder=1)
        ax.scatter(np.zeros(len(ps)), ps[cols[0]], s=16,
                   color=VERTEX if free else '.6', lw=0, alpha=.7, zorder=3)
        ax.scatter(np.ones(len(ps)), ps[cols[1]], s=16,
                   color=IPS if free else '.6', lw=0, alpha=.7, zorder=3)
        for x, c, col in [(0, cols[0], VERTEX), (1, cols[1], IPS)]:
            m, se = ps[c].mean(), stats.sem(ps[c])
            ax.errorbar(x, m, yerr=se, fmt='D', ms=7,
                        color=col if free else '.4', mec='0.15', mew=1.4,
                        elinewidth=1.5, capsize=0, zorder=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Vertex', 'IPS'])
        ax.set_xlim(-.4, 1.4)
        ax.set_ylabel('Per-subject mean' if j == 0 else '')
        if free and np.isfinite(diff).sum() > 2:
            t, p = stats.ttest_1samp(diff[np.isfinite(diff)], 0)
            ax.set_title(f'Δ = {np.nanmean(diff):+.4f}   t({len(ps)-1}) = {t:+.2f}, '
                         f'p = {p:.3f}', fontsize=7.5,
                         color='0.15' if p < .05 else '0.45', pad=3)
        else:
            ax.set_title('not free in this model', fontsize=7.5, color='.55', pad=3)

    for j in range(len(PARAMS)):
        if axes[0, j].get_visible():
            axes[0, j].text(-0.16, 1.06, 'abcd'[j], transform=axes[0, j].transAxes,
                            fontsize=12, fontweight='bold', va='bottom', ha='right')
    free_txt = ', '.join(varying) if varying else 'nothing (fully pooled)'
    fig.suptitle(f'Model m{model_label} — free per session: {free_txt}   ·   {roi}   ·   '
                 f'n = {n_sub} subjects, {len(d)} voxels   ·   main fit, no folds held out',
                 fontsize=9)
    sns.despine(fig=fig, offset=3)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='notes/data')
    p.add_argument('--model_label', default=5, type=int)
    p.add_argument('--roi', default='NPCr2cm-cluster')
    p.add_argument('--log_x', action='store_true', default=True)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    out = a.out or f'notes/figures/prf_params_m{a.model_label}_{a.roi}'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    main(a.data_dir, a.model_label, a.roi, out, a.log_x)
