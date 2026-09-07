"""Supplementary: the cTBS contrast for every parameter, one participant at a time.

The group panels of Figure 5 are hierarchical means. A hierarchical mean can sit
credibly away from zero while most individual participants straddle it, and it
can equally look null while a subgroup moves hard. This shows the spread behind
each group estimate: one column per free parameter that carries a cTBS term,
participants sorted by their own posterior median, with the group posterior
drawn on the same axis for scale.

    python -m tms_risk.behavior.scripts.plot_subject_params --model_label log-power-n2psd
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]
IPS, VERTEX = '#d62728', '#2ca02c'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

#: parameter name -> short axis title
PRETTY = {
    'log_n1_power_sd7': 'ν 1st @ 7 CHF', 'log_n1_power_sd112': 'ν 1st @ 112 CHF',
    'log_n2_power_sd7': 'ν 2nd @ 7 CHF', 'log_n2_power_sd112': 'ν 2nd @ 112 CHF',
    'log_perc_power_sd7': 'ν perc @ 7', 'log_perc_power_sd112': 'ν perc @ 112',
    'log_mem_power_sd7': 'ν mem @ 7', 'log_mem_power_sd112': 'ν mem @ 112',
    'log_risky_prior_sd': 'Risky prior σ', 'log_safe_prior_sd': 'Safe prior σ',
    'log_risky_prior_mu': 'Risky prior μ', 'log_safe_prior_mu': 'Safe prior μ',
}


def main(data_dir, out_stem, label):
    dd = Path(data_dir)
    d = pd.read_csv(dd / 'subject_params' / f'subject_params.{label}.tsv', **READ)
    pars = [p for p in d.parameter.unique()]
    n = len(pars)
    fig, axes = plt.subplots(1, n, figsize=(1.65 * n + .4, 3.0),
                             constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, par in zip(axes, pars):
        q = d[d.parameter == par]
        g = q[q.subject == 'GROUP'].iloc[0]
        s_ = q[q.subject != 'GROUP'].sort_values('mid').reset_index(drop=True)
        y = np.arange(len(s_))
        # each participant's own 95% CrI, coloured by the direction of the shift
        for i, r in s_.iterrows():
            col = IPS if r.mid > 0 else VERTEX
            ax.plot([r.lo, r.hi], [i, i], color=col, lw=.7, alpha=.45,
                    solid_capstyle='butt', zorder=1)
        ax.scatter(s_.mid, y, s=7, c=[IPS if v > 0 else VERTEX for v in s_.mid],
                   zorder=3, lw=0)
        ax.axvline(0, color='0.45', lw=.9, zorder=0)
        # the group posterior, on the same scale, as a band across the column
        ax.axvspan(g.lo, g.hi, color='0.35', alpha=.13, lw=0, zorder=0)
        ax.axvline(g.mid, color='0.2', lw=1.3, zorder=2)
        ax.set_title(PRETTY.get(par, par.replace('log_', '')), fontsize=7)
        # one line, above the panel: the group estimate and how unanimous the
        # participants are about it. Below the axis it collided with the ticks.
        frac = float((s_.mid > 0).mean())
        ax.text(.5, 1.002,
                f'group {g.mid:+.3f} (p {min(g.p_gt0, 1 - g.p_gt0):.2f})'
                f'  ·  {frac:.0%} above 0',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=5.7,
                color='0.4')
    axes[0].set_ylabel('Participants, sorted by their own estimate')
    axes[0].set_yticks([])
    for ax in axes:
        ax.set_xlabel('IPS − vertex (log units)')
    # one glyph key, at figure level under the panels, so it sits on no data
    ents = [('Group posterior median', 'line', '0.2'),
            ('95% CrI on the group', 'band', '0.35'),
            ('Participant, IPS > vertex', 'dot', IPS),
            ('Participant, vertex > IPS', 'dot', VERTEX)]
    kx = fig.add_axes([0.02, -0.03, 0.96, 0.045])
    kx.set_xlim(0, 1); kx.set_ylim(0, 1); kx.axis('off')
    x0 = .0
    for lab_, kind, col in ents:
        if kind == 'line':
            kx.plot([x0, x0 + .018], [.5, .5], color=col, lw=1.3)
        elif kind == 'band':
            kx.add_patch(plt.Rectangle((x0, .28), .018, .44, facecolor=col,
                                       alpha=.20, lw=0))
        else:
            kx.plot(x0 + .009, .5, 'o', ms=3, color=col)
        kx.text(x0 + .026, .5, lab_, fontsize=5.8, va='center', color='0.4')
        x0 += .026 + .0092 * len(lab_)
    fig.suptitle(f'Per-participant cTBS contrast · {label}', fontsize=8)
    sns.despine(fig=fig, offset=3, trim=False)
    for ax in axes:
        ax.spines['left'].set_visible(False)
    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out_stem}.pdf')
    fig.savefig(f'{out_stem}.png', dpi=200)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n2psd')
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    main(a.data_dir, a.out_stem or str(REPO / f'notes/figures/supp_subject_params_{a.model_label}'),
         a.model_label)
