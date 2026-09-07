"""Figure 4, candidate revision (2026-08-19): the v9 layout with panel B in NATURAL space.

Same four panels as plot_fig4_model.py (A posterior predictive checks, B noise
function, C relative cTBS effect, D ELPD ladder). Only B changes: natural axes with
two references — pure Weber (proportionality through the origin, the actual content
of Weber's law) and "Weber + floor" (affine) — replacing the log-log "slope 0.48 vs
slope 1" framing, which conflates a compressive power law with an additive noise
floor. The nu_1 dashed line of the original is dropped: with the affine reference
also dashed the two are confusable, and the memory gap (~0.1 CHF) belongs in text.

Earlier candidate panels (PPC of the cTBS effect by safe payoff; model-free delta-P
by risky payoff; the P(regional increase) annotation on C) were set aside — see the
git history of this file and notes/v9_plan.md 5c.

    python -m tms_risk.behavior.scripts.plot_fig4_new
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tms_risk.behavior.scripts.plot_fig4_model import (
    BOLD, DIFF, FLEX, IPS, VERTEX, WEBER, ppc_panel, shorten)

# plot_fig4_model runs sns.set_context('paper') at import, which clobbers the
# rcParams; re-assert the ones that matter.
mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'mathtext.fontset': 'stixsans',
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def main(data_dir, table, label, weber_label, out_stem):
    data = Path(data_dir)
    t = pd.read_csv(table, sep='\t', index_col=0).sort_values('elpd_diff')
    t['base'] = t.name.map(shorten)
    t['weber'] = t.index.str.startswith('weber')
    t['family'] = t.index.str.extract(r'(?:flexible|weber)([12])')[0].values
    t['short'] = [('Weber: ' if r.weber else 'Flexible: ') + r.base
                  for _, r in t.iterrows()]
    dup = t.short.duplicated(keep=False)
    t.loc[dup, 'short'] = [f'{r.short} ({r.family})' for _, r in t[dup].iterrows()]

    n_models = len(t)
    H_A, H_BC, H_D = 1.30, 1.45, .145 * n_models
    PAD_TOP, GAP_A, GAP_BC, PAD_BOT = .60, .80, .72, .38
    H = PAD_TOP + H_A + GAP_A + H_BC + GAP_BC + H_D + PAD_BOT
    fig = plt.figure(figsize=(7.25, H))

    def band(top_in, height_in):
        return dict(top=1 - top_in / H, bottom=1 - (top_in + height_in) / H)

    row_a = band(PAD_TOP, H_A)
    row_bc = band(PAD_TOP + H_A + GAP_A, H_BC)
    row_d = band(PAD_TOP + H_A + GAP_A + H_BC + GAP_BC, H_D)

    PAIRS = [(.075, .495), (.575, .995)]
    gs_a = [fig.add_gridspec(1, 2, left=l, right=r, wspace=.12, **row_a)
            for l, r in PAIRS]
    gs_bc = fig.add_gridspec(1, 2, left=.095, right=.975, wspace=.30, **row_bc)
    gs_d = fig.add_gridspec(1, 1, left=.28, right=.985, **row_d)

    # --- A: posterior predictive check, Weber against Flexible (unchanged)
    ppc_axes = [fig.add_subplot(gs_a[m][0, o]) for m in (0, 1) for o in (0, 1)]
    misses = ppc_panel(ppc_axes, data, label, weber_label)
    for (l, r), nm in zip(PAIRS, ['Weber PMC', 'Flexible PMC']):
        fig.text((l + r) / 2, row_a['top'] + .19 / H, nm, ha='center', va='bottom',
                 fontsize=9.5, color='.1')
        fig.text((l + r) / 2, row_a['bottom'] - .42 / H, 'Stake (CHF)', ha='center',
                 va='bottom', fontsize=8.5)

    # --- B: noise function in NATURAL space, Weber and Weber+floor references
    ax_b = fig.add_subplot(gs_bc[0, 0])
    c = pd.read_csv(data / f'pmcpars_curves.{label}.tsv', sep='\t')
    TERM = 'perceptual_noise_sd'
    for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
        s_ = c[(c.term == TERM) & (c.stimulation == stim)].sort_values('payoff')
        ax_b.fill_between(s_.payoff, s_.lo, s_.hi, color=colr, alpha=.16, lw=0,
                          zorder=1)
        ax_b.plot(s_.payoff, s_.nu, color=colr, zorder=2)
    v = c[(c.term == TERM) & (c.stimulation == 'vertex')].sort_values('payoff')
    x_, nu_ = v.payoff.values, v.nu.values
    w_ = 1 / (((v.hi - v.lo) / 2).values ** 2)
    b_web = np.linalg.lstsq((x_ * np.sqrt(w_))[:, None], nu_ * np.sqrt(w_),
                            rcond=None)[0][0]
    A_ = np.vstack([np.ones_like(x_), x_]).T
    a_aff, b_aff = np.linalg.lstsq(A_ * np.sqrt(w_)[:, None], nu_ * np.sqrt(w_),
                                   rcond=None)[0]
    xr = np.linspace(0, 112, 100)
    ax_b.plot(xr, b_web * xr, ls=':', color='.6', lw=.8, zorder=0)
    ax_b.plot(xr, a_aff + b_aff * xr, ls='--', color='.45', lw=.8, zorder=0)
    ax_b.text(.03, .96, 'Weber + floor', transform=ax_b.transAxes, fontsize=6.2,
              color='.4', va='top')
    ax_b.text(.03, .87, 'Weber (k·payoff)', transform=ax_b.transAxes, fontsize=6.2,
              color='.55', va='top')
    ax_b.text(.97, .26, 'IPS', transform=ax_b.transAxes, fontsize=7.2, color=IPS,
              ha='right')
    ax_b.text(.97, .16, 'Vertex', transform=ax_b.transAxes, fontsize=7.2,
              color=VERTEX, ha='right')
    ax_b.set_xlim(0, 118)
    ax_b.set_ylim(0, 7.2)
    ax_b.set_xticks([0, 28, 56, 84, 112])
    ax_b.set_yticks([0, 2, 4, 6])
    ax_b.set_xlabel('Payoff (CHF)')
    ax_b.set_ylabel(r'Noise $\nu$ (CHF)')

    # --- C: the increase as a percentage, with its credible interval
    ax_c = fig.add_subplot(gs_bc[0, 1])
    rel = pd.read_csv(data / f'pmcpars_relative.{label}.tsv', sep='\t')
    rel = rel[rel.term == 'perceptual_noise_sd'].sort_values('payoff')
    ax_c.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax_c.fill_between(rel.payoff, rel.lo, rel.hi, color=DIFF, alpha=.16, lw=0,
                      zorder=1)
    ax_c.plot(rel.payoff, rel.pct, color=DIFF, zorder=2)
    ax_c.set_xscale('log')
    ax_c.set_xticks([7, 14, 28, 56, 112])
    ax_c.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_c.minorticks_off()
    ax_c.set_xlabel('Payoff (CHF)')
    ax_c.set_ylabel(r'$\Delta$ noise, IPS − vertex (%)')

    # --- D: ELPD ladder (unchanged)
    ax_d = fig.add_subplot(gs_d[0, 0])
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        colr = WEBER if r.weber else FLEX
        ax_d.errorbar(r.elpd_diff, yi, xerr=r.dse, fmt='o', color=colr, ms=4.4,
                      lw=0, elinewidth=1.1, capsize=0, zorder=3)
    ax_d.axvline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([r.short for _, r in t.iterrows()], fontsize=7)
    ax_d.set_xlabel('ELPD cost vs the best model (nats)')
    ax_d.set_ylim(-.8, len(t) - .2)
    ax_d.invert_xaxis()
    ax_d.annotate('Shown in A–C', xy=(t.elpd_diff.iloc[0], y[0]),
                  xytext=(t.elpd_diff.iloc[2], y[0] + .45), fontsize=6.5,
                  color='.35', ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-.2',
                                  color='.5', lw=.6))

    sns.despine(fig=fig, offset=4)
    fig.canvas.draw()
    for ax_l, ax_r, yf, letter, title in [
            (ppc_axes[0], ppc_axes[3], row_a['top'] + .36 / H, 'A',
             'Posterior predictive checks'),
            (ax_b, ax_b, row_bc['top'] + .15 / H, 'B',
             'Noise as a function of magnitude'),
            (ax_c, ax_c, row_bc['top'] + .15 / H, 'C', 'Effect of cTBS on noise'),
            (ax_d, ax_d, row_d['top'] + .07 / H, 'D', 'Model comparison')]:
        fig.text(.008 if letter != 'C' else .507, yf, letter, fontsize=11,
                 va='bottom', ha='left', **BOLD)
        fig.text((ax_l.get_position().x0 + ax_r.get_position().x1) / 2, yf + .006,
                 title, fontsize=8.5, color='.1', va='bottom', ha='center', **BOLD)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf')
    print(f'  weber ref slope {b_web:.4f}; affine floor {a_aff:.2f} + {b_aff:.4f}x')
    print('  PPC misses: ' + (', '.join(f'{k} {v}' for k, v in misses.items())
                              or 'none'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--table',
                        default='/Users/gdehol/git/tms_risk/notes/data/table1_all16.tsv')
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--weber_label', default='weber2nf')
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/fig4_new')
    a = parser.parse_args()
    main(a.data_dir, a.table, a.label, a.weber_label, a.out)
