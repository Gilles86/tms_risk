"""Figure 5 of the preprint, redrawn from a refitted trace.

Same layout as the published panel: two blocks (A total representational noise,
B perceived risky-safe EV ratio), each a 3 x 2 grid of

    rows  vertex stimulation / IPS stimulation / IPS-vs-vertex effect (ratio)
    cols  risky first / risky second

over the (safe payoff, risky-safe EV ratio) decision space, with contour lines.

Input is `decision_space.<label>.tsv` from `plot_decision_space`, which evaluates
the fitted model on a synthetic grid covering that space for every subject, order
and stimulation condition.

    python -m tms_risk.behavior.scripts.plot_fig5_style --label flexible1nf
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

P_RISKY = 0.55                      # the risky option's win probability
ROWS = ['Vertex stimulation', 'IPS stimulation', 'IPS − Vertex effect\n(ratio)']
COLS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 6, 'xtick.labelsize': 5.5,
    'ytick.labelsize': 5.5, 'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.6, 'axes.labelpad': 1.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 1.8, 'ytick.major.size': 1.8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')

BLOCKS = {
    'noise': dict(title='Total representational noise', cmap='inferno',
                  keys=('tot_nu_vertex', 'tot_nu_ips'), fmt='%.1f', dfmt='%.3f'),
    'percept': dict(title='Percept of risky-safe EV-ratio', cmap='mako',
                    keys=('ratio_vertex', 'ratio_ips'), fmt='%.2f', dfmt='%.2f'),
}


def total_noise(space, curves):
    """sqrt(nu_1^2 + nu_2^2) over the grid -- the preprint's Fig-5A quantity.

    Computed straight from the fitted noise functions, so unlike the model's
    internal `diff_sd` it does not depend on the choice rule and is therefore
    comparable across bauer versions.
    """
    def nu(term, cond, x):
        c = curves[(curves.term == term) & (curves.stimulation == cond)]
        c = c.sort_values('payoff')
        return np.interp(x, c.payoff, c.nu)

    n_safe = space.n_safe.values
    n_risky = space.n_safe.values * space.ratio.values
    first_is_risky = (space.order == 'Risky first').values
    n1 = np.where(first_is_risky, n_risky, n_safe)
    n2 = np.where(first_is_risky, n_safe, n_risky)
    for cond in ['vertex', 'ips']:
        space[f'tot_nu_{cond}'] = np.sqrt(nu('n1_evidence_sd', cond, n1) ** 2 +
                                          nu('n2_evidence_sd', cond, n2) ** 2)
    return space


def panel(fig, ax, X, Y, Z, cmap, vmin, vmax, fmt, contour_color, levels=6,
          center=None):
    if center is not None:
        c = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = center - c, center + c
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
    cs = ax.contour(X, Y, Z, levels=levels, colors=contour_color, linewidths=.45)
    ax.clabel(cs, fmt=fmt, fontsize=4.2, inline=True)
    ax.set_xticks([7, 10, 14, 20, 28])
    ax.set_yticks([0.6, 1.1, 1.6, 2.2])
    cb = fig.colorbar(im, ax=ax, pad=.025, aspect=11, fraction=.075)
    cb.ax.tick_params(labelsize=4.6, length=1.2, width=.5)
    cb.outline.set_linewidth(.4)
    return im


def draw_block(fig, gs0, space, spec, letter):
    gs = gs0.subgridspec(3, 2, hspace=.42, wspace=.52)
    k_v, k_i = spec['keys']
    lo = min(space[k_v].min(), space[k_i].min())
    hi = max(space[k_v].max(), space[k_i].max())
    dlo, dhi = space.cause_block.min(), space.cause_block.max()

    for r in range(3):
        for c, order in enumerate(COLS):
            ax = fig.add_subplot(gs[r, c])
            o = space[space.order == order]
            key = {0: k_v, 1: k_i}.get(r)
            if r < 2:
                piv = o.pivot(index='ratio', columns='n_safe', values=key)
                panel(fig, ax, piv.columns.values, piv.index.values * P_RISKY,
                      piv.values, spec['cmap'], lo, hi, spec['fmt'], 'white')
            else:
                piv = o.pivot(index='ratio', columns='n_safe', values='cause_block')
                panel(fig, ax, piv.columns.values, piv.index.values * P_RISKY,
                      piv.values, 'RdBu_r', dlo, dhi,
                      spec['dfmt'], 'black', center=1.0)
            if r == 0:
                ax.set_title(order, fontsize=6, color='0.15', pad=2.5)
            if r == 2:
                ax.set_xlabel('Safe payoff')
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel('Risky-safe ratio (EV)', fontsize=5.4)
                ax.text(-.72, .5, ROWS[r], transform=ax.transAxes, fontsize=5.8,
                        color='0.15', ha='center', va='center', rotation=90,
                        linespacing=1.3)
                if r == 0:
                    ax.text(-.88, 1.22, letter, transform=ax.transAxes, fontsize=11,
                            fontweight='bold', va='bottom', ha='left')
            else:
                ax.set_yticklabels([])


def bare_grid(n=24):
    """The (safe payoff, payoff ratio, order) grid, for the noise block alone."""
    n_safe = np.linspace(7, 28, n)
    ratio = np.linspace(1., 4., n)
    return pd.DataFrame([{'order': o, 'n_safe': s_, 'ratio': r}
                         for o in COLS for r in ratio for s_ in n_safe])


def main(data_dir, label, out_stem, noise_only=False):
    curves = pd.read_csv(Path(data_dir) / f'pmcpars_curves.{label}.tsv', sep='\t')
    ds = Path(data_dir) / f'decision_space.{label}.tsv'
    if noise_only or not ds.exists():
        if not noise_only:
            print(f'{ds.name} not found -- drawing the noise block only')
        noise_only = True
        space = bare_grid()
    else:
        space = pd.read_csv(ds, sep='\t')
        missing = [k for k in ['ratio_vertex', 'ratio_ips'] if k not in space]
        if missing:
            raise SystemExit(f'{label} decision_space TSV lacks {missing}; re-run '
                             'plot_decision_space with the current script')
    space = total_noise(space, curves)

    blocks = {'noise': BLOCKS['noise']} if noise_only else BLOCKS
    fig = plt.figure(figsize=(3.9 if noise_only else 7.25, 4.6))
    outer = fig.add_gridspec(1, len(blocks), wspace=.42,
                             left=.18 if noise_only else .095, right=.985,
                             top=.90, bottom=.085)
    for i, (name, spec) in enumerate(blocks.items()):
        s = space.copy()
        k_v, k_i = spec['keys']
        s['cause_block'] = s[k_i] / s[k_v]
        draw_block(fig, outer[0, i], s, spec, 'AB'[i])
        fig.text(.60 if noise_only else .30 + .50 * i, .965, spec['title'],
                 ha='center', va='center', fontsize=8, fontweight='bold')

    sns.despine(fig=fig, offset=1)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible1nf')
    parser.add_argument('--out', default=None)
    parser.add_argument('--noise_only', action='store_true',
                        help='draw only block A, which needs no decision-space grid')
    args = parser.parse_args()
    main(args.data_dir, args.label, args.out or
         f'/Users/gdehol/git/tms_risk/notes/figures/fig5_style.{args.label}',
         args.noise_only)
