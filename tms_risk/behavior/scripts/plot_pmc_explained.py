"""One figure for the whole causal chain: cTBS -> noise -> percept -> choice.

Every panel is drawn from a TSV written by another script in this folder, so the
figure can be rebuilt locally without touching the 1 GB traces.

    row 1  the cause          nu(payoff) per condition; the cTBS contrast; LOO
    row 2  the transformation percept transfer function; per-option shift; which
                              channel of the model actually moves choices
    row 3  the behaviour      where in the decision space it bites, and the
                              observed order-specific effect against two models

Source data (all in notes/data/, <label> defaults to flexible1nf -- the converged
constrained refit against current bauer):

    pmcpars_curves.<label>.tsv        noise functions + cTBS contrast
    summary_flexible1.loo.tsv         ArviZ LOO over the nested variants
    pmc_percepts.<label>.tsv          perceived EV per objective payoff
    pmc_channels_by_ratio.<label>.tsv Delta P(risky) per model channel
    decision_space.<label>.tsv        Delta P(risky) over (safe payoff x ratio)
    localnoise_delta_by_ratio.tsv     observed Delta P + hierarchical probit fit
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS, MODEL = '#2ca02c', '#d62728', '#1f6fb4'
SAFE, RISKY, PROBIT = '#2b7a8c', '#7b4f9d', '0.55'
TERM = {'n1_evidence_sd': 'First-presented', 'n2_evidence_sd': 'Second-presented'}

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def letter(ax, s, x=-0.24, y=1.10):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10.5, fontweight='bold',
            va='bottom', ha='right')


def logx(ax, lo=7, hi=28):
    ax.set_xscale('log')
    ax.set_xticks([t for t in [7, 10, 14, 20, 28, 56, 112] if lo <= t <= hi])
    ax.set_xlim(lo, hi)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


def main(data_dir, label, out_stem, x_hi):
    dd = Path(data_dir)
    curves = pd.read_csv(dd / f'pmcpars_curves.{label}.tsv', sep='\t')
    loo = pd.read_csv(dd / 'summary_flexible1.loo.tsv', sep='\t', index_col=0)
    perc = pd.read_csv(dd / f'pmc_percepts.{label}.tsv', sep='\t')
    chan = pd.read_csv(dd / f'pmc_channels_by_ratio.{label}.tsv', sep='\t')
    space = pd.read_csv(dd / f'decision_space.{label}.tsv', sep='\t')
    obs = pd.read_csv(dd / 'localnoise_delta_by_ratio.tsv', sep='\t')
    ppc = pd.read_csv(dd / f'ppc_fig3a.{label}.tsv', sep='\t')
    priors = (pd.read_csv(dd / f'pmcpars_priors.{label}.tsv', sep='\t')
                .query('level == "group"').set_index('parameter')['mean'])

    fig = plt.figure(figsize=(7.25, 7.9))
    gs = fig.add_gridspec(3, 3, hspace=.62, wspace=.58,
                          left=.085, right=.965, top=.945, bottom=.055)

    # ================================================================= a  nu(payoff)
    ax = fig.add_subplot(gs[0, 0])
    for term, ls in [('n1_evidence_sd', '--'), ('n2_evidence_sd', '-')]:
        for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
            s = curves[(curves.term == term) & (curves.stimulation == cond)]
            ax.plot(s.payoff, s.nu, color=col, ls=ls, lw=1.3)
            if ls == '-':
                ax.fill_between(s.payoff, s.lo, s.hi, color=col, alpha=.15, lw=0)
    ax.set_ylabel('Representational noise ν (CHF)')
    ax.set_xlabel('Payoff magnitude (CHF)')
    ax.set_title('cTBS makes the representation\nnoisier', fontsize=7.5, color='0.15')
    logx(ax, hi=x_hi)
    top_a = float(curves[(curves.stimulation != 'ips - vertex') &
                         (curves.payoff <= x_hi)].hi.max()) * 1.32
    ax.set_ylim(0, top_a)
    ax.text(7.2, top_a * .97, 'IPS', color=IPS, fontsize=7, va='top')
    ax.text(7.2, top_a * .88, 'Vertex', color=VERTEX, fontsize=7, va='top')
    ax.text(x_hi * .98, top_a * .03, 'Solid: second-presented\nDashed: first-presented',
            fontsize=6, color='0.4', ha='right', va='bottom')
    letter(ax, 'a')

    # ============================================================ b  the cTBS contrast
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color='0.75', lw=.6, ls='--', zorder=0)
    for term, col, ls in [('n1_evidence_sd', '0.35', '--'),
                          ('n2_evidence_sd', MODEL, '-')]:
        s = curves[(curves.term == term) & (curves.stimulation == 'ips - vertex')]
        ax.fill_between(s.payoff, s.lo, s.hi, color=col, alpha=.16, lw=0)
        ax.plot(s.payoff, s.nu, color=col, ls=ls, lw=1.4)
    ax.set_ylabel('Δ ν, IPS − vertex (CHF)')
    ax.set_xlabel('Payoff magnitude (CHF)')
    ax.set_title('The increase is credible and\nroughly uniform', fontsize=7.5, color='0.15')
    logx(ax, hi=x_hi)
    d_ = curves[(curves.stimulation == 'ips - vertex') & (curves.payoff <= x_hi)]
    lo_b, hi_b = float(d_.lo.min()), float(d_.hi.max())
    pad = (hi_b - lo_b) * .18
    ax.set_ylim(lo_b - pad * .4, hi_b + pad)
    ax.text(7.2, hi_b + pad * .85, 'Second-presented', color=MODEL, fontsize=6.5,
            va='top')
    ax.text(7.2, hi_b + pad * .1, 'First-presented', color='0.35', fontsize=6.5,
            va='top')
    letter(ax, 'b')

    # ================================================================== c  LOO
    ax = fig.add_subplot(gs[0, 2])
    nice = {'flexible1_noisefix.head': 'Both options\n(full)',
            'flexible1_noisefix_first.head': 'First option\nonly',
            'flexible1_noisefix_second.head': 'Second option\nonly',
            'flexible1_noisefix_null.head': 'No cTBS effect\non noise'}
    l = loo.loc[[k for k in nice if k in loo.index]].copy()
    l['name'] = [nice[i] for i in l.index]
    l = l.sort_values('elpd_diff', ascending=False).reset_index(drop=True)
    y = np.arange(len(l))
    ax.errorbar(-l.elpd_diff, y, xerr=l.dse, fmt='o', ms=5, color=MODEL,
                elinewidth=1.2, capsize=2.2, lw=0, zorder=3)
    ax.axvline(0, color='0.75', lw=.6, ls='--', zorder=0)
    for yy, v, se in zip(y, l.elpd_diff, l.dse):
        if v > 0:
            ax.text(-v, yy + .28, f'−{v:.0f} ± {se:.0f}', fontsize=6,
                    color='0.3', ha='center', va='bottom')
    ax.set_yticks(y); ax.set_yticklabels(l.name, fontsize=6.5)
    ax.set_ylim(-.6, len(l) - .25)
    ax.set_xlabel('Δ ELPD vs. best model')
    ax.set_title('Both options need their own\ncTBS noise term', fontsize=7.5, color='0.15')
    letter(ax, 'c', x=-0.66, y=1.02)

    # ==================================================== d  percept transfer function
    ax = fig.add_subplot(gs[1, 0])
    hi, top = perc.objective_ev.max() * 1.06, 12.5
    ax.plot([0, top], [0, top], color='0.75', lw=.8, ls=(0, (4, 3)), zorder=1)
    ax.text(top * .96, top * .96, 'Veridical', fontsize=6, color='0.5',
            rotation=52, rotation_mode='anchor', ha='right', va='bottom')
    for opt, col in [('safe', SAFE), ('risky', RISKY)]:
        s = perc[perc.option == opt].sort_values('objective_ev')
        ax.plot(s.objective_ev, s.vertex, color=col, lw=1.4, marker='o', ms=3.4)
        ax.plot(s.objective_ev, s.ips, color=col, lw=1.4, ls=':', marker='o',
                ms=3.4, mfc='white')
    for nm, key, col in [('Safe prior μ', 'safe_prior_mu', SAFE),
                         ('Risky prior μ (× p)', 'risky_prior_mu', RISKY)]:
        v = priors[key] * (0.55 if key.startswith('risky') else 1.0)
        ax.axhline(v, color=col, lw=.7, ls=':', zorder=0)
        ax.text(hi * .30, v - .18, nm, fontsize=5.8, color=col, ha='left', va='top')
    ax.set_xlabel('Objective expected value (CHF)')
    ax.set_ylabel('Perceived expected value (CHF)')
    ax.set_title('Percepts collapse toward priors\nthat sit below the payoffs',
                 fontsize=7.5, color='0.15')
    ax.set_xlim(0, hi); ax.set_ylim(0, top)
    ax.text(hi * .40, 11.9, 'Safe', color=SAFE, fontsize=7, va='top')
    ax.text(hi * .40, 10.9, 'Risky', color=RISKY, fontsize=7, va='top')
    ax.text(hi * .99, 11.9, 'Solid: vertex\nDotted: IPS', fontsize=6, color='0.4',
            ha='right', va='top')
    letter(ax, 'd')

    # ============================================ e  per-option shift in perceived EV
    ax = fig.add_subplot(gs[1, 1])
    ax.axhline(0, color='0.75', lw=.6, ls='--', zorder=0)
    for opt, col in [('safe', SAFE), ('risky', RISKY)]:
        s = perc[perc.option == opt].sort_values('n_safe')
        ax.fill_between(s.n_safe, s.lo, s.hi, color=col, alpha=.16, lw=0)
        ax.plot(s.n_safe, s.delta, color=col, lw=1.4, marker='o', ms=3.4)
    ax.set_xscale('log')
    ax.set_xticks([7, 10, 14, 20, 28])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim(6.7, 29.5)
    ax.set_xlabel('Safe payoff on the trial (CHF)')
    ax.set_ylabel('Δ perceived EV, IPS − vertex (CHF)')
    ax.set_title('The safe option loses more\nthan the risky one', fontsize=7.5,
                 color='0.15')
    ax.set_ylim(-1.0, .08)
    ax.text(7.1, -.95, 'Risky-second trials', fontsize=6, color='0.4', va='bottom')
    for opt, col, nm, dy in [('risky', RISKY, 'Risky', .10), ('safe', SAFE, 'Safe', -.13)]:
        s = perc[perc.option == opt].sort_values('n_safe')
        ax.text(s.n_safe.iloc[-1] * .99, s.delta.iloc[-1] + dy, nm, color=col,
                fontsize=7, ha='right', va='bottom' if dy > 0 else 'top')
    letter(ax, 'e')

    # ======================================================== f  channel decomposition
    ax = fig.add_subplot(gs[1, 2])
    xs = {b: i for i, b in enumerate(sorted(chan.bin.unique(),
                                           key=lambda b: int(b.rstrip('%'))))}
    ax.axhline(0, color='0.75', lw=.6, ls='--', zorder=0)
    for order, ls, mk in [('Risky second', '-', 'o'), ('Risky first', '--', 's')]:
        for ch, col, lw in [('full', MODEL, 1.5), ('noise_only', '#c88b2a', 1.1)]:
            s = chan[(chan.order == order) & (chan.channel == ch)].copy()
            s['x'] = s.bin.map(xs)
            s = s.sort_values('x')
            ax.plot(s.x, s.delta, color=col, ls=ls, lw=lw, marker=mk, ms=3)
            if ch == 'full' and order == 'Risky second':
                ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.15, lw=0)
    ax.set_xticks(list(xs.values())); ax.set_xticklabels(list(xs), fontsize=6.5)
    ax.set_xlabel('Risky/safe ratio (bin)')
    ax.set_ylabel('Δ P(chose risky), IPS − vertex')
    ax.set_title('The effect is a prior-driven bias,\nnot extra randomness',
                 fontsize=7.5, color='0.15')
    ax.set_ylim(-.012, .105)
    ax.text(0.98, .97, 'Blue: full model\nOrange: randomness channel only\n'
                       'Solid: risky second   Dashed: risky first',
            transform=ax.transAxes, fontsize=5.6, color='0.35', va='top', ha='right',
            linespacing=1.5)
    letter(ax, 'f')

    # ============================================== g/h  where in the decision space
    vals = space.effect.values
    lim = np.nanmax(np.abs(vals))
    for col, name in enumerate(['Risky first', 'Risky second']):
        ax = fig.add_subplot(gs[2, col])
        o = space[space.order == name]
        piv = o.pivot(index='ratio', columns='n_safe', values='effect')
        pv = o.pivot(index='ratio', columns='n_safe', values='p_vertex')
        im = ax.pcolormesh(piv.columns.values, piv.index.values, piv.values,
                           cmap='RdBu_r', shading='gouraud', vmin=-lim, vmax=lim)
        cs = ax.contour(pv.columns.values, pv.index.values, pv.values, levels=[.5],
                        colors='k', linewidths=1.0)
        ax.clabel(cs, fmt={.5: 'Indiff.'}, fontsize=5.5, inline=True)
        fr = np.sort(ppc[ppc.order == name].frac.unique())
        ax.scatter(np.tile([7, 10, 14, 20, 28], len(fr)), np.repeat(fr, 5),
                   s=4, facecolor='none', edgecolor='0.2', lw=.45, zorder=4)
        ax.set_xticks([7, 14, 21, 28]); ax.set_yticks([1, 2, 3, 4])
        ax.set_xlabel('Safe payoff (CHF)')
        if col == 0:
            ax.set_ylabel('Risky/safe payoff ratio')
        ax.set_title(name, fontsize=7.5, color='0.15')
        if col == 1:
            cb = fig.colorbar(im, ax=ax, pad=.04, aspect=17, fraction=.055)
            cb.ax.set_title('Δ P(risky)\nIPS − vertex', fontsize=5.8, color='0.2',
                            pad=3, linespacing=1.3)
            cb.ax.tick_params(labelsize=5.5, length=1.8)
            cb.outline.set_linewidth(.5)
        letter(ax, 'g' if col == 0 else 'h')

    # ================================== i  observed effect vs probit vs cognitive model
    sub = gs[2, 2].subgridspec(2, 1, hspace=.28)
    for r, name in enumerate(['Risky first', 'Risky second']):
        ax = fig.add_subplot(sub[r])
        oo = obs[obs.order == name].copy()
        oo['x'] = oo.bin.map(xs)
        oo = oo.sort_values('x')
        cc = chan[(chan.order == name) & (chan.channel == 'full')].copy()
        cc['x'] = cc.bin.map(xs)
        cc = cc.sort_values('x')
        ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
        ax.fill_between(oo.x, oo.probit_lo, oo.probit_hi, color=PROBIT, alpha=.30,
                        lw=0, zorder=1)
        ax.plot(oo.x, oo.probit, color=PROBIT, lw=1.1, zorder=2)
        ax.fill_between(cc.x, cc.lo, cc.hi, color=MODEL, alpha=.22, lw=0, zorder=2)
        ax.plot(cc.x, cc.delta, color=MODEL, lw=1.3, zorder=3)
        ax.errorbar(oo.x, oo.delta, yerr=[oo.delta - oo.ci_lo, oo.ci_hi - oo.delta],
                    fmt='o', ms=3.2, color='0.1', lw=0, elinewidth=.8, capsize=0,
                    zorder=4)
        ax.set_ylim(-.13, .27)
        ax.set_yticks([0, .1, .2])
        ax.set_xticks(list(xs.values()))
        ax.set_title(name, fontsize=7, color='0.15', pad=2)
        if r == 1:
            ax.set_xticklabels(list(xs), fontsize=6.5)
            ax.set_xlabel('Risky/safe ratio (bin)')
        else:
            ax.set_xticklabels([])
            ax.text(.97, .93, 'Points: data', transform=ax.transAxes, fontsize=5.8,
                    color='0.1', ha='right', va='top')
            ax.text(.97, .78, 'Blue: Flexible PMC', transform=ax.transAxes,
                    fontsize=5.8, color=MODEL, ha='right', va='top')
            ax.text(.97, .63, 'Grey: probit', transform=ax.transAxes, fontsize=5.8,
                    color='0.4', ha='right', va='top')
            letter(ax, 'i', x=-0.34, y=1.24)
            ax.set_ylabel('Δ P(chose risky)', y=-0.12, labelpad=1)

    sns.despine(fig=fig, offset=2)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible1nf')
    parser.add_argument('--x_hi', default=28., type=float)
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/pmc_explained')
    args = parser.parse_args()
    main(args.data_dir, args.label, args.out, args.x_hi)
