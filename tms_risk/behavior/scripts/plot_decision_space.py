"""Where in the decision space do the cTBS-induced distortions actually matter?

Figure 5B shows the perceptual distortion over the (safe payoff x risky/safe ratio)
space, but a distortion only changes behaviour where the psychometric function is
steep. This evaluates the fitted Flexible PMC on a grid over that space and separates
the two factors:

    cause      the shift in the perceived risky/safe EV ratio, IPS / vertex
    leverage   how much a shift in the decision variable moves P(risky) --
               |dP/dm| at vertex, which peaks along the indifference contour
    effect     the resulting Delta P(risky), i.e. cause x leverage

The indifference contour (where vertex P(risky) = 0.5) is drawn on all three, and the
30 cells the design actually sampled are marked, because only those carry data.

    python -m tms_risk.behavior.scripts.plot_decision_space \\
        --bauer_path /tmp/bauer_ecc6454 --model_label flexible2
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss

REPO = Path(__file__).resolve().parents[3]
_bp = None
for _i, _a in enumerate(sys.argv):
    if _a == '--bauer_path':
        _bp = sys.argv[_i + 1]
sys.path.insert(0, _bp or str(REPO / 'libs' / 'bauer'))
sys.path.insert(0, str(REPO / 'tms_risk' / 'behavior'))

import arviz as az            # noqa: E402
import pymc as pm             # noqa: E402
import matplotlib as mpl      # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
import seaborn as sns         # noqa: E402
from fit_model import get_data    # noqa: E402
from tms_risk.behavior.scripts.decompose_pmc_channels import (  # noqa: E402
    build_flexible, tap_get_diff_dist, tap_get_posterior)

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def grid_paradigm(df, safes, ratios, subjects):
    """A synthetic paradigm covering the decision space, for every subject, order and
    stimulation condition. The payoff range is kept identical to the real data so the
    spline knots (anchored to min/max of n1 and n2) are unchanged."""
    template = df.iloc[0]
    rows = []
    trial = 0
    for subject in subjects:
        for rf in [True, False]:
            for stim in ['vertex', 'ips']:
                for s in safes:
                    for r in ratios:
                        nr = s * r
                        rows.append({
                            'subject': subject, 'run': 1, 'trial_nr': trial,
                            'n1': nr if rf else s, 'n2': s if rf else nr,
                            'p1': 0.55 if rf else 1.0, 'p2': 1.0 if rf else 0.55,
                            'n_safe': s, 'n_risky': nr, 'frac': r,
                            'risky_first': rf, 'stimulation_condition': stim,
                            'choice': True, 'chose_risky': True,
                            'log(risky/safe)': np.log(r),
                        })
                        trial += 1
    g = pd.DataFrame(rows)
    for col in df.columns:
        if col not in g.columns:
            g[col] = template[col]
    return g.set_index(['subject', 'run', 'trial_nr'])


def main(bids_folder, model_label, bauer_path, out_stem, n_draws, n_grid, data_dir,
         trace_dir=None, tag=None):
    df = get_data(bids_folder)
    tdir = Path(trace_dir) if trace_dir else Path(bids_folder) / 'derivatives' / 'cogmodels'
    idata = az.from_netcdf(tdir / f'model-{model_label}_trace.netcdf')
    m_ = re.fullmatch(r'flexible([12])(\.\d)?(_noisefix)?(\.\w+)?', model_label)
    if not m_:
        raise SystemExit(f'unsupported label {model_label!r}')
    family = int(m_.group(1))
    order = 5 if m_.group(2) is None else int(m_.group(2)[1:])
    model_label = tag or model_label
    print(f'family {family}, spline order {order}')

    safes = np.linspace(7, 28, n_grid)
    ratios = np.linspace(1.0, 4.0, n_grid)
    subjects = np.asarray(idata.posterior.coords['subject'].values)
    par = grid_paradigm(df, safes, ratios, subjects)
    print(f'grid: {n_grid}x{n_grid} cells, {len(subjects)} subjects, {len(par)} rows')
    print(f'payoff range in grid [{par[["n1","n2"]].min().min():.1f}, '
          f'{par[["n1","n2"]].max().max():.1f}] vs real '
          f'[{df[["n1","n2"]].min().min():.1f}, {df[["n1","n2"]].max().max():.1f}]')

    tap_get_diff_dist()
    tap_get_posterior()
    model = build_flexible(par.copy(), spline_order=order, family=family)
    model.build_estimation_model()
    n_draw = idata.posterior.sizes['draw']
    keep = np.linspace(0, n_draw - 1, max(1, n_draws // idata.posterior.sizes['chain'])).astype(int)
    det = pm.compute_deterministics(idata.posterior.isel(draw=keep),
                                    model=model.estimation_model,
                                    var_names=['diff_mu', 'diff_sd',
                                               'post_mu_1', 'post_mu_2'],
                                    merge_dataset=False, progressbar=False)

    m = -det['diff_mu'].stack(sample=('chain', 'draw')).values
    s = det['diff_sd'].stack(sample=('chain', 'draw')).values
    mu1 = det['post_mu_1'].stack(sample=('chain', 'draw')).values
    mu2 = det['post_mu_2'].stack(sample=('chain', 'draw')).values
    good = np.isfinite(m).all(0) & np.isfinite(s).all(0)
    m, s, mu1, mu2 = m[:, good], s[:, good], mu1[:, good], mu2[:, good]
    print(f'{good.sum()} usable draws')

    p = par.reset_index()
    ev1 = mu1 * p['p1'].values[:, None]
    ev2 = mu2 * p['p2'].values[:, None]
    rf = p['risky_first'].values[:, None]
    ev_r = np.where(rf, ev1, ev2)
    ev_s = np.where(rf, ev2, ev1)
    ips = (p['stimulation_condition'] == 'ips').values

    def cell_mean(vals, sel):
        """Average over subjects and draws, reshape to (ratio, safe)."""
        t = p.loc[sel, ['n_safe', 'frac']].copy()
        t['v'] = vals[sel].mean(1)
        piv = t.groupby(['frac', 'n_safe'])['v'].mean().unstack('n_safe')
        return piv.values, piv.columns.values, piv.index.values

    out = {}
    for rf_val, name in [(True, 'Risky first'), (False, 'Risky second')]:
        base = (p['risky_first'] == rf_val).values
        sv, si = base & ~ips, base & ips
        # `m` is EV2 - EV1, so norm.cdf(m/s) is P(choose the SECOND option). The
        # risky option is second only when risky_first is False, so on risky-first
        # trials that probability belongs to the safe option and has to be flipped
        # before it can be called P(chose risky). Without this, p_vertex and effect
        # carry the wrong sign on half the design.
        pv = ss.norm.cdf(m / s)
        pv = np.where(p['risky_first'].values[:, None], 1.0 - pv, pv)
        p_v, _, _ = cell_mean(pv, sv)
        p_i, xs, ys = cell_mean(pv, si)
        # cause: perceived risky/safe EV ratio, IPS relative to vertex
        rat = ev_r / ev_s
        r_v, _, _ = cell_mean(rat, sv)
        r_i, _, _ = cell_mean(rat, si)
        # leverage: |dP/dm| at vertex
        lev, _, _ = cell_mean(ss.norm.pdf(m / s) / s, sv)
        # per-condition absolutes, for the preprint's Fig-5 style panels
        n_v, _, _ = cell_mean(s, sv)
        n_i, _, _ = cell_mean(s, si)
        evr_v, _, _ = cell_mean(ev_r, sv)
        evr_i, _, _ = cell_mean(ev_r, si)
        evs_v, _, _ = cell_mean(ev_s, sv)
        evs_i, _, _ = cell_mean(ev_s, si)
        out[name] = dict(x=xs, y=ys, p_vertex=p_v, cause=r_i / r_v,
                         leverage=lev, effect=p_i - p_v,
                         noise_vertex=n_v, noise_ips=n_i,
                         ratio_vertex=r_v, ratio_ips=r_i,
                         ev_risky_vertex=evr_v, ev_risky_ips=evr_i,
                         ev_safe_vertex=evs_v, ev_safe_ips=evs_i)

    # -------------------------------------------------------------------- plot
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.9), constrained_layout=True,
                            sharex=True, sharey=True)
    bins = df[~df.risky_first].groupby('bin(risky/safe)', observed=True)['frac'].mean()
    cells_x = np.tile([7, 10, 14, 20, 28], len(bins))
    cells_y = np.repeat(bins.values, 5)

    specs = [('cause', 'Perceived risky/safe ratio\nIPS / vertex', 'RdBu_r', None),
             ('leverage', 'Leverage\n|dP/dm| at vertex', 'mako', None),
             ('effect', 'Δ P(chose risky)\nIPS − vertex', 'RdBu_r', None)]
    for row, name in enumerate(['Risky first', 'Risky second']):
        o = out[name]
        for col, (key, title, cmap, _) in enumerate(specs):
            ax = axes[row, col]
            z = o[key]
            if cmap == 'RdBu_r':
                c = np.nanmax(np.abs(z - (1 if key == 'cause' else 0)))
                center = 1 if key == 'cause' else 0
                im = ax.pcolormesh(o['x'], o['y'], z, cmap=cmap, shading='gouraud',
                                   vmin=center - c, vmax=center + c)
            else:
                im = ax.pcolormesh(o['x'], o['y'], z, cmap=cmap, shading='gouraud')
            cs = ax.contour(o['x'], o['y'], o['p_vertex'], levels=[0.5],
                            colors='k', linewidths=1.1, linestyles='-')
            ax.clabel(cs, fmt={0.5: 'Indifference'}, fontsize=6, inline=True)
            ax.scatter(cells_x, cells_y, s=5, facecolor='none', edgecolor='0.25',
                       linewidth=0.5, zorder=4)
            fig.colorbar(im, ax=ax, pad=0.02, aspect=18)
            if row == 0:
                ax.set_title(title, fontsize=8, color='0.2')
            if col == 0:
                ax.set_ylabel(f'{name}\nRisky/safe ratio')
            if row == 1:
                ax.set_xlabel('Safe payoff (CHF)')
            ax.set_xticks([7, 14, 21, 28]); ax.set_yticks([1, 2, 3, 4])

    axes[1, 2].annotate('Design cells sit\non the ridge', xy=(12, 1.9),
                        xytext=(15.5, 3.4), fontsize=6.5, color='0.15',
                        ha='left', va='center',
                        arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.25',
                                        color='0.2', lw=0.6))
    sns.despine(fig=fig, offset=2)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')

    rows = []
    for name, o in out.items():
        for i, yy in enumerate(o['y']):
            for j, xx in enumerate(o['x']):
                rows.append({'order': name, 'n_safe': xx, 'ratio': yy,
                             **{k: o[k][i, j] for k in
                                ['p_vertex', 'cause', 'leverage', 'effect',
                                 'noise_vertex', 'noise_ips',
                                 'ratio_vertex', 'ratio_ips',
                                 'ev_risky_vertex', 'ev_risky_ips',
                                 'ev_safe_vertex', 'ev_safe_ips']}})
    tsv = Path(data_dir) / f'decision_space.{model_label}.tsv'
    tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tsv, sep='\t', index=False)
    print(f'wrote {tsv}')

    # ------------------------------------------------- combined figure with curves
    curves_figure(df, idata, model_label, order, out_stem, keep, out, family, data_dir)


def curves_figure(df, idata, model_label, order, out_stem, keep, maps, family, data_dir):
    """Top: where the distortion matters (risky-second). Bottom: the psychometric
    function at each safe payoff, model against data."""
    VERTEX, IPS = '#2ca02c', '#d62728'
    safes = [7., 10., 14., 20., 28.]
    ratios = np.linspace(1.15, 3.6, 26)
    subjects = np.asarray(idata.posterior.coords['subject'].values)
    par = grid_paradigm(df, safes, ratios, subjects)
    model = build_flexible(par.copy(), spline_order=order, family=family)
    model.build_estimation_model()
    det = pm.compute_deterministics(idata.posterior.isel(draw=keep),
                                    model=model.estimation_model,
                                    var_names=['diff_mu', 'diff_sd'],
                                    merge_dataset=False, progressbar=False)
    m = -det['diff_mu'].stack(sample=('chain', 'draw')).values
    sd = det['diff_sd'].stack(sample=('chain', 'draw')).values
    ok = np.isfinite(m).all(0) & np.isfinite(sd).all(0)
    pr = ss.norm.cdf(m[:, ok] / sd[:, ok])

    p = par.reset_index()
    sel2 = (~p['risky_first']).values
    mod = p.loc[sel2, ['n_safe', 'frac', 'stimulation_condition']].copy()
    pv = pr[sel2]
    mod['mean'] = pv.mean(1)
    mod['lo'] = np.quantile(pv, .025, axis=1)
    mod['hi'] = np.quantile(pv, .975, axis=1)
    mod = mod.groupby(['stimulation_condition', 'n_safe', 'frac']).mean().reset_index()

    # observed, within subject
    b = df[~df.risky_first].reset_index()
    b['bin'] = b['bin(risky/safe)'].astype(str)
    g = (b.groupby(['subject', 'n_safe', 'bin', 'stimulation_condition'])['chose_risky']
          .mean().groupby(['n_safe', 'bin', 'stimulation_condition'])
          .agg(['mean', 'sem']).reset_index())
    xpos = b.groupby(['n_safe', 'bin'])['frac'].mean().rename('frac')
    g = g.join(xpos, on=['n_safe', 'bin'])

    fig = plt.figure(figsize=(7.25, 5.0))
    gs = fig.add_gridspec(2, 15, hspace=.55, wspace=1.9,
                          left=.07, right=.97, top=.93, bottom=.09)
    o = maps['Risky second']
    specs = [('cause', 'Perceived risky/safe ratio\nIPS / vertex', 'RdBu_r'),
             ('leverage', 'Leverage |dP/dm|\nat vertex', 'mako'),
             ('effect', 'Δ P(chose risky)\nIPS − vertex', 'RdBu_r')]
    for col, (key, title, cmap) in enumerate(specs):
        ax = fig.add_subplot(gs[0, col * 5:(col + 1) * 5])
        z = o[key]
        if cmap == 'RdBu_r':
            centre = 1 if key == 'cause' else 0
            c = np.nanmax(np.abs(z - centre))
            im = ax.pcolormesh(o['x'], o['y'], z, cmap=cmap, shading='gouraud',
                               vmin=centre - c, vmax=centre + c)
        else:
            im = ax.pcolormesh(o['x'], o['y'], z, cmap=cmap, shading='gouraud')
        cs = ax.contour(o['x'], o['y'], o['p_vertex'], levels=[.5], colors='k',
                        linewidths=1.1)
        ax.clabel(cs, fmt={.5: 'Indiff.'}, fontsize=5.5, inline=True)
        for sv in safes:
            ax.axvline(sv, color='0.25', lw=.4, alpha=.5, zorder=3)
        fig.colorbar(im, ax=ax, pad=.03, aspect=14)
        ax.set_title(title, fontsize=7.5, color='0.2')
        ax.set_xticks([7, 14, 21, 28]); ax.set_yticks([1, 2, 3, 4])
        ax.set_xlabel('Safe payoff (CHF)', fontsize=7.5)
        if col == 0:
            ax.set_ylabel('Risky/safe ratio', fontsize=7.5)
        else:
            ax.set_yticklabels([])
        if col == 0:
            ax.text(-.34, 1.10, 'a', transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='bottom', ha='right')

    for i, sv in enumerate(safes):
        ax = fig.add_subplot(gs[1, i * 3:(i + 1) * 3])
        for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
            mm = mod[(mod.stimulation_condition == stim) & (mod.n_safe == sv)].sort_values('frac')
            ax.fill_between(mm.frac, mm.lo, mm.hi, color=colr, alpha=.15, lw=0)
            ax.plot(mm.frac, mm['mean'], color=colr, lw=1.3)
            gg = g[(g.stimulation_condition == stim) & (g.n_safe == sv)]
            ax.errorbar(gg.frac, gg['mean'], yerr=gg['sem'], fmt='o', color=colr,
                        ms=3.2, lw=0, elinewidth=.8, capsize=0, zorder=3)
        ax.axhline(.5, color='0.8', lw=.6, ls='--', zorder=0)
        ax.set_ylim(.05, 1.0); ax.set_xlim(1.1, 3.65)
        ax.set_xticks([1.5, 2.5, 3.5])
        ax.set_title(f'Safe = {sv:.0f} CHF', fontsize=7.5, color='0.2')
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('P(chose risky)', fontsize=7.5)
            ax.text(-.42, 1.12, 'b', transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='bottom', ha='right')
        if i == 2:
            ax.set_xlabel('Risky/safe payoff ratio', fontsize=7.5)
        if i == 4:
            ax.text(3.55, .18, 'IPS', color=IPS, fontsize=7, ha='right')
            ax.text(3.55, .08, 'Vertex', color=VERTEX, fontsize=7, ha='right')
    sns.despine(fig=fig, offset=2)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}_curves.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}_curves.pdf')

    # source data, so the figure can be rebuilt locally without the 1 GB trace
    dd = Path(data_dir)
    mod.assign(source='model').to_csv(
        dd / f'psychometric_model.{model_label}.tsv', sep='\t', index=False)
    g.assign(source='observed').to_csv(
        dd / f'psychometric_observed.{model_label}.tsv', sep='\t', index=False)
    print(f'wrote {dd}/psychometric_{{model,observed}}.{model_label}.tsv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--model_label', default='flexible2')
    parser.add_argument('--bauer_path', default=None)
    parser.add_argument('--n_draws', default=80, type=int)
    parser.add_argument('--n_grid', default=16, type=int)
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/decision_space')
    parser.add_argument('--trace_dir', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()
    main(args.bids_folder, args.model_label, args.bauer_path, args.out,
         args.n_draws, args.n_grid, args.data_dir, args.trace_dir, args.tag)
