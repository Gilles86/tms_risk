"""Figure 3A, but as a posterior predictive check of the cognitive model.

The published Fig 3A shows observed choice proportions against the *psychophysical*
(probit) model. This does the same against the fitted Flexible PMC, so you can see
whether the cognitive model reproduces the order-specific cTBS effect it is supposed
to explain.

    python -m tms_risk.behavior.scripts.plot_ppc_fig3a \\
        --model_label flexible1 --bauer_path /tmp/bauer_ecc6454
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
_bp = None
for _i, _a in enumerate(sys.argv):
    if _a == '--bauer_path':
        _bp = sys.argv[_i + 1]
sys.path.insert(0, _bp or str(REPO / 'libs' / 'bauer'))
sys.path.insert(0, str(REPO / 'tms_risk' / 'behavior'))

import arviz as az                # noqa: E402
import pymc as pm                 # noqa: E402
import matplotlib as mpl          # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
import seaborn as sns             # noqa: E402
from fit_model import get_data    # noqa: E402
from tms_risk.behavior.scripts.fit_pmc_noisefix import (SUFFIX_REGRESSORS,  # noqa: E402
                                                        build as build_model)


def rebuild(df, idata, model_label):
    """Rebuild the exact model a trace was fitted with.

    `compute_deterministics` needs a graph whose free variables match the posterior,
    so the regressor set, family, spline count and spline degree all have to agree.
    Every fit from 2026-07 on stamps those into `posterior.attrs`, which is more
    reliable than re-deriving them from the label; the regex is the fallback for
    older traces.
    """
    a = idata.posterior.attrs
    m = re.fullmatch(r'(flexible|weber)([12])(\.\d)?_noisefix(_\w+)?(\.\w+)?', model_label)
    if m is None:                                   # pre-refit labels, e.g. `flexible2`
        m2 = re.fullmatch(r'flexible([12])(\.\d)?', model_label)
        if m2 is None:
            raise SystemExit(f'cannot infer a model from label {model_label!r}')
        noise, family, order_, suffix = 'flexible', int(m2.group(1)), \
            (5 if m2.group(2) is None else int(m2.group(2)[1:])), ''
    else:
        noise = 'weber' if m.group(1) == 'weber' else 'flexible'
        family, suffix = int(m.group(2)), (m.group(4) or '')
        order_ = 5 if m.group(3) is None else int(m.group(3)[1:])

    family = int(a.get('tms_risk_family', family))
    order_ = int(a.get('tms_risk_spline_order', order_)) or order_
    degree = int(a.get('tms_risk_spline_degree', 3))
    if 'tms_risk_noise' in a:
        noise = a['tms_risk_noise']
    if 'tms_risk_regressors' in a:
        regs = [r for r in a['tms_risk_regressors'].split(',') if r]
    else:
        regs = SUFFIX_REGRESSORS[family][suffix]
    print(f'rebuilding {model_label}: noise={noise} family={family} '
          f'splines={order_} degree={degree} regressors={regs or "none"}')
    return build_model(df, regs, spline_order=order_, family=family,
                       spline_degree=degree, noise=noise)

VERTEX, IPS = '#2ca02c', '#d62728'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def main(bids_folder, model_label, out_stem, n_draws, trace_dir=None, tag=None):
    df = get_data(bids_folder)
    tdir = Path(trace_dir) if trace_dir else Path(bids_folder) / 'derivatives' / 'cogmodels'
    idata = az.from_netcdf(tdir / f'model-{model_label}_trace.netcdf')
    model = rebuild(df.copy(), idata, model_label)
    model_label = tag or model_label

    model.build_estimation_model(save_p_choice=True)
    keep = np.linspace(0, idata.posterior.sizes['draw'] - 1,
                       max(1, n_draws // idata.posterior.sizes['chain'])).astype(int)
    det = pm.compute_deterministics(idata.posterior.isel(draw=keep),
                                    model=model.estimation_model, var_names=['p'],
                                    merge_dataset=False, progressbar=False)
    p2 = det['p'].stack(sample=('chain', 'draw')).values      # P(choose option 2)
    good = np.isfinite(p2).all(0)
    p2 = p2[:, good]
    print(f'{p2.shape[1]} usable draws')

    d = df.reset_index().copy()
    d['bin'] = d['bin(risky/safe)'].astype(str)
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['stim'] = d['stimulation_condition']
    # option 2 is the risky one exactly when the risky option came second
    p_risky = np.where((~d['risky_first']).values[:, None], p2, 1 - p2)

    # observed: within-subject means, then across subjects
    obs = (d.assign(y=d['chose_risky'].astype(float))
             .groupby(['subject', 'order', 'bin', 'stim'])['y'].mean()
             .groupby(['order', 'bin', 'stim']).agg(['mean', 'sem']).reset_index())
    xpos = d.groupby(['order', 'bin'])['frac'].mean().rename('frac')
    obs = obs.join(xpos, on=['order', 'bin'])

    # model: same aggregation, per draw, so the band is a real predictive interval
    keys = ['subject', 'order', 'bin', 'stim']
    idx = pd.MultiIndex.from_frame(d[keys])
    per_draw = (pd.DataFrame(p_risky, index=idx)
                  .groupby(level=keys).mean()
                  .groupby(['order', 'bin', 'stim']).mean())
    mod = pd.DataFrame({'mean': per_draw.mean(1),
                        'lo': per_draw.quantile(.025, axis=1),
                        'hi': per_draw.quantile(.975, axis=1)}).reset_index()
    mod = mod.join(xpos, on=['order', 'bin'])

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, nm in zip(axes, ['Risky first', 'Risky second']):
        for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
            mm = mod[(mod.order == nm) & (mod.stim == stim)].sort_values('frac')
            ax.fill_between(mm.frac, mm.lo, mm.hi, color=col, alpha=.20, lw=0, zorder=1)
            ax.plot(mm.frac, mm['mean'], color=col, lw=1.3, zorder=2)
            oo = obs[(obs.order == nm) & (obs.stim == stim)].sort_values('frac')
            ax.errorbar(oo.frac, oo['mean'], yerr=oo['sem'], fmt='o', color=col,
                        ms=4.5, lw=0, elinewidth=1.0, capsize=0, zorder=4)
        ax.axhline(.5, color='0.8', lw=.6, ls='--', zorder=0)
        ax.set_ylim(.15, .95)
        ax.set_yticks([.2, .4, .6, .8])
        ax.set_ylabel('P(chose risky)')
        ax.text(.03, .95, nm, transform=ax.transAxes, fontsize=8.5,
                color='0.2', va='top')
    axes[1].set_xlabel('Risky/safe payoff ratio')
    axes[1].set_xticks([1.5, 2.0, 2.5, 3.0])
    axes[0].text(3.28, .29, 'IPS', color=IPS, fontsize=8, ha='right')
    axes[0].text(3.28, .21, 'Vertex', color=VERTEX, fontsize=8, ha='right')
    axes[1].annotate('cTBS effect only here', xy=(1.85, .45), xytext=(2.15, .27),
                     fontsize=7.5, color='0.3', ha='left', va='center',
                     arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3',
                                     color='0.4', lw=.6))
    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf')

    out = mod.merge(obs.rename(columns={'mean': 'observed', 'sem': 'observed_sem'}),
                    on=['order', 'bin', 'stim', 'frac'])
    out.to_csv(Path(out_stem).parent.parent / 'data' /
               f'ppc_fig3a.{model_label}.tsv', sep='\t', index=False)

    # The same check broken down by safe payoff as well. The ratio bins are formed
    # WITHIN each safe payoff, because the ratios available differ across them -- a
    # shared binning would leave cells empty at the extremes. Four bins keeps ~100
    # trials and ~28 subjects per cell.
    d['sbin'] = (d.groupby('n_safe')['frac']
                 .transform(lambda v: pd.qcut(v, 4, labels=False, duplicates='drop')))
    keys_s = ['subject', 'order', 'n_safe', 'sbin', 'stim']
    grp_s = ['order', 'n_safe', 'sbin', 'stim']
    obs_s = (d.assign(y=d['chose_risky'].astype(float))
               .groupby(keys_s)['y'].mean()
               .groupby(grp_s).agg(['mean', 'sem']).reset_index())
    xpos_s = d.groupby(['order', 'n_safe', 'sbin'])['frac'].mean().rename('frac')
    obs_s = obs_s.join(xpos_s, on=['order', 'n_safe', 'sbin'])
    idx_s = pd.MultiIndex.from_frame(d[keys_s])
    per_draw_s = (pd.DataFrame(p_risky, index=idx_s)
                    .groupby(level=keys_s).mean()
                    .groupby(grp_s).mean())
    mod_s = pd.DataFrame({'mean': per_draw_s.mean(1),
                          'lo': per_draw_s.quantile(.025, axis=1),
                          'hi': per_draw_s.quantile(.975, axis=1)}).reset_index()
    mod_s = mod_s.join(xpos_s, on=['order', 'n_safe', 'sbin'])
    out_s = mod_s.merge(
        obs_s.rename(columns={'mean': 'observed', 'sem': 'observed_sem'}),
        on=['order', 'n_safe', 'sbin', 'stim', 'frac'])
    out_s.to_csv(Path(out_stem).parent.parent / 'data' /
                 f'ppc_by_safe.{model_label}.tsv', sep='\t', index=False)
    print(f'wrote ppc_by_safe.{model_label}.tsv  ({len(out_s)} cells)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--model_label', default='flexible1')
    parser.add_argument('--bauer_path', default=None)
    parser.add_argument('--n_draws', default=200, type=int)
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/ppc_fig3a')
    parser.add_argument('--trace_dir', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()
    main(args.bids_folder, args.model_label, args.out, args.n_draws,
         args.trace_dir, args.tag)
