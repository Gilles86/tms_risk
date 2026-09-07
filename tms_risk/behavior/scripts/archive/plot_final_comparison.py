"""Where the log-space PMC model set stands: fit, convergence, and cTBS effect.

Three questions, three panels, because they turn out to have different answers:

  a  Which models fit?          ELPD relative to the best, with paired dSE.
  b  Which of those converged?  r-hat against ELPD -- several of the best-fitting
                                cells never sampled, so the ladder alone misleads.
  c  What does cTBS do?         The region-integrated noise effect (7-28 CHF and
                                28-112 CHF), computed per draw, per channel.

Reads notes/data/ploo/*.npz (pointwise ELPD) and notes/data/delta/regions.*.tsv.

    python -m tms_risk.behavior.scripts.plot_final_comparison
"""
import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

# Noise form -> colour. Blue/orange/green/grey for model contrasts; red and green
# stay reserved for IPS/vertex elsewhere, so this palette avoids them.
FORM = {'bs3': ('5-df spline', '#3B5BA5'), 'p2': ('Affine in log', '#D8801F'),
        'pl': ('Power law', '#7B4FA8'), 'gw': ('Gen. Weber', '#5D8C3F'),
        'w': ('Weber (constant)', '#9C9C9C')}
N = 8335


def parse(label):
    m = re.fullmatch(r'lfx2-(bs3|bs2|cr3|gw|pl)-(fm|sm|m2|m3|w|sd\d)-(dp|tp)-'
                     r'(null|b|bm|t|m)(-p[1-5])?(-i)?(-fx)?(-\w+)?', label)
    if not m:
        return None
    basis, mem, hp, tms, pdf, ind, fx, extra = m.groups()
    form = basis if basis in ('gw', 'pl') else ('w' if mem == 'w' else
                                                ('p2' if pdf else 'bs3'))
    place = {'null': 'no cTBS', 'bm': 'both', 'b': '2nd option' if ind else 'perceptual',
             'm': '1st option' if ind else 'memory', 't': 'total'}[tms]
    name = f"{FORM[form][0]} · {place}"
    if ind:
        name += ' · n1/n2'
    if fx:
        name += ' · fixed slope'
    return dict(label=label, form=form, place=place, name=name,
                null=(tms == 'null'), fx=bool(fx), color=FORM[form][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ploo_dir', default='notes/data/ploo')
    ap.add_argument('--delta_dir', default='notes/data/delta')
    ap.add_argument('--out', default='notes/figures/final_comparison')
    a = ap.parse_args()

    E, M = {}, {}
    for p in glob.glob(f'{a.ploo_dir}/*.npz'):
        z = np.load(p); m = json.loads(str(z['meta']))
        E[m['label']], M[m['label']] = z['elpd_i'], m
    rows = [dict(**parse(l), elpd=E[l].sum(), rhat=M[l]['max_rhat'],
                 p_loo=M[l]['p_loo'], div=M[l]['divergences'])
            for l in E if parse(l) and ('p2' in l or l in (
                'lfx2-bs3-m2-dp-bm', 'lfx2-bs3-m2-dp-null',
                'lfx2-bs3-w-dp-bm', 'lfx2-bs3-w-dp-null'))]
    t = pd.DataFrame(rows).sort_values('elpd', ascending=False).reset_index(drop=True)
    best = t.label.iloc[0]

    def dse(lab):
        d = E[lab] - E[best]
        return np.sqrt(N * np.var(d, ddof=1)) if lab != best else 0.0
    t['d'] = t.elpd - t.elpd.max()
    t['dse'] = [dse(l) for l in t.label]
    t['ok'] = t.rhat <= 1.01

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.5), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.5, 1, 1.1]))

    # -- a: the ladder ------------------------------------------------------
    ax = axes[0]
    y = np.arange(len(t))[::-1]
    for yi, r in zip(y, t.itertuples()):
        ax.errorbar(r.d, yi, xerr=r.dse, color=r.color if r.ok else '.75',
                    lw=.9, capsize=0, zorder=2)
        ax.plot(r.d, yi, 'o', ms=4.6, mfc=r.color if r.ok else 'white',
                mec=r.color if r.ok else '.65', mew=1.1, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(t.name, fontsize=5.9)
    ax.axvline(0, color='.75', lw=.7, ls='--', zorder=0)
    ax.set_xlabel('ΔELPD vs best (nats)')
    ax.set_title('Fit', fontsize=8)
    ax.text(.02, .02, 'Open marker: r̂ > 1.01', transform=ax.transAxes,
            fontsize=5.9, color='.45')

    # -- b: convergence vs fit ---------------------------------------------
    ax = axes[1]
    for r in t.itertuples():
        ax.plot(r.rhat, r.elpd, 'o', ms=5.5, mfc=r.color if r.ok else 'white',
                mec=r.color, mew=1.1)
    ax.axvline(1.01, color='#b0453b', lw=.9, ls='--')
    ax.text(1.012, t.elpd.min(), 'r̂ = 1.01', fontsize=6.2, color='#b0453b',
            rotation=90, va='bottom')
    ax.set_xlabel('Max r̂ (group parameters)'); ax.set_ylabel('ELPD (nats)')
    ax.set_title('Convergence', fontsize=8)
    for form, (nm, col) in FORM.items():
        if (t.form == form).any():
            ax.plot([], [], 'o', ms=4.5, color=col, label=nm)
    ax.legend(loc='lower right', fontsize=5.9, handletextpad=.3)

    # -- c: the cTBS effect, with intervals ---------------------------------
    ax = axes[2]
    reg = []
    for f in glob.glob(f'{a.delta_dir}/regions.*.tsv'):
        lab = re.sub(r'.*regions\.|\.tsv', '', f)
        info = parse(lab)
        if not info or info['null']:
            continue
        d = pd.read_csv(f, sep='\t')
        d = d[d.curve.isin(['memory', 'perceptual', 'n1 (first)', 'n2 (second)'])]
        d = d[d['mean'].abs() > 1e-9]
        for _, r in d.iterrows():
            reg.append(dict(name=f"{info['name']}\n{r.curve}", region=r.region,
                            mean=r['mean'], lo=r.lo, hi=r.hi, color=info['color']))
    rg = pd.DataFrame(reg)
    if len(rg):
        rg = rg[rg.region == '7-28 CHF'].drop_duplicates('name')
        yy = np.arange(len(rg))[::-1]
        ax.axvline(0, color='.75', lw=.7, ls='--', zorder=0)
        for yi, r in zip(yy, rg.itertuples()):
            ax.hlines(yi, r.lo, r.hi, color=r.color, lw=1.2)
            ax.plot(r.mean, yi, 'o', ms=4.4, color=r.color)
        ax.set_yticks(yy); ax.set_yticklabels(rg.name, fontsize=5.6)
    ax.set_xlabel('Δ noise SD, IPS − vertex\n(mean over 7–28 CHF)')
    ax.set_title('cTBS effect at low payoffs', fontsize=8)

    sns.despine(fig=fig, offset=3)
    for ax_, letter in zip(axes, 'abc'):
        ax_.text(-0.10, 1.06, letter, transform=ax_.transAxes, fontsize=8,
                 family='Arial', fontweight='bold', va='bottom', ha='left')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(t[['name', 'label', 'elpd', 'd', 'dse', 'rhat', 'p_loo', 'div']]
          .to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print(f'\nwrote {a.out}.pdf')


if __name__ == '__main__':
    main()
