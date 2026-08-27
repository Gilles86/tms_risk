"""Model-comparison overview: what the 46-trace ladder actually shows.

One story, four panels: cTBS raised representational noise in every model family
(b), the prior is load-bearing in all of them (d), and the natural-space model's
apparently competitive fit was an artefact of chains that never converged --
fit it in coordinates where it samples and the log-space model wins outright (a, c).

Reads only `notes/data/ploo/*.npz` (per-trace pointwise ELPD, 67 KB each), so it
needs no trace, no bauer and no GPU. Paired dSE throughout: the SE of the
trial-by-trial ELPD difference, not of the two ELPDs separately -- the models
share their between-subject variance, so the unpaired `se` (~46 nats here)
understates every comparison roughly fivefold.

    python -m tms_risk.behavior.scripts.plot_model_ladder
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

# Blue/orange for model contrasts, per the repo's colour convention -- red/green is
# reserved for stimulation (IPS/vertex) and must not appear on a model figure.
C_LOG = '#3B5BA5'      # log-space (ratio-scale observer)
C_NAT = '#D8801F'      # natural-space (linear-scale observer)
C_POW = '#7F7F7F'      # power-law
C_BAD = '#B9B9B9'      # failed the convergence gate

N_OBS = 8335

# label -> (display name, family colour). Curated: one row per question a reviewer
# would ask, not one row per fit.
LADDER = [
    ('lfx2-bs3-m2-dp-bm',                 'Log-Flexible PMC (primary)',        C_LOG),
    ('lfx2-bs2-m2-dp-b',                  'Log-Flexible, perceptual only',     C_LOG),
    ('lfx2-bs2-m3-dp-bm',                 'Log-Flexible, alternative basis',   C_LOG),
    ('flexible1_noisefix.head',           'Flexible PMC, natural space',       C_NAT),
    ('flexible2_noisefix.head',           'Flexible PMC, natural (bimodal)',   C_NAT),
    ('lfx2-bs3-w-dp-bm',                  'Log-Weber PMC',                     C_LOG),
    ('power2',                            'Power-law PMC',                     C_POW),
    ('lfx2-bs3-m2-dp-null',               'Log-Flexible, no cTBS effect',      C_LOG),
    ('flexible1_noisefix_null.head',      'Natural space, no cTBS effect',     C_NAT),
    ('lfx2-bs3-w-dp-null',                'Log-Weber, no cTBS effect',         C_LOG),
]

# (display name, TMS model, matched null, colour)
EXISTENCE = [
    ('Log-Flexible',        'lfx2-bs3-m2-dp-bm',            'lfx2-bs3-m2-dp-null',           C_LOG),
    ('Log-Weber',           'lfx2-bs3-w-dp-bm',             'lfx2-bs3-w-dp-null',            C_LOG),
    ('Natural space',       'flexible1_noisefix.head',      'flexible1_noisefix_null.head',  C_NAT),
    ('Power-law',           'power2',                       'power2_null',                   C_POW),
]

# (display name, model with prior, matched model without, colour)
PRIOR = [
    ('Log-space',    'lfx2-bs3-m2-dp-bm', 'lfx2-bs3-m3-dp-bm-op', C_LOG),
    ('Power-law 2',  'power2',            'power2_flat',          C_POW),
    ('Power-law 1',  'power1',            'power1_flat',          C_POW),
]

# Panel c: the convergence trap. (label, colour, annotate?)
TRAP = [
    ('lfx2-bs3-m2-dp-bm',            C_LOG),
    ('lfx2-bs2-m2-dp-b',             C_LOG),
    ('lfx2-bs2-m3-dp-bm',            C_LOG),
    ('lfx2-bs3-w-dp-bm',             C_LOG),
    ('flexible1_noisefix.head',      C_NAT),
    ('flexible1_noisefix_first.head', C_NAT),
    ('flexible1_noisefix_second.head', C_NAT),
    ('flexible2_noisefix.head',      C_NAT),
    ('flexible2_noisefix_perception.head', C_NAT),
    ('flexible2_noisefix_memory.head', C_NAT),
    ('flexible2.6_noisefix.head',    C_NAT),
]


def load(ploo_dir):
    elpd, meta = {}, {}
    for p in sorted(glob.glob(str(Path(ploo_dir) / '*.npz'))):
        z = np.load(p)
        m = json.loads(str(z['meta']))
        elpd[m['label']] = z['elpd_i']
        meta[m['label']] = m
    if not elpd:
        raise SystemExit(f'no .npz files in {ploo_dir}')
    hashes = {m['obs_hash'] for m in meta.values()}
    if len(hashes) > 1:
        raise SystemExit('traces score different observations -- comparison is void')
    return elpd, meta


def contrast(elpd, a, b):
    """Paired ELPD difference and its dSE."""
    d = elpd[a] - elpd[b]
    s = d.sum()
    dse = np.sqrt(len(d) * np.var(d, ddof=1))
    return s, dse


def panel_ladder(ax, elpd, meta, ref):
    rows = [(lab, name, col) for lab, name, col in LADDER if lab in elpd]
    vals = [contrast(elpd, lab, ref) for lab, _, _ in rows]
    order = np.argsort([v[0] for v in vals])
    y = np.arange(len(rows))

    for yi, idx in zip(y, order):
        lab, name, col = rows[idx]
        d, dse = vals[idx]
        ok = meta[lab]['max_rhat'] <= 1.01 and meta[lab]['min_ess'] >= 400
        face = col if ok else 'white'
        ax.errorbar(d, yi, xerr=dse, color=col if ok else C_BAD, lw=0.9,
                    zorder=2, capsize=0)
        ax.plot(d, yi, 'o', ms=5, mfc=face, mec=col if ok else C_BAD,
                mew=1.1, zorder=3)
        ax.text(4, yi, name, va='center', ha='left', fontsize=6.5,
                color='0.15' if ok else '0.45')

    ax.axvline(0, color='0.7', lw=0.6, ls='--', zorder=0)
    ax.set_yticks([])
    ax.set_xlim(-148, 96)
    ax.set_xticks([-125, -100, -75, -50, -25, 0])
    ax.set_xlabel('ΔELPD vs primary (nats)')
    ax.spines['left'].set_visible(False)
    ax.set_ylim(-0.9, len(rows) - 0.1)
    ax.text(-146, len(rows) - 1.15, 'Open marker: chains not converged',
            fontsize=6.2, color='0.45', ha='left', va='top')


def panel_existence(ax, elpd):
    rows = [(n, a, b, c) for n, a, b, c in EXISTENCE if a in elpd and b in elpd]
    y = np.arange(len(rows))[::-1]
    for yi, (name, a, b, col) in zip(y, rows):
        d, dse = contrast(elpd, a, b)
        ax.errorbar(d, yi, xerr=dse, color=col, lw=1.0, capsize=0, zorder=2)
        ax.plot(d, yi, 'o', ms=5, mfc=col, mec=col, zorder=3)
        ax.text(d, yi + 0.28, f'{d / dse:.0f} dSE', fontsize=6.2,
                ha='center', va='bottom', color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.axvline(0, color='0.7', lw=0.6, ls='--', zorder=0)
    ax.set_xlim(-8, 145)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel('ΔELPD, cTBS effect vs matched null')
    ax.set_ylim(-0.7, len(rows) - 0.3)


def panel_trap(ax, elpd, meta):
    for lab, col in TRAP:
        if lab not in elpd:
            continue
        m = meta[lab]
        div = max(m['divergences'], 0.7)          # 0 divergences -> plot at the floor
        ax.plot(div, elpd[lab].sum(), 'o', ms=5.5, mfc=col, mec='white', mew=0.6,
                zorder=3, alpha=.95)
    ax.axvline(200, color='0.7', lw=0.6, ls='--', zorder=0)
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['0', '10', '100', '1000'])
    ax.set_xlabel('Divergent transitions (of 20,000)')
    ax.set_ylabel('ELPD (nats)')
    ax.set_xlim(0.45, 5200)
    ax.set_ylim(-4246, -4138)
    ax.annotate('Better fit, but\nthe chains failed',
                xy=(2283, -4161.0), xytext=(60, -4142),
                fontsize=6.5, color=C_NAT, ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', color=C_NAT, lw=1.0,
                                mutation_scale=8, shrinkA=3, shrinkB=8,
                                relpos=(0.5, 0.0),
                                connectionstyle='angle3,angleA=0,angleB=70'))
    ax.annotate('Same model,\ncoordinates that sample',
                xy=(10, -4184.6), xytext=(1.05, -4210),
                fontsize=6.5, color=C_NAT, ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', color=C_NAT, lw=1.0,
                                mutation_scale=8, shrinkA=3, shrinkB=8,
                                relpos=(0.5, 1.0),
                                connectionstyle='angle3,angleA=0,angleB=-70'))
    ax.text(1.15, -4146.5, 'Log-space', fontsize=6.5, color=C_LOG,
            ha='left', va='center')


def panel_prior(ax, elpd):
    rows = [(n, a, b, c) for n, a, b, c in PRIOR if a in elpd and b in elpd]
    x = np.arange(len(rows))
    for xi, (name, a, b, col) in zip(x, rows):
        d, dse = contrast(elpd, a, b)
        ax.bar(xi, d, width=.6, color=col, alpha=.85, lw=0)
        ax.errorbar(xi, d, yerr=dse, color='0.15', lw=0.9, capsize=0, zorder=3)
        ax.text(xi, d + 42, f'{d / dse:.0f} dSE', ha='center', va='bottom',
                fontsize=6.2, color='0.15')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylabel('ELPD cost of removing\nthe prior (nats)')
    ax.set_ylim(0, 1000)
    ax.set_yticks([0, 250, 500, 750])
    ax.set_xlim(-0.6, len(rows) - 0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ploo_dir', default='notes/data/ploo')
    ap.add_argument('--reference', default='lfx2-bs3-m2-dp-bm')
    ap.add_argument('--out', default='notes/figures/model_ladder')
    args = ap.parse_args()

    elpd, meta = load(args.ploo_dir)
    print(f'{len(elpd)} traces, identical {N_OBS} observations')

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    panel_ladder(axes[0, 0], elpd, meta, args.reference)
    panel_existence(axes[0, 1], elpd)
    panel_trap(axes[1, 0], elpd, meta)
    panel_prior(axes[1, 1], elpd)

    import seaborn as sns
    sns.despine(fig=fig, offset=4, trim=False)
    axes[0, 0].spines['left'].set_visible(False)

    for ax, letter in zip(axes.ravel(), 'abcd'):
        ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=8,
                family='Arial', fontweight='bold', va='bottom', ha='right')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {args.out}.pdf / .png')


if __name__ == '__main__':
    main()
