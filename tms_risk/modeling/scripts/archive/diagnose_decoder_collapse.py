"""Diagnostic: is the decoder collapsing to the grid mean?

For each (subject, true stimulus, condition) we have `mean_E` — the average
posterior mean of the decoded magnitude across `n_simulations` forward
draws of the encoding model. If the decoder works, mean_E ≈ true. If
the decoder is noise-dominated, mean_E ≈ grid_mean (uniform-prior
attractor), regardless of true. The expected-uncertainty figure's
"minimum near n=58" only makes sense in the second regime — at n=58
the grid mean happens to coincide with the truth.

This diagnostic plots mean_E vs true magnitude per subject, with the
identity line (y=x) and the grid mean as reference. If subject curves
hug y=x → decoder has signal. If they hug the horizontal at the grid
mean → confirmed collapse.
"""
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tms_risk.utils.data import get_all_behavior
from tms_risk.modeling.scripts.plot_spherical_expected_uncertainty import (
    SPHERICAL_ROOT, COLOR, N_MIN, N_MAX, X_TICKS, apply_style,
    load_mc_decode, join_stimulation_condition,
)

GRID_MIN, GRID_MAX = 7, 28 * 4   # the stimulus_range used in monte_carlo_decode
GRID_MEAN = (GRID_MIN + GRID_MAX) / 2


def main(n_voxels: int = 100):
    apply_style()

    mc = load_mc_decode()
    mc = mc[mc['n_voxels'] == n_voxels]
    mc = join_stimulation_condition(mc)
    mc = mc[(mc['value'] >= N_MIN) & (mc['value'] <= N_MAX)].copy()

    n_ips = mc[mc.stimulation_condition == 'ips']['subject'].nunique()
    n_vtx = mc[mc.stimulation_condition == 'vertex']['subject'].nunique()
    print(f'mc: {len(mc):,} rows · IPS subj={n_ips} · Vertex subj={n_vtx}')

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.4), sharey=True,
                              constrained_layout=True)

    for ax, cond in zip(axes, ('ips', 'vertex')):
        sub = mc[mc.stimulation_condition == cond]
        c = COLOR[cond]

        # Identity (perfect decoding) and grid-mean attractor (collapse).
        x_grid = np.array([N_MIN, N_MAX], dtype=float)
        ax.plot(x_grid, x_grid, ls='--', color='0.3', lw=0.8, zorder=2)
        ax.axhline(GRID_MEAN, ls=':', color='0.5', lw=0.7, zorder=1)

        # Per-subject curves (thin, translucent)
        for subj, sd in sub.groupby('subject'):
            sd = sd.sort_values('value')
            ax.plot(sd['value'], sd['mean_E'], color=c, lw=0.6, alpha=0.30, zorder=3)

        # Group mean
        agg = sub.groupby('value')['mean_E'].agg(['mean', 'sem']).reset_index()
        ax.fill_between(agg['value'], agg['mean'] - agg['sem'],
                         agg['mean'] + agg['sem'], color=c, alpha=0.25,
                         linewidth=0, zorder=4)
        ax.plot(agg['value'], agg['mean'], color=c, lw=1.6, zorder=5)

        # Annotations
        ax.annotate('Identity (y = x)\n— perfect decoding',
                     xy=(20, 20), xytext=(20, 12),
                     textcoords='data', ha='left', va='top',
                     fontsize=7, color='0.3',
                     arrowprops=dict(arrowstyle='-', color='0.5', lw=0.5,
                                      connectionstyle='arc3,rad=-0.2'))
        ax.annotate(f'Grid mean ({GRID_MEAN:.0f})\n— collapse attractor',
                     xy=(70, GRID_MEAN), xytext=(60, GRID_MEAN + 10),
                     textcoords='data', ha='right', va='bottom',
                     fontsize=7, color='0.5',
                     arrowprops=dict(arrowstyle='-', color='0.6', lw=0.5,
                                      connectionstyle='arc3,rad=0.2'))

        ax.set_xlabel('True magnitude (n)')
        if ax is axes[0]:
            ax.set_ylabel('Decoded mean (mean_E)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(X_TICKS)
        ax.set_yticks(X_TICKS)
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlim(N_MIN, N_MAX)
        ax.set_ylim(N_MIN, N_MAX)
        ax.set_aspect('equal')

        # Title with condition + subject count
        n_subj = sub['subject'].nunique()
        ax.set_title(f"{cond.capitalize()}-TMS  (n = {n_subj} subjects)",
                      fontsize=9, color=c)
        sns.despine(ax=ax, offset=3, trim=True)

    # Panel letters
    for ax, letter in zip(axes, 'AB'):
        ax.text(-0.18, 1.05, letter, transform=ax.transAxes,
                 fontsize=11, fontweight='bold', va='bottom', ha='right')

    fig.suptitle('Diagnostic: decoded mean vs true magnitude — is the decoder collapsing?',
                  fontsize=10, y=1.04)
    fig.text(
        0.5, -0.04,
        f'n_voxels = {n_voxels}  ·  ROI = NPC12r  ·  '
        f'each thin line = one subject  ·  bold line + band = group mean ± SEM',
        ha='center', va='top', fontsize=7.5, color='0.4',
    )

    out = Path('notes/figures/decoder_collapse_diagnostic')
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=200)
    print(f'wrote {out}.{{pdf,png}}')

    # Quick numeric summary: regression slope of mean_E on true magnitude
    # within each subject. Slope ≈ 1 = perfect. Slope ≈ 0 = total collapse.
    slopes = []
    for (cond, subj), sub in mc.groupby(['stimulation_condition', 'subject']):
        x = sub['value'].values; y = sub['mean_E'].values
        slope = np.polyfit(np.log(x), np.log(y), 1)[0]   # log-slope
        slopes.append({'condition': cond, 'subject': subj, 'slope': slope})
    slopes = pd.DataFrame(slopes)
    print('\nLog-slope of decoded mean on true magnitude per subject:')
    print(slopes.groupby('condition')['slope'].describe().round(3))


if __name__ == '__main__':
    main()
