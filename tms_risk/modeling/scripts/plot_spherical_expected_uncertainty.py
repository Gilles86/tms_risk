"""Plot the spherical-Ω expected uncertainty across the number line, IPS vs vertex.

Reads spherical Monte-Carlo decode TSVs from
``derivatives/monte_carlo_decode.denoise.spherical/sub-*/ses-*/func/*.tsv``,
joins each (subject, session) to its TMS condition via the behavior
data, and plots:

1. Decoded SD vs true magnitude, mean ± SEM per condition (IPS / vertex).
2. The IPS − vertex difference curve, showing where in number space
   parietal TMS hurts decoding acuity most relative to the vertex sham.
3. (Optional 3rd panel) the decoder bias curve, since spherical-Ω
   shrinkage toward the grid mean is still substantial.

Run::

    ~/mambaforge/envs/tms_risk_unified_test/bin/python -m \
        tms_risk.modeling.scripts.plot_spherical_expected_uncertainty
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tms_risk.utils.data import get_all_behavior
from tms_risk.behavior.utils import stimulation_palette


SPHERICAL_ROOT = Path(
    '/data/ds-tmsrisk/derivatives/monte_carlo_decode.denoise.spherical'
)
# Use the project's canonical IPS / vertex palette (seaborn default
# colors 2 and 3, alphabetical → ips=green, vertex=red), matching the
# PPC plots in tms_risk.behavior.utils.
COLOR = {'ips': stimulation_palette[0], 'vertex': stimulation_palette[1]}

# Most experimental stimuli are ≤ ~80 (q99 = 96, q95 = 68). The decoder
# evaluates over 7–111 but the tails are sparsely sampled and dominated
# by grid-mean shrinkage; cap the figure for readability.
N_MIN, N_MAX = 7, 80
# Natural-magnitude x-tick locations on the log axis.
X_TICKS = [7, 10, 15, 20, 30, 50, 80]


def load_mc_decode(root: Path = SPHERICAL_ROOT) -> pd.DataFrame:
    """Load every TSV in `root`, parse (subject, session, roi, n_voxels)
    from the filename and concat."""
    rows = []
    for tsv in root.glob('sub-*/ses-*/func/*_mc_decode.tsv'):
        m = re.match(
            r'sub-(\d+)_ses-(\d)_roi-([^_]+)_nvoxels-(\d+)_mc_decode',
            tsv.stem,
        )
        if not m:
            continue
        df = pd.read_csv(tsv, sep='\t')
        df['subject'] = int(m.group(1))
        df['session'] = int(m.group(2))
        df['roi'] = m.group(3)
        rows.append(df)
    if not rows:
        raise SystemExit(f'No TSVs found under {root}')
    out = pd.concat(rows, ignore_index=True)
    # Realised simulate-and-decode error per stimulus. `mean_abs_error` =
    # average |posterior_mean - true_value| across n_simulations forward
    # draws from the fitted model + residual covariance. This is the actual
    # decoding error we'd expect for a fresh trial, NOT the posterior width
    # the decoder claims (= sqrt(var_E)).
    out['expected_error'] = out['mean_abs_error']
    out['rmse'] = np.sqrt(out['var_E'] + out['mean_error']**2)
    out['bias'] = out['mean_error']
    return out


def join_stimulation_condition(
    mc: pd.DataFrame, bids_folder: str = '/data/ds-tmsrisk'
) -> pd.DataFrame:
    beh = get_all_behavior(bids_folder=bids_folder)
    cond = (beh.reset_index()[['subject', 'session', 'stimulation_condition']]
                .drop_duplicates())
    return mc.merge(cond, on=['subject', 'session'], how='left')


def main():
    mc = load_mc_decode()
    mc = join_stimulation_condition(mc)
    # The TMS contrast lives in sessions 2 & 3; session 1 is the baseline
    # PRF-mapping session and won't be in the mc_decode output anyway.
    mc = mc[mc['stimulation_condition'].isin(['ips', 'vertex'])].copy()
    print(f'mc: {len(mc):,} rows  |  '
          f"subjects: ips={mc[mc.stimulation_condition=='ips']['subject'].nunique()}, "
          f"vertex={mc[mc.stimulation_condition=='vertex']['subject'].nunique()}")

    # Cap to the experimentally-meaningful range (most trials ≤ 80).
    mc = mc[(mc['value'] >= N_MIN) & (mc['value'] <= N_MAX)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)

    def _style_x(ax):
        ax.set_xscale('log')
        ax.set_xticks(X_TICKS)
        ax.get_xaxis().set_major_formatter(
            plt.matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()

    # Panel 1 — realised decoding error vs true magnitude
    ax = axes[0]
    for cond, sub in mc.groupby('stimulation_condition'):
        agg = sub.groupby('value')['expected_error'].agg(['mean', 'sem']).reset_index()
        ax.fill_between(agg['value'], agg['mean'] - agg['sem'],
                        agg['mean'] + agg['sem'], alpha=0.25, color=COLOR[cond])
        ax.plot(agg['value'], agg['mean'], color=COLOR[cond], lw=2,
                label=cond.capitalize())
    ax.set_xlabel('True magnitude (n)')
    ax.set_ylabel('Expected decoding error\n(mean |decoded − true|, natural units)')
    ax.set_title('Realised decoding error\n(lower = sharper readout)')
    ax.legend(title='TMS condition', loc='upper left', frameon=False)
    _style_x(ax)

    # Panel 2 — IPS − vertex difference curve (paired within-subject)
    ax = axes[1]
    paired = mc.pivot_table(index=['subject', 'value'],
                             columns='stimulation_condition',
                             values='expected_error', aggfunc='mean').reset_index()
    paired['diff'] = paired['ips'] - paired['vertex']
    diff_agg = paired.groupby('value')['diff'].agg(['mean', 'sem']).reset_index()
    ax.fill_between(diff_agg['value'], diff_agg['mean'] - diff_agg['sem'],
                    diff_agg['mean'] + diff_agg['sem'], alpha=0.25, color='dimgray')
    ax.plot(diff_agg['value'], diff_agg['mean'], color='black', lw=2)
    ax.axhline(0, ls='--', color='k', alpha=0.4)
    ax.set_xlabel('True magnitude (n)')
    ax.set_ylabel('Expected error: IPS − Vertex')
    ax.set_title('TMS effect on decoding error\n(positive = IPS-TMS hurts)')
    _style_x(ax)

    # Panel 3 — bias (sanity check for grid-mean collapse)
    ax = axes[2]
    for cond, sub in mc.groupby('stimulation_condition'):
        agg = sub.groupby('value')['bias'].agg(['mean', 'sem']).reset_index()
        ax.fill_between(agg['value'], agg['mean'] - agg['sem'],
                        agg['mean'] + agg['sem'], alpha=0.25, color=COLOR[cond])
        ax.plot(agg['value'], agg['mean'], color=COLOR[cond], lw=2,
                label=cond.capitalize())
    ax.axhline(0, ls='--', color='k', alpha=0.4)
    ax.set_xlabel('True magnitude (n)')
    ax.set_ylabel('Decoded mean − true magnitude')
    ax.set_title('Decoder bias\n(>0 = pulled toward grid mean)')
    _style_x(ax)

    plt.suptitle(
        f'Spherical-Ω expected uncertainty across the number line — '
        f'NPC12r, n_voxels=100, n_simulations=1000\n'
        f"({mc[mc.stimulation_condition=='ips']['subject'].nunique()} IPS subjects · "
        f"{mc[mc.stimulation_condition=='vertex']['subject'].nunique()} Vertex subjects, "
        f'preliminary — sweep still running)',
        fontsize=12, y=1.05,
    )
    plt.tight_layout()
    out = Path('notes/figures/spherical_expected_uncertainty')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=130, bbox_inches='tight')
    print(f'wrote {out}.{{pdf,png}}')


if __name__ == '__main__':
    main()
