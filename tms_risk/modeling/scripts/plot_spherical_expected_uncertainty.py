"""Plot the spherical-Ω expected-uncertainty across the number line — IPS vs vertex.

Reads spherical Monte-Carlo decode TSVs from
``derivatives/monte_carlo_decode.denoise.spherical/sub-*/ses-*/func/*.tsv``,
joins each (subject, session) to its TMS condition via the behavior
data, and plots:

1. Realised decoding error (mean |decoded − true|) vs true magnitude
   per condition (IPS = red, Vertex = green).
2. The IPS − Vertex paired-difference curve with a Maris-Oostenveld
   cluster-based permutation test (sign-flip null, two-sided, 1000
   permutations) marking magnitudes where parietal-cTBS measurably
   degrades decoding precision.
3. Decoder bias per condition (sanity check for grid-mean shrinkage).

Style follows the **scientific-figures** house guide: Helvetica, spines
offset/trimmed, ticks outward, direct condition labels, no legend frame,
PDF + SVG output with editable fonts.
"""
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from tms_risk.utils.data import get_all_behavior


# ── Paths and constants ─────────────────────────────────────────────────────
SPHERICAL_ROOT = Path(
    '/data/ds-tmsrisk/derivatives/monte_carlo_decode.denoise.spherical'
)
DEFAULT_N_VOXELS = 100      # primary figure uses n=100 (50 / 250 = robustness)

# IPS = stimulated (parietal cTBS) → red
# Vertex = sham / unstimulated control → green
# NOT what the alphabetical mapping of stimulation_palette gives you, so we
# hardcode here (see notes/memory: feedback_ips_vertex_palette).
COLOR = {'ips': '#d62728', 'vertex': '#2ca02c'}

# Cap to the experimentally meaningful range; q99 of stimuli is ~96 but the
# tails are dominated by grid-mean shrinkage and noise.
N_MIN, N_MAX = 7, 80
X_TICKS = [7, 10, 15, 20, 30, 50, 80]


# ── Scientific-figures rcParams ─────────────────────────────────────────────
def apply_style():
    mpl.rcParams.update({
        'font.family':       'Helvetica',
        'font.sans-serif':   ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
        'font.size':         9,
        'axes.labelsize':    10,
        'axes.titlesize':    10,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   8,
        'mathtext.fontset':  'stixsans',
        'axes.linewidth':    0.8,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.labelpad':     4,
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'xtick.major.size':  3,
        'ytick.major.size':  3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth':   1.2,
        'lines.markersize':  4,
        'patch.linewidth':   0.5,
        'legend.frameon':    False,
        'legend.handlelength': 1.5,
        'pdf.fonttype':      42,
        'ps.fonttype':       42,
        'svg.fonttype':      'none',
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.02,
    })
    sns.set_context('paper')


# ── Loading + joining ───────────────────────────────────────────────────────
def load_mc_decode(root: Path = SPHERICAL_ROOT) -> pd.DataFrame:
    rows = []
    for tsv in root.glob('sub-*/ses-*/func/*_mc_decode.tsv'):
        m = re.match(
            r'sub-(\d+)_ses-(\d)_roi-([^_]+)_nvoxels-(\d+)_mc_decode',
            tsv.stem,
        )
        if not m:
            continue
        df = pd.read_csv(tsv, sep='\t')
        df['subject']  = int(m.group(1))
        df['session']  = int(m.group(2))
        df['roi']      = m.group(3)
        df['n_voxels'] = int(m.group(4))
        rows.append(df)
    if not rows:
        raise SystemExit(f'No TSVs found under {root}')
    out = pd.concat(rows, ignore_index=True)
    # The user's headline metric is the *empirical variance of the
    # decoded point estimate across forward draws* (var_E in
    # get_expected_uncertainty's output). This is the precision of the
    # point estimate, independent of bias. mean_abs_error keeps the bias
    # term and inherits the grid-mean-collapse artefact.
    # `expected_variance` = empirical Var(decoded_mean) across n_simulations.
    # `expected_abs_error`  = mean |decoded − true|; kept for the diagnostic
    # but NOT used in the headline figure.
    out['expected_variance']   = out['var_E']
    out['expected_abs_error']  = out['mean_abs_error']
    out['bias']              = out['mean_error']
    return out


def join_stimulation_condition(
    mc: pd.DataFrame, bids_folder: str = '/data/ds-tmsrisk'
) -> pd.DataFrame:
    beh = get_all_behavior(bids_folder=bids_folder)
    cond = (beh.reset_index()[['subject', 'session', 'stimulation_condition']]
                .drop_duplicates())
    out = mc.merge(cond, on=['subject', 'session'], how='left')
    return out[out['stimulation_condition'].isin(['ips', 'vertex'])].copy()


# ── Cluster-based permutation test (Maris-Oostenveld 2007) ──────────────────
def cluster_perm_test(
    paired: pd.DataFrame,
    n_perm: int = 1000,
    alpha_cluster: float = 0.05,
    rng_seed: int = 0,
):
    """Sign-flip cluster permutation test on a per-subject paired-difference
    curve ``ips − vertex`` over a stimulus grid.

    ``paired`` must have one row per (subject, value) with columns 'ips',
    'vertex', and 'diff' (= ips − vertex). Subjects with both conditions
    available at the same set of stimulus values are kept.

    Returns
    -------
    dict with keys:
        - ``stimuli``       (n_x,)   stimulus grid
        - ``t_obs``         (n_x,)   observed t(x) over subjects
        - ``t_thresh``      scalar   cluster-forming threshold (|t|)
        - ``clusters``      list[(start_idx, end_idx_inclusive, sum_t, p)]
        - ``null_max_mass`` (n_perm,) max |cluster sum| under sign-flip H0
        - ``cluster_alpha`` α level used for the cluster forming step
    """
    # Pivot to (subject × stimulus) array of within-subject diffs
    wide = paired.pivot(index='subject', columns='value', values='diff')
    wide = wide.dropna(axis=0, how='any')     # subjects with full coverage only
    stimuli = wide.columns.values.astype(float)
    D = wide.values                            # (n_subj, n_x)
    n_subj, n_x = D.shape
    if n_subj < 2:
        return None

    # Cluster-forming threshold = two-sided t critical at p < alpha_cluster
    t_thresh = stats.t.ppf(1 - alpha_cluster / 2, df=n_subj - 1)

    def t_curve(diffs):
        m = diffs.mean(axis=0)
        s = diffs.std(axis=0, ddof=1) / np.sqrt(diffs.shape[0])
        # Avoid div-by-zero where SEM == 0 (constant diff across subjects).
        return np.where(s > 0, m / s, 0.0)

    def find_clusters(t_vec, thresh):
        """Return list of (start, end_inclusive, signed_sum) for contiguous
        runs where |t| > thresh, signed by the cluster's sign."""
        out = []
        above = np.abs(t_vec) > thresh
        if not above.any():
            return out
        # Identify contiguous runs
        runs = np.diff(np.concatenate([[False], above, [False]]).astype(int))
        starts = np.where(runs ==  1)[0]
        ends   = np.where(runs == -1)[0] - 1
        for s, e in zip(starts, ends):
            signed_sum = float(t_vec[s:e + 1].sum())
            out.append((int(s), int(e), signed_sum))
        return out

    t_obs    = t_curve(D)
    obs_clusters = find_clusters(t_obs, t_thresh)

    # Permutations: random sign flips per subject
    rng = np.random.default_rng(rng_seed)
    null_max_mass = np.zeros(n_perm)
    for k in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=n_subj)[:, None]
        t_perm = t_curve(D * flips)
        perm_clusters = find_clusters(t_perm, t_thresh)
        null_max_mass[k] = max((abs(m) for _, _, m in perm_clusters), default=0.0)

    # P-value per observed cluster = fraction of perms with max-|mass| ≥ |obs|
    clusters_out = []
    for s, e, m in obs_clusters:
        p = float((null_max_mass >= abs(m)).mean())
        clusters_out.append({'start': s, 'end': e, 'mass': m, 'p': p})

    return {
        'stimuli':        stimuli,
        't_obs':          t_obs,
        't_thresh':       t_thresh,
        'clusters':       clusters_out,
        'null_max_mass':  null_max_mass,
        'cluster_alpha':  alpha_cluster,
        'n_subj':         n_subj,
    }


# ── Plot ────────────────────────────────────────────────────────────────────
def _style_x(ax, ticks=X_TICKS, x_min=N_MIN, x_max=N_MAX):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xlim(x_min, x_max)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()


def main(n_voxels: int = DEFAULT_N_VOXELS, n_perm: int = 1000):
    apply_style()

    mc = load_mc_decode()
    mc = mc[mc['n_voxels'] == n_voxels]
    mc = join_stimulation_condition(mc)
    mc = mc[(mc['value'] >= N_MIN) & (mc['value'] <= N_MAX)].copy()

    n_ips = mc[mc.stimulation_condition == 'ips']['subject'].nunique()
    n_vtx = mc[mc.stimulation_condition == 'vertex']['subject'].nunique()
    print(f'mc: {len(mc):,} rows · n_voxels={n_voxels} · IPS subj={n_ips} · Vertex subj={n_vtx}')

    # Per-subject × stimulus paired diff
    paired = mc.pivot_table(
        index=['subject', 'value'], columns='stimulation_condition',
        values='expected_variance', aggfunc='mean',
    ).reset_index()
    paired['diff'] = paired['ips'] - paired['vertex']
    n_paired = paired.dropna(subset=['ips', 'vertex'])['subject'].nunique()
    print(f'paired subjects (both IPS + Vertex sessions): {n_paired}')

    perm = cluster_perm_test(paired.dropna(subset=['ips', 'vertex']), n_perm=n_perm)

    # ── Figure: 3-panel row, double-column width ──
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.6), constrained_layout=True)
    ax1, ax2, ax3 = axes

    # PANEL 1 — Realised decoding error per condition. Direct-label both
    # curves at small magnitudes (where IPS sits clearly above Vertex)
    # rather than at the right endpoint (where the curves converge into
    # the grid-mean attractor and labels overlap).
    label_x = 10
    for cond in ('ips', 'vertex'):
        sub = mc[mc.stimulation_condition == cond]
        agg = sub.groupby('value')['expected_variance'].agg(['mean', 'sem']).reset_index()
        ax1.fill_between(agg['value'], agg['mean'] - agg['sem'],
                         agg['mean'] + agg['sem'], alpha=0.20, color=COLOR[cond],
                         linewidth=0)
        ax1.plot(agg['value'], agg['mean'], color=COLOR[cond], lw=1.4, zorder=3)
        # Tag both curves with their condition name at the same x and a
        # small vertical offset, so they're never on top of each other.
        y_at_label = agg.loc[agg['value'].sub(label_x).abs().idxmin(), 'mean']
        dy = 4 if cond == 'ips' else -10
        ax1.annotate(cond.capitalize(),
                     xy=(label_x, y_at_label), xytext=(0, dy),
                     textcoords='offset points', color=COLOR[cond],
                     fontsize=9, ha='center',
                     fontweight='bold' if cond == 'ips' else 'normal')
    ax1.set_xlabel('True magnitude (n)')
    ax1.set_ylabel('Expected variance of\ndecoded estimate (natural²)')
    _style_x(ax1)
    sns.despine(ax=ax1, offset=5, trim=True)

    # PANEL 2 — IPS − Vertex difference with cluster permutation
    diff_agg = paired.dropna(subset=['ips', 'vertex']).groupby('value')['diff'].agg(['mean', 'sem']).reset_index()
    ax2.axhline(0, ls='--', color='0.6', lw=0.6, zorder=0)
    ax2.fill_between(diff_agg['value'], diff_agg['mean'] - diff_agg['sem'],
                     diff_agg['mean'] + diff_agg['sem'], alpha=0.25,
                     color='0.4', linewidth=0)
    ax2.plot(diff_agg['value'], diff_agg['mean'], color='black', lw=1.4, zorder=3)
    # Highlight significant clusters
    if perm is not None:
        for cl in perm['clusters']:
            x0, x1 = perm['stimuli'][cl['start']], perm['stimuli'][cl['end']]
            inside = (diff_agg['value'] >= x0) & (diff_agg['value'] <= x1)
            color = '#d62728' if cl['mass'] > 0 else '#3B5BA5'
            ax2.fill_between(diff_agg.loc[inside, 'value'],
                             diff_agg.loc[inside, 'mean'] - diff_agg.loc[inside, 'sem'],
                             diff_agg.loc[inside, 'mean'] + diff_agg.loc[inside, 'sem'],
                             alpha=0.55, color=color, linewidth=0, zorder=2)
            if cl['p'] < 0.05:
                # Place a marker above/below the cluster
                y_top = (diff_agg.loc[inside, 'mean'] + diff_agg.loc[inside, 'sem']).max()
                ax2.annotate(f'p = {cl["p"]:.3f}',
                             xy=(np.sqrt(x0 * x1), y_top), xytext=(0, 8),
                             textcoords='offset points', ha='center',
                             fontsize=7, color=color, fontweight='bold')
        # Subtle in-panel note on the test
        ax2.text(0.02, 0.98,
                 f'Cluster perm. test\n|t| > {perm["t_thresh"]:.2f}, n_perm = {n_perm}',
                 transform=ax2.transAxes, ha='left', va='top', fontsize=6.5,
                 color='0.4')
    ax2.set_xlabel('True magnitude (n)')
    ax2.set_ylabel('Expected variance\n(IPS − Vertex)')
    _style_x(ax2)
    sns.despine(ax=ax2, offset=5, trim=True)

    # PANEL 3 — Decoder bias (sanity check)
    for cond in ('ips', 'vertex'):
        sub = mc[mc.stimulation_condition == cond]
        agg = sub.groupby('value')['bias'].agg(['mean', 'sem']).reset_index()
        ax3.fill_between(agg['value'], agg['mean'] - agg['sem'],
                         agg['mean'] + agg['sem'], alpha=0.20, color=COLOR[cond],
                         linewidth=0)
        ax3.plot(agg['value'], agg['mean'], color=COLOR[cond], lw=1.4, zorder=3)
    ax3.axhline(0, ls='--', color='0.6', lw=0.6, zorder=0)
    ax3.set_xlabel('True magnitude (n)')
    ax3.set_ylabel('Decoder bias\n(decoded − true)')
    _style_x(ax3)
    sns.despine(ax=ax3, offset=5, trim=True)

    # Panel letters
    for ax, letter in zip(axes, 'ABC'):
        ax.text(-0.18, 1.05, letter, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom', ha='right')

    # Footer with subject counts
    fig.text(
        0.5, -0.05,
        f'n_voxels = {n_voxels}  ·  ROI = NPC12r  ·  '
        f'{n_paired} paired subjects (IPS + Vertex)  ·  '
        f'shaded bands = ±1 SEM across subjects',
        ha='center', va='top', fontsize=7.5, color='0.4',
    )

    out = Path('notes/figures/spherical_expected_uncertainty')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=200)
    fig.savefig(out.with_suffix('.svg'))
    print(f'wrote {out}.{{pdf,png,svg}}')

    # Dump cluster table for the caption
    if perm is not None:
        print('\nCluster permutation results:')
        for cl in perm['clusters']:
            x0, x1 = perm['stimuli'][cl['start']], perm['stimuli'][cl['end']]
            print(f"  n={int(x0)}..{int(x1)}  mass={cl['mass']:+.2f}  p={cl['p']:.3f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n_voxels', type=int, default=DEFAULT_N_VOXELS)
    p.add_argument('--n_perm', type=int, default=1000)
    args = p.parse_args()
    main(n_voxels=args.n_voxels, n_perm=args.n_perm)
