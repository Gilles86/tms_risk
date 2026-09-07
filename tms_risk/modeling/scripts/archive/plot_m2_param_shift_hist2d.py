"""Per-voxel 2D histograms: how do m2 PRF parameters shift IPS vs Vertex?

The companion ``plot_m2_tms_param_shifts.py`` collapses each subject to one
(IPS, Vertex) mean per parameter. Here we keep every signal voxel: for each
TMS subject the IPS-stimulation and Vertex-sham sessions share the same voxel
grid (same NPC12r mask), so each voxel has a paired (IPS, Vertex) value for
every parameter. Pooling those voxels across subjects, we draw a 2D histogram
per parameter (IPS on x, Vertex on y) with the identity line — mass off the
diagonal is the IPS-vs-Vertex shift, resolved at the voxel level.

Signal voxels = cvR² > 0 (session-agnostic in the regression PRF, so a single
threshold applies to both sessions). Restricted to NPC12r.

Caches the pooled per-voxel table to TSV so the figure can be replotted
locally (`--from_tsv`) without reloading raw PRF NIfTIs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from tms_risk.utils.data import Subject, get_all_behavior
from tms_risk.modeling.scripts.plot_spherical_expected_uncertainty import apply_style


SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31,
            34, 35, 36, 37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
ROI = 'NPC12r'
PARAMS = ['mu', 'sd', 'amplitude', 'baseline']
PAR_TITLE = {'mu': 'μ (log n)', 'sd': 'σ', 'amplitude': 'Amplitude',
             'baseline': 'Baseline'}


def collect(bids_folder='/data/ds-tmsrisk', model_label=2):
    """Long table: one row per (subject, voxel), columns = each param's IPS
    and Vertex value. Signal voxels only (cvR² > 0)."""
    beh = get_all_behavior(bids_folder=bids_folder)
    cond_map = (beh.reset_index()[['subject', 'session', 'stimulation_condition']]
                    .drop_duplicates())

    frames = []
    for sid in SUBJECTS:
        sub = Subject(sid, bids_folder=bids_folder)
        cond = cond_map[cond_map.subject == sid]
        ses_of = {}
        for c in ('ips', 'vertex'):
            hit = cond[cond.stimulation_condition == c]['session'].values
            if len(hit):
                ses_of[c] = int(hit[0])
        if {'ips', 'vertex'} - ses_of.keys():
            continue

        per_cond = {}
        ok = True
        for c, ses in ses_of.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pars = sub.get_prf_parameters(model_label=model_label,
                                                  session=ses, roi=ROI)
            except FileNotFoundError:
                ok = False
                break
            if any(p not in pars.columns for p in PARAMS) or 'cvr2' not in pars:
                ok = False
                break
            per_cond[c] = pars
        if not ok:
            continue

        ips, vtx = per_cond['ips'], per_cond['vertex']
        # cvR² is session-agnostic; both copies are equal. Signal = cvR² > 0.
        signal = (ips['cvr2'] > 0) & (vtx['cvr2'] > 0)
        if signal.sum() == 0:
            continue
        df = pd.DataFrame({'subject': sid}, index=ips.index[signal])
        for p in PARAMS:
            df[f'{p}_ips'] = ips.loc[signal, p].values
            df[f'{p}_vertex'] = vtx.loc[signal, p].values
        frames.append(df.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)


def plot(df, model_label=2, out='notes/figures/m2_param_shift_hist2d'):
    apply_style()
    n = len(PARAMS)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4),
                             constrained_layout=True)

    for ax, p in zip(axes, PARAMS):
        x = df[f'{p}_ips'].values
        y = df[f'{p}_vertex'].values
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        # Common robust limits (1–99th pct over both axes) so the identity
        # line is the visual reference.
        lo, hi = np.percentile(np.concatenate([x, y]), [1, 99])
        bins = np.linspace(lo, hi, 60)
        h = ax.hist2d(x, y, bins=[bins, bins], norm=LogNorm(), cmap='mako')
        ax.plot([lo, hi], [lo, hi], color='#d62728', lw=1.0, ls='--',
                zorder=3)  # identity
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel('IPS')
        ax.set_ylabel('Vertex' if ax is axes[0] else '')
        # Paired voxel-level shift: mean (Vertex − IPS) and % below diagonal
        d = y - x
        frac_below = float(np.mean(y < x))
        ax.set_title(f"{PAR_TITLE[p]}\nΔ(V−IPS) = {d.mean():+.3f}  ·  "
                     f"{frac_below*100:.0f}% below",
                     fontsize=9)
        cb = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)

    n_vox = len(df)
    n_sub = df['subject'].nunique()
    fig.suptitle(f'm{model_label} per-voxel PRF parameters: IPS vs Vertex '
                 f'({ROI} signal voxels, n={n_sub} subjects, {n_vox} voxels)',
                 fontsize=11, y=1.06)
    fig.text(0.5, -0.04,
             'Each panel pools all signal voxels (cvR² > 0) across subjects  ·  '
             'red dashed = identity (no shift)  ·  colour = voxel count (log)',
             ha='center', va='top', fontsize=8, color='0.4')

    out_root = Path(out)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_root.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_root.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f'wrote {out_root}.{{pdf,png}}')


def main(model_label=2, bids_folder='/data/ds-tmsrisk', from_tsv=False):
    tsv = Path(f'notes/data/m{model_label}_param_shift_voxels.tsv')
    if from_tsv:
        df = pd.read_csv(tsv, sep='\t')
    else:
        df = collect(bids_folder=bids_folder, model_label=model_label)
        tsv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tsv, sep='\t', index=False)
        print(f'wrote {tsv}  ({len(df)} voxels)')
    plot(df, model_label=model_label,
         out=f'notes/figures/m{model_label}_param_shift_hist2d')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_label', type=int, default=2)
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--from_tsv', action='store_true',
                   help='Replot from the cached per-voxel TSV')
    args = p.parse_args()
    main(model_label=args.model_label, bids_folder=args.bids_folder,
         from_tsv=args.from_tsv)
