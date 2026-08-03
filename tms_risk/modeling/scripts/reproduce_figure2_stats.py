"""Reproduce the five nPRF statistics in the paper's Figure-2 paragraph.

All five reproduce to 3-4 decimals. The recipe is not obvious and is easy to get wrong,
which is why it lives here rather than in a notebook:

    tree      derivatives/encoding_model.denoise.smoothed        (mu, sd, amplitude, r2)
              derivatives/encoding_model.cv.denoise.smoothed     (cvr2)
              -- the OLD log-space tree. Commit `ba58cb1` switched
              `analyze_encoding_model.ipynb` to `get_prf_parameters(model_label=1)`,
              under which mu / sd / r2 / cvr2 are session-invariant BY CONSTRUCTION,
              so four of these five statistics cannot be computed there at all.
    roi       NPCr2cm-cluster (the 2 cm stimulation cluster)
    mask      cvR2 > 0 in EITHER arm, i.e. `(cvr2.unstack(arm) > 0).any(axis=1)`
              -- notebook cell 7. Keeping a voxel only where ITS OWN session passes
              unbalances the two arms and none of the numbers come out.
    aggregate per-subject MEAN over voxels, then a paired t-test across 35 subjects
              (the median gives different descriptives; see below)

Published values (v8) and what this script returns:

    [1] amplitude          1.3015 -> 1.0416   t(34) = 1.9924   p1 = 0.027
    [2] preferred numer.   t(34) = 1.0069, p2 = 0.32   (see the descriptive caveat)
    [3] dispersion         0.7733 -> 0.9083   t(34) = 1.2780   p2 = 0.21
    [4] explained variance 0.0681 -> 0.0493   t(34) = 2.0644   p1 = 0.023
    [5] prop. cvR2 > 0     0.1113 -> 0.0752   t(34) = 1.9893   p1 = 0.027   (no mask)

Two caveats that survive the reproduction, both already in `notes/v8_stats_check.md`:

  * The manuscript's natural-space preferred numerosities (14.8 / 17.8) are the median
    across subjects of the per-subject MEAN of exp(mu); this script's mean-of-log
    aggregation gives 10.10 / 11.28. The t and p match exactly either way -- only the
    descriptive differs.
  * For BOTH mu and sd the larger value is IPS, not vertex. In a paragraph whose
    convention is vertex -> parietal, "from 0.9 to 0.77" reads as a decrease when the
    test statistic is positive, i.e. a (non-significant) increase.

    python -m tms_risk.modeling.scripts.reproduce_figure2_stats --out_tsv notes/data/figure2_stats.tsv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from scipy import stats

from tms_risk.utils.data import get_all_behavior

warnings.filterwarnings('ignore')

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31, 34, 35, 36,
            37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
PUBLISHED = {
    'amplitude': dict(vertex=1.3015, ips=1.0416, t=1.9924, p1=0.027),
    'mu': dict(t=1.0069, p2=0.3211),
    'sd': dict(vertex=0.7733, ips=0.9083, t=1.2780, p2=0.2099),
    'r2': dict(vertex=0.0681, ips=0.0493, t=2.0644, p1=0.023),
    'prop_cvr2_pos': dict(vertex=0.1113, ips=0.0752, t=1.9893, p1=0.027),
}


def collect(bids_folder, roi, subjects):
    deriv = Path(bids_folder) / 'derivatives'
    par_dir, cv_dir = (deriv / 'encoding_model.denoise.smoothed',
                       deriv / 'encoding_model.cv.denoise.smoothed')
    beh = get_all_behavior(bids_folder=bids_folder).reset_index()
    arm = {(int(r.subject), int(r.session)): r.stimulation_condition
           for r in beh[['subject', 'session', 'stimulation_condition']]
           .drop_duplicates().itertuples()}

    rows = []
    for s in subjects:
        sid = f'{s:02d}'
        mask_fn = (deriv / 'ips_masks' / f'sub-{sid}' / 'func' / 'ses-1'
                   / f'sub-{sid}_space-T1w_desc-{roi}_mask.nii.gz')
        if not mask_fn.exists():
            continue
        masker = NiftiMasker(mask_img=str(mask_fn))
        masker.fit()
        for ses in (2, 3):
            a = arm.get((s, ses))
            if a not in ('ips', 'vertex'):
                continue
            vals, ok = {}, True
            for par, root in [('mu', par_dir), ('sd', par_dir), ('amplitude', par_dir),
                              ('r2', par_dir), ('cvr2', cv_dir)]:
                fn = (root / f'sub-{sid}' / f'ses-{ses}' / 'func'
                      / f'sub-{sid}_ses-{ses}_desc-{par}.optim_space-T1w_pars.nii.gz')
                if not fn.exists():
                    ok = False
                    break
                try:
                    vals[par] = masker.transform(str(fn)).squeeze()
                except Exception:
                    ok = False
                    break
            if ok:
                rows.append(pd.DataFrame({'subject': s, 'arm': a, **vals}))
    d = pd.concat(rows, ignore_index=True)
    d['vox'] = d.groupby(['subject', 'arm']).cumcount()
    return d


def notebook_mask(d):
    """Cell 7: keep a voxel where cvR2 > 0 in EITHER arm, so the pairing survives."""
    w = d.pivot_table(index=['subject', 'vox'], columns='arm', values='cvr2')
    keep = (w > 0).any(axis=1)
    return d.set_index(['subject', 'vox']).loc[keep[keep].index].reset_index()


def paired(x, col, agg='mean'):
    ps = x.pivot_table(index='subject', columns='arm', values=col, aggfunc=agg).dropna()
    t, p = stats.ttest_rel(ps['ips'], ps['vertex'])
    # Descriptives are the MEDIAN across subjects of the per-subject mean -- that is
    # what the manuscript quotes (1.3015 / 1.0416 etc.). The test is on the means.
    return dict(vertex=ps['vertex'].median(), ips=ps['ips'].median(), t=t, p2=p,
                p1=p / 2, n=len(ps))


def main(bids_folder, roi, out_tsv, subjects, dump_voxels=None):
    d = collect(bids_folder, roi, subjects)
    if dump_voxels:
        Path(dump_voxels).parent.mkdir(parents=True, exist_ok=True)
        out = d.copy()
        out['in_mask'] = out.set_index(['subject', 'vox']).index.isin(
            notebook_mask(d).set_index(['subject', 'vox']).index)
        out.to_csv(dump_voxels, sep='\t', index=False)
        print(f'wrote {dump_voxels}  ({len(out)} voxel-sessions)')
    thr = notebook_mask(d)
    print(f'{roi}: {len(d)} voxel-sessions, {d.subject.nunique()} subjects; '
          f'notebook mask keeps {len(thr)}')

    rows = []
    for par, label in [('amplitude', '[1] amplitude'), ('mu', '[2] preferred numerosity'),
                       ('sd', '[3] dispersion (log sd)'),
                       ('r2', '[4] explained variance')]:
        r = paired(thr, par)
        rows.append(dict(statistic=label, parameter=par, **r))
    # [5] uses NO mask -- it is a property of all voxels in the ROI
    prop = (d.assign(pos=d.cvr2 > 0)
              .pivot_table(index='subject', columns='arm', values='pos', aggfunc='mean')
              .dropna())
    t, p = stats.ttest_rel(prop['ips'], prop['vertex'])
    rows.append(dict(statistic='[5] proportion cvR2 > 0', parameter='prop_cvr2_pos',
                     vertex=prop['vertex'].mean(), ips=prop['ips'].mean(),  # [5] is a mean
                     t=t, p2=p, p1=p / 2, n=len(prop)))

    out = pd.DataFrame(rows)
    for _, r in out.iterrows():
        pub = PUBLISHED.get(r.parameter, {})
        flag = ''
        if 't' in pub:
            flag = ' OK' if abs(abs(r.t) - pub['t']) < 0.01 else ' MISMATCH'
        print(f'  {r.statistic:28s} vertex {r.vertex:8.4f} -> ips {r.ips:8.4f}  '
              f't({int(r.n)-1}) = {r.t:+.4f}  p2 = {r.p2:.4f}  p1 = {r.p1:.4f}{flag}')
    # natural-space descriptive for mu, both aggregations
    w = thr.pivot_table(index='subject', columns='arm', values='mu', aggfunc='mean')
    print(f'  preferred numerosity, natural space (mean of log, then exp): '
          f'vertex {np.exp(w["vertex"]).median():.2f} -> ips {np.exp(w["ips"]).median():.2f}')
    thr2 = thr.assign(mu_nat=np.exp(thr.mu))
    w2 = thr2.pivot_table(index='subject', columns='arm', values='mu_nat', aggfunc='mean')
    print(f'  preferred numerosity, natural space (mean of exp)          : '
          f'vertex {w2["vertex"].median():.2f} -> ips {w2["ips"].median():.2f}'
          '   <- the manuscript\'s 14.8 / 17.8')

    if out_tsv:
        Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_tsv, sep='\t', index=False)
        print(f'wrote {out_tsv}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--roi', default='NPCr2cm-cluster')
    p.add_argument('--out_tsv', default='notes/data/figure2_stats.tsv')
    p.add_argument('--subjects', default='')
    p.add_argument('--dump_voxels', default=None,
                   help='also write the per-voxel old-tree table, for plot_figure2.py')
    a = p.parse_args()
    subs = [int(x) for x in a.subjects.split(',')] if a.subjects else SUBJECTS
    main(a.bids_folder, a.roi, a.out_tsv, subs, a.dump_voxels)
