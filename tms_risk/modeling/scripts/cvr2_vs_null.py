"""What fraction of voxels does each encoding model actually beat the NULL on?

The pipeline's "non-noise" criterion is cvR2 > 0, but cvR2 = 0 is not the null: braincoder's
`get_rsq` puts the held-out fold's OWN mean in the denominator, so 0 means "as good as
already knowing the test fold's mean". A real null predictor only has the TRAINING mean,
and sits around -0.018 (see `null_cvr2.py`). This compares each model against that null
**voxel by voxel**, under the identical leave-one-run-out folds, which is the honest
version of the question.

    python -m tms_risk.modeling.scripts.cvr2_vs_null --roi NPCr2cm-cluster
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from tms_risk.utils.data import Subject

warnings.filterwarnings('ignore')

SUBS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31, 34, 35, 36,
        37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]


def rsq(data, pred):
    ssr = ((data - pred) ** 2).sum(0)
    sst = ((data - data.mean(0)) ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = 1 - ssr / sst
    out[sst == 0] = np.nan
    return out


def main(bids_folder, roi, models, cv_suffix, out_tsv):
    bids, deriv = Path(bids_folder), Path(bids_folder) / 'derivatives'
    rows = []
    for s in SUBS:
        sid = f'{s:02d}'
        try:
            sub = Subject(sid, bids_folder=bids)
            par = sub.get_paradigm().reset_index('session')[['log(n1)', 'session']]
            masker = NiftiMasker(mask_img=sub.get_volume_mask(session=1, roi=None,
                                                              epi_space=True))
            data = np.vstack([
                masker.fit_transform(str(
                    deriv / 'glm_stim1.denoise.smoothed' / f'sub-{sid}' / f'ses-{ses}'
                    / 'func' / f'sub-{sid}_ses-{ses}_task-task_space-T1w_desc-stims1_pe.nii.gz'))
                for ses in (2, 3)]).astype(np.float32)
            keep = masker.transform(sub.get_volume_mask(session=1, roi=roi,
                                                        epi_space=True)).squeeze() > 0
        except Exception as e:
            print(f'  sub-{sid}: {type(e).__name__}'); continue

        # per-voxel null, same folds the CV used
        runs = par.index.get_level_values('run')
        folds = []
        for r in sorted(np.unique(runs)):
            te, tr = runs == r, runs != r
            folds.append(rsq(data[te],
                             np.repeat(data[tr].mean(0)[None, :], te.sum(), axis=0)))
        null = np.nanmean(np.stack(folds), axis=0)[keep]

        row = dict(subject=s, n_voxels=int(keep.sum()), null_mean=np.nanmean(null))
        for m in models:
            fn = (deriv / f'encoding_model2.model-{m}.smoothed.cv{cv_suffix}'
                  / f'sub-{sid}' / f'sub-{sid}_desc-cvr2.optim_space-T1w_pars.nii.gz')
            if not fn.exists():
                continue
            mv = masker.transform(str(fn)).squeeze()[keep]
            ok = np.isfinite(mv) & np.isfinite(null)
            row[f'm{m}_mean'] = np.nanmean(mv[ok])
            row[f'm{m}_beats_null'] = np.mean(mv[ok] > null[ok])
            row[f'm{m}_gt0'] = np.mean(mv[ok] > 0)
        rows.append(row)
        print(f'  sub-{sid} done', flush=True)

    R = pd.DataFrame(rows)
    tag = cv_suffix if cv_suffix else '(none -- whatever is in .cv on THIS machine)'
    print(f'\n=== {roi}, cv dir suffix {tag}, n = {len(R)} subjects ===')
    print(f'  null cvR2 (mean over voxels, then subjects): {R.null_mean.mean():+.5f}')
    print(f'\n  {"model":6s} {"mean cvR2":>10s} {"% beating NULL":>16s} {"% cvR2 > 0":>12s}')
    for m in models:
        if f'm{m}_mean' not in R:
            print(f'  m{m:<5d} {"— not on disk":>10s}'); continue
        c = R[[f'm{m}_mean', f'm{m}_beats_null', f'm{m}_gt0']].dropna()
        print(f'  m{m:<5d} {c[f"m{m}_mean"].mean():+10.5f} '
              f'{100*c[f"m{m}_beats_null"].mean():15.1f}% {100*c[f"m{m}_gt0"].mean():11.1f}%'
              f'   (n={len(c)})')
    if out_tsv:
        R.to_csv(out_tsv, sep='\t', index=False)
        print(f'\nwrote {out_tsv}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--roi', default='NPCr2cm-cluster')
    p.add_argument('--models', default='0,1,2')
    p.add_argument('--cv_suffix', default='',
                   help='e.g. ".iter10" for the preserved under-converged fits')
    p.add_argument('--out_tsv', default='notes/data/cvr2_vs_null.tsv')
    a = p.parse_args()
    main(a.bids_folder, a.roi, [int(x) for x in a.models.split(',')], a.cv_suffix,
         a.out_tsv)
